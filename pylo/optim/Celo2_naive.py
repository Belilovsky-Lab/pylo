"""Celo2_naive: PyTorch implementation of the CELO2 learned optimizer.

Paper: "Celo2: Towards Learned Optimization Free Lunch"
       https://arxiv.org/abs/2602.19142

Ported from the JAX/Optax implementation in snippets/celo2/celo2_optax.py.

This implements the full CELO2 variant:
  - Learned MLP update rule for 2D hidden-layer parameters
  - Newton-Schulz orthogonalization on 2D parameter updates
  - Inline AdamW for 1D params, embeddings, and input/output layers
"""

import os
from typing import Optional

import numpy as np
import torch
from torch.optim import Optimizer

from pylo.models.Celo2_MLP import Celo2MLP
from pylo.util.newton_schulz import orthogonalize_newton_schulz


def factored_dims(shape):
    """Whether to use a factored second moment estimator.

    Returns the two largest dimension indices for shapes with ndim >= 2, else None.
    """
    if len(shape) < 2:
        return None
    sorted_dims = np.argsort(shape)
    return int(sorted_dims[-2]), int(sorted_dims[-1])


def safe_rsqrt(x):
    return torch.rsqrt(torch.maximum(x, torch.tensor(1e-9, dtype=x.dtype, device=x.device)))


def second_moment_normalizer(x, axis, eps=1e-9):
    rms = torch.mean(x ** 2, dim=axis, keepdim=True)
    return x * torch.rsqrt(eps + rms)


def _classify_params_list(named_params):
    """Classify parameters as AdamW or CELO2.

    Returns a dict mapping parameter index to bool (True = AdamW).

    Uses AdamW for:
    - 1D params (biases, layernorms, etc.)
    - Embedding layers (name contains 'embed')
    - First 2D+ parameter (input layer)
    - Last 2D+ parameter (output layer)
    """
    result = {}
    # Find first and last 2D+ param indices
    indices_2d = [i for i, (name, p) in enumerate(named_params) if p.ndim >= 2]

    for i, (name, p) in enumerate(named_params):
        if p.ndim <= 1:
            result[i] = True
        elif "embed" in name.lower():
            result[i] = True
        elif indices_2d and (i == indices_2d[0] or i == indices_2d[-1]):
            result[i] = True
        else:
            result[i] = False

    return result


def _is_adamw_param(name, idx, total_params, p):
    """Simple check for a single param (used in tests)."""
    if p.ndim <= 1:
        return True
    if "embed" in name.lower():
        return True
    return False


class Celo2_naive(Optimizer):
    """CELO2 learned optimizer (naive/CPU implementation).

    For 2D hidden-layer parameters: uses a learned MLP to compute update
    directions, followed by Newton-Schulz orthogonalization.

    For 1D params, embeddings, and input/output layers: uses inline AdamW.

    Args:
        params: Model parameters (can be parameter groups or an iterable).
        lr: Learning rate.
        weight_decay: Weight decay coefficient.
        momentum_decays: Decay rates for momentum accumulators.
        rms_decays: Decay rates for RMS accumulators.
        adafactor_decays: Decay rates for factored accumulators.
        exp_mult: Magnitude exponential multiplier (0.0 in default CELO2).
        rmsmult: Output scaling multiplier.
        ns_coeffs: Newton-Schulz iteration coefficients.
        ns_iters: Number of Newton-Schulz iterations.
        adam_betas: Beta1, Beta2 for the AdamW branch.
        adam_eps: Epsilon for AdamW branch.
        checkpoint_path: Path to converted .pt checkpoint.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        momentum_decays=(0.9, 0.99, 0.999),
        rms_decays=(0.95,),
        adafactor_decays=(0.9, 0.99, 0.999),
        exp_mult: float = 0.0,
        rmsmult: float = 1.0,
        ns_coeffs=(3.4445, -4.7750, 2.0315),
        ns_iters: int = 5,
        adam_betas=(0.9, 0.95),
        adam_eps: float = 1e-8,
        checkpoint_path: Optional[str] = None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
        )

        # Collect named params to determine adamw vs celo2
        param_list = list(params)

        # Handle param groups vs flat list
        if len(param_list) > 0 and isinstance(param_list[0], dict):
            # Already param groups — user is responsible for grouping
            super().__init__(param_list, defaults)
        else:
            super().__init__([{"params": param_list}], defaults)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Store config
        self.momentum_decays = torch.tensor(momentum_decays, dtype=torch.float32)
        self.rms_decays = torch.tensor(rms_decays, dtype=torch.float32)
        self.adafactor_decays = torch.tensor(adafactor_decays, dtype=torch.float32)
        self.exp_mult = exp_mult
        self.rmsmult = rmsmult
        self.ns_coeffs = torch.tensor(ns_coeffs, dtype=torch.float32)
        self.ns_iters = ns_iters
        self.adam_betas = adam_betas
        self.adam_eps = adam_eps

        # Load MLP
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "models", "celo2_weights.pt",
            )
        self.network = Celo2MLP.from_pretrained_file(checkpoint_path).to(self.device)
        for p in self.network.parameters():
            p.requires_grad = False
        self.network.eval()

    def _classify_params(self):
        """Build a mapping from param id to whether it uses adamw."""
        if hasattr(self, "_param_is_adamw"):
            return self._param_is_adamw

        # Fallback: use index-based heuristics with generic names
        all_params = []
        for group in self.param_groups:
            for p in group["params"]:
                all_params.append(p)

        named = [(f"param_{i}", p) for i, p in enumerate(all_params)]
        classification = _classify_params_list(named)

        self._param_is_adamw = {}
        for idx, p in enumerate(all_params):
            self._param_is_adamw[id(p)] = classification[idx]

        return self._param_is_adamw

    def set_param_names(self, model):
        """Set parameter names from a model for better input/output layer detection.

        Call this after creating the optimizer if you want name-based filtering:
            optimizer = Celo2_naive(model.parameters())
            optimizer.set_param_names(model)
        """
        # Build ordered list of (name, param) matching optimizer param order
        param_id_to_name = {}
        for name, p in model.named_parameters():
            param_id_to_name[id(p)] = name

        all_params = []
        for group in self.param_groups:
            for p in group["params"]:
                all_params.append(p)

        named = [(param_id_to_name.get(id(p), f"param_{i}"), p)
                 for i, p in enumerate(all_params)]
        classification = _classify_params_list(named)

        self._param_is_adamw = {}
        for idx, p in enumerate(all_params):
            self._param_is_adamw[id(p)] = classification[idx]

    def _init_celo2_state(self, p):
        """Initialize CELO2 accumulator state for a parameter."""
        state = {}
        p_shape = p.shape
        device = p.device
        n_mom = len(self.momentum_decays)
        n_rms = len(self.rms_decays)
        n_fac = len(self.adafactor_decays)

        # Momentum: shape + (n_mom,)
        state["mom"] = torch.zeros(p_shape + (n_mom,), dtype=torch.float32, device=device)

        # RMS: shape + (n_rms,)
        state["rms"] = torch.zeros(p_shape + (n_rms,), dtype=torch.float32, device=device)

        # Factored accumulators
        f_dims = factored_dims(p_shape)
        if f_dims is not None:
            d1, d0 = f_dims
            # v_row: remove d0 dimension, add n_fac
            vr_shape = tuple(s for i, s in enumerate(p_shape) if i != d0) + (n_fac,)
            vc_shape = tuple(s for i, s in enumerate(p_shape) if i != d1) + (n_fac,)
            state["fac_vec_row"] = torch.zeros(vr_shape, dtype=torch.float32, device=device)
            state["fac_vec_col"] = torch.zeros(vc_shape, dtype=torch.float32, device=device)
            state["fac_vec_v"] = torch.tensor([], dtype=torch.float32, device=device)
        else:
            state["fac_vec_row"] = torch.tensor([], dtype=torch.float32, device=device)
            state["fac_vec_col"] = torch.tensor([], dtype=torch.float32, device=device)
            state["fac_vec_v"] = torch.zeros(p_shape + (n_fac,), dtype=torch.float32, device=device)

        state["step"] = 0
        return state

    def _init_adamw_state(self, p):
        """Initialize AdamW state for a parameter."""
        return {
            "exp_avg": torch.zeros_like(p),
            "exp_avg_sq": torch.zeros_like(p),
            "step": 0,
        }

    def _update_factored(self, state, grad, p_shape):
        """Update factored accumulators and return factored-normalized gradient."""
        device = grad.device
        n_fac = len(self.adafactor_decays)
        f_dims = factored_dims(p_shape)
        epsilon = 1e-30

        fac_vec_row = state["fac_vec_row"]
        fac_vec_col = state["fac_vec_col"]
        fac_vec_v = state["fac_vec_v"]

        # Expand gradient with decay dimension
        g_expanded = grad.unsqueeze(-1).expand(p_shape + (n_fac,))
        grad_sqr = g_expanded * g_expanded + epsilon

        if f_dims is not None:
            d1, d0 = f_dims
            decay = self.adafactor_decays.to(device)
            mixing = 1.0 - decay

            # Reshape decay for broadcasting
            decay_shape = [1] * (len(p_shape) + 1)  # +1 for the fac dim
            decay_shape[-1] = n_fac
            decay = decay.view(decay_shape)
            mixing = mixing.view(decay_shape)

            new_v_row = decay.squeeze(d0) * fac_vec_row + mixing.squeeze(d0) * torch.mean(grad_sqr, dim=d0)
            new_v_col = decay.squeeze(d1) * fac_vec_col + mixing.squeeze(d1) * torch.mean(grad_sqr, dim=d1)

            reduced_d1 = d1 - 1 if d1 > d0 else d1
            row_col_mean = torch.mean(new_v_row, dim=reduced_d1, keepdim=True)
            row_factor = safe_rsqrt(new_v_row / (row_col_mean + 1e-9))
            col_factor = safe_rsqrt(new_v_col)

            fac_g = g_expanded * row_factor.unsqueeze(d0) * col_factor.unsqueeze(d1)

            state["fac_vec_row"] = new_v_row
            state["fac_vec_col"] = new_v_col
            return fac_g
        else:
            decay = self.adafactor_decays.to(device)
            mixing = 1.0 - decay

            decay_shape = [1] * (len(p_shape) + 1)
            decay_shape[-1] = n_fac
            decay = decay.view(decay_shape)
            mixing = mixing.view(decay_shape)

            new_v = decay * fac_vec_v + mixing * grad_sqr
            fac_g = g_expanded * safe_rsqrt(new_v + 1e-9)

            state["fac_vec_v"] = new_v
            return fac_g

    def _build_input_groups(self, p, grad, state, p_shape):
        """Build the 14 input feature groups for the MLP."""
        device = p.device
        mom = state["mom"]
        rms = state["rms"]
        fac_g = state["_fac_g"]  # computed in _update_factored

        rsqrt_rms = torch.rsqrt(rms + 1e-8)

        inps = []
        inps.append(grad.unsqueeze(-1))                              # 0: g
        inps.append(torch.clamp(grad, -0.1, 0.1).unsqueeze(-1))     # 1: clip_g
        inps.append(p.unsqueeze(-1))                                 # 2: p
        inps.append(mom)                                              # 3: m
        inps.append(rms)                                              # 4: rms
        inps.append(mom * rsqrt_rms)                                  # 5: m*rsqrt
        inps.append(rsqrt_rms)                                        # 6: rsqrt
        inps.append(fac_g)                                            # 7: fac_g
        inps.append(grad.unsqueeze(-1) * rsqrt_rms)                  # 8: g*rsqrt

        # Factored features
        f_dims = factored_dims(p_shape)
        if f_dims is not None:
            d1, d0 = f_dims
            fac_vec_row = state["fac_vec_row"]
            fac_vec_col = state["fac_vec_col"]

            rp_row = [1] * (len(p_shape) + 1)
            rp_row[d0] = p_shape[d0]
            row_feat = fac_vec_row.unsqueeze(d0).repeat(rp_row)

            rp_col = [1] * (len(p_shape) + 1)
            rp_col[d1] = p_shape[d1]
            col_feat = fac_vec_col.unsqueeze(d1).repeat(rp_col)

            inps.append(row_feat)                                     # 9: row_feat
            inps.append(col_feat)                                     # 10: col_feat
            inps.append(torch.rsqrt(row_feat + 1e-8))               # 11: rsqrt_row
            inps.append(torch.rsqrt(col_feat + 1e-8))               # 12: rsqrt_col

            # fac_mom_mult
            reduced_d1 = d1 - 1 if d1 > d0 else d1
            row_col_mean = torch.mean(fac_vec_row, dim=reduced_d1, keepdim=True)
            row_factor = safe_rsqrt(fac_vec_row / (row_col_mean + 1e-9))
            col_factor = safe_rsqrt(fac_vec_col)
            fac_mom_mult = mom * row_factor.unsqueeze(d0) * col_factor.unsqueeze(d1)
            inps.append(fac_mom_mult)                                 # 13: fac_mom_mult
        else:
            fac_vec_v = state["fac_vec_v"]
            inps.append(fac_vec_v)                                    # 9: v_full (as row_feat)
            inps.append(fac_vec_v)                                    # 10: v_full (as col_feat)
            inps.append(torch.rsqrt(fac_vec_v + 1e-8))              # 11: rsqrt_v
            inps.append(torch.rsqrt(fac_vec_v + 1e-8))              # 12: rsqrt_v
            fac_mom_mult = mom * torch.pow(fac_vec_v + 1e-6, -0.5)
            inps.append(fac_mom_mult)                                 # 13: fac_mom_mult

        return inps

    @torch.no_grad()
    def step(self, loss=None):
        param_is_adamw = self._classify_params()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                is_adamw = param_is_adamw.get(id(p), p.ndim <= 1)

                if is_adamw:
                    self._adamw_step(p, grad, state, lr, weight_decay)
                else:
                    self._celo2_step(p, grad, state, lr, weight_decay)

    def _adamw_step(self, p, grad, state, lr, weight_decay):
        """Inline AdamW update."""
        if len(state) == 0:
            state.update(self._init_adamw_state(p))

        state["step"] += 1
        step = state["step"]
        beta1, beta2 = self.adam_betas
        eps = self.adam_eps

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]

        # Decoupled weight decay
        if weight_decay > 0:
            p.mul_(1 - lr * weight_decay)

        # Update biased moments
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # Bias correction
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        step_size = lr / bias_correction1
        denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

        p.addcdiv_(exp_avg, denom, value=-step_size)

    def _celo2_step(self, p, grad, state, lr, weight_decay):
        """CELO2 learned optimizer update."""
        if len(state) == 0:
            state.update(self._init_celo2_state(p))

        state["step"] += 1
        device = p.device
        p_shape = p.shape

        # Clip gradients
        grad = torch.clamp(grad, -1000.0, 1000.0)

        # Update momentum accumulators: mom = decay * mom + (1 - decay) * grad
        mom_decay = self.momentum_decays.to(device)
        decay_shape = [1] * len(p_shape) + [len(mom_decay)]
        mom_decay_bc = mom_decay.view(decay_shape)
        state["mom"].mul_(mom_decay_bc).add_((1 - mom_decay_bc) * grad.unsqueeze(-1))

        # Update RMS accumulators: rms = decay * rms + (1 - decay) * grad^2
        rms_decay = self.rms_decays.to(device)
        rms_decay_bc = rms_decay.view([1] * len(p_shape) + [len(rms_decay)])
        state["rms"].mul_(rms_decay_bc).add_((1 - rms_decay_bc) * (grad.unsqueeze(-1) ** 2))

        # Update factored accumulators
        fac_g = self._update_factored(state, grad, p_shape)
        state["_fac_g"] = fac_g

        # Build input features
        input_groups = self._build_input_groups(p, grad, state, p_shape)

        # Normalize inputs (second moment normalization across spatial dims)
        axis = list(range(len(p_shape)))
        if len(axis) >= 2:
            norm_axis = axis[-2:]
        else:
            norm_axis = axis
        input_groups = [second_moment_normalizer(inp, axis=norm_axis) for inp in input_groups]

        # MLP forward pass
        mlp_out = self.network([inp.to(self.network.first_layer.bias.device) for inp in input_groups])
        mlp_out = mlp_out.to(device)

        direction = mlp_out[..., 0]
        magnitude_param = mlp_out[..., 1]

        # Compute step
        step = direction * torch.exp(magnitude_param * self.exp_mult)

        # Newton-Schulz orthogonalization for 2D+ params
        if len(p_shape) >= 2:
            step = orthogonalize_newton_schulz(
                step, self.ns_coeffs.to(device), self.ns_iters
            )

        # Output normalization
        step = second_moment_normalizer(step, axis=norm_axis)

        # Scale
        step = step * self.rmsmult

        # Ensure shape matches parameter
        step = step.reshape(p_shape)

        # Apply update
        p.add_(step, alpha=-lr)

        # Weight decay (decoupled, applied after learned step)
        if weight_decay > 0:
            p.add_(p, alpha=-lr * weight_decay)

        # Clean up temporary state
        del state["_fac_g"]
