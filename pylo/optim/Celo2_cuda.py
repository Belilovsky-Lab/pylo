"""Celo2_cuda: CUDA-accelerated CELO2 learned optimizer.

The CUDA kernel handles feature construction, second-moment normalization,
and MLP inference per-element. Newton-Schulz orthogonalization, output
normalization, weight decay, and the AdamW branch remain in Python.
"""

import os
from typing import Optional

import numpy as np
import torch
from torch.optim import Optimizer

import celo2_cuda_kernel

from pylo.models.Celo2_MLP import Celo2MLP
from pylo.util.newton_schulz import orthogonalize_newton_schulz


def factored_dims(shape):
    """Return the two largest dimension indices for shapes with ndim >= 2, else None."""
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
    """Classify parameters as AdamW or CELO2."""
    result = {}
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


class Celo2_cuda(Optimizer):
    """CELO2 learned optimizer (CUDA-accelerated).

    For 2D hidden-layer parameters: CUDA kernel computes per-element MLP step,
    then Python applies Newton-Schulz orthogonalization and normalization.

    For 1D params, embeddings, and input/output layers: uses inline AdamW (Python).
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
        defaults = dict(lr=lr, weight_decay=weight_decay)

        param_list = list(params)
        if len(param_list) > 0 and isinstance(param_list[0], dict):
            super().__init__(param_list, defaults)
        else:
            super().__init__([{"params": param_list}], defaults)

        self.device = torch.device("cuda")

        # Store config
        self.momentum_decays = torch.tensor(momentum_decays, dtype=torch.float32, device=self.device)
        self.rms_decays = torch.tensor(rms_decays, dtype=torch.float32, device=self.device)
        self.adafactor_decays = torch.tensor(adafactor_decays, dtype=torch.float32, device=self.device)
        self.exp_mult = exp_mult
        self.rmsmult = rmsmult
        self.ns_coeffs = torch.tensor(ns_coeffs, dtype=torch.float32, device=self.device)
        self.ns_iters = ns_iters
        self.adam_betas = adam_betas
        self.adam_eps = adam_eps

        # Second-moment buffer for the CUDA kernel (reused across params)
        self.second_moment = torch.zeros(30, dtype=torch.float32, device=self.device)

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
        if hasattr(self, "_param_is_adamw"):
            return self._param_is_adamw

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
        """Set parameter names from a model for better input/output layer detection."""
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

    # ── State initialization ─────────────────────────────────────────────────

    def _init_celo2_state(self, p):
        """Initialize CELO2 accumulator state for a parameter.

        Unlike the naive version which uses trailing decay dimensions,
        the CUDA kernel expects leading decay dimensions: (n_decays,) + p.shape.
        """
        state = {}
        p_shape = p.shape
        device = p.device
        n_mom = len(self.momentum_decays)
        n_rms = len(self.rms_decays)
        n_fac = len(self.adafactor_decays)

        # Momentum: (n_mom,) + p.shape — leading decay dim for contiguous access in kernel
        state["mom"] = torch.zeros((n_mom,) + p_shape, dtype=torch.float32, device=device)

        # RMS: p.shape (single decay rate, no extra dim needed)
        state["rms"] = torch.zeros(p_shape, dtype=torch.float32, device=device)

        # Factored accumulators: (n_fac,) + reduced_shape
        f_dims = factored_dims(p_shape)
        if f_dims is not None:
            d1, d0 = f_dims
            vr_shape = list(p_shape)
            vr_shape[d0] = 1
            vr_shape = [n_fac] + vr_shape
            vc_shape = list(p_shape)
            vc_shape[d1] = 1
            vc_shape = [n_fac] + vc_shape
            state["fac_vec_row"] = torch.zeros(vr_shape, dtype=torch.float32, device=device)
            state["fac_vec_col"] = torch.zeros(vc_shape, dtype=torch.float32, device=device)
            state["fac_vec_v"] = torch.tensor([], dtype=torch.float32, device=device)
        else:
            state["fac_vec_row"] = torch.tensor([], dtype=torch.float32, device=device)
            state["fac_vec_col"] = torch.tensor([], dtype=torch.float32, device=device)
            state["fac_vec_v"] = torch.zeros((n_fac,) + p_shape, dtype=torch.float32, device=device)

        state["step"] = 0
        return state

    def _init_adamw_state(self, p):
        return {
            "exp_avg": torch.zeros_like(p),
            "exp_avg_sq": torch.zeros_like(p),
            "step": 0,
        }

    # ── Accumulator updates (Python, before kernel call) ─────────────────────

    def _update_accumulators(self, state, grad, p_shape):
        """Update momentum, RMS, and factored accumulators in-place.

        Uses the same lerp_ pattern as VeLO CUDA for clean broadcasting
        with leading decay dimensions: (n_decays,) + p_shape.
        """
        device = grad.device
        epsilon = 1e-30

        # Momentum: (n_mom,) + p.shape — lerp towards grad[None, ...]
        mom_decay = self.momentum_decays.to(device).view(-1, *[1] * len(p_shape))
        state["mom"].lerp_(grad[None, ...], (1 - mom_decay).to(grad.dtype))

        # RMS: p.shape — single decay rate
        rms_decay = self.rms_decays[0].item()
        state["rms"].lerp_(grad ** 2, 1 - rms_decay)

        # Factored accumulators: (n_fac,) + reduced_shape
        f_dims = factored_dims(p_shape)
        fac_decay = self.adafactor_decays.to(device).view(-1, *[1] * len(p_shape))
        grad_sqr = grad * grad + epsilon

        if f_dims is not None:
            d1, d0 = f_dims
            # fac_vec_row: d0 reduced (set to 1), shape (3, ..., 1, ...)
            state["fac_vec_row"].lerp_(
                grad_sqr.mean(dim=d0, keepdim=True)[None, ...],
                (1 - fac_decay).to(state["fac_vec_row"].dtype),
            )
            # fac_vec_col: d1 reduced (set to 1), shape (3, 1, ..., ...)
            state["fac_vec_col"].lerp_(
                grad_sqr.mean(dim=d1, keepdim=True)[None, ...],
                (1 - fac_decay).to(state["fac_vec_col"].dtype),
            )
        else:
            state["fac_vec_v"].lerp_(
                grad_sqr[None, ...],
                (1 - fac_decay).to(state["fac_vec_v"].dtype),
            )

    def _compute_factors(self, state, p_shape):
        """Compute row_factor and col_factor from factored accumulators."""
        f_dims = factored_dims(p_shape)

        if f_dims is not None:
            d1, d0 = f_dims
            fac_vec_row = state["fac_vec_row"]
            fac_vec_col = state["fac_vec_col"]

            reduced_d1 = (d1 + 1) - 1 if (d1 + 1) > (d0 + 1) else (d1 + 1)
            row_col_mean = torch.mean(fac_vec_row, dim=reduced_d1, keepdim=True)
            row_factor = safe_rsqrt(fac_vec_row / (row_col_mean + 1e-9))
            col_factor = safe_rsqrt(fac_vec_col)
            dc, dr = d1, d0
            vector_like = 0
        else:
            fac_vec_row = state["fac_vec_v"]
            fac_vec_col = state["fac_vec_v"]
            row_factor = safe_rsqrt(fac_vec_row + 1e-9)
            col_factor = torch.ones_like(row_factor)
            dc, dr = 0, 0
            vector_like = 1

        return row_factor, col_factor, fac_vec_row, fac_vec_col, dc, dr, vector_like

    # ── Step methods ─────────────────────────────────────────────────────────

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
        """Inline AdamW update (same as naive)."""
        if len(state) == 0:
            state.update(self._init_adamw_state(p))

        state["step"] += 1
        step = state["step"]
        beta1, beta2 = self.adam_betas
        eps = self.adam_eps

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]

        if weight_decay > 0:
            p.mul_(1 - lr * weight_decay)

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        step_size = lr / bias_correction1
        denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

        p.addcdiv_(exp_avg, denom, value=-step_size)

    def _celo2_step(self, p, grad, state, lr, weight_decay):
        """CELO2 learned optimizer update using CUDA kernel."""
        if len(state) == 0:
            state.update(self._init_celo2_state(p))

        state["step"] += 1
        p_shape = p.shape

        # Clip gradients
        grad = torch.clamp(grad, -1000.0, 1000.0)

        # Update accumulators (Python — same logic as naive)
        self._update_accumulators(state, grad, p_shape)

        # Compute row/col factors for the kernel
        row_factor, col_factor, fac_vec_row, fac_vec_col, dc, dr, vector_like = \
            self._compute_factors(state, p_shape)

        # Flatten momentum from (3, *p_shape) for kernel's contiguous access
        mom_flat = state["mom"]
        rms_flat = state["rms"]

        # Reset second-moment buffer
        self.second_moment.zero_()

        # Allocate step output
        step_out = torch.empty_like(p)

        # Extract MLP weights
        input_weights = self.network.first_layer.weight.to(grad.dtype)
        input_bias = self.network.first_layer.bias.to(grad.dtype)
        hidden_weights = self.network.hidden.weight.to(grad.dtype)
        hidden_bias = self.network.hidden.bias.to(grad.dtype)
        output_weights = self.network.output.weight.to(grad.dtype)
        output_bias = self.network.output.bias.to(grad.dtype)

        # Call CUDA kernel
        celo2_cuda_kernel.celo2_kernel(
            grad,
            p,
            mom_flat,
            rms_flat,
            row_factor,
            col_factor,
            fac_vec_row,
            fac_vec_col,
            self.second_moment,
            input_weights,
            input_bias,
            hidden_weights,
            hidden_bias,
            output_weights,
            output_bias,
            step_out,
            self.exp_mult,
            1e-8,       # epsilon
            dc,
            dr,
            vector_like,
        )

        step = step_out

        # Newton-Schulz orthogonalization for 2D+ params (Python)
        if len(p_shape) >= 2:
            step = orthogonalize_newton_schulz(
                step, self.ns_coeffs, self.ns_iters
            )

        # Output normalization
        axis = list(range(len(p_shape)))
        norm_axis = axis[-2:] if len(axis) >= 2 else axis
        step = second_moment_normalizer(step, axis=norm_axis)

        # Scale
        step = step * self.rmsmult

        # Ensure shape matches parameter
        step = step.reshape(p_shape)

        # Apply update
        p.add_(step, alpha=-lr)

        # Weight decay (decoupled)
        if weight_decay > 0:
            p.add_(p, alpha=-lr * weight_decay)
