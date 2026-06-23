"""CELO2_naive: a PyTorch port of the CELO2 learned optimizer.

This is a faithful, pure-PyTorch reimplementation of the inference-time forward
pass of the original JAX/optax CELO2 optimizer
(``scaling_l2o/src/learned_optimizers/celo2_optax.py``,
https://arxiv.org/abs/2602.19142).

The optimizer keeps, per 2D+ parameter, three families of accumulators:
  * momentum at several decays (default ``(0.9, 0.99, 0.999)``),
  * an RMS / second-moment accumulator (default ``(0.95,)``),
  * an Adafactor-style factored second moment (default ``(0.9, 0.99, 0.999)``).

For each 2D+ parameter it builds a stack of features, feeds them through the
:class:`~pylo.models.CELO2_MLP.CELO2MLP` (split-input MLP), optionally
orthogonalizes the resulting update via Newton-Schulz iteration, and normalizes
it. 1D parameters (biases, norms) and embeddings are updated with AdamW.

Unlike the original shared-accumulator design (still used by
:class:`pylo.optim.ELO_CELO2_naive.ELO_CELO2_naive`), the AdamW branch here keeps
its *own* ``exp_avg`` / ``exp_avg_sq`` moments, fully decoupled from the learned
optimizer's momentum/RMS accumulators. This makes the AdamW betas/eps independent
hyper-parameters and avoids allocating the (unused) CELO2 accumulators for 1D
parameters.

No learning-rate schedule is built in: ``lr`` is used as-is, so an external
``torch.optim.lr_scheduler`` can drive the schedule (warmup, cosine, etc.).
"""

from typing import Optional

import numpy as np
import torch
from torch.optim import Optimizer

from pylo.models.CELO2_MLP import CELO2MLP


# =============================================================================
# Shared math helpers (mirroring celo2_optax.py)
# =============================================================================

def factored_dims(shape):
    """The two largest dims used for the factored second moment, or None."""
    if len(shape) < 2:
        return None
    sorted_dims = np.argsort(shape)
    return int(sorted_dims[-2]), int(sorted_dims[-1])


def safe_rsqrt(x):
    return torch.rsqrt(torch.clamp(x, min=1e-9))


def second_moment_normalizer(x, axis, eps=1e-9):
    mean_squared = torch.mean(x ** 2, dim=axis, keepdim=True)
    return x * torch.rsqrt(eps + mean_squared)


def orthogonalize_via_newton_schulz(x, ns_coeffs, ns_steps=5, eps=1e-8):
    """Newton-Schulz orthogonalization over the last two dims of ``x``."""
    transposed = False
    if x.shape[-2] > x.shape[-1]:
        x = x.transpose(-2, -1)
        transposed = True
    x = x / (torch.linalg.norm(x, dim=(-2, -1), keepdim=True) + eps)
    c0, c1, c2 = ns_coeffs
    for _ in range(ns_steps):
        a = x @ x.transpose(-2, -1)
        b = c1 * a + c2 * (a @ a)
        x = c0 * x + b @ x
    if transposed:
        x = x.transpose(-2, -1)
    return x


class CELO2_naive(Optimizer):
    """Pure-PyTorch CELO2 learned optimizer.

    Args:
        params: Iterable of parameters or ``param_groups``. A group may carry an
            ``is_embedding=True`` flag to force its (2D) parameters onto the
            AdamW path, mirroring the ``'embed'`` routing of the JAX version.
        lr: Base learning rate for the CELO2 (2D+) path. No schedule is applied
            internally; drive any warmup/cosine schedule with an external
            ``torch.optim.lr_scheduler``.
        weight_decay: Decoupled weight decay for the CELO2 (2D+) path.
        adam_lr_mult, adam_weight_decay, adam_betas, adam_eps, use_adamw_for_1d:
            AdamW configuration for 1D / embedding parameters. The AdamW moments
            are maintained independently of the CELO2 accumulators.
            ``adam_weight_decay`` defaults to ``weight_decay`` when None.
        orthogonalize: Apply Newton-Schulz orthogonalization to 2D+ updates
            (set False for the "celo2-base" variant).
        clip_grad, clip_norm: Optional global-norm gradient clipping.
        ff_hidden_size, ff_hidden_layers, initial_momentum_decays,
            initial_rms_decays, initial_adafactor_decays, exp_mult, rmsmult,
            ns_coeffs, ns_iters, ns_eps: CELO2 model / accumulator configuration.
        grad_clip_val: Element-wise gradient clamp applied before preprocessing
            (matches ``celo2_optax`` which clamps to ``[-1000, 1000]``).
        hf_key: HuggingFace Hub id to load the CELO2MLP weights from.
        checkpoint_path: Local path to a converted CELO2MLP ``state_dict`` (.pt).
        network: An already-constructed :class:`CELO2MLP` (overrides the above).
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        # Weight decay
        weight_decay=0.0,
        # AdamW for 1D / embedding params (independent accumulators)
        adam_lr_mult=1.0,
        adam_weight_decay=None,
        adam_betas=(0.9, 0.95),
        adam_eps=1e-8,
        use_adamw_for_1d=True,
        # CELO2 backbone
        orthogonalize=True,
        clip_grad=False,
        clip_norm=1.0,
        ff_hidden_size=8,
        ff_hidden_layers=2,
        initial_momentum_decays=(0.9, 0.99, 0.999),
        initial_rms_decays=(0.95,),
        initial_adafactor_decays=(0.9, 0.99, 0.999),
        exp_mult=0.0,
        rmsmult=1.0,
        param_scale_mult=False,
        ns_coeffs=(3.4445, -4.7750, 2.0315),
        ns_iters=5,
        ns_eps=1e-8,
        grad_clip_val=1000.0,
        # Weights
        hf_key: Optional[str] = "DiamondXL/celo2",
        checkpoint_path: Optional[str] = None,
        network: Optional[CELO2MLP] = None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            adam_lr_mult=adam_lr_mult,
            adam_weight_decay=(
                adam_weight_decay if adam_weight_decay is not None else weight_decay
            ),
            use_adamw_for_1d=use_adamw_for_1d,
            orthogonalize=orthogonalize,
            clip_grad=clip_grad,
            clip_norm=clip_norm,
            exp_mult=exp_mult,
            rmsmult=rmsmult,
            param_scale_mult=param_scale_mult,
            grad_clip_val=grad_clip_val,
            is_embedding=False,
        )
        super().__init__(params, defaults)

        # Place state / meta-network on the same device as the parameters, so
        # CPU params on a GPU machine don't trigger device-mismatch errors.
        first_param = next(p for g in self.param_groups for p in g["params"])
        self.device = first_param.device.type

        self.initial_momentum_decays = torch.tensor(
            initial_momentum_decays, dtype=torch.float32, device=self.device
        )
        self.initial_rms_decays = torch.tensor(
            initial_rms_decays, dtype=torch.float32, device=self.device
        )
        self.initial_adafactor_decays = torch.tensor(
            initial_adafactor_decays, dtype=torch.float32, device=self.device
        )
        self.adam_beta1, self.adam_beta2 = (float(b) for b in adam_betas)
        self.adam_eps = adam_eps
        self.ns_coeffs = tuple(float(c) for c in ns_coeffs)
        self.ns_iters = ns_iters
        self.ns_eps = ns_eps

        # Precedence: explicit network > local checkpoint > HuggingFace Hub.
        if network is not None:
            self.network = network
        elif checkpoint_path is not None:
            self.network = CELO2MLP(
                hidden_size=ff_hidden_size, hidden_layers=ff_hidden_layers
            )
            self.network.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu")
            )
        elif hf_key is not None:
            self.network = CELO2MLP.from_pretrained(hf_key)
        else:
            self.network = CELO2MLP(
                hidden_size=ff_hidden_size, hidden_layers=ff_hidden_layers
            )
        self.network = self.network.to(self.device)

    # ---------------------------------------------------------------- helpers
    def _global_grad_scale(self, clip_norm):
        """optax.clip_by_global_norm scale factor over all parameter grads."""
        total = torch.zeros((), device=self.device)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    total = total + p.grad.detach().float().pow(2).sum()
        global_norm = torch.sqrt(total)
        denom = torch.clamp(global_norm, min=clip_norm)
        return (clip_norm / denom).item() if denom > 0 else 1.0

    def _init_celo2_state(self, state, p_shape):
        """Allocate the CELO2 momentum / RMS / factored accumulators."""
        n_mom = self.initial_momentum_decays.shape[-1]
        n_rms = self.initial_rms_decays.shape[-1]
        n_fac = self.initial_adafactor_decays.shape[-1]
        state["mom"] = torch.zeros(p_shape + (n_mom,), device=self.device)
        state["rms"] = torch.zeros(p_shape + (n_rms,), device=self.device)
        f_dims = factored_dims(p_shape)
        if f_dims is not None:
            d1, d0 = f_dims
            full = p_shape + (n_fac,)
            vr = tuple(d for i, d in enumerate(full) if i != d0)
            vc = tuple(d for i, d in enumerate(full) if i != d1)
            state["fac_vec_row"] = torch.zeros(vr, device=self.device)
            state["fac_vec_col"] = torch.zeros(vc, device=self.device)
            state["fac_vec_v"] = torch.empty(0, device=self.device)
        else:
            state["fac_vec_row"] = torch.empty(0, device=self.device)
            state["fac_vec_col"] = torch.empty(0, device=self.device)
            state["fac_vec_v"] = torch.zeros(p_shape + (n_fac,), device=self.device)

    def _update_accumulators(self, state, batch_g, p_shape):
        """Advance momentum, RMS and factored accumulators in place; return fac_g."""
        beta_m = self.initial_momentum_decays
        beta_rms = self.initial_rms_decays
        beta_fac = self.initial_adafactor_decays

        # momentum: m = decay * m + (1 - decay) * g
        mom = state["mom"]
        mom.mul_(beta_m).add_((1 - beta_m) * batch_g)
        # rms: rms = clip(decay) * rms + (1 - clip(decay)) * g^2
        crms = torch.clamp(beta_rms, 0.0, 1.0)
        rms = state["rms"]
        rms.mul_(crms).add_((1 - crms) * (batch_g ** 2))

        # factored Adafactor accumulator (over the adafactor decays)
        cdec = torch.clamp(beta_fac, 0.0, 1.0)
        mixing = 1.0 - cdec
        g_rep = batch_g.repeat([1] * len(p_shape) + [beta_fac.shape[-1]])
        grad_sqr = g_rep * g_rep + 1e-30
        f_dims = factored_dims(p_shape)
        if f_dims is not None:
            d1, d0 = f_dims
            new_v_row = cdec * state["fac_vec_row"] + mixing * grad_sqr.mean(dim=d0)
            new_v_col = cdec * state["fac_vec_col"] + mixing * grad_sqr.mean(dim=d1)
            reduced_d1 = d1 - 1 if d1 > d0 else d1
            row_col_mean = new_v_row.mean(dim=reduced_d1, keepdim=True)
            row_factor = safe_rsqrt(new_v_row / (row_col_mean + 1e-9))
            col_factor = safe_rsqrt(new_v_col)
            fac_g = g_rep * row_factor.unsqueeze(d0) * col_factor.unsqueeze(d1)
            state["fac_vec_row"], state["fac_vec_col"] = new_v_row, new_v_col
        else:
            new_v = cdec * state["fac_vec_v"] + mixing * grad_sqr
            fac_g = g_rep * safe_rsqrt(new_v + 1e-9)
            state["fac_vec_v"] = new_v
        return mom, rms, fac_g

    def _adamw_step(self, p, grad, state, lr, weight_decay):
        """Independent AdamW update (own exp_avg / exp_avg_sq moments)."""
        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)
            state["adam_step"] = 0

        state["adam_step"] += 1
        t = state["adam_step"]
        beta1, beta2 = self.adam_beta1, self.adam_beta2

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        m_bc = exp_avg / (1.0 - beta1 ** t)
        v_bc = exp_avg_sq / (1.0 - beta2 ** t)
        adam_step = m_bc / (torch.sqrt(v_bc) + self.adam_eps)
        p.add_(adam_step + weight_decay * p, alpha=-lr)

    def _celo2_step(self, p, grad, state, group):
        """The CELO2 MLP update for a single (2D+) parameter, shape == p.shape."""
        p_shape = tuple(p.shape)
        batch_p = p.unsqueeze(-1)
        batch_g = grad.unsqueeze(-1)

        mom, rms, fac_g = self._update_accumulators(state, batch_g, p_shape)
        rsqrt = torch.rsqrt(rms + 1e-8)

        inps = [
            batch_g,
            torch.clamp(batch_g, -0.1, 0.1),
            batch_p,
            mom,
            rms,
            mom * rsqrt,
            rsqrt,
            fac_g,
            batch_g * rsqrt,
        ]

        f_dims = factored_dims(p_shape)
        if f_dims is not None:
            d1, d0 = f_dims
            v_row, v_col = state["fac_vec_row"], state["fac_vec_col"]
            rp_row = [1] * (1 + len(p_shape))
            rp_col = [1] * (1 + len(p_shape))
            rp_row[d0] = p_shape[d0]
            rp_col[d1] = p_shape[d1]
            row_feat = v_row.unsqueeze(d0).repeat(rp_row)
            col_feat = v_col.unsqueeze(d1).repeat(rp_col)
            inps.extend([
                row_feat,
                col_feat,
                torch.rsqrt(row_feat + 1e-8),
                torch.rsqrt(col_feat + 1e-8),
            ])
            reduced_d1 = d1 - 1 if d1 > d0 else d1
            row_col_mean = v_row.mean(dim=reduced_d1, keepdim=True)
            row_factor = safe_rsqrt(v_row / (row_col_mean + 1e-9))
            col_factor = safe_rsqrt(v_col)
            fac_mom_mult = mom * row_factor.unsqueeze(d0) * col_factor.unsqueeze(d1)
            inps.append(fac_mom_mult)

        axis = list(range(len(p_shape)))[-2:]
        inps = [second_moment_normalizer(i, axis=axis) for i in inps]

        out = self.network(inps)
        direction = out[..., 0]
        magnitude = out[..., 1]

        mag_param = torch.exp(magnitude * group["exp_mult"])
        if group["param_scale_mult"]:
            param_scale = torch.sqrt(torch.mean(p ** 2) + 1e-9)
            step = direction * (param_scale * mag_param)
        else:
            step = direction * mag_param

        if group["orthogonalize"] and step.dim() >= 2:
            step = orthogonalize_via_newton_schulz(
                step, self.ns_coeffs, self.ns_iters, self.ns_eps
            )
        step = second_moment_normalizer(step, axis=axis)
        step = step * group["rmsmult"]
        return step

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, loss=None):
        # Global-norm clipping needs all grads up front.
        grad_scales = {}
        for group in self.param_groups:
            if group["clip_grad"]:
                grad_scales[id(group)] = self._global_grad_scale(group["clip_norm"])

        for group in self.param_groups:
            lr = group["lr"]
            adam_lr = group["adam_lr_mult"] * lr
            scale = grad_scales.get(id(group), 1.0)
            clip_val = group["grad_clip_val"]
            use_adamw_for_1d = group["use_adamw_for_1d"]
            force_adam = group["is_embedding"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if scale != 1.0:
                    grad = grad * scale
                grad = torch.clamp(grad, -clip_val, clip_val)
                p_shape = tuple(p.shape)

                state = self.state[p]
                is_1d = p.dim() <= 1 or force_adam
                if use_adamw_for_1d and is_1d:
                    self._adamw_step(
                        p, grad, state, adam_lr, group["adam_weight_decay"]
                    )
                else:
                    if "mom" not in state:
                        self._init_celo2_state(state, p_shape)
                    step = self._celo2_step(p, grad, state, group)
                    p.add_(step + group["weight_decay"] * p, alpha=-lr)
        return loss
