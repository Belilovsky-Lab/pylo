"""ELO_CELO2_naive: the ELO-CELO2 learned optimizer (PyTorch).

ELO-CELO2 fuses a CELO2 MLP backbone with an ELO expert mechanism *during
meta-training*. At inference / meta-test time the expert trajectory and the IMT
losses are disabled (``meta_train=False``), so the parameter update reduces
exactly to the CELO2 forward pass: CELO2 MLP steps for 2D+ parameters and AdamW
for 1D parameters.

Unlike :class:`pylo.optim.CELO2_naive.CELO2_naive` (which gives AdamW its own
independent moments), ELO-CELO2 keeps the original *shared-accumulator* design:
the AdamW branch for 1D parameters reuses ``momentum[..., 0]`` / ``rms[..., -1]``
of the same momentum/RMS accumulators that feed the CELO2 MLP, matching the JAX
ELO-CELO2 (``config/learned_optimizer/elo_celo2.py``). Because of that this class
no longer inherits from ``CELO2_naive`` -- it is a standalone copy that only
differs in the AdamW accumulator wiring and in the ELO-CELO2 default
hyper-parameters (nonzero weight decay, gradient clipping enabled, ELO-CELO2
checkpoint). The distinct learned weights live in the loaded checkpoint.

No learning-rate schedule is built in: ``lr`` is used as-is, so an external
``torch.optim.lr_scheduler`` can drive the schedule.
"""

from typing import Optional

import torch
from torch.optim import Optimizer

from pylo.models.CELO2_MLP import CELO2MLP
from pylo.optim.CELO2_naive import (
    factored_dims,
    safe_rsqrt,
    second_moment_normalizer,
    orthogonalize_via_newton_schulz,
)


class ELO_CELO2_naive(Optimizer):
    """Inference-time ELO-CELO2 optimizer (CELO2 forward, shared-accumulator AdamW)."""

    def __init__(
        self,
        params,
        lr=1e-3,
        # ELO-CELO2 defaults differ from CELO2 here:
        weight_decay=0.1,
        clip_grad=True,
        clip_norm=1.0,
        # AdamW for 1D params (shares momentum[0] / rms[-1] accumulators)
        adam_lr_mult=1.0,
        adam_weight_decay=None,
        use_adamw_for_1d=True,
        # CELO2 backbone
        orthogonalize=True,
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
        hf_key: Optional[str] = "DiamondXL/elo-celo2",
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

        # Place state / meta-network on the same device as the parameters.
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

    def _init_shared_state(self, state, p_shape):
        """Allocate the (shared) CELO2 momentum / RMS / factored accumulators."""
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

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, loss=None):
        # Global-norm clipping needs all grads up front.
        grad_scales = {}
        for group in self.param_groups:
            if group["clip_grad"]:
                grad_scales[id(group)] = self._global_grad_scale(group["clip_norm"])

        for group in self.param_groups:
            group["step"] = group.get("step", 0) + 1
            t = group["step"]
            lr = group["lr"]
            adam_lr = group["adam_lr_mult"] * lr
            beta1 = float(self.initial_momentum_decays[0])
            beta2 = float(self.initial_rms_decays[-1])
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
                if len(state) == 0:
                    self._init_shared_state(state, p_shape)

                is_1d = p.dim() <= 1 or force_adam
                if use_adamw_for_1d and is_1d:
                    # AdamW over the shared momentum[0] / rms[-1] accumulators.
                    batch_g = grad.unsqueeze(-1)
                    mom, rms, _ = self._update_accumulators(state, batch_g, p_shape)
                    m_bc = mom[..., 0] / (1.0 - beta1 ** t)
                    v_bc = rms[..., -1] / (1.0 - beta2 ** t)
                    adam_step = m_bc / (torch.sqrt(v_bc) + 1e-8)
                    p.add_(adam_step + group["adam_weight_decay"] * p, alpha=-adam_lr)
                else:
                    step = self._celo2_step(p, grad, state, group)
                    p.add_(step + group["weight_decay"] * p, alpha=-lr)
        return loss
