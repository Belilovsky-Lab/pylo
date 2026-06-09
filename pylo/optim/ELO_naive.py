"""ELO_naive: a PyTorch port of the ELO (Adafactor-MLP) learned optimizer.

ELO trains an Adafactor-style MLP learned optimizer with an auxiliary "expert"
mechanism. The expert trajectory and the imitation (IMT) losses are only active
during meta-training; at inference (``meta_train=False``) the parameter update
reduces to the plain Adafactor-MLP forward pass. This file ports that
inference-time forward pass.

The per-parameter features and the meta-model are identical to
:class:`pylo.optim.AdafacLO_naive.AdafacLO_naive` / :class:`pylo.models.Meta_MLP.MetaMLP`
(39 input features, 2 outputs). ELO differs only in:

  * raw accumulator decays (no decay reparameterization),
  * a warmup-then-constant (optionally cosine) learning-rate schedule, and
  * the update rule ``p -= scheduled_lr * (direction * exp(magnitude * exp_mult)
    + weight_decay * p)`` (the schedule plays the role of ``step_mult``).

Reference: ``scaling_l2o/src/learned_optimizers/elo_adfac_mlp_lopt.py``.
"""

from typing import Optional

import numpy as np
import torch
from torch.optim import Optimizer

from pylo.models.Meta_MLP import MetaMLP
from pylo.optim.AdafacLO_naive import (
    factored_dims,
    init_factors,
    safe_rsqrt,
    second_moment_normalizer,
    tanh_embedding,
    update_factors,
)


class ELO_naive(Optimizer):
    """Pure-PyTorch ELO (Adafactor-MLP) learned optimizer.

    Args:
        params: Iterable of parameters or ``param_groups``.
        num_steps: Total number of inner training steps (defines the warmup /
            cosine schedule). Required.
        exp_mult, step_mult: Magnitude exponent multiplier and base step size
            (``step_mult`` is the post-warmup learning rate).
        init_lr, warmup_fraction, warmup_steps: Linear warmup from ``init_lr`` to
            ``step_mult``. ``warmup_fraction`` (fraction of ``num_steps``) takes
            priority over ``warmup_steps`` when > 0.
        use_lo_cosine_scheduler, step_mult_min: If enabled, cosine-decay the
            post-warmup learning rate from ``step_mult`` down to ``step_mult_min``.
        weight_decay: Decoupled weight decay (scaled by the schedule).
        hidden_size, hidden_layers: MetaMLP geometry. Note ELO counts hidden
            *weight* layers, so the original ``hidden_layers=2`` maps to
            ``MetaMLP(hidden_layers=1)`` (input + one hidden + output).
        initial_momentum_decays, initial_rms_decays, initial_adafactor_decays:
            Raw accumulator decays (used directly, no reparameterization).
        clip_grad, clip_norm: Optional global-norm gradient clipping.
        hf_key: HuggingFace Hub id for the MetaMLP weights.
        checkpoint_path: Local path to a converted MetaMLP ``state_dict`` (.pt).
        network: An already-constructed :class:`MetaMLP` (overrides the above).
    """

    def __init__(
        self,
        params,
        num_steps,
        exp_mult=0.001,
        step_mult=0.001,
        init_lr=0.0,
        warmup_fraction=0.05,
        warmup_steps=0,
        use_lo_cosine_scheduler=False,
        step_mult_min=1e-4,
        weight_decay=0.0,
        hidden_size=32,
        hidden_layers=1,
        initial_momentum_decays=(0.9, 0.99, 0.999),
        initial_rms_decays=(0.999,),
        initial_adafactor_decays=(0.9, 0.99, 0.999),
        clip_grad=False,
        clip_norm=1.0,
        hf_key: Optional[str] = "DiamondXL/elo",
        checkpoint_path: Optional[str] = None,
        network: Optional[MetaMLP] = None,
    ):
        if num_steps is None:
            raise ValueError("ELO_naive requires num_steps for the LR schedule.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        defaults = dict(
            num_steps=num_steps,
            exp_mult=exp_mult,
            step_mult=step_mult,
            init_lr=init_lr,
            warmup_fraction=warmup_fraction,
            warmup_steps=warmup_steps,
            use_lo_cosine_scheduler=use_lo_cosine_scheduler,
            step_mult_min=step_mult_min,
            weight_decay=weight_decay,
            clip_grad=clip_grad,
            clip_norm=clip_norm,
        )
        super().__init__(params, defaults)

        self.initial_momentum_decays = torch.tensor(
            initial_momentum_decays, dtype=torch.float32, device=self.device
        )
        self.initial_rms_decays = torch.tensor(
            initial_rms_decays, dtype=torch.float32, device=self.device
        )
        self.initial_adafactor_decays = torch.tensor(
            initial_adafactor_decays, dtype=torch.float32, device=self.device
        )

        # Precedence: explicit network > local checkpoint > HuggingFace Hub.
        if network is not None:
            self.network = network
        elif checkpoint_path is not None:
            self.network = MetaMLP(
                input_size=39, hidden_size=hidden_size, hidden_layers=hidden_layers
            )
            self.network.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        elif hf_key is not None:
            self.network = MetaMLP.from_pretrained(hf_key)
        else:
            self.network = MetaMLP(
                input_size=39, hidden_size=hidden_size, hidden_layers=hidden_layers
            )
        self.network = self.network.to(self.device)

    def _scheduled_lr(self, iteration, group):
        """Warmup → constant (or cosine) schedule, matching the JAX version."""
        num_steps = group["num_steps"]
        step_mult = group["step_mult"]
        init_lr = group["init_lr"]
        if group["warmup_fraction"] > 0:
            warmup_n = group["warmup_fraction"] * num_steps
        else:
            warmup_n = group["warmup_steps"]
        warmup_n = max(float(warmup_n), 1.0)
        it = float(iteration)

        if group["use_lo_cosine_scheduler"]:
            frac = min(max(it / max(num_steps - 1, 1), 0.0), 1.0)
            step_mult_min = group["step_mult_min"]
            base = step_mult_min + (step_mult - step_mult_min) * 0.5 * (
                1.0 + np.cos(np.pi * frac)
            )
        else:
            base = step_mult

        warmup_lr = init_lr + (step_mult - init_lr) * min(it / warmup_n, 1.0)
        return warmup_lr if it < warmup_n else base

    def _global_grad_scale(self, clip_norm):
        total = torch.zeros((), device=self.device)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    total = total + p.grad.detach().float().pow(2).sum()
        denom = torch.clamp(torch.sqrt(total), min=clip_norm)
        return (clip_norm / denom).item() if denom > 0 else 1.0

    @torch.no_grad()
    def step(self, loss=None):
        grad_scales = {}
        for group in self.param_groups:
            if group["clip_grad"]:
                grad_scales[id(group)] = self._global_grad_scale(group["clip_norm"])

        for group in self.param_groups:
            iteration = group.get("step", 0)  # 0-indexed, as in the JAX update
            group["step"] = iteration + 1
            exp_mult = group["exp_mult"]
            weight_decay = group["weight_decay"]
            scheduled_lr = self._scheduled_lr(iteration, group)
            scale = grad_scales.get(id(group), 1.0)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if scale != 1.0:
                    grad = grad * scale
                grad = torch.nan_to_num(grad)
                p_shape = p.shape

                beta_m = self.initial_momentum_decays
                beta_rms = self.initial_rms_decays
                beta_adafactor = self.initial_adafactor_decays

                state = self.state[p]
                if len(state) == 0:
                    state["mom"] = torch.zeros(p_shape + (beta_m.shape[-1],)).to(self.device)
                    state["rms"] = torch.zeros(p_shape + (beta_rms.shape[-1],)).to(self.device)
                    fr, fc, fv = init_factors(p)
                    state["fac_vec_row"] = fr.to(self.device)
                    state["fac_vec_col"] = fc.to(self.device)
                    state["fac_vec_v"] = fv.to(self.device)

                batch_p = p.unsqueeze(-1)
                batch_g = grad.unsqueeze(-1)

                training_step_feature = tanh_embedding(iteration).to(self.device)
                axis = list(range(len(p_shape)))
                for _ in axis:
                    beta_m = beta_m[None, ...]
                    beta_rms = beta_rms[None, ...]
                    beta_adafactor = beta_adafactor[None, ...]
                    training_step_feature = training_step_feature[None, ...]
                training_step_feature = training_step_feature.repeat(p_shape + (1,))

                mom = state["mom"]
                rms = state["rms"]
                mom.mul_(beta_m).add_((1 - beta_m) * batch_g)
                rms.mul_(beta_rms).add_((1 - beta_rms) * (batch_g ** 2))
                (
                    state["fac_vec_col"],
                    state["fac_vec_row"],
                    state["fac_vec_v"],
                    fac_g,
                ) = update_factors(
                    state["fac_vec_col"],
                    state["fac_vec_row"],
                    state["fac_vec_v"],
                    batch_g,
                    p_shape,
                    beta_adafactor,
                )
                fac_vec_col = state["fac_vec_col"]
                fac_vec_row = state["fac_vec_row"]
                fac_vec_v = state["fac_vec_v"]

                rsqrt = torch.rsqrt(rms + 1e-6)
                inps = [batch_g, batch_p, mom, rms, mom * rsqrt, rsqrt, fac_g]

                f_dims = factored_dims(p_shape)
                if f_dims is not None:
                    d1, d0 = f_dims
                    rp_row = [1] * (1 + len(p_shape))
                    rp_col = [1] * (1 + len(p_shape))
                    rp_row[d0] = p_shape[d0]
                    rp_col[d1] = p_shape[d1]
                    row_feat = fac_vec_row.unsqueeze(d0).repeat(rp_row)
                    col_feat = fac_vec_col.unsqueeze(d1).repeat(rp_col)
                    inps.extend([
                        row_feat,
                        col_feat,
                        torch.rsqrt(row_feat + 1e-8),
                        torch.rsqrt(col_feat + 1e-8),
                    ])
                    reduced_d1 = d1 - 1 if d1 > d0 else d1
                    row_col_mean = fac_vec_row.mean(dim=reduced_d1, keepdim=True)
                    row_factor = safe_rsqrt(fac_vec_row / (row_col_mean + 1e-9))
                    col_factor = safe_rsqrt(fac_vec_col)
                    fac_mom_mult = mom * row_factor.unsqueeze(d0) * col_factor.unsqueeze(d1)
                    inps.append(fac_mom_mult)
                else:
                    inps.extend([
                        fac_vec_v,
                        fac_vec_v,
                        torch.rsqrt(fac_vec_v + 1e-8),
                        torch.rsqrt(fac_vec_v + 1e-8),
                    ])
                    fac_mom_mult = mom * torch.pow(fac_vec_v + 1e-6, -0.5)
                    inps.append(fac_mom_mult)

                inps = torch.cat(inps, dim=-1)
                inps = second_moment_normalizer(inps, axis=axis)
                inp_stack = torch.cat([inps, training_step_feature], dim=-1)

                direction, magnitude = self.network(inp_stack).split(1, dim=-1)
                step = (direction * torch.exp(magnitude * exp_mult)).squeeze(-1)
                p.add_(step + weight_decay * p, alpha=-scheduled_lr)
        return loss
