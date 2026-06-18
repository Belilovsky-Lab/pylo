"""CELO2_CUDA: CUDA-accelerated CELO2 learned optimizer.

This is the fused-kernel counterpart of :class:`pylo.optim.CELO2_naive.CELO2_naive`.
The per-element heavy lifting -- CELO2 feature construction, per-channel
second-moment normalization, and the split-input MLP inference -- is done by the
``celo2_cuda_kernel.celo2_kernel`` CUDA extension. Everything that is cheap or
matrix-shaped stays in Python and is shared verbatim with the naive optimizer:

  * the warmup + cosine LR schedule (``CELO2_naive._lr_schedule``),
  * global-norm gradient clipping,
  * Newton-Schulz orthogonalization of the 2D+ update,
  * the post-MLP second-moment normalization + rmsmult scaling, and
  * the inline AdamW branch for 1D / embedding parameters.

The kernel only computes the *raw* per-element step
``direction * exp(magnitude * exp_mult)``; it deliberately does not touch the
parameter, so the Python side can orthogonalize/normalize/apply exactly as the
naive code does. Correctness is checked against the (JAX-aligned) naive optimizer
in ``tests/test_celo2_cuda.py``.

Layout: accumulators use the *leading* decay axis ``(n_decay,) + shape`` expected
by the kernel (the naive optimizer uses a trailing axis); the factored
second-moment handling mirrors :mod:`pylo.optim.AdafacLO_cuda`, whose
``exp_avg_sq_r/c`` + ``row_factor/col_factor`` layout is exactly what the kernel
indexes.

Only ``ff_hidden_layers == 2`` (one split input layer + one hidden + one output,
i.e. 30 -> 8 -> 8 -> 3) is supported by the hardcoded kernel.
"""

from typing import Optional

import numpy as np
import torch
from torch.optim import Optimizer

import celo2_cuda_kernel

from pylo.models.CELO2_MLP import CELO2MLP
from pylo.optim.CELO2_naive import (
    factored_dims,
    safe_rsqrt,
    second_moment_normalizer,
    orthogonalize_via_newton_schulz,
)

# Must match the kernel's INPUT_DIM / HIDDEN_DIM / OUTPUT_DIM.
_KERNEL_INPUT_DIM = 30
_KERNEL_HIDDEN_DIM = 8
_KERNEL_OUTPUT_DIM = 3


class CELO2_CUDA(Optimizer):
    """CUDA CELO2 learned optimizer. Drop-in for :class:`CELO2_naive`.

    The constructor signature mirrors :class:`CELO2_naive`. See that class for a
    description of the hyper-parameters.
    """

    def __init__(
        self,
        params,
        num_steps,
        # LR schedule
        init_lr=0.0,
        peak_lr=1e-3,
        warmup_steps=0,
        warmup_fraction=0.05,
        end_lr=0.0,
        # Weight decay
        weight_decay=0.0,
        # AdamW for 1D params
        adam_lr_mult=1.0,
        adam_weight_decay=None,
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
        if num_steps is None:
            raise ValueError("CELO2_CUDA requires num_steps for the LR schedule.")
        if not torch.cuda.is_available():
            raise RuntimeError("CELO2_CUDA requires a CUDA device.")

        self.device = torch.device("cuda")

        defaults = dict(
            num_steps=num_steps,
            init_lr=init_lr,
            peak_lr=peak_lr,
            warmup_steps=warmup_steps,
            warmup_fraction=warmup_fraction,
            end_lr=end_lr,
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

        # Fixed CELO2 decays (no learned offset, unlike AdafacLO), clamped to [0, 1].
        self.beta_m = torch.clamp(
            torch.tensor(initial_momentum_decays, dtype=torch.float32, device=self.device),
            0.0, 1.0,
        )
        self.beta_rms = torch.clamp(
            torch.tensor(initial_rms_decays, dtype=torch.float32, device=self.device),
            0.0, 1.0,
        )
        self.beta_fac = torch.clamp(
            torch.tensor(initial_adafactor_decays, dtype=torch.float32, device=self.device),
            0.0, 1.0,
        )
        self.n_mom = self.beta_m.shape[-1]
        self.n_rms = self.beta_rms.shape[-1]
        self.n_fac = self.beta_fac.shape[-1]
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
            self.network.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        elif hf_key is not None:
            self.network = CELO2MLP.from_pretrained(hf_key)
        else:
            self.network = CELO2MLP(
                hidden_size=ff_hidden_size, hidden_layers=ff_hidden_layers
            )
        self.network = self.network.to(self.device)

        # The kernel hardcodes a 30 -> 8 -> 8 -> 3 MLP (one split input layer, one
        # hidden, one output): validate the loaded network matches that shape.
        if sum(self.network.feature_dims) != _KERNEL_INPUT_DIM:
            raise ValueError(
                f"CELO2_CUDA kernel expects feature_dims summing to "
                f"{_KERNEL_INPUT_DIM}, got {sum(self.network.feature_dims)}."
            )
        if self.network.hidden_size != _KERNEL_HIDDEN_DIM or len(self.network.dense_w) != 2:
            raise ValueError(
                "CELO2_CUDA kernel only supports ff_hidden_size=8, ff_hidden_layers=2."
            )

        # Pre-stack the split first layer into a single (30, 8) matrix: the kernel
        # treats it as a plain (in, out) dense layer (CELO2MLP.forward computes
        # sum_i feature_i @ w0[i] == concat(features) @ cat(w0)).
        self.w_in = torch.cat([w.detach() for w in self.network.w0], dim=0).contiguous()
        self.b_in = self.network.b0.detach().contiguous()
        self.w_h = self.network.dense_w[0].detach().contiguous()
        self.b_h = self.network.dense_b[0].detach().contiguous()
        self.w_out = self.network.dense_w[1].detach().contiguous()
        self.b_out = self.network.dense_b[1].detach().contiguous()

        # CELO2 offsets the 1-indexed step counter by 1 (schedule(0) on first step);
        # ELO_CELO2_CUDA overrides this to 0.
        self._lr_offset = 1

    # ---------------------------------------------------------------- helpers
    def _lr_schedule(self, step, group):
        """Warmup + cosine decay learning rate (identical to CELO2_naive)."""
        num_steps = group["num_steps"]
        warmup = group["warmup_steps"]
        if group["warmup_fraction"] > 0:
            warmup = group["warmup_fraction"] * num_steps
        warmup_f = max(float(warmup), 1.0)
        decay_f = max(float(num_steps), 1.0)
        step = float(step)
        peak_lr, init_lr, end_lr = group["peak_lr"], group["init_lr"], group["end_lr"]
        if step < warmup_f:
            return init_lr + (peak_lr - init_lr) * min(step / warmup_f, 1.0)
        progress = min(max((step - warmup_f) / max(decay_f - warmup_f, 1.0), 0.0), 1.0)
        return end_lr + (peak_lr - end_lr) * 0.5 * (1.0 + np.cos(np.pi * progress))

    def _global_grad_scale(self, clip_norm):
        total = torch.zeros((), device=self.device)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    total = total + p.grad.detach().float().pow(2).sum()
        global_norm = torch.sqrt(total)
        denom = torch.clamp(global_norm, min=clip_norm)
        return (clip_norm / denom).item() if denom > 0 else 1.0

    # ----------------------------------------------------- accumulator update
    def _init_state(self, state, p):
        shape = tuple(p.shape)
        state["mom"] = torch.zeros((self.n_mom,) + shape, device=self.device)
        state["rms"] = torch.zeros(shape, device=self.device)
        f_dims = factored_dims(shape)
        if f_dims is not None:
            dc, dr = f_dims
            r_shape = list(shape)
            r_shape[dr] = 1
            c_shape = list(shape)
            c_shape[dc] = 1
            state["fac_r"] = torch.zeros([self.n_fac] + r_shape, device=self.device)
            state["fac_c"] = torch.zeros([self.n_fac] + c_shape, device=self.device)

    def _update_mom_rms(self, state, grad):
        """EMA of momentum (leading layout) and the single rms accumulator."""
        m = state["mom"]
        m.lerp_(grad[None, ...], (1 - self.beta_m).view([-1] + [1] * grad.dim()))
        rms = state["rms"]
        rms.lerp_(grad * grad, float(1 - self.beta_rms[-1]))
        return m, rms

    def _update_factored(self, state, grad):
        """Mirror AdafacLO_cuda / CELO2_naive factored second-moment update.

        Returns (fac_r, fac_c, row_factor, col_factor, dc, dr).
        """
        grad_sqr = grad * grad + 1e-30
        dc, dr = factored_dims(tuple(grad.shape))
        beta_fac = self.beta_fac.view([-1] + [1] * grad.dim())
        fac_r = state["fac_r"]
        fac_c = state["fac_c"]
        fac_r.lerp_(grad_sqr.mean(dim=dr, keepdim=True)[None, ...], 1 - beta_fac)
        fac_c.lerp_(grad_sqr.mean(dim=dc, keepdim=True)[None, ...], 1 - beta_fac)
        # Match CELO2_naive: row_col_mean reduces the dc axis of the (kept-dim)
        # row accumulator. In this leading-decay layout the dc parameter axis sits
        # at tensor position dc + 1 (the decay axis is prepended, all param dims
        # are kept via keepdim above).
        row_col_mean = fac_r.mean(dim=dc + 1, keepdim=True)
        row_factor = safe_rsqrt(fac_r / (row_col_mean + 1e-9))
        col_factor = safe_rsqrt(fac_c)
        return fac_r, fac_c, row_factor, col_factor, dc, dr

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, loss=None):
        grad_scales = {}
        for group in self.param_groups:
            if group["clip_grad"]:
                grad_scales[id(group)] = self._global_grad_scale(group["clip_norm"])

        for group in self.param_groups:
            group["step"] = group.get("step", 0) + 1
            t = group["step"]
            lr = self._lr_schedule(t - self._lr_offset, group)
            adam_lr = group["adam_lr_mult"] * lr
            beta1 = float(self.beta_m[0])
            beta2 = float(self.beta_rms[-1])
            scale = grad_scales.get(id(group), 1.0)
            clip_val = group["grad_clip_val"]
            use_adamw_for_1d = group["use_adamw_for_1d"]
            force_adam = group["is_embedding"]
            exp_mult = group["exp_mult"]

            w_in = self.w_in
            b_in = self.b_in
            w_h = self.w_h
            b_h = self.b_h
            w_out = self.w_out
            b_out = self.b_out

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if scale != 1.0:
                    grad = grad * scale
                grad = torch.clamp(grad, -clip_val, clip_val)

                state = self.state[p]
                if len(state) == 0:
                    self._init_state(state, p)

                m, rms = self._update_mom_rms(state, grad)

                is_1d = p.dim() <= 1 or force_adam
                if use_adamw_for_1d and is_1d:
                    # Inline AdamW using the shared momentum[0] / rms accumulators.
                    m_bc = m[0] / (1.0 - beta1 ** t)
                    v_bc = rms / (1.0 - beta2 ** t)
                    adam_step = m_bc / (torch.sqrt(v_bc) + 1e-8)
                    p.add_(adam_step + group["adam_weight_decay"] * p, alpha=-adam_lr)
                    continue

                # CELO2 (2D+) path: kernel computes the raw per-element step.
                fac_r, fac_c, row_factor, col_factor, dc, dr = self._update_factored(
                    state, grad
                )

                second_moment = torch.zeros(
                    _KERNEL_INPUT_DIM, dtype=torch.float32, device=self.device
                )
                step_out = torch.empty_like(p)
                dt = grad.dtype
                celo2_cuda_kernel.celo2_kernel(
                    grad.contiguous(),
                    p,
                    m.to(dt).contiguous(),
                    rms.to(dt).contiguous(),
                    fac_r.to(dt).contiguous(),
                    fac_c.to(dt).contiguous(),
                    row_factor.to(dt).contiguous(),
                    col_factor.to(dt).contiguous(),
                    second_moment,
                    w_in.to(dt),
                    b_in.to(dt),
                    w_h.to(dt),
                    b_h.to(dt),
                    w_out.to(dt),
                    b_out.to(dt),
                    step_out,
                    float(exp_mult),
                    int(dc),
                    int(dr),
                    0,
                )

                step = step_out
                if group["orthogonalize"] and step.dim() >= 2:
                    step = orthogonalize_via_newton_schulz(
                        step, self.ns_coeffs, self.ns_iters, self.ns_eps
                    )
                axis = list(range(p.dim()))[-2:]
                step = second_moment_normalizer(step, axis=axis)
                step = step * group["rmsmult"]
                p.add_(step + group["weight_decay"] * p, alpha=-lr)

        return loss
