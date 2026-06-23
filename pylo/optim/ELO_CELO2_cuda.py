"""ELO_CELO2_CUDA: the ELO-CELO2 learned optimizer (CUDA-accelerated).

CUDA counterpart of :class:`pylo.optim.ELO_CELO2_naive.ELO_CELO2_naive`. At
inference time ELO-CELO2 reduces exactly to the CELO2 forward pass, fused into the
``celo2_cuda_kernel`` extension for the 2D+ path.

Like the naive ELO-CELO2 (and unlike :class:`pylo.optim.CELO2_cuda.CELO2_CUDA`),
the AdamW branch for 1D parameters reuses the *shared* momentum[0] / rms
accumulators that feed the CELO2 MLP. This class is therefore a standalone copy
of the original shared-accumulator CELO2 CUDA optimizer with the ELO-CELO2
defaults (nonzero weight decay, gradient clipping, ELO-CELO2 checkpoint, larger
AdamW LR multiplier); it no longer inherits ``CELO2_CUDA``.

No learning-rate schedule is built in: ``lr`` is used as-is, so an external
``torch.optim.lr_scheduler`` can drive the schedule.
"""

from typing import Optional

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


class ELO_CELO2_CUDA(Optimizer):
    """Inference-time ELO-CELO2 optimizer (CUDA CELO2 forward, shared-accumulator AdamW)."""

    def __init__(
        self,
        params,
        lr=1e-3,
        # ELO-CELO2 defaults differ from CELO2 here:
        weight_decay=0.1,
        clip_grad=True,
        clip_norm=1.0,
        # AdamW for 1D params (shares momentum[0] / rms accumulators)
        adam_lr_mult=20.0,
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
        if not torch.cuda.is_available():
            raise RuntimeError("ELO_CELO2_CUDA requires a CUDA device.")

        self.device = torch.device("cuda")

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
                f"ELO_CELO2_CUDA kernel expects feature_dims summing to "
                f"{_KERNEL_INPUT_DIM}, got {sum(self.network.feature_dims)}."
            )
        if self.network.hidden_size != _KERNEL_HIDDEN_DIM or len(self.network.dense_w) != 2:
            raise ValueError(
                "ELO_CELO2_CUDA kernel only supports ff_hidden_size=8, ff_hidden_layers=2."
            )

        # Pre-stack the split first layer into a single (30, 8) matrix.
        self.w_in = torch.cat([w.detach() for w in self.network.w0], dim=0).contiguous()
        self.b_in = self.network.b0.detach().contiguous()
        self.w_h = self.network.dense_w[0].detach().contiguous()
        self.b_h = self.network.dense_b[0].detach().contiguous()
        self.w_out = self.network.dense_w[1].detach().contiguous()
        self.b_out = self.network.dense_b[1].detach().contiguous()

    # ---------------------------------------------------------------- helpers
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
            lr = group["lr"]
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
                    # AdamW over the shared momentum[0] / rms accumulators.
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
