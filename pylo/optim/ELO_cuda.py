"""ELO_CUDA: CUDA-accelerated ELO (Adafactor-MLP) learned optimizer.

This is the fused-kernel counterpart of :class:`pylo.optim.ELO_naive.ELO_naive`.
ELO and AdafacLO share the *identical* MetaMLP feature set (39 inputs, 2 outputs)
and per-element update, so this optimizer reuses the same fused
``cuda_lo.learned_optimizer_kernel`` extension as
:class:`pylo.optim.AdafacLO_cuda.AdafacLO_CUDA`. The kernel constructs the
features, runs the split MLP, computes ``direction * exp(magnitude * exp_mult)``
and applies it to the parameter in-place.

ELO differs from AdafacLO only in:

  * raw accumulator decays (no ``param_to_decay(decay_to_param(.) + offset)``
    reparameterization -- the decays are used directly), and
  * the update scale: ELO uses ``lr`` directly (no separate ``step_mult``), and
    decoupled weight decay scaled by ``lr``.

No learning-rate schedule is built in: ``lr`` is used as-is, so an external
``torch.optim.lr_scheduler`` can drive the schedule. Correctness is checked
against the (JAX-aligned) naive optimizer in ``tests/test_elo_cuda.py``.

The kernel hardcodes a 39 -> 32 -> 32 -> 2 MLP (``input`` / ``linear_0`` /
``output``), i.e. ``hidden_size=32`` and ``hidden_layers=1``.
"""

from typing import Optional, Tuple

import torch
from torch.optim import Optimizer

import cuda_lo

from pylo.models.Meta_MLP import MetaMLP


def _get_scalar_dtype():
    return torch.float64


def safe_rsqrt(x):
    return torch.rsqrt(
        torch.maximum(x, torch.tensor(1e-9, dtype=x.dtype, device=x.device))
    )


def _factored_dims(shape: Tuple[int, ...]) -> Optional[Tuple[int, int]]:
    """Two largest axes (dc, dr) for the factored second moment, or None for <2D."""
    if len(shape) < 2:
        return None
    sorted_dims = sorted(((x, i) for i, x in enumerate(shape)))
    return int(sorted_dims[-2][1]), int(sorted_dims[-1][1])


class ELO_CUDA(Optimizer):
    """CUDA ELO (Adafactor-MLP) learned optimizer. Drop-in for :class:`ELO_naive`.

    Args:
        params: Iterable of parameters or ``param_groups``.
        lr: Base learning rate (the ELO ``step_mult``). No schedule is applied
            internally; drive any warmup/cosine schedule with an external
            ``torch.optim.lr_scheduler``.
        exp_mult: Magnitude exponent multiplier for the MLP output.
        weight_decay: Decoupled weight decay (scaled by ``lr``).
        hidden_size, hidden_layers: MetaMLP geometry. The fused kernel only
            supports ``hidden_size=32`` and ``hidden_layers=1`` (39->32->32->2).
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
        lr=0.001,
        exp_mult=0.001,
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
        if not torch.cuda.is_available():
            raise RuntimeError("ELO_CUDA requires a CUDA device.")

        self.device = torch.device("cuda")

        defaults = dict(
            lr=lr,
            exp_mult=exp_mult,
            weight_decay=weight_decay,
            clip_grad=clip_grad,
            clip_norm=clip_norm,
        )
        super().__init__(params, defaults)

        # Raw decays (no reparameterization, unlike AdafacLO), clamped to [0, 1].
        self.beta_m = torch.clamp(
            torch.tensor(initial_momentum_decays, dtype=torch.float32, device=self.device),
            0.0, 1.0,
        )
        self.beta_rms = torch.clamp(
            torch.tensor(initial_rms_decays, dtype=torch.float32, device=self.device),
            0.0, 1.0,
        )
        self.beta_adafactor = torch.clamp(
            torch.tensor(initial_adafactor_decays, dtype=torch.float32, device=self.device),
            0.0, 1.0,
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
        for p in self.network.parameters():
            p.requires_grad = False

        # The kernel hardcodes a 39 -> 32 -> 32 -> 2 MLP (input / linear_0 / output).
        net = self.network.network
        if not (hasattr(net, "input") and hasattr(net, "linear_0") and hasattr(net, "output")):
            raise ValueError(
                "ELO_CUDA kernel requires a MetaMLP with input/linear_0/output "
                "layers (hidden_layers=1)."
            )
        if hasattr(net, "linear_1"):
            raise ValueError(
                "ELO_CUDA kernel only supports hidden_layers=1 (39->32->32->2)."
            )

    # ---------------------------------------------------------------- helpers
    def _global_grad_scale(self, clip_norm):
        """optax.clip_by_global_norm scale factor over all parameter grads."""
        total = torch.zeros((), device=self.device)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    total = total + p.grad.detach().float().pow(2).sum()
        denom = torch.clamp(torch.sqrt(total), min=clip_norm)
        return (clip_norm / denom).item() if denom > 0 else 1.0

    def _init_state(self, state, grad):
        state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())
        shape = grad.shape
        f_dims = _factored_dims(tuple(shape))
        if f_dims is not None:
            dc, dr = f_dims
            row_shape = list(shape)
            row_shape[dr] = 1
            col_shape = list(shape)
            col_shape[dc] = 1
            state["exp_avg_sq_r"] = grad.new_zeros([3] + row_shape)
            state["exp_avg_sq_c"] = grad.new_zeros([3] + col_shape)
        else:
            state["exp_avg_sq_r"] = grad.new_zeros((3,) + tuple(shape))
            state["exp_avg_sq_c"] = grad.new_zeros((3,) + tuple(shape))
        state["exp_avg_sq"] = torch.zeros_like(grad, memory_format=torch.preserve_format)
        state["exp_avg"] = grad.new_zeros((3,) + tuple(shape))

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, loss=None):
        grad_scales = {}
        for group in self.param_groups:
            if group["clip_grad"]:
                grad_scales[id(group)] = self._global_grad_scale(group["clip_norm"])

        for group in self.param_groups:
            lr = group["lr"]
            exp_mult = group["exp_mult"]
            weight_decay = group["weight_decay"]
            scale = grad_scales.get(id(group), 1.0)

            w_in = self.network.network.input.weight
            b_in = self.network.network.input.bias
            w_h = self.network.network.linear_0.weight
            b_h = self.network.network.linear_0.bias
            w_out = self.network.network.output.weight
            b_out = self.network.network.output.bias

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Sparse gradients not supported")

                grad = p.grad
                if scale != 1.0:
                    grad = grad * scale
                grad = torch.nan_to_num(grad)

                state = self.state[p]
                if len(state) == 0:
                    self._init_state(state, grad)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg_sq_r = state["exp_avg_sq_r"]
                exp_avg_sq_c = state["exp_avg_sq_c"]
                step_t = state["step"]
                step_t += 1

                eps = 1e-7 if grad.dtype == torch.float16 else 1e-30
                grad_sqr = torch.square(grad) + eps
                one_minus_beta2 = (1 - self.beta_rms).to(grad_sqr.dtype)

                f_dims = _factored_dims(tuple(grad.shape))
                if f_dims is not None:
                    dc, dr = f_dims
                    beta_fac = 1 - self.beta_adafactor.view(-1, *[1] * grad.dim()).to(grad_sqr.dtype)
                    exp_avg_sq_r.lerp_(grad_sqr.mean(dim=dr, keepdim=True)[None, ...], beta_fac)
                    exp_avg_sq_c.lerp_(grad_sqr.mean(dim=dc, keepdim=True)[None, ...], beta_fac)
                    exp_avg_sq.lerp_(grad_sqr, one_minus_beta2)
                    reduce_dc = dc - 1 if dc > dr else dc
                    row_col_mean = exp_avg_sq_r.mean(dim=reduce_dc, keepdim=True)
                    row_factor = safe_rsqrt(exp_avg_sq_r / (row_col_mean + 1e-9))
                    col_factor = safe_rsqrt(exp_avg_sq_c)
                    vector_like = 0
                else:
                    dc = dr = 0
                    beta_fac = 1 - self.beta_adafactor.view(-1, 1).to(grad_sqr.dtype)
                    exp_avg_sq_r.lerp_(grad_sqr[None, ...], beta_fac)
                    exp_avg_sq_c.lerp_(grad_sqr[None, ...], beta_fac)
                    exp_avg_sq.lerp_(grad_sqr, one_minus_beta2)
                    row_factor = safe_rsqrt(exp_avg_sq_r + 1e-9)
                    col_factor = torch.ones_like(row_factor)
                    vector_like = 1

                exp_avg.lerp_(
                    grad[None, ...],
                    (1 - self.beta_m.view([-1] + [1] * grad.dim())).to(grad.dtype),
                )

                # ELO_naive applies decoupled weight decay against the *pre-step*
                # parameter (p -= lr * (step + wd * p)); the kernel updates p
                # in-place, so snapshot p first to reproduce that exactly.
                p_before = p.detach().clone() if weight_decay > 0 else None

                second_moment = torch.zeros([28], device=self.device)
                cuda_lo.learned_optimizer_kernel(
                    grad,
                    p,
                    exp_avg,
                    exp_avg_sq,
                    exp_avg_sq_r,
                    exp_avg_sq_c,
                    row_factor,
                    col_factor,
                    second_moment,
                    w_in.to(grad.dtype),
                    b_in.to(grad.dtype),
                    w_h.to(grad.dtype),
                    b_h.to(grad.dtype),
                    w_out.to(grad.dtype),
                    b_out.to(grad.dtype),
                    lr,        # ELO update scale (no separate step_mult)
                    1.0,       # step_mult == 1.0 for ELO
                    exp_mult,
                    1e-6,      # rms rsqrt epsilon
                    step_t - 1,
                    0.0,       # weight decay handled in Python below
                    dc,
                    dr,
                    vector_like,
                )

                if weight_decay > 0:
                    p.add_(p_before, alpha=-weight_decay * lr)

        return loss
