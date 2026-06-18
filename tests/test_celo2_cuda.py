"""Numerical-alignment tests for the CUDA CELO2 / ELO-CELO2 optimizers.

We validate the CUDA implementation against the pure-PyTorch naive one. The naive
optimizer is itself aligned to the original JAX/optax CELO2 (see test_celo2.py),
so CUDA-vs-naive agreement transitively implies CUDA-vs-JAX agreement, without
needing a JAX environment.

Both optimizers are fed the *same* meta-model weights (``network=net``), the same
initial parameters, and the same gradient sequence; we then compare the parameter
trajectories step by step. The tolerance (~1e-3) is looser than the CPU-vs-JAX
1e-4 because the kernel accumulates features/MLP in fp32 with fast-math rsqrt/exp.

These tests require a CUDA device and the compiled ``celo2_cuda_kernel``
extension; they skip otherwise.
"""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CELO2 CUDA optimizer requires a GPU"
)

# Skip cleanly if the extension hasn't been built.
pytest.importorskip("celo2_cuda_kernel")

from pylo.models.CELO2_MLP import CELO2MLP
from pylo.optim.CELO2_naive import CELO2_naive
from pylo.optim.ELO_CELO2_naive import ELO_CELO2_naive
from pylo.optim.CELO2_cuda import CELO2_CUDA
from pylo.optim.ELO_CELO2_cuda import ELO_CELO2_CUDA

DEVICE = torch.device("cuda")

# Matrix params in both orientations (R<C and R>C, to exercise both dc/dr index
# paths) plus a 1D param to exercise the shared-accumulator AdamW branch.
SHAPES = [(64, 32), (32, 64), (32,)]


def _make_init(seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return [torch.randn(s, generator=g, device=DEVICE) for s in SHAPES]


def _make_grads(n_steps, seed, scale=3.0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return [
        [torch.randn(s, generator=g, device=DEVICE) * scale for s in SHAPES]
        for _ in range(n_steps)
    ]


def _run(opt_class, init_vals, grads_seq, net, **cfg):
    params = [torch.nn.Parameter(v.clone()) for v in init_vals]
    opt = opt_class(params, num_steps=cfg.pop("num_steps", 50), network=net, **cfg)
    traj = []
    for grads in grads_seq:
        for p, gg in zip(params, grads):
            p.grad = gg.clone()
        opt.step()
        traj.append([p.detach().clone() for p in params])
    return traj


def _max_traj_diff(traj_a, traj_b):
    worst = 0.0
    for step_a, step_b in zip(traj_a, traj_b):
        for a, b in zip(step_a, step_b):
            worst = max(worst, torch.max(torch.abs(a - b)).item())
    return worst


@pytest.mark.parametrize("orthogonalize", [True, False])
@pytest.mark.parametrize("exp_mult", [0.0, 0.01])
def test_celo2_cuda_matches_naive(orthogonalize, exp_mult):
    torch.manual_seed(0)
    np.random.seed(0)
    net = CELO2MLP().to(DEVICE)  # shared random-but-fixed meta-model

    init_vals = _make_init(seed=1)
    grads_seq = _make_grads(n_steps=50, seed=2)
    cfg = dict(
        num_steps=100,
        peak_lr=1e-2,
        weight_decay=0.1,
        clip_grad=True,
        clip_norm=1.0,
        orthogonalize=orthogonalize,
        exp_mult=exp_mult,
    )

    naive_traj = _run(CELO2_naive, init_vals, grads_seq, net, **cfg)
    cuda_traj = _run(CELO2_CUDA, init_vals, grads_seq, net, **cfg)

    assert _max_traj_diff(naive_traj, cuda_traj) < 1e-5


def test_elo_celo2_cuda_matches_naive():
    """ELO-CELO2 (different defaults + _lr_offset=0) stays aligned."""
    torch.manual_seed(0)
    np.random.seed(0)
    net = CELO2MLP().to(DEVICE)

    init_vals = _make_init(seed=3)
    grads_seq = _make_grads(n_steps=50, seed=4)
    cfg = dict(num_steps=100, peak_lr=1e-3)

    naive_traj = _run(ELO_CELO2_naive, init_vals, grads_seq, net, **cfg)
    cuda_traj = _run(ELO_CELO2_CUDA, init_vals, grads_seq, net, **cfg)

    assert _max_traj_diff(naive_traj, cuda_traj) < 1e-5


def test_celo2_cuda_single_matrix_alignment():
    """Tight check on a single matrix param (pure CELO2 path, no AdamW)."""
    torch.manual_seed(0)
    np.random.seed(0)
    net = CELO2MLP().to(DEVICE)

    g = torch.Generator(device=DEVICE).manual_seed(7)
    init = [torch.randn(48, 24, generator=g, device=DEVICE)]
    grads = [[torch.randn(48, 24, generator=g, device=DEVICE) * 2.0] for _ in range(50)]
    cfg = dict(num_steps=100, peak_lr=1e-2, orthogonalize=True)

    naive_traj = _run(CELO2_naive, init, grads, net, **cfg)
    cuda_traj = _run(CELO2_CUDA, init, grads, net, **cfg)

    assert _max_traj_diff(naive_traj, cuda_traj) < 1e-5
