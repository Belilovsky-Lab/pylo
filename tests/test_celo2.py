"""Tests for the CELO2 / ELO-CELO2 learned optimizers."""

import functools
import importlib.util
import os

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from pylo.models.CELO2_MLP import CELO2MLP
from pylo.optim import CELO2_naive, ELO_CELO2_naive
from pylo.optim.CELO2_naive import factored_dims

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCALING_L2O = "/home/mila/h/huangx/scaling_l2o"
LO_DIR = os.path.join(SCALING_L2O, "src/learned_optimizers")
CELO2_OPTAX = os.path.join(LO_DIR, "celo2_optax.py")


@functools.lru_cache(maxsize=1)
def _load_jax_lopts():
    """Import the real JAX Celo2LOpt / ELO_Celo2LOpt without their package init.

    The ``learned_optimizers`` package __init__ pulls in heavy modules; here we
    register a throwaway package so the modules' relative imports resolve, then
    load only the files we need. Cached so the ``@gin.configurable`` classes are
    registered exactly once per process (re-loading would raise a gin error).
    """
    import sys
    import types

    pkg = types.ModuleType("_lo_ref")
    pkg.__path__ = [LO_DIR]
    sys.modules["_lo_ref"] = pkg

    def load(name, fn):
        spec = importlib.util.spec_from_file_location(
            f"_lo_ref.{name}", os.path.join(LO_DIR, fn)
        )
        m = importlib.util.module_from_spec(spec)
        sys.modules[f"_lo_ref.{name}"] = m
        spec.loader.exec_module(m)
        return m

    load("celo2_optax", "celo2_optax.py")
    load("elo_adfac_mlp_lopt", "elo_adfac_mlp_lopt.py")
    return load("celo2_lopt", "celo2_lopt.py"), load("elo_celo2", "elo_celo2.py")


def _theta_to_celo2mlp(theta):
    leaves = {}

    def flat(t, pre=""):
        if isinstance(t, dict):
            for k, v in t.items():
                flat(v, f"{pre}/{k}" if pre else k)
        else:
            leaves[pre.split("/")[-1]] = np.asarray(t)

    flat(theta)
    m = CELO2MLP()
    sd = m.state_dict()
    for i in range(len(m.w0)):
        sd[f"w0.{i}"] = torch.tensor(leaves[f"w0__{i}"], dtype=torch.float32)
    sd["b0"] = torch.tensor(leaves["b0"], dtype=torch.float32)
    sd["dense_w.0"] = torch.tensor(leaves["w1"], dtype=torch.float32)
    sd["dense_b.0"] = torch.tensor(leaves["b1"], dtype=torch.float32)
    sd["dense_w.1"] = torch.tensor(leaves["w2"], dtype=torch.float32)
    sd["dense_b.1"] = torch.tensor(leaves["b2"], dtype=torch.float32)
    m.load_state_dict(sd)
    return m


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 8)  # 2D weight + 1D bias
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.mark.parametrize("optimizer_class", [CELO2_naive, ELO_CELO2_naive])
def test_step_runs_and_updates(optimizer_class):
    """Optimizer runs end-to-end and produces finite, non-trivial updates."""
    torch.manual_seed(0)
    model = SmallNet().to(DEVICE)
    before = [p.clone() for p in model.parameters()]

    optimizer = optimizer_class(model.parameters(), num_steps=50, peak_lr=1e-2)
    x = torch.randn(16, 10, device=DEVICE)
    y = torch.randn(16, 1, device=DEVICE)

    for _ in range(5):
        optimizer.zero_grad()
        loss = F.mse_loss(model(x), y)
        loss.backward()
        optimizer.step(loss)

    for p in model.parameters():
        assert torch.isfinite(p).all(), "non-finite parameter after step"
    moved = any(not torch.allclose(b, p) for b, p in zip(before, model.parameters()))
    assert moved, "optimizer did not update any parameters"


def test_higher_rank_parameter():
    """A >2D parameter (conv-like) goes through the orthogonalization path."""
    torch.manual_seed(0)
    w = nn.Parameter(torch.randn(3, 4, 5, device=DEVICE))
    optimizer = CELO2_naive([w], num_steps=20, peak_lr=1e-2)
    for _ in range(3):
        optimizer.zero_grad()
        (w.sum() ** 2).backward()
        optimizer.step()
    assert torch.isfinite(w).all()


def test_celo2_resume(tmp_path):
    """State-dict save/load reproduces the exact continuation trajectory."""
    torch.manual_seed(42)
    np.random.seed(42)
    net = CELO2MLP().to(DEVICE)  # shared meta-model (random but fixed)

    model = SmallNet().to(DEVICE)
    optimizer = CELO2_naive(model.parameters(), num_steps=50, peak_lr=1e-2, network=net)

    x = torch.randn(32, 10, device=DEVICE)
    y = torch.randn(32, 1, device=DEVICE)

    n_steps = 5
    for _ in range(n_steps):
        optimizer.zero_grad()
        F.mse_loss(model(x), y).backward()
        optimizer.step()

    ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    path = tmp_path / "celo2_ckpt.pt"
    torch.save(ckpt, path)

    # Continue with the original.
    orig_final = []
    for _ in range(n_steps):
        optimizer.zero_grad()
        F.mse_loss(model(x), y).backward()
        optimizer.step()
    orig_final = [p.clone().detach() for p in model.parameters()]

    # Restart from checkpoint and continue.
    loaded_model = SmallNet().to(DEVICE)
    loaded_opt = CELO2_naive(
        loaded_model.parameters(), num_steps=50, peak_lr=1e-2, network=net
    )
    ckpt = torch.load(path)
    loaded_model.load_state_dict(ckpt["model"])
    loaded_opt.load_state_dict(ckpt["optimizer"])
    for _ in range(n_steps):
        loaded_opt.zero_grad()
        F.mse_loss(loaded_model(x), y).backward()
        loaded_opt.step()
    loaded_final = [p.clone().detach() for p in loaded_model.parameters()]

    for a, b in zip(orig_final, loaded_final):
        assert torch.max(torch.abs(a - b)).item() < 1e-5


@pytest.mark.skipif(
    not os.path.isfile(CELO2_OPTAX), reason="scaling_l2o JAX source not available"
)
@pytest.mark.parametrize("ortho", [True, False])
@pytest.mark.parametrize("shape", [(7, 5), (4, 9), (6, 6)])
def test_jax_numerical_alignment(ortho, shape):
    """The 2D CELO2 update matches the reference JAX implementation bitwise-ish."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    spec = importlib.util.spec_from_file_location("celo2_optax_ref", CELO2_OPTAX)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)

    r, c = shape
    tr = ref.Celo2Transformation(orthogonalize=ortho)
    theta = tr.init_meta_params(jax.random.PRNGKey(7))
    tr_t = ref.Celo2Transformation(theta=theta, orthogonalize=ortho)

    params = {"w": jnp.asarray(np.random.randn(r, c).astype(np.float32))}
    jstate = tr_t.init(params)

    # Convert theta -> torch CELO2MLP.
    leaves = {}

    def flat(t, pre=""):
        if isinstance(t, dict):
            for k, v in t.items():
                flat(v, f"{pre}/{k}" if pre else k)
        else:
            leaves[pre.split("/")[-1]] = np.asarray(t)

    flat(theta)
    model = CELO2MLP()
    sd = model.state_dict()
    for i in range(len(model.w0)):
        sd[f"w0.{i}"] = torch.tensor(leaves[f"w0__{i}"], dtype=torch.float32)
    sd["b0"] = torch.tensor(leaves["b0"], dtype=torch.float32)
    sd["dense_w.0"] = torch.tensor(leaves["w1"], dtype=torch.float32)
    sd["dense_b.0"] = torch.tensor(leaves["b1"], dtype=torch.float32)
    sd["dense_w.1"] = torch.tensor(leaves["w2"], dtype=torch.float32)
    sd["dense_b.1"] = torch.tensor(leaves["b2"], dtype=torch.float32)
    model.load_state_dict(sd)

    p = torch.nn.Parameter(torch.tensor(np.asarray(params["w"]), dtype=torch.float32))
    opt = CELO2_naive([p], num_steps=100, orthogonalize=ortho, network=model)
    opt.device = "cpu"
    opt.initial_momentum_decays = opt.initial_momentum_decays.cpu()
    opt.initial_rms_decays = opt.initial_rms_decays.cpu()
    opt.initial_adafactor_decays = opt.initial_adafactor_decays.cpu()
    group = opt.param_groups[0]

    state = opt.state[p]
    state["mom"] = torch.zeros((r, c, 3))
    state["rms"] = torch.zeros((r, c, 1))
    d1, d0 = factored_dims((r, c))
    full = (r, c, 3)
    state["fac_vec_row"] = torch.zeros(tuple(d for i, d in enumerate(full) if i != d0))
    state["fac_vec_col"] = torch.zeros(tuple(d for i, d in enumerate(full) if i != d1))
    state["fac_vec_v"] = torch.empty(0)

    for _ in range(5):
        g = np.random.randn(r, c).astype(np.float32)
        jstep, jstate = tr_t.update({"w": jnp.asarray(g)}, jstate, params)
        tstep = opt._celo2_step(p, torch.tensor(g), state, group).detach().numpy()
        assert np.max(np.abs(np.asarray(jstep["w"]) - tstep)) < 1e-4


@pytest.mark.skipif(
    not os.path.isfile(CELO2_OPTAX), reason="scaling_l2o JAX source not available"
)
@pytest.mark.parametrize("which", ["celo2", "elo_celo2"])
def test_jax_end_to_end_alignment(which):
    """The FULL step() (1D AdamW + warmup/cosine LR + weight decay + global-norm
    clipping) matches the reference JAX optimizer over a multi-step trajectory.

    Uses a model with a 2D weight and a 1D bias, nonzero weight decay, and
    enabled gradient clipping with large grads, so every code path is exercised.
    """
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    celo2_lopt, elo_celo2 = _load_jax_lopts()

    N = 60
    cfg = dict(
        init_lr=0.0, peak_lr=1e-3, warmup_fraction=0.05, end_lr=0.0,
        weight_decay=0.1, adam_lr_mult=1.0, use_adamw_for_1d=True,
        clip_grad=True, clip_norm=1.0,
    )
    spec = importlib.util.spec_from_file_location("celo2_optax_ref", CELO2_OPTAX)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    theta = ref.Celo2Transformation(orthogonalize=True).init_meta_params(
        jax.random.PRNGKey(0)
    )
    net = _theta_to_celo2mlp(theta)

    if which == "celo2":
        jlopt = celo2_lopt.Celo2LOpt(
            orthogonalize=True, adam_beta1=0.9, adam_beta2=0.95, **cfg
        )
        topt_cls = CELO2_naive
    else:
        jlopt = elo_celo2.ELO_Celo2LOpt(
            orthogonalize=True, ff_hidden_size=8, ff_hidden_layers=2,
            initial_momentum_decays=(0.9, 0.99, 0.999),
            initial_rms_decays=(0.95,),
            initial_adafactor_decays=(0.9, 0.99, 0.999),
            meta_train=False, **cfg,
        )
        topt_cls = ELO_CELO2_naive

    rng = np.random.RandomState(0)
    W = rng.randn(7, 5).astype(np.float32)
    B = rng.randn(5).astype(np.float32)
    jopt = jlopt.opt_fn(theta)
    jstate = jopt.init({"w": jnp.asarray(W), "b": jnp.asarray(B)}, num_steps=N)

    pw = torch.nn.Parameter(torch.tensor(W))
    pb = torch.nn.Parameter(torch.tensor(B))
    topt = topt_cls(
        [pw, pb], num_steps=N, peak_lr=1e-3, warmup_fraction=0.05, end_lr=0.0,
        weight_decay=0.1, clip_grad=True, clip_norm=1.0, network=net,
    )
    topt.device = "cpu"
    for a in ("initial_momentum_decays", "initial_rms_decays", "initial_adafactor_decays"):
        setattr(topt, a, getattr(topt, a).cpu())

    gr = np.random.RandomState(1)
    for _ in range(12):
        gw = (gr.randn(7, 5) * 3).astype(np.float32)
        gb = (gr.randn(5) * 3).astype(np.float32)
        out = jopt.update(jstate, {"w": jnp.asarray(gw), "b": jnp.asarray(gb)}, loss=jnp.asarray(0.0))
        jstate = out[0] if isinstance(out, tuple) else out
        pw.grad = torch.tensor(gw)
        pb.grad = torch.tensor(gb)
        topt.step()
        assert np.max(np.abs(np.asarray(jstate.params["w"]) - pw.detach().numpy())) < 1e-4
        assert np.max(np.abs(np.asarray(jstate.params["b"]) - pb.detach().numpy())) < 1e-4
