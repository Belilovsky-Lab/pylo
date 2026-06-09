"""Tests for the ELO (Adafactor-MLP) learned optimizer."""

import importlib.util
import os

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from pylo.models.Meta_MLP import MetaMLP
from pylo.optim import ELO_naive

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCALING_L2O = "/home/mila/h/huangx/scaling_l2o"
ELO_SRC = os.path.join(SCALING_L2O, "src/learned_optimizers/elo_adfac_mlp_lopt.py")


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def test_step_runs_and_updates():
    torch.manual_seed(0)
    model = SmallNet().to(DEVICE)
    before = [p.clone() for p in model.parameters()]
    optimizer = ELO_naive(model.parameters(), num_steps=50, step_mult=1e-2)
    x = torch.randn(16, 10, device=DEVICE)
    y = torch.randn(16, 1, device=DEVICE)
    for _ in range(6):
        optimizer.zero_grad()
        F.mse_loss(model(x), y).backward()
        optimizer.step()
    for p in model.parameters():
        assert torch.isfinite(p).all()
    assert any(not torch.allclose(b, p) for b, p in zip(before, model.parameters()))


def test_elo_resume(tmp_path):
    torch.manual_seed(42)
    net = MetaMLP(input_size=39, hidden_size=32, hidden_layers=1).to(DEVICE)

    model = SmallNet().to(DEVICE)
    optimizer = ELO_naive(model.parameters(), num_steps=50, step_mult=1e-2, network=net)
    x = torch.randn(32, 10, device=DEVICE)
    y = torch.randn(32, 1, device=DEVICE)

    n_steps = 5
    for _ in range(n_steps):
        optimizer.zero_grad()
        F.mse_loss(model(x), y).backward()
        optimizer.step()

    ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    path = tmp_path / "elo_ckpt.pt"
    torch.save(ckpt, path)

    for _ in range(n_steps):
        optimizer.zero_grad()
        F.mse_loss(model(x), y).backward()
        optimizer.step()
    orig_final = [p.clone().detach() for p in model.parameters()]

    loaded_model = SmallNet().to(DEVICE)
    loaded_opt = ELO_naive(loaded_model.parameters(), num_steps=50, step_mult=1e-2, network=net)
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
    not os.path.isfile(ELO_SRC), reason="scaling_l2o JAX source not available"
)
def test_jax_numerical_alignment():
    """ELO_naive matches the reference JAX ELO_AdafacMLPLOpt (meta_train=False)."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    spec = importlib.util.spec_from_file_location("elo_adfac_ref", ELO_SRC)
    elo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(elo)

    N = 50
    lopt = elo.ELO_AdafacMLPLOpt(
        meta_train=False, exp_mult=0.001, step_mult=0.001, init_lr=0.0,
        warmup_fraction=0.05, weight_decay=0.0, hidden_size=32, hidden_layers=2,
        initial_momentum_decays=(0.9, 0.99, 0.999), initial_rms_decays=(0.999,),
        initial_adafactor_decays=(0.9, 0.99, 0.999), clip_grad=False,
        use_lo_cosine_scheduler=False,
    )
    theta = lopt.init(jax.random.PRNGKey(0))
    jopt = lopt.opt_fn(theta)

    rng = np.random.RandomState(0)
    W = rng.randn(7, 5).astype(np.float32)
    B = rng.randn(5).astype(np.float32)
    jstate = jopt.init({"w": jnp.asarray(W), "b": jnp.asarray(B)}, num_steps=N)

    # Convert theta -> torch MetaMLP (transpose haiku (in,out) -> nn.Linear (out,in)).
    leaves = {}

    def flat(t, pre=""):
        if isinstance(t, dict):
            for k, v in t.items():
                flat(v, f"{pre}/{k}" if pre else k)
        else:
            leaves[pre.split("/")[-1]] = np.asarray(t)

    flat(theta)
    net = MetaMLP(input_size=39, hidden_size=32, hidden_layers=1)
    sd = net.state_dict()
    for j, dst in enumerate(["network.input", "network.linear_0", "network.output"]):
        sd[f"{dst}.weight"] = torch.tensor(leaves[f"w{j}"]).t().contiguous()
        sd[f"{dst}.bias"] = torch.tensor(leaves[f"b{j}"])
    net.load_state_dict(sd)

    pw = torch.nn.Parameter(torch.tensor(W))
    pb = torch.nn.Parameter(torch.tensor(B))
    topt = ELO_naive([pw, pb], num_steps=N, network=net)
    topt.device = "cpu"
    for a in ("initial_momentum_decays", "initial_rms_decays", "initial_adafactor_decays"):
        setattr(topt, a, getattr(topt, a).cpu())

    gr = np.random.RandomState(1)
    for _ in range(8):
        gw = gr.randn(7, 5).astype(np.float32)
        gb = gr.randn(5).astype(np.float32)
        jstate = jopt.update(jstate, {"w": jnp.asarray(gw), "b": jnp.asarray(gb)}, loss=jnp.asarray(0.0))
        pw.grad = torch.tensor(gw)
        pb.grad = torch.tensor(gb)
        topt.step()
        assert np.max(np.abs(np.asarray(jstate.params["w"]) - pw.detach().numpy())) < 1e-5
        assert np.max(np.abs(np.asarray(jstate.params["b"]) - pb.detach().numpy())) < 1e-5
