"""Compare Celo2 naive (Python) vs CUDA implementations.

Runs both optimizers on identical inputs for several steps and reports
max absolute difference in parameter values and per-step timing.

Usage:
    python scripts/compare_celo2_naive_cuda.py [--steps 10] [--seed 42]
"""

import argparse
import copy
import time

import torch
import torch.nn as nn

from pylo.optim.Celo2_naive import Celo2_naive
from pylo.optim.Celo2_cuda import Celo2_cuda


class SmallMLP(nn.Module):
    """Small model with input/hidden/output layers to exercise both AdamW and CELO2 branches."""
    def __init__(self, d_in=32, d_hidden=64, d_out=16):
        super().__init__()
        self.input_layer = nn.Linear(d_in, d_hidden)
        self.hidden1 = nn.Linear(d_hidden, d_hidden)
        self.hidden2 = nn.Linear(d_hidden, d_hidden)
        self.output_layer = nn.Linear(d_hidden, d_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        return self.output_layer(x)


def compare(steps: int = 10, seed: int = 42, verbose: bool = True):
    torch.manual_seed(seed)
    device = torch.device("cuda")

    # Create two identical models
    model_naive = SmallMLP().to(device)
    model_cuda = copy.deepcopy(model_naive).to(device)

    # Create optimizers
    opt_naive = Celo2_naive(model_naive.parameters(), lr=1e-3)
    opt_naive.set_param_names(model_naive)

    opt_cuda = Celo2_cuda(model_cuda.parameters(), lr=1e-3)
    opt_cuda.set_param_names(model_cuda)

    # Fixed synthetic data
    torch.manual_seed(seed + 1)
    X = torch.randn(64, 32, device=device)
    Y = torch.randn(64, 16, device=device)
    loss_fn = nn.MSELoss()

    # Warm-up CUDA
    _ = model_cuda(X)
    torch.cuda.synchronize()

    naive_times = []
    cuda_times = []
    max_diffs = []

    for step in range(steps):
        # ── Naive step ───────────────────────────────────────────────
        opt_naive.zero_grad()
        loss_n = loss_fn(model_naive(X), Y)
        loss_n.backward()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        opt_naive.step()
        torch.cuda.synchronize()
        naive_times.append(time.perf_counter() - t0)

        # ── CUDA step ────────────────────────────────────────────────
        opt_cuda.zero_grad()
        loss_c = loss_fn(model_cuda(X), Y)
        loss_c.backward()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        opt_cuda.step()
        torch.cuda.synchronize()
        cuda_times.append(time.perf_counter() - t0)

        # ── Compare parameters ───────────────────────────────────────
        step_max_diff = 0.0
        for (name_n, p_n), (name_c, p_c) in zip(
            model_naive.named_parameters(), model_cuda.named_parameters()
        ):
            diff = (p_n - p_c).abs().max().item()
            step_max_diff = max(step_max_diff, diff)
            if verbose and diff > 1e-4:
                print(f"  Step {step} | {name_n}: max diff = {diff:.6e}")

        max_diffs.append(step_max_diff)

        if verbose:
            print(
                f"Step {step:3d} | "
                f"loss naive={loss_n.item():.6f}  cuda={loss_c.item():.6f} | "
                f"max param diff={step_max_diff:.6e} | "
                f"time naive={naive_times[-1]*1000:.2f}ms  cuda={cuda_times[-1]*1000:.2f}ms"
            )

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Steps:              {steps}")
    print(f"Max param diff:     {max(max_diffs):.6e}")
    print(f"Final param diff:   {max_diffs[-1]:.6e}")
    print(f"Avg naive time:     {sum(naive_times)/len(naive_times)*1000:.2f} ms")
    print(f"Avg CUDA time:      {sum(cuda_times)/len(cuda_times)*1000:.2f} ms")
    print(f"Speedup:            {sum(naive_times)/sum(cuda_times):.2f}x")

    ok = max(max_diffs) < 1e-3
    print(f"\nNumerical match:    {'PASS' if ok else 'FAIL'} (threshold: 1e-3)")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Celo2 naive vs CUDA")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    ok = compare(steps=args.steps, seed=args.seed, verbose=not args.quiet)
    exit(0 if ok else 1)
