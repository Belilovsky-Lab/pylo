# CELO2 PyTorch Implementation Plan

**Last updated:** 2026-04-01

## Status: Phase 1 Complete (Naive Implementation)

All core components implemented and tested. 8/8 tests passing.

| Component | Status | File |
|-----------|--------|------|
| Checkpoint conversion | Done | `scripts/convert_celo2_checkpoint.py` |
| Celo2 MLP model | Done | `pylo/models/Celo2_MLP.py` |
| Converted weights | Done | `pylo/models/celo2_weights.pt` |
| Newton-Schulz util | Done | `pylo/util/newton_schulz.py` |
| Celo2_naive optimizer | Done | `pylo/optim/Celo2_naive.py` |
| Package registration | Done | `pylo/optim/__init__.py`, `pylo/__init__.py` |
| Tests (8/8 passing) | Done | `tests/test_celo2.py` |

### Verification Results (2026-03-31)
- MLP forward pass: JAX vs PyTorch max diff **~1.8e-7**
- Newton-Schulz: JAX vs PyTorch max diff **~1e-6**
- Linear regression convergence: loss drops **>90%** in 300 steps
- MLP classification convergence: accuracy improves from **20%** (random) to **>50%**
- State save/load roundtrip: **exact match**

---

## Goal
Port CELO2 (the full version with Newton-Schulz orthogonalization + AdamW for biases/embeddings) from JAX/Optax to PyTorch, integrated into the pylo framework. Convert the pretrained `theta.state` checkpoint to PyTorch format.

## Background

**CELO2** (from `snippets/celo2/celo2_optax.py`) is a learned optimizer with:
- An MLP that computes per-parameter update steps from gradient statistics
- Newton-Schulz orthogonalization applied to 2D (matrix) parameter updates
- AdamW used for 1D params (biases, embeddings) instead of the learned rule
- Meta-trained on 4 small image MLP tasks, scales to large models (tested up to GPT-3 1.3B)

**Key architectural differences from existing AdafacLO:**

| Aspect | AdafacLO | CELO2 |
|--------|----------|-------|
| MLP size | 39→32→32→2 | 30→8→8→3 |
| First layer | Single weight matrix | 14 separate per-input-group matrices |
| Output dims | 2 (direction, magnitude) | 3 (direction, magnitude, unused) |
| exp_mult | 0.001 | 0.0 (magnitude=1 always) |
| Orthogonalization | None | Newton-Schulz (5 iters) for 2D params |
| Output normalization | None | Second-moment normalization |
| Decay computation | param_to_decay(decay_to_param(initial) + learned_offset) | Raw values (0.9, 0.99, 0.999 etc.) |
| 1D param handling | Same MLP for all params | AdamW for biases/embeddings |
| rms_decays | (0.999,) | (0.95,) |
| Bias correction | No | No (flag exists but default False) |

## Checkpoint Structure

The pretrained weights at `snippets/celo2/celo2/theta.state` (Flax serialized) contain:

```
ff_mod_stack/~/b0:      (8,)    — bias layer 0
ff_mod_stack/~/b1:      (8,)    — bias layer 1
ff_mod_stack/~/b2:      (3,)    — bias layer 2 (output)
ff_mod_stack/~/w0__0:   (1, 8)  — first layer weight for input group 0 (g)
ff_mod_stack/~/w0__1:   (1, 8)  — input group 1 (clip_g)
ff_mod_stack/~/w0__2:   (1, 8)  — input group 2 (p)
ff_mod_stack/~/w0__3:   (3, 8)  — input group 3 (m, 3 momentum decays)
ff_mod_stack/~/w0__4:   (1, 8)  — input group 4 (rms)
ff_mod_stack/~/w0__5:   (3, 8)  — input group 5 (m*rsqrt)
ff_mod_stack/~/w0__6:   (1, 8)  — input group 6 (rsqrt)
ff_mod_stack/~/w0__7:   (3, 8)  — input group 7 (fac_g)
ff_mod_stack/~/w0__8:   (1, 8)  — input group 8 (g*rsqrt)
ff_mod_stack/~/w0__9:   (3, 8)  — input group 9 (row_feat)
ff_mod_stack/~/w0__10:  (3, 8)  — input group 10 (col_feat)
ff_mod_stack/~/w0__11:  (3, 8)  — input group 11 (rsqrt_row)
ff_mod_stack/~/w0__12:  (3, 8)  — input group 12 (rsqrt_col)
ff_mod_stack/~/w0__13:  (3, 8)  — input group 13 (fac_mom_mult)
ff_mod_stack/~/w1:      (8, 8)  — hidden layer 1 weight
ff_mod_stack/~/w2:      (8, 3)  — output layer weight
```

**JAX→PyTorch weight conversion:**
- JAX linear layers use `x @ W + b` with W shape `(in, out)`
- PyTorch `nn.Linear` uses `x @ W.T + b` with W shape `(out, in)`
- So all weight matrices must be **transposed**: `W_pytorch = W_jax.T`

## Input Features (30-dim for factored/2D params)

The MLP receives 14 input groups concatenated along last dim:

| # | Feature | Dim | Description |
|---|---------|-----|-------------|
| 0 | g | 1 | Raw gradient |
| 1 | clip_g | 1 | Clipped gradient [-0.1, 0.1] |
| 2 | p | 1 | Parameter value |
| 3 | m | 3 | Momentum (3 decay rates) |
| 4 | rms | 1 | Second moment (1 decay rate) |
| 5 | m*rsqrt | 3 | Momentum × rsqrt(rms) |
| 6 | rsqrt | 1 | rsqrt(rms) |
| 7 | fac_g | 3 | Factored-normalized gradient (3 decay rates) |
| 8 | g*rsqrt | 1 | Gradient × rsqrt(rms) |
| 9 | row_feat | 3 | Factored row statistics (or v_full for 1D) |
| 10 | col_feat | 3 | Factored col statistics (or v_full for 1D) |
| 11 | rsqrt_row | 3 | rsqrt(row_feat) |
| 12 | rsqrt_col | 3 | rsqrt(col_feat) |
| 13 | fac_mom_mult | 3 | Momentum × factored preconditioner |

All inputs are second-moment normalized across spatial dims before MLP forward pass.

---

## What Was Implemented (Phase 1)

### Celo2 MLP Model (`pylo/models/Celo2_MLP.py`)
- `Celo2MLP` class inheriting from `nn.Module`
- First layer: 14 separate `nn.Parameter` weight matrices (one per input group), outputs summed + shared bias + ReLU
- Hidden layer: `nn.Linear(8, 8)` + ReLU
- Output layer: `nn.Linear(8, 3)`
- `from_pretrained_file(path)` class method to load converted `.pt` checkpoint

### Checkpoint Conversion (`scripts/convert_celo2_checkpoint.py`)
- Loads `theta.state` via Flax deserialization
- Transposes all weight matrices (JAX→PyTorch convention)
- Saves as `pylo/models/celo2_weights.pt`

### Newton-Schulz Orthogonalization (`pylo/util/newton_schulz.py`)
- `orthogonalize_newton_schulz(x, ns_coeffs, ns_steps=5, eps=1e-8)`
- Handles rows > cols case via transpose
- Coefficients: `(3.4445, -4.7750, 2.0315)`

### Celo2_naive Optimizer (`pylo/optim/Celo2_naive.py`)
- **Parameter classification** via `_classify_params_list()`:
  - `ndim <= 1` → AdamW
  - `ndim >= 2` + first/last 2D param (input/output layer) → AdamW
  - Name contains "embed" → AdamW
  - All other 2D+ params → CELO2
  - `set_param_names(model)` for name-based detection
- **CELO2 branch** (`_celo2_step`):
  1. Clip gradients to [-1000, 1000]
  2. Update momentum (3 decays: 0.9, 0.99, 0.999)
  3. Update RMS (1 decay: 0.95)
  4. Update factored accumulators (3 decays: 0.9, 0.99, 0.999)
  5. Build 14 input feature groups
  6. Second-moment normalize inputs
  7. MLP forward → direction, magnitude
  8. Step = direction × exp(magnitude × 0.0) = direction
  9. Newton-Schulz orthogonalization (2D params)
  10. Second-moment normalize output
  11. Scale by rmsmult (1.0)
  12. p = p - lr × step
  13. Weight decay (decoupled)
- **AdamW branch** (`_adamw_step`):
  - Standard AdamW with beta1=0.9, beta2=0.95, eps=1e-8
  - Bias-corrected, decoupled weight decay

### Tests (`tests/test_celo2.py`) — 8/8 passing
1. `test_1d_params_get_adamw` — 1D params routed to AdamW
2. `test_input_output_layers_get_adamw` — first/last 2D params routed to AdamW
3. `test_hidden_layers_get_celo2` — hidden 2D weights routed to CELO2
4. `test_embedding_gets_adamw` — "embed" in name → AdamW
5. `test_linear_regression_converges` — loss drops >90% in 300 steps
6. `test_no_nan_in_parameters` — no NaN/Inf after 50 steps
7. `test_mlp_classification_converges` — accuracy >50% on synthetic 5-class task
8. `test_optimizer_state_roundtrip` — save/load produces identical training trajectory

---

## Design Decisions (Resolved)

1. **Parameter grouping**: Automatic by `ndim`. `ndim <= 1` → AdamW. `ndim >= 2` → CELO2, **except** first and last 2D layers (input/output) which also get AdamW. Detection uses a mixture of parameter index position and name matching (e.g., contains "embed", is first/last 2D param in parameter list).
2. **AdamW**: Self-contained inline implementation within the optimizer class. No dependency on `torch.optim.AdamW`.
3. **Checkpoint hosting**: Local `.pt` file only for now. No HuggingFace push.
4. **LR / weight decay**: Standard PyTorch pattern — `lr` as constructor arg, users attach `lr_scheduler` externally. Weight decay applied additively after the learned step.
5. **Scope**: Naive (CPU-compatible) version only. CUDA kernel deferred.

---

## Next Steps (Phase 2 — Not Yet Started)

- [ ] Full numerical match test: compare single optimizer step JAX vs PyTorch end-to-end
- [ ] Benchmark on real tasks (e.g., CIFAR-10 ResNet, small GPT)
- [ ] CUDA kernel for performance (`pylo/csrc/celo2_kernel.cu`)
- [ ] Push converted checkpoint to HuggingFace for `from_pretrained()` loading
- [ ] Add `Celo2_naive` to pylo documentation (index.rst, usage.rst)
- [ ] Support for `celo2-base` variant (orthogonalize=False, MLP for all params)

---

## File Structure

```
pylo/
├── models/
│   ├── Celo2_MLP.py              # CELO2 MLP model (30→8→8→3)
│   ├── celo2_weights.pt           # Converted pretrained weights
│   └── ...
├── optim/
│   ├── Celo2_naive.py             # CELO2 optimizer (naive/CPU)
│   ├── __init__.py                # Updated: exports Celo2_naive
│   └── ...
├── util/
│   ├── newton_schulz.py           # Newton-Schulz orthogonalization
│   └── ...
├── __init__.py                    # Updated: exports Celo2_naive
scripts/
├── convert_celo2_checkpoint.py    # JAX→PyTorch checkpoint converter
tests/
├── test_celo2.py                  # All CELO2 tests (8 tests)
```
