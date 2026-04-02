# CELO2 CUDA Speedup Plan

**Created:** 2026-04-01

## Current Architecture

The `Celo2_cuda` optimizer step for each CELO2 parameter currently does:

```
Python: clip grad, update accumulators (lerp_), compute row/col factors, extract MLP weights
  |
  v
CUDA kernel 1: accumulate squared features (second-moment stats)
CUDA kernel 2: normalize features, MLP forward, write per-element step
  |
  v
Python: Newton-Schulz (5 matrix multiplies), output normalization, scaling, param update
```

**Bottlenecks** (ordered by likely impact):
1. Python-side overhead: per-parameter loop, accumulator updates, factor computation
2. Two separate kernel launches per parameter (launch overhead)
3. Newton-Schulz in Python (5 iterations of matmul per 2D param)
4. MLP weights recast `.to(grad.dtype)` every step
5. `step_out` allocation every step
6. Feature recomputation (computed twice: once in kernel 1, once in kernel 2)

---

## Speedup Strategies

### 1. Fuse Accumulator Updates into the CUDA Kernel

**Impact: HIGH** | **Effort: MEDIUM**

Currently `_update_accumulators()` runs 3 separate `lerp_` calls in Python (momentum, RMS, factored). These are simple element-wise ops that can be fused into `celo2_apply_kernel`, eliminating 3+ kernel launches per parameter and the associated Python overhead.

The kernel would:
- Read old momentum/RMS/factored state
- Compute new values in registers
- Write updated state back to global memory
- Use the updated values directly for feature construction

This also removes `_compute_factors()` from Python — row/col factor computation moves into the kernel.

### 2. Fuse Kernel 1 and Kernel 2 (Single-Kernel with Cooperative Groups)

**Impact: HIGH** | **Effort: HIGH**

The two-kernel design exists because kernel 1 needs a global reduction (sum of squared features across all elements) before kernel 2 can normalize. Options:

- **a) Cooperative groups (`cudaLaunchCooperativeKernel`)**: Use `grid.sync()` as a global barrier between the accumulation and application phases within a single kernel launch. Eliminates one kernel launch and the redundant feature recomputation.

- **b) Two-pass single-launch with persistent threads**: Launch once, first pass accumulates to shared→global, `__threadfence()` + atomic counter for completion, second pass reads and applies.

- **c) Approximate normalization**: Use the previous step's second-moment statistics (stale by 1 step). Eliminates kernel 1 entirely — single kernel does everything. The staleness is negligible in practice since second moments change slowly. This is the simplest approach.

### 3. Multi-Parameter Batching (Eliminate Per-Param Python Loop)

**Impact: HIGH** | **Effort: HIGH**

Currently the Python `step()` loops over each parameter individually, calling the kernel once per param. This causes:
- Many small kernel launches (bad GPU utilization for small tensors)
- Python loop overhead between launches
- Separate `second_moment.zero_()` + allocation per param

**Approach**: Batch all CELO2 parameters into a single kernel launch:
- Pre-flatten all CELO2 params, grads, and state into contiguous buffers (or use pointer arrays)
- Launch one kernel that processes all parameters
- Use segment descriptors (start offset, size, factored dims) to tell each thread block which parameter it belongs to
- Accumulate second moments per-parameter-segment using cooperative groups or atomics

### 4. Newton-Schulz CUDA Kernel

**Impact: MEDIUM-HIGH** | **Effort: MEDIUM**

Newton-Schulz is 5 iterations of: `A = X @ X^T`, `B = c1*A + c2*(A@A)`, `X = c0*X + B@X`.

Currently runs in Python using PyTorch matmul. For small matrices (e.g., 64x64) this is memory-bound with poor GPU utilization. Options:

- **a) Custom CUDA kernel**: Use shared memory for small matrices. Each thread block handles one matrix. Avoids launch overhead of 5 separate matmul calls.

- **b) `torch.compile` / Triton**: JIT-compile the Newton-Schulz loop. Zero implementation effort, moderate speedup from kernel fusion.

- **c) Fuse into apply kernel for small matrices**: If the matrix fits in shared memory (e.g., 64x64 = 16KB in float32), load the entire step output, run NS iterations in shared memory, and write back. Avoids a separate kernel entirely.

### 5. Cache MLP Weight Dtype Casts

**Impact: LOW-MEDIUM** | **Effort: LOW**

Currently `.to(grad.dtype)` is called on 6 MLP weight tensors every step. Since the MLP weights are frozen, cache them at init time in the expected dtype:

```python
# In __init__, after loading network:
self._mlp_weights_f32 = {
    "input_w": self.network.first_layer.weight.data.contiguous(),
    "input_b": self.network.first_layer.bias.data.contiguous(),
    ...
}
```

### 6. Pre-Allocate Step Output Buffer

**Impact: LOW** | **Effort: LOW**

`step_out = torch.empty_like(p)` allocates a new tensor every step. Pre-allocate during state init and reuse:

```python
state["step_out"] = torch.empty_like(p)
```

### 7. Fuse Output Normalization + Scaling + Parameter Update

**Impact: MEDIUM** | **Effort: LOW-MEDIUM**

After the kernel writes `step_out`, Python runs:
```python
step = second_moment_normalizer(step, axis=norm_axis)  # element-wise + reduction
step = step * self.rmsmult                              # element-wise
p.add_(step, alpha=-lr)                                 # element-wise
p.add_(p, alpha=-lr * weight_decay)                     # element-wise
```

That's 4 separate element-wise passes. Fuse into a single CUDA kernel or write a small custom kernel that:
1. Reads `step_out`, computes RMS normalization
2. Applies scaling, learning rate, and weight decay
3. Writes updated `p` directly

### 8. Use `__ldg` / Read-Only Cache for State Tensors

**Impact: LOW** | **Effort: LOW**

The apply kernel already uses `__ldg` for MLP weights. Extend to momentum/RMS/factored state reads (which are read-only if accumulator updates are fused into a separate write path or done in a prior phase).

### 9. Register Pressure / Occupancy Tuning

**Impact: LOW-MEDIUM** | **Effort: MEDIUM**

Current kernel uses 30 floats for features + 8 for activations + 8 for hidden + 3 for output = ~49 registers per thread, plus MLP weights loaded via `__ldg`. Profile with `ncu` to check:
- Actual register usage and occupancy
- Whether `__launch_bounds__` would help
- Optimal `BLOCK_SIZE` (currently 256, may be sub-optimal for this register count)

### 10. Half-Precision (FP16/BF16) Support

**Impact: MEDIUM** | **Effort: MEDIUM**

The kernel already dispatches via `AT_DISPATCH_FLOATING_TYPES_AND_HALF` but accumulates in the input dtype. For half-precision training:
- Keep feature accumulation in float32 (already done for `second_moment`)
- MLP computation can stay in float16 (8x8 matmuls are small, precision loss minimal)
- Enables AMP training with CELO2

Currently AMP is force-disabled. This would unlock it.

### 11. Fuse Gradient Clipping into Kernel

**Impact: LOW** | **Effort: LOW**

`grad = torch.clamp(grad, -1000.0, 1000.0)` is a separate kernel launch. The feature construction already clips to [-0.1, 0.1] for feature[1]. Add the [-1000, 1000] clamp at the point of reading `grad[idx]` in the kernel and skip the Python clamp entirely.

### 12. Multi-Stream Overlap

**Impact: LOW-MEDIUM** | **Effort: MEDIUM**

Launch CELO2 kernels for different parameters on different CUDA streams to overlap execution. Particularly useful when parameters vary widely in size (small params finish while large ones are still running).

---

## Recommended Priority Order

| Priority | Strategy | Impact | Effort |
|----------|----------|--------|--------|
| 1 | Cache MLP weight casts (#5) | Low-Med | Low |
| 2 | Pre-allocate step buffer (#6) | Low | Low |
| 3 | Fuse grad clipping into kernel (#11) | Low | Low |
| 4 | Fuse accumulator updates into kernel (#1) | High | Medium |
| 5 | Stale second-moments / single kernel (#2c) | High | Medium |
| 6 | Fuse post-kernel ops (#7) | Medium | Low-Med |
| 7 | Newton-Schulz kernel or torch.compile (#4b) | Med-High | Low-Med |
| 8 | Multi-parameter batching (#3) | High | High |
| 9 | Register / occupancy tuning (#9) | Low-Med | Medium |
| 10 | FP16/BF16 support (#10) | Medium | Medium |
| 11 | Cooperative groups single kernel (#2a) | High | High |
| 12 | Multi-stream overlap (#12) | Low-Med | Medium |

Items 1-3 are quick wins. Item 4-5 give the biggest bang for the buck. Items 8 and 11 are the most impactful but require significant refactoring.

---

## Profiling Checklist

Before optimizing, profile with `ncu` (NVIDIA Nsight Compute) to identify the actual bottleneck:

```bash
ncu --set full -o celo2_profile python scripts/compare_celo2_naive_cuda.py --steps 3
```

Key metrics to check:
- [ ] Kernel launch overhead vs compute time (are we launch-bound?)
- [ ] Memory throughput (are we bandwidth-bound?)
- [ ] Occupancy (are we register/shared-memory limited?)
- [ ] Python-side time vs kernel time (use `torch.cuda.Event` for timing)
- [ ] Per-parameter kernel time distribution (small vs large params)
