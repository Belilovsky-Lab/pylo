# CELO2 LLM Training Experiment Report

**Date:** 2026-04-05  
**Status:** Complete — full 3.2B token dataset, 1 epoch (6100 steps), both optimizers finished

## Setup

| | Value |
|---|---|
| **Model** | LLaMA-style, 137.8M params (16 layers, 768 hidden, 12 heads, SwiGLU, RoPE, RMSNorm) |
| **Data** | FineWeb-Edu, 3.20B pre-tokenized tokens (1.56M sequences of 2048) |
| **Tokenizer** | Mistral-7B-v0.1 (32k vocab, BPE) |
| **Hardware** | 8x NVIDIA RTX A6000 (48GB), runs executed sequentially |
| **Distributed** | DDP (DistributedDataParallel) |
| **Batch** | 256 sequences/step (8 GPU x 8 micro-batch x 4 grad_accum) |
| **Tokens/step** | 524,288 (~0.52M) |
| **Total steps** | 6100 (~3.2B tokens, ~1 epoch) |
| **LR schedule** | Cosine decay with 600-step linear warmup |
| **Gradient clipping** | 1.0 (global norm) |
| **Activation checkpointing** | Yes (per decoder layer) |

### Optimizer Configs

| | CELO2 CUDA | AdamW |
|---|---|---|
| **Peak learning rate** | 1e-3 | 3e-4 |
| **Weight decay** | 0.0 | 0.0 |
| **Betas** | N/A (learned MLP) | (0.9, 0.95) |
| **Newton-Schulz** | 5 iterations | N/A |

---

## Results

### Eval Perplexity Over Training

| Step | CELO2 Eval PPL | AdamW Eval PPL | Winner |
|------|---------------|----------------|--------|
| 500 | **192.2** | 445.2 | CELO2 (2.3x lower) |
| 1000 | **126.6** | 209.2 | CELO2 (1.7x lower) |
| 1500 | **111.4** | 146.8 | CELO2 (1.3x lower) |
| 2000 | **104.5** | 124.4 | CELO2 (1.2x lower) |
| 2500 | **99.4** | 111.1 | CELO2 (1.1x lower) |
| 3000 | **95.8** | 102.8 | CELO2 |
| 3500 | 99.7 | **97.3** | AdamW |
| 4000 | 1598.3 ⚠️ | **93.0** | AdamW |
| 4500 | 148.9 | **89.8** | AdamW |
| 5000 | 110.6 | **87.7** | AdamW |
| 5500 | 101.0 | **86.3** | AdamW |
| **6000** | **96.4** | **85.4** | **AdamW** |

### Final Results

| Metric | CELO2 CUDA | AdamW |
|---|---|---|
| **Best eval PPL** | **95.8** (step 3000) | **85.4** (step 6000) |
| **Final eval PPL** | 96.4 | **85.4** |
| **Final train loss** | 4.64 | 4.53 |
| **Total wall time** | 13.18 hours | 12.76 hours |
| **Avg throughput** | 67,427 tok/s | 69,640 tok/s |
| **Step time** | 7.73s | 7.51s |
| **Optimizer step** | 259ms (3.3%) | 17ms (0.2%) |

---

## Key Findings

### 1. CELO2 dominates early, AdamW wins in the long run

**CELO2 converges dramatically faster in the first ~3000 steps:**
- At step 500: CELO2 PPL 192 vs AdamW 445 (2.3x better)
- At step 1000: CELO2 PPL 127 vs AdamW 209 (1.7x better)  
- At step 3000 (1 epoch): CELO2 PPL 95.8 vs AdamW 102.8

**Then the advantage reverses:**
- Around step 3100-4000, CELO2 encounters a major instability (eval PPL spikes to 1598!)
- AdamW has no such spike and continues smooth improvement
- Final: AdamW PPL 85.4 vs CELO2 PPL 96.4 — AdamW is 1.13x better

### 2. CELO2 has a training instability at the data epoch boundary

Looking at the train loss trajectory:
- Steps 1-3100: smooth convergence from loss 9.5 → 4.61 (eval PPL 95.8)
- **Step 3150: loss spike to 6.05**, then recovery to 4.65
- **Step 3900-4000: larger spike to 7.6** (eval PPL 1598 at step 4000)
- Steps 4000-6100: gradual recovery to loss 4.64

This behavior is **not an epoch boundary** (6100 steps = 1 full epoch). It's unclear what causes it — possibly the DataLoader's shuffle period, a numerical instability in Newton-Schulz, or interaction between the learned optimizer and specific data batches.

### 3. Throughput is essentially equal

Both optimizers run at ~68-70k tokens/sec on 8 GPUs. CELO2's 259ms optimizer step is only 3.3% of the 7.73s total step time. The earlier observation of "2x throughput difference" on 4-GPU runs was likely an artifact (possibly DataLoader worker count or CUDA memory contention).

### 4. Perplexity is now in a reasonable range

Final eval PPL of 85.4 (AdamW) and 95.8 (CELO2 at its best) is a major improvement over the earlier undertrained runs (PPL 127-132). For a 138M param model on ~3B tokens, this is approaching the expected ballpark though still higher than fully-converged models (Chinchilla-optimal would train longer).

---

## Training Loss Trajectory

**CELO2 CUDA:**

| Step | Train Loss | Train PPL | LR |
|------|-----------|-----------|-----|
| 500 | 5.38 | 217 | 8.3e-4 |
| 1000 | 4.93 | 139 | 9.9e-4 |
| 1500 | 4.79 | 120 | 9.3e-4 |
| 2000 | 4.73 | 114 | 8.1e-4 |
| 2500 | 4.65 | 105 | 6.3e-4 |
| 3000 | 4.62 | 102 | 4.4e-4 |
| 3100 | **6.05 ⚠️** | 423 | 4.1e-4 |
| 4000 | **7.48 ⚠️** | 1772 | 2.0e-4 |
| 5000 | 4.79 | 120 | 1.9e-4 |
| 6000 | 4.65 | 105 | 1.0e-4 |

**AdamW:**

| Step | Train Loss | Train PPL | LR |
|------|-----------|-----------|-----|
| 500 | — | — | 2.5e-4 |
| 1000 | — | — | 3.0e-4 |
| 2000 | — | — | 2.6e-4 |
| 3000 | — | — | 1.8e-4 |
| 4000 | — | — | 1.0e-4 |
| 5000 | — | — | 4.8e-5 |
| 6000 | 4.53 | 93 | 3.0e-5 |

---

## Comparison at Matched Wall Time

Both runs took ~13 hours for 6100 steps (identical throughput), so step-matched comparison = wall-time-matched:

- **First 6 hours** (step 3000): **CELO2 wins** (95.8 vs 102.8)
- **Full 13 hours** (step 6000): **AdamW wins** (85.4 vs 96.4)

---

## Previous Experiment (1000 steps, 85M tokens)

On the smaller 85M token dataset with 1000 steps:
- CELO2: eval PPL 126.2 (1.84h)
- AdamW: eval PPL 276.4 (1.81h)
- CELO2 won by 2.2x

With more data and longer training, the gap closes and eventually reverses.

---

## Limitations & Observations

- Did not sweep LR for either optimizer — CELO2 at 1e-3, AdamW at 3e-4
- CELO2's instability at step 3100-4000 needs investigation (may be fixable with lower LR at that phase, or gradient clipping tuning)
- Only ~1 epoch trained — for truly optimal training, larger data + multiple epochs would help
- CELO2 warmup is to LR 1e-3 which is 3.3x AdamW's peak; a fairer comparison would sweep both

## Next Steps

- **Investigate CELO2 instability at step 3100-4000**: Was it a specific batch? Newton-Schulz issue? LR-too-high issue?
- LR sweep for both optimizers
- Test with `exp_mult > 0` to see if CELO2 magnitude output stabilizes late-training dynamics
- Compare CELO2 naive vs CUDA at scale to verify kernel correctness doesn't contribute to the instability
- Run with gradient clipping lower (e.g., 0.5) to see if it prevents the spike

---

## Reproducibility

```bash
# Build CUDA extensions
python setup.py install --cuda

# Pre-tokenize 3.2B tokens (requires HF auth for speed)
# Cached at outputs/llm-fineweb/data/train_tokens_3B.bin

# CELO2 CUDA (8 GPUs, full epoch):
torchrun --nproc_per_node=8 examples/llm-fineweb/train_llama.py \
    --optimizer celo2_cuda --max_steps 6100 --batch_size 8 --grad_accum_steps 4 --warmup_steps 600

# AdamW baseline (8 GPUs, full epoch):
torchrun --nproc_per_node=8 examples/llm-fineweb/train_llama.py \
    --optimizer adamw --max_steps 6100 --batch_size 8 --grad_accum_steps 4 --warmup_steps 600
```
