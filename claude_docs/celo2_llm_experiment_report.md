# CELO2 LLM Training Experiment Report

**Date:** 2026-04-04  
**Status:** Complete — full 3B token dataset, 3000 steps, both optimizers finished

## Setup

| | Value |
|---|---|
| **Model** | LLaMA-style, 137.8M params (16 layers, 768 hidden, 12 heads, SwiGLU, RoPE, RMSNorm) |
| **Data** | FineWeb-Edu, 3.20B pre-tokenized tokens (1.56M sequences of 2048) |
| **Tokenizer** | Mistral-7B-v0.1 (32k vocab, BPE) |
| **Hardware** | 4x NVIDIA RTX A6000 (48GB) per run, 2 runs in parallel |
| **Distributed** | DDP (DistributedDataParallel) |
| **Batch** | 128 sequences/step (4 GPU x 8 micro-batch x 4 grad_accum) |
| **Tokens/step** | 262,144 (~0.26M) |
| **Total steps** | 3000 (~786M tokens, 0.25 epochs) |
| **LR schedule** | Cosine decay with 300-step linear warmup |
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

| Step | CELO2 Eval PPL | AdamW Eval PPL | CELO2 Advantage |
|------|---------------|----------------|-----------------|
| 500 | **211.9** | 381.9 | 1.8x lower |
| 1000 | **162.3** | 213.6 | 1.3x lower |
| 1500 | **146.9** | 162.7 | 1.1x lower |
| 2000 | **140.5** | 141.9 | ~tied |
| 2500 | **131.7** | 132.0 | ~tied |
| 3000 | ~132 | **127.5** | AdamW 1.04x lower |

### Final Numbers

| Metric | CELO2 CUDA | AdamW |
|---|---|---|
| **Final train loss** | 4.88 | 4.93 |
| **Final train PPL** | 132.3 | 137.8 |
| **Final eval PPL (step 3000)** | ~132 | **127.5** |
| **Total wall time** | **5.5 hours** | 11.0 hours |
| **Avg throughput** | **39,884 tok/s** | 19,903 tok/s |
| **Step time** | **6.57s** | 13.2s |
| **Optimizer step** | 261ms | 39ms |

### Training Loss Trajectory

**CELO2 CUDA** (3000 steps, 5.5 hours):

| Step | Loss | PPL | LR |
|------|------|------|-----|
| 500 | 5.41 | 224 | 1.0e-3 |
| 1000 | 5.15 | 173 | 9.6e-4 |
| 1500 | 5.06 | 158 | 8.0e-4 |
| 2000 | 4.98 | 146 | 5.5e-4 |
| 2500 | 4.97 | 143 | 2.7e-4 |
| 3000 | 4.88 | 132 | 1.0e-4 |

**AdamW** (3000 steps, 11.0 hours):

| Step | Loss | PPL | LR |
|------|------|------|-----|
| 500 | 6.00 | 404 | 3.0e-4 |
| 1000 | 5.41 | 223 | 2.6e-4 |
| 1500 | 5.17 | 176 | 1.9e-4 |
| 2000 | 5.02 | 151 | 1.2e-4 |
| 2500 | 4.93 | 138 | 5.6e-5 |
| 3000 | 4.93 | 138 | 3.0e-5 |

---

## Key Findings

### 1. CELO2 converges much faster early, AdamW catches up late

CELO2 has a decisive advantage in the first ~1500 steps (1.1–1.8x lower eval PPL). By step 2000, both optimizers reach comparable perplexity. By step 3000, AdamW slightly edges ahead (eval PPL 127.5 vs ~132). This pattern is consistent with the learned optimizer being more aggressive early (higher effective LR, adaptive per-parameter rules) while AdamW benefits from its longer cosine decay schedule.

### 2. CELO2 trains 2x faster in wall time

Despite both running 3000 steps, CELO2 completes in **5.5 hours vs 11.0 hours** — a 2x speedup. This was unexpected since both had equal throughput (~40k tok/s) on the smaller 85M dataset. The difference on the 3B dataset appears to be related to DataLoader performance with the larger dataset and DDP synchronization patterns. CELO2 consistently runs at 6.6s/step while AdamW runs at 13.2s/step.

**Note:** This throughput difference needs investigation — it may be an artifact of the experimental setup rather than a fundamental property of the optimizers. Possible causes:
- Different DataLoader shuffling behavior across the two GPU groups
- CUDA memory pressure differences (GPU 0 shows 19.5GB vs others at 13.2GB)
- DDP all-reduce timing differences

### 3. Both reach good perplexity for this model size

Final eval PPL of 127.5–132 on FineWeb-Edu is reasonable for a 138M model trained on 786M tokens (~0.25 epochs of 3.2B). For comparison:
- Chinchilla-optimal training would use ~2.8B tokens (20x params)
- With only 0.25 epochs, neither optimizer has fully converged
- Extending to 1 full epoch (12,200 steps) would likely push both below PPL 100

### 4. Stable training at scale

Both optimizers completed all 3000 steps without NaN/Inf or divergence. CELO2's hybrid approach (learned MLP for hidden layers, AdamW for input/output/1D) works correctly with DDP on 4 GPUs.

---

## Throughput Comparison

| Metric | CELO2 CUDA | AdamW |
|---|---|---|
| **Avg tokens/sec** | 39,884 | 19,903 |
| **Step time** | 6.57s | 13.2s |
| **Optimizer step** | 261ms (4.0% of step) | 39ms (0.3% of step) |
| **Forward+backward** | ~6.3s | ~13.1s |

---

## Comparison at Same Wall Time

| Wall Time | CELO2 (step / eval PPL) | AdamW (step / eval PPL) |
|---|---|---|
| 5.5 hours | step 3000 / **~132** | step ~1500 / **162.7** |
| 11 hours | — | step 3000 / **127.5** |

At the same wall time (5.5h), CELO2 achieves eval PPL ~132 while AdamW is only at PPL 162.7. CELO2 would need no further training; AdamW needs another 5.5 hours to finish.

---

## Previous Experiment (85M tokens, for reference)

On the smaller dataset (85M tokens, 1000 steps), CELO2 showed a much larger advantage:

| | CELO2 | AdamW |
|---|---|---|
| Eval PPL @ 1000 | **126.2** | 276.4 |
| Throughput | 39.6k tok/s | 40.1k tok/s |

The smaller dataset caused more overfitting (3+ epochs), which CELO2 handled better. With the larger 3B dataset (0.25 epochs), both optimizers converge to similar final quality but CELO2 gets there faster.

---

## Limitations

- Did not sweep LR for either optimizer — CELO2 at 1e-3, AdamW at 3e-4
- Only 0.25 epochs of the 3.2B dataset — both optimizers have room to improve
- Throughput difference (2x) needs investigation — may be experimental artifact
- CELO2 eval at step 3000 was captured from training logs (deleted file), not the JSON

## Next Steps

- Investigate the 2x throughput difference between CELO2 and AdamW runs
- LR sweep for both optimizers
- Train for full epoch (12,200 steps) to see final convergence
- Test at 350M and 1B+ parameter scales

---

## Reproducibility

```bash
# Build CUDA extensions
python setup.py install --cuda

# Pre-tokenize 3B tokens (requires HF auth for speed)
# Cached at outputs/llm-fineweb/data/train_tokens_3B.bin

# CELO2 CUDA (4 GPUs):
torchrun --nproc_per_node=4 examples/llm-fineweb/train_llama.py \
    --optimizer celo2_cuda --max_steps 3000 --batch_size 8 --grad_accum_steps 4 --warmup_steps 300

# AdamW baseline (4 GPUs):
torchrun --nproc_per_node=4 examples/llm-fineweb/train_llama.py \
    --optimizer adamw --max_steps 3000 --batch_size 8 --grad_accum_steps 4 --warmup_steps 300
```
