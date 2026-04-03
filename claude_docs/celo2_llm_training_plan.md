# CELO2 LLM Training Plan: 150M LLaMA on FineWeb

**Created:** 2026-04-03  
**Status:** Draft — awaiting review

## Objective

Train a 150M parameter LLaMA-style language model on 3B tokens from FineWeb-Edu using the CELO2 CUDA optimizer across 8x NVIDIA RTX A6000 (48GB each). Goal is to validate CELO2 at scale beyond vision tasks.

---

## Hardware

- 8x NVIDIA RTX A6000, 49GB VRAM each
- Total VRAM: ~384GB
- Interconnect: assumed PCIe (no NVLink) — favors FSDP/DDP over tensor parallelism

---

## Model: LLaMA 150M

Use HuggingFace `transformers.LlamaForCausalLM` with a custom config sized to ~150M params.

| Hyperparameter | Value | Notes |
|---|---|---|
| `hidden_size` | 768 | |
| `intermediate_size` | 2048 | ~2.67x hidden (LLaMA convention) |
| `num_hidden_layers` | 12 | |
| `num_attention_heads` | 12 | head_dim = 64 |
| `num_key_value_heads` | 12 | Full MHA (no GQA at this scale) |
| `max_position_embeddings` | 2048 | |
| `vocab_size` | 32000 | LLaMA tokenizer |
| `rms_norm_eps` | 1e-5 | |
| `rope_theta` | 10000 | |

Estimated param count: ~150M (embedding ~24.6M, transformer ~100M, lm_head tied ~24.6M).

**Why HF LlamaForCausalLM**: Pre-built, well-tested, includes RoPE, RMSNorm, SwiGLU, and KV-cache. No need to reimplement.

---

## Data: FineWeb-Edu

Use `HuggingFaceFW/fineweb-edu` from HuggingFace Datasets (streaming mode to avoid full download).

| Setting | Value |
|---|---|
| Dataset | `HuggingFaceFW/fineweb-edu` |
| Subset | `sample-10BT` (10B token subset, we use first 3B) |
| Tokenizer | `meta-llama/Llama-2-7b-hf` tokenizer (32k vocab, BPE) |
| Sequence length | 2048 |
| Loading | Streaming (`datasets` library, no full download) |

**Token budget**: 3B tokens = ~1.46M sequences of length 2048.

**Data pipeline**:
1. Stream from HF Hub → tokenize on-the-fly with `AutoTokenizer`
2. Pack sequences with concatenation + `eos` separator (no padding waste)
3. Chunk into fixed 2048-length blocks
4. `DataLoader` with `DistributedSampler` for multi-GPU

---

## Distributed Strategy: DDP

Use PyTorch DDP (DistributedDataParallel) for multi-GPU training.

| Setting | Value | Rationale |
|---|---|---|
| Strategy | DDP (`torch.nn.parallel.DistributedDataParallel`) | Simple, well-tested, no optimizer state sharding complications |
| Mixed precision | **Disabled** | CELO2 needs float32 gradients; use float32 compute throughout |
| Gradient sync | Automatic (DDP all-reduce) | Averaged gradients across 8 GPUs before optimizer step |
| Activation checkpointing | Per decoder layer | Saves memory, enables larger micro batch |

**Why DDP**: 150M model + optimizer state fits comfortably on a single 48GB A6000. DDP is the simplest multi-GPU approach and has zero interaction issues with custom optimizers — each GPU gets a full copy of params and state, CELO2 runs identically to single-GPU. No sharding/unsharding complications.

**CELO2 + DDP**: Works out of the box. Call `set_param_names()` on the unwrapped model (`model.module`). DDP only touches gradient all-reduce, not the optimizer step.

---

## Training Hyperparameters

| Hyperparameter | Value | Notes |
|---|---|---|
| Global batch size | 512 sequences (1M tokens) | 64 per GPU × 8 GPUs |
| Micro batch size | 8 per GPU | With gradient accumulation steps = 8 |
| Sequence length | 2048 | |
| Total steps | 3000 | 3B tokens / 1M tokens per step |
| Learning rate | 1e-3 | CELO2 default; sweep if needed |
| LR schedule | Cosine decay to 1e-4 | With 300 step warmup (linear) |
| Weight decay | 0.0 | Start without; add if overfitting |
| Gradient clipping | 1.0 (global norm) | Before CELO2's internal [-1000] clamp |
| Eval interval | Every 100 steps | |
| Checkpoint interval | Every 500 steps | |

**Tokens/step**: 512 × 2048 = 1,048,576 ≈ 1M  
**Total steps for 3B tokens**: 3,000,000,000 / 1,048,576 ≈ 2861 → round to 3000

---

## Script Structure

Single training script: `examples/llm-fineweb/train_llama.py`

```
examples/llm-fineweb/
├── train_llama.py         # Main training script (torchrun entry point)
├── config.py              # Model + training config dataclass
└── README.md              # Usage instructions
```

### Key components of `train_llama.py`:

1. **Config**: Dataclass with all hyperparameters, CLI-overridable
2. **Data**: HF `datasets.load_dataset(streaming=True)` → tokenize → pack → DataLoader
3. **Model**: `LlamaForCausalLM(LlamaConfig(...))` → DDP wrap
4. **Optimizer**: `Celo2_cuda` (or `Celo2_naive` via flag) with `set_param_names`
5. **Training loop**: Standard forward/backward/step with gradient accumulation
6. **Logging**: wandb (optional) + stdout + JSON log file
7. **Checkpointing**: Standard `torch.save` on rank 0 (DDP — full state on every GPU)
8. **Eval**: Perplexity on held-out FineWeb split every N steps

### Launch command:

```bash
torchrun --nproc_per_node=8 examples/llm-fineweb/train_llama.py \
    --optimizer celo2_cuda \
    --lr 1e-3 \
    --batch_size 8 \
    --grad_accum_steps 8 \
    --max_steps 3000 \
    --eval_interval 100
```

---

## Dependencies to Install

```bash
pip install transformers datasets tokenizers wandb tqdm
```

The `transformers` library provides `LlamaForCausalLM`, `LlamaConfig`, `AutoTokenizer`.  
The `datasets` library provides streaming access to FineWeb.

---

## Estimated Resource Usage

| Resource | Per GPU | Notes |
|---|---|---|
| Model params (fp32) | ~600MB | Full copy on each GPU (DDP) |
| Optimizer state (CELO2) | ~3.5GB | Full copy on each GPU (mom×3, RMS, factored accumulators) |
| Activations (with checkpointing) | ~4GB | Per-layer recomputation |
| Gradient buffers | ~600MB | All-reduced by DDP |
| **Total VRAM** | **~8.7GB** | Well within 48GB |

Plenty of headroom — can increase batch size or model size if needed.

**Estimated throughput**: ~50-80k tokens/sec across 8 A6000s (conservative; depends on CELO2 kernel overhead vs AdamW). Full 3B token run: ~10-17 hours.

---

## Baseline Comparison

Run the same setup with AdamW to establish a baseline:

```bash
torchrun --nproc_per_node=8 examples/llm-fineweb/train_llama.py \
    --optimizer adamw \
    --lr 3e-4 \
    --batch_size 8 \
    --grad_accum_steps 8 \
    --max_steps 3000
```

Track: loss curves, final perplexity, tokens/sec, optimizer step time.

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Slow data loading from HF streaming | Pre-tokenize first epoch to disk cache; use multiple dataloader workers |
| OOM with gradient accumulation | Activation checkpointing already planned; reduce micro batch if needed |
| CELO2 diverges at this scale | Start with naive version to verify, then switch to CUDA. Have AdamW baseline for comparison. |
| DDP gradient sync overhead | 150M params → ~600MB all-reduce per step; fast on PCIe with 8 GPUs. Not a bottleneck. |

---

## Implementation Steps

1. [ ] Install dependencies (`transformers`, `datasets`, `tokenizers`, `wandb`)
2. [ ] Create `examples/llm-fineweb/` directory and config
3. [ ] Implement data pipeline (streaming FineWeb → tokenized DataLoader)
4. [ ] Implement model setup (LlamaConfig → LlamaForCausalLM → DDP)
5. [ ] Integrate CELO2 optimizer with `set_param_names(model.module)`
6. [ ] Implement training loop with gradient accumulation + eval
7. [ ] Test single-GPU run (1 step) to verify pipeline
8. [ ] Test 8-GPU DDP run (10 steps) to verify distributed setup
9. [ ] Run AdamW baseline (3000 steps)
10. [ ] Run CELO2 CUDA (3000 steps)
11. [ ] Compare loss curves, perplexity, throughput
