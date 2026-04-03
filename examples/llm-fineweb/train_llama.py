"""Train a 150M LLaMA-style model on FineWeb-Edu with CELO2 or AdamW.

Usage:
    # Single GPU test:
    python examples/llm-fineweb/train_llama.py --max_steps 1

    # 8-GPU DDP:
    torchrun --nproc_per_node=8 examples/llm-fineweb/train_llama.py --optimizer celo2_cuda

    # AdamW baseline:
    torchrun --nproc_per_node=8 examples/llm-fineweb/train_llama.py --optimizer adamw
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from itertools import chain
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset

from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from datasets import load_dataset

# Add project root to path for pylo imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ── Config ───────────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    # Model
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    max_position_embeddings: int = 2048
    vocab_size: int = 32000
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True

    # Data
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_subset: str = "sample-10BT"
    tokenizer_name: str = "mistralai/Mistral-7B-v0.1"
    seq_length: int = 2048

    # Training
    optimizer: str = "celo2_cuda"  # "celo2_cuda", "celo2_naive", "adamw"
    lr: float = 1e-3
    adamw_lr: float = 3e-4
    weight_decay: float = 0.0
    max_steps: int = 3000
    batch_size: int = 8  # micro batch per GPU
    grad_accum_steps: int = 8
    warmup_steps: int = 300
    lr_min_ratio: float = 0.1  # min_lr = lr * ratio
    max_grad_norm: float = 1.0

    # Eval & logging
    eval_interval: int = 100
    eval_steps: int = 20
    log_interval: int = 10
    checkpoint_interval: int = 500
    output_dir: str = "outputs/llm-fineweb"
    wandb_project: str = "celo2-llm"
    no_wandb: bool = False

    # System
    activation_checkpointing: bool = True
    seed: int = 42
    compile_model: bool = False


def parse_args() -> TrainConfig:
    config = TrainConfig()
    parser = argparse.ArgumentParser()
    for k, v in asdict(config).items():
        t = type(v)
        if t == bool:
            parser.add_argument(f"--{k}", action="store_true", default=v)
        else:
            parser.add_argument(f"--{k}", type=t, default=v)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


# ── Data ─────────────────────────────────────────────────────────────────────


import numpy as np


class MemmapTokenDataset(torch.utils.data.Dataset):
    """Memory-mapped dataset of pre-tokenized packed sequences."""

    def __init__(self, data_path: str, seq_length: int, rank: int = 0, world_size: int = 1):
        self.seq_length = seq_length
        self.tokens = np.memmap(data_path, dtype=np.uint16, mode="r")
        # Number of full sequences we can make
        self.n_sequences = len(self.tokens) // (seq_length + 1)
        # Shard across GPUs
        self.rank = rank
        self.world_size = world_size
        self.shard_size = self.n_sequences // world_size
        self.offset = rank * self.shard_size

    def __len__(self):
        return self.shard_size

    def __getitem__(self, idx):
        start = (self.offset + idx) * (self.seq_length + 1)
        chunk = self.tokens[start : start + self.seq_length + 1].astype(np.int64)
        return {
            "input_ids": torch.from_numpy(chunk[:-1]),
            "labels": torch.from_numpy(chunk[1:]),
        }


def prepare_data(config: TrainConfig, tokenizer, target_tokens: int = 800_000_000):
    """Pre-tokenize FineWeb to a memmap file. Only runs on rank 0."""
    data_dir = Path(config.output_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "train_tokens.bin"

    if data_path.exists():
        tokens = np.memmap(str(data_path), dtype=np.uint16, mode="r")
        print(f"Using cached tokenized data: {len(tokens)/1e9:.2f}B tokens at {data_path}")
        return str(data_path)

    print(f"Pre-tokenizing FineWeb to {data_path} (target: {target_tokens/1e9:.1f}B tokens)...")
    ds = load_dataset(config.dataset_name, config.dataset_subset, split="train", streaming=True)

    # Tokenize in chunks and write to memmap
    all_tokens = []
    total = 0
    t0 = time.time()

    for i, example in enumerate(ds):
        tokens = tokenizer(example["text"], add_special_tokens=False, return_attention_mask=False)["input_ids"]
        all_tokens.extend(tokens + [tokenizer.eos_token_id])
        total = len(all_tokens)

        if i % 10000 == 0 and i > 0:
            elapsed = time.time() - t0
            rate = total / elapsed
            print(f"  {total/1e6:.1f}M tokens ({i} docs) | {rate/1e6:.2f}M tok/s | {elapsed:.0f}s")

        if total >= target_tokens:
            break

    # Align to seq_length + 1
    n_seqs = total // (config.seq_length + 1)
    total_aligned = n_seqs * (config.seq_length + 1)
    all_tokens = all_tokens[:total_aligned]

    # Write as uint16 (vocab_size=32000 fits in uint16)
    arr = np.array(all_tokens, dtype=np.uint16)
    mm = np.memmap(str(data_path), dtype=np.uint16, mode="w+", shape=arr.shape)
    mm[:] = arr[:]
    mm.flush()
    del mm, arr, all_tokens

    print(f"Saved {total_aligned/1e9:.2f}B tokens ({n_seqs} sequences) to {data_path}")
    print(f"Time: {(time.time()-t0)/60:.1f} min")
    return str(data_path)


# ── Model ────────────────────────────────────────────────────────────────────


def create_model(config: TrainConfig) -> LlamaForCausalLM:
    model_config = LlamaConfig(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_position_embeddings=config.max_position_embeddings,
        vocab_size=config.vocab_size,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        tie_word_embeddings=config.tie_word_embeddings,
        use_cache=False,  # Not needed for training
    )
    model = LlamaForCausalLM(model_config)
    return model


# ── Optimizer ────────────────────────────────────────────────────────────────


def create_optimizer(config: TrainConfig, model: torch.nn.Module):
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.adamw_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "celo2_cuda":
        from pylo.optim.Celo2_cuda import Celo2_cuda
        opt = Celo2_cuda(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        opt.set_param_names(model)
        return opt
    elif config.optimizer == "celo2_naive":
        from pylo.optim.Celo2_naive import Celo2_naive
        opt = Celo2_naive(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        opt.set_param_names(model)
        return opt
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


# ── LR Schedule ──────────────────────────────────────────────────────────────


def get_lr(step: int, config: TrainConfig) -> float:
    """Cosine decay with linear warmup."""
    base_lr = config.adamw_lr if config.optimizer == "adamw" else config.lr
    min_lr = base_lr * config.lr_min_ratio

    if step < config.warmup_steps:
        return base_lr * step / config.warmup_steps

    if step >= config.max_steps:
        return min_lr

    progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ── Training ─────────────────────────────────────────────────────────────────


def train(config: TrainConfig):
    # ── Distributed setup ────────────────────────────────────────────────
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = rank == 0

    # ── Seed ─────────────────────────────────────────────────────────────
    torch.manual_seed(config.seed + rank)

    # ── Output dir ───────────────────────────────────────────────────────
    run_name = f"{config.optimizer}_lr{config.lr}_bs{config.batch_size * config.grad_accum_steps * world_size}"
    output_dir = Path(config.output_dir) / run_name
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Wandb ────────────────────────────────────────────────────────────
    if is_main and not config.no_wandb:
        try:
            import wandb
            wandb.init(
                project=config.wandb_project,
                name=run_name,
                config=asdict(config),
            )
        except Exception as e:
            print(f"wandb init failed: {e}, continuing without wandb")
            config.no_wandb = True

    # ── Tokenizer ────────────────────────────────────────────────────────
    if is_main:
        print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Data ─────────────────────────────────────────────────────────────
    if is_main:
        print("Preparing data...")
    # Pre-tokenize on rank 0, then all ranks wait
    if is_main:
        data_path = prepare_data(config, tokenizer)
    if distributed:
        dist.barrier()
    if not is_main:
        data_path = str(Path(config.output_dir) / "data" / "train_tokens.bin")

    train_dataset = MemmapTokenDataset(data_path, config.seq_length, rank=rank, world_size=world_size)
    if is_main:
        print(f"Dataset: {len(train_dataset)} sequences per GPU")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # ── Model ────────────────────────────────────────────────────────────
    if is_main:
        print("Creating model...")
    model = create_model(config)
    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"Model parameters: {n_params / 1e6:.1f}M")

    model = model.to(device)

    # Activation checkpointing
    if config.activation_checkpointing:
        from torch.utils.checkpoint import checkpoint
        model.gradient_checkpointing_enable()

    # ── Optimizer (create before DDP so set_param_names sees unwrapped model) ──
    if is_main:
        print(f"Creating optimizer: {config.optimizer}")
    optimizer = create_optimizer(config, model)

    # ── DDP ──────────────────────────────────────────────────────────────
    if distributed:
        model = DDP(model, device_ids=[local_rank])

    # Optional torch.compile
    if config.compile_model:
        model = torch.compile(model)

    # ── Training loop ────────────────────────────────────────────────────
    global_batch_size = config.batch_size * config.grad_accum_steps * world_size
    tokens_per_step = global_batch_size * config.seq_length

    if is_main:
        print(f"Global batch size: {global_batch_size} sequences ({tokens_per_step / 1e6:.2f}M tokens/step)")
        print(f"Total steps: {config.max_steps}")
        print(f"Total tokens: {config.max_steps * tokens_per_step / 1e9:.2f}B")
        print("=" * 60)

    train_iter = iter(train_loader)
    model.train()

    log_data = []
    total_tokens = 0
    running_loss = 0.0
    running_steps = 0
    t_start = time.time()

    for step in range(1, config.max_steps + 1):
        step_t0 = time.time()

        # Update LR
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Gradient accumulation
        optimizer.zero_grad()
        accum_loss = 0.0

        for micro_step in range(config.grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            if distributed and micro_step < config.grad_accum_steps - 1:
                with model.no_sync():
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss / config.grad_accum_steps
                    loss.backward()
            else:
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / config.grad_accum_steps
                loss.backward()

            accum_loss += loss.item()

        # Gradient clipping
        if config.max_grad_norm > 0:
            raw_model = model.module if distributed else model
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), config.max_grad_norm)

        # Optimizer step
        opt_t0 = time.time()
        optimizer.step()
        torch.cuda.synchronize()
        opt_time = time.time() - opt_t0

        total_tokens += tokens_per_step
        running_loss += accum_loss
        running_steps += 1
        step_time = time.time() - step_t0

        # ── Logging ──────────────────────────────────────────────────────
        if step % config.log_interval == 0 and is_main:
            avg_loss = running_loss / running_steps
            tokens_sec = tokens_per_step / step_time
            elapsed = time.time() - t_start

            entry = {
                "step": step,
                "loss": round(avg_loss, 4),
                "ppl": round(math.exp(min(avg_loss, 20)), 2),
                "lr": round(lr, 6),
                "tokens_sec": round(tokens_sec),
                "opt_ms": round(opt_time * 1000, 1),
                "step_ms": round(step_time * 1000, 1),
                "total_tokens": total_tokens,
                "elapsed_min": round(elapsed / 60, 1),
            }
            log_data.append(entry)

            print(
                f"step {step:5d} | loss {avg_loss:.4f} | ppl {entry['ppl']:8.2f} | "
                f"lr {lr:.2e} | tok/s {tokens_sec:,.0f} | opt {opt_time*1000:.0f}ms | "
                f"step {step_time*1000:.0f}ms | {elapsed/60:.1f}min"
            )

            if not config.no_wandb:
                try:
                    import wandb
                    wandb.log(entry, step=step)
                except Exception:
                    pass

            running_loss = 0.0
            running_steps = 0

        # ── Eval ─────────────────────────────────────────────────────────
        if step % config.eval_interval == 0 and is_main:
            eval_loss = evaluate(model, data_path, config, device)
            print(f"  [eval] step {step} | loss {eval_loss:.4f} | ppl {math.exp(min(eval_loss, 20)):.2f}")
            if not config.no_wandb:
                try:
                    import wandb
                    wandb.log({"eval_loss": eval_loss, "eval_ppl": math.exp(min(eval_loss, 20))}, step=step)
                except Exception:
                    pass
            model.train()

        # ── Checkpoint ───────────────────────────────────────────────────
        if step % config.checkpoint_interval == 0 and is_main:
            ckpt_path = output_dir / f"checkpoint_{step}.pt"
            raw_model = model.module if distributed else model
            torch.save({
                "step": step,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(config),
                "total_tokens": total_tokens,
            }, ckpt_path)
            print(f"  [ckpt] saved to {ckpt_path}")

    # ── Save final logs ──────────────────────────────────────────────────
    if is_main:
        log_path = output_dir / "train_log.json"
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        print(f"\nTraining complete. Logs saved to {log_path}")
        total_time = time.time() - t_start
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Total tokens: {total_tokens/1e9:.2f}B")
        print(f"Avg throughput: {total_tokens/total_time:,.0f} tokens/sec")

    if distributed:
        dist.destroy_process_group()


@torch.no_grad()
def evaluate(model, data_path, config, device):
    """Run eval on a held-out slice of the data (last N sequences)."""
    model.eval()
    # Use the tail of the dataset as eval (not seen during training due to sharding)
    tokens = np.memmap(data_path, dtype=np.uint16, mode="r")
    n_total = len(tokens) // (config.seq_length + 1)
    eval_start = n_total - config.eval_steps * config.batch_size
    eval_start = max(eval_start, 0)

    total_loss = 0.0
    count = 0
    for i in range(config.eval_steps):
        start = (eval_start + i * config.batch_size) * (config.seq_length + 1)
        batch_tokens = []
        for b in range(config.batch_size):
            s = start + b * (config.seq_length + 1)
            chunk = tokens[s : s + config.seq_length + 1].astype(np.int64)
            batch_tokens.append(chunk)
        batch_tokens = np.stack(batch_tokens)
        input_ids = torch.from_numpy(batch_tokens[:, :-1]).to(device)
        labels = torch.from_numpy(batch_tokens[:, 1:]).to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        total_loss += outputs.loss.item()
        count += 1

    return total_loss / max(count, 1)


if __name__ == "__main__":
    config = parse_args()
    train(config)
