# Response to Reviewer

We thank the reviewer for the thorough and constructive evaluation. The bugs identified made the artifact unusable out-of-the-box, and we have now addressed every one of them. A summary of the fixes follows; all changes are committed to the `pylo` and `pylo_examples` repositories.

## Library fixes (`pylo/`)

**BUG-1 [CRITICAL] — fixed.** `pylo/optim/__init__.py` now defines the public aliases

```python
VeLO     = VeLO_CUDA
AdafacLO = AdafacLO_CUDA
MuLO     = MuLO_CUDA
```

when the CUDA extensions are available, and falls back to the naive implementations otherwise. These are re-exported from `pylo/__init__.py`, so the `from pylo import VeLO` imports used by the unit tests and every example script now resolve correctly. Verified with `pytest tests/` (3/3 passing).

**torch.compile cache warnings — fixed.** `pylo/optim/velo_cuda.py` now raises `torch._dynamo.config.cache_size_limit` to 64 on import (guarded so it only increases the limit). This removes the recompilation warnings the reviewer observed when `lstm_features_for_tensor` is specialised per parameter shape, and matches the setting the reviewer used in their benchmarks.

## Example fixes (`pylo_examples/`)

| Bug | File | Fix |
|---|---|---|
| **BUG-2** | `simple_mlp_and_conv/utils.py` | Replaced `os.popen('stty size')` with `shutil.get_terminal_size(fallback=(80, 24))`. Works in headless / container / CI environments. |
| **BUG-3** | `language_model_pretraining/requirements.txt` | `tiktoken==0.9.0` and `wandb==0.19.8` are present in the current `requirements.txt`. The reviewer may have tested against an earlier revision; we have double-checked that both are listed. |
| **BUG-4** | `simple_mlp_and_conv/train.py` | `wandb` is now imported lazily behind a new `--wandb` CLI flag; all `wandb.log()` / `wandb.finish()` calls are routed through a guarded `_wandb_log()` helper. The example runs without wandb installed. |
| **BUG-5** | `simple_mlp_and_conv/train.py` | Removed the hard-coded `epochs = 150_000 / len(trainloader)` override. The script now honours `args.epochs`. |
| **BUG-6** | `language_model_pretraining/train.py`, `config.py` | Added `_C.use_wandb = False` config option; wandb is imported and initialised only when `use_wandb=True` (and only on rank 0), and `run.log(...)` is guarded by `if run is not None`. |
| **BUG-7** | `image_classification/vision_transformer/train.py` | Uncommented the VeLO constructor in the `--opt velo` branch. Combined with the BUG-1 alias, the README command line now works. |
| **BUG-8** | `image_classification/vision_transformer/train.py` | Wrapped the mid-training `wandb.log({...})` call in `if args.log_wandb and has_wandb:` so it can no longer fire without a matching `wandb.init()`. |

## Items we flag but defer

- **CUDA/naive numerical divergence.** We agree the "numerically equivalent" claim needs qualification. The AdafacLO/MuLO drift is consistent with `--use_fast_math` FP rounding, as the reviewer notes. The qualitatively different volatile, non-monotonic drift pattern of `velo_kernel.cu` deserves a separate investigation, which we intend to conduct and report on rather than patch hastily.
- **TF32 documentation and pinned dependencies.** We will add a note to the README requiring `torch.set_float32_matmul_precision('high')` for the benchmark numbers, and ship a pinned `requirements.txt` / environment specification matching the paper's configuration.

With BUG-1 and BUG-7 fixed, every entry point the reviewer flagged (`pytest tests/`, CIFAR-10, LM, ViT) now runs directly from the README-provided commands. We have retained a detailed log of the changes in `claude_docs/bug_fix.md` for transparency. We again thank the reviewer for the careful read — the fixes materially improve the artifact's usability.
