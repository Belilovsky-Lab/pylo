PyLO is a PyTorch library that ports learned optimizers (neural networks trained to optimize other neural networks) from JAX to PyTorch. The library provides three learned optimizer families — AdafacLO (small_fc_lopt), VeLO, and MuLO — with both naive PyTorch implementations and fused CUDA kernel implementations. The paper's central claim is that the CUDA kernels achieve 80–86% speedup over the naive implementations while remaining numerically equivalent, making learned optimizers practical for standard PyTorch training workflows.

The artifact consists of two repositories: pylo (the core library with CUDA extensions) and pylo_examples (three training examples: CIFAR-10 MLP/ResNet, language model pre-training, and Vision Transformer on ImageNet/Imagenette).

Comments for authors
Evaluation Setup
Hardware: NVIDIA A100-SXM4-80GB (single GPU)
Software: Python 3.11.15, PyTorch 2.7.1+cu118, CUDA 11.8 (nvcc), timm 1.0.26
Artifacts Available
The source code for both pylo and pylo_examples is provided. Pre-trained learned optimizer weights are hosted on HuggingFace Hub and download automatically. The CUDA extensions compile successfully. However, no pinned dependency versions are provided — setup.py requires only torch (unpinned) and python>=3.6. No Dockerfile, environment.yml, or requirements.txt with pinned versions exists. This makes exact reproduction dependent on the reviewer's environment choices.

Score: Accept (4) — Artifacts are available and complete, but lack environment specifications for reproducibility.

Artifacts Functional
Critical Finding: Zero entry points work out-of-the-box
When artifact was tested with no modifications. Every entry point fails:

Entry Point	Result	Blocking Bug
Unit tests (pytest tests/)	FAIL	BUG-1: ImportError: cannot import name 'VeLO' from 'pylo'
CIFAR-10 example	FAIL	BUG-2: ValueError from stty size in headless env
Language model example	FAIL	BUG-8: Missing tiktoken dependency, then BUG-1
ViT example	FAIL	BUG-1: Missing VeLO export
Bug Classification (numbered in discovery order)
Library bugs (pylo/):

BUG-1 [CRITICAL]: pylo/optim/__init__.py exports VeLO_naive and VeLO_CUDA but never defines a VeLO alias. All example scripts and the unit test import VeLO, which doesn't exist. This is a one-line fix (VeLO = VeLO_CUDA in __init__.py) but blocks the entire artifact.
[SIGNIFICANT]: CUDA kernels fail torch.allclose at float32 defaults at every checkpoint over 2000 gradient-isolated steps. VeLO kernel shows volatile, non-monotonic divergence (see Results Reproduced below).
[MINOR]: @torch.compile on VeLO's LSTM feature function causes recompilation warnings with default cache_size_limit=8.
Example bugs (pylo_examples/) — discovered during fix-and-rerun:

BUG-2: os.popen('stty size') crashes in headless/container environments (CIFAR-10 utils.py)
BUG-3: tiktoken not listed in any dependency file for the LM example
BUG-4, BUG-6: Unconditional import wandb / wandb.init() in CIFAR-10 and LM examples; crashes without wandb installed
BUG-5: --epochs CLI argument silently ignored; hardcoded to ~383 epochs (CIFAR-10 train.py)
BUG-7: --opt velo branch in ViT train.py has its constructor commented out — the README command uses --opt velo but this is dead code
BUG-8: wandb.log() called without wandb.init() in ViT example
After applying minimal fixes for all bugs, all three examples train successfully and produce expected learning curves.

Score: Fell below expectations (3) — The core library design is sound and the CUDA kernels build correctly, but the artifact requires fixing 9 bugs (including 1 critical library bug) before any example can run. The README-provided commands will not work on the unchanged artifact.

Results Reproduced
Step-Time Speedup (Table 2)
Benchmark setup: ViT-B/16 (vit_base_patch16_224), batch size 32, matching the paper's Table 2 configuration. Timing uses torch.cuda.Event (GPU-side measurement, same as paper). Each trial runs 15 warmup steps followed by 50 measured steps; 3 trials with seeds 42, 43, 44. TF32 enabled via torch.set_float32_matmul_precision('high'). torch._dynamo.config.cache_size_limit set to 64 to avoid VeLO recompilation warnings. MuLO excluded because it requires µP-parameterized models which standard ViT does not have (paper's Table 2 also excludes MuLO).

Important: TF32 must be enabled to match the paper's forward/backward times. Without it, forward time is ~86ms (5x slower than reported 17.5ms). This is not documented in the paper or repository.

Optimizer	Our Opt (ms)	Paper Opt (ms)	Our Speedup	Paper Speedup
Adam	4.03±0.01	4.90	—	—
AdafacLO naive	594.03±0.99	756.80	—	—
AdafacLO CUDA	109.24±0.69	99.59	81.6%	86%
VeLO naive	548.55±0.76	585.11	—	—
VeLO CUDA	145.39±0.64	113.58	73.5%	80%
The speedup trend is confirmed — CUDA kernels are substantially faster than naive implementations. However, absolute CUDA kernel times are 10–28% slower than reported, and speedup percentages are 5–7 percentage points lower than paper claims. Naive implementations run faster than reported (possibly due to PyTorch 2.7.1 optimizations), which narrows the speedup ratio. Results are highly reproducible (<1% standard deviation across 3 trials).

Numerical Equivalence (Paper Claim: "fused CUDA kernels are numerically equivalent")
Test setup: A gradient-isolated methodology is used to test strictly the optimizer kernel, eliminating the "butterfly effect" where small parameter divergences produce different gradients on the next step. For each optimizer family, two identical model copies are created (torch.manual_seed(42)). At each step, forward/backward is computed on the naive model only, and the resulting gradients are cloned to the CUDA model before both call .step(). This ensures both optimizers receive identical inputs at every step — any parameter divergence is purely from the kernel. Dynamic batches (new random input each step) test across varied gradient distributions. NaN/Inf checks run after every CUDA step. 2000 steps, reporting every 100. Architectures: MLP (427K params), ResNet-18 (11M), ViT-small/16 (22M), MuMLP (427K, µP). Metrics follow standards for fused kernel verification: torch.allclose pass/fail with float32 defaults (rtol=1.3e-6, atol=1e-5), max parameter absolute diff, and per-step increment analysis.

Optimizer	Architecture	allclose	ParamAbsDiff @2000	Per-step incr (avg±std)
AdafacLO	MLP (427K)	0%	2.68e-02	1.3e-03 ± 4.9e-04
AdafacLO	ResNet-18 (11M)	0%	3.45e-01	1.7e-02 ± 9.6e-03
AdafacLO	ViT-small (22M)	0%	3.29e-01	1.6e-02 ± 4.5e-03
VeLO	MLP (427K)	0%	1.02e-01	5.0e-03 ± 3.0e-02
VeLO	ResNet-18 (11M)	0%	5.03e-01	2.4e-02 ± 3.4e-02
VeLO	ViT-small (22M)	0%	7.28e-01	3.6e-02 ± 3.3e-02
MuLO	MuMLP (427K)	0%	2.61e-02	1.3e-03 ± 6.6e-04
No optimizer passes torch.allclose at any checkpoint — by PyTorch's own float32 tolerance, none are numerically equivalent.

AdafacLO and MuLO (both using learned_optimizer.cu) show predictable, monotonic drift: AdafacLO's per-step increment decelerates over time, MuLO's is nearly constant (std = 7% of mean). This is consistent with accumulated FP rounding from --use_fast_math. The magnitude is modest (0.03–0.35 at 2000 steps).

VeLO (velo_kernel.cu) shows a qualitatively different pattern — volatile, non-monotonic drift. The per-step increment standard deviation exceeds the mean (e.g., MLP: avg 5.0e-03, std 3.0e-02), with frequent negative increments where ParamAbsDiff decreases between checkpoints (MLP peaks at 0.228 at step 700, then drops to 0.102 at step 2000). This oscillating behavior — where the CUDA kernel sometimes drifts closer to the naive implementation — is distinct from the steady drift of AdafacLO/MuLO. The overall magnitude (0.10–0.73) is comparable to AdafacLO on similar architectures, but the behavioral pattern suggests the VeLO kernel computes a meaningfully different optimization trajectory.

End-to-End Training
All three examples were run following their READMEs (adapted for single GPU, no wandb):

Example	Follows README	After Fixes	Result
CIFAR-10 MLP (VeLO, 5 epochs)	Mostly	4 bugs fixed	Val acc: 46.4% — training works
Language Model (VeLO, 200 steps)	Partially (1 GPU, smaller model)	3 bugs fixed	Loss: 10.93→10.80 — training works
ViT Imagenette (VeLO_CUDA, 5 epochs)	Mostly (1 GPU, no aug-repeats)	3 bugs fixed	Val top-1: 47.1%, top-5: 89.0% — training works
Training curves show expected behavior in all cases. Full 100-epoch and multi-GPU reproduction was not attempted due to time and hardware constraints.

Checkpoint/Resume
Optimizer checkpoint/resume test passes for both VeLO_CUDA and AdafacLO_CUDA (after fixing the VeLO import bug). Parameters match within 1e-5 tolerance after loading and continuing training.

Score: Fell below expectations (3) — The speedup trend (primary claim) is confirmed but with lower absolute numbers. The numerical equivalence claim is not supported by torch.allclose testing over 2000 steps — all optimizers fail at every checkpoint and VeLO additionally shows a qualitatively different (volatile, non-monotonic) drift pattern. End-to-end training works but only after substantial bug fixes. Single-GPU evaluation means multi-GPU claims (Table 3) could not be verified.

Fixes Applied (this branch)
The following library-side fixes have been implemented in the `pylo/` package
(the `pylo_examples/` fixes are tracked separately below):

- **BUG-1 [CRITICAL] — fixed.** `pylo/optim/__init__.py` now defines
  `VeLO = VeLO_CUDA`, `AdafacLO = AdafacLO_CUDA`, `MuLO = MuLO_CUDA` when the
  CUDA extensions are available, and falls back to the naive implementations
  when they are not. The same three names are re-exported from
  `pylo/__init__.py` so `from pylo import VeLO` (used by the unit tests and
  every example script) now works. Verified with `pytest tests/` — 3/3 pass.
- **MINOR (torch.compile cache warning) — fixed.** `pylo/optim/velo_cuda.py`
  now bumps `torch._dynamo.config.cache_size_limit` to 64 on import (guarded
  so it only raises the limit, never lowers it). This removes the
  recompilation warnings emitted by the `@torch.compile`-decorated
  `lstm_features_for_tensor` when it is specialised per parameter shape.

Not addressed here (left to the kernel authors):

- **SIGNIFICANT (CUDA/naive numerical divergence).** Requires investigation of
  `velo_kernel.cu` and the `--use_fast_math` compile flag; outside the scope
  of a pure library bug-fix pass.

Example-side fixes (BUG-2 … BUG-8) are applied in the sibling
`pylo_examples/` repository — see "pylo_examples fixes" below.

pylo_examples fixes (sibling repo `/home/paul/workspace/pylo_examples`)
All of the example-side bugs from the review have now been fixed in the
sibling `pylo_examples/` checkout:

- **BUG-2 — fixed.**
  `simple_mlp_and_conv/utils.py`: replaced
  `_, term_width = os.popen('stty size', 'r').read().split()` with
  `term_width = shutil.get_terminal_size(fallback=(80, 24)).columns`. The
  progress bar now works under nohup / docker / CI where no tty is
  attached.
- **BUG-3 — already satisfied.**
  `language_model_pretraining/requirements.txt` already lists
  `tiktoken==0.9.0` (and `wandb==0.19.8`); no change required there. The
  reviewer may have been using an older revision of the requirements
  file.
- **BUG-4 — fixed.**
  `simple_mlp_and_conv/train.py`: removed the unconditional
  `import wandb` and `wandb.init()`, added a `--wandb` CLI flag, and
  routed every `wandb.log(...)` call through a `_wandb_log()` helper
  that is a no-op when `--wandb` is not passed. `wandb.finish()` is
  likewise guarded.
- **BUG-5 — fixed.**
  `simple_mlp_and_conv/train.py`: replaced the hard-coded
  `epochs = 150_000 / len(trainloader)` (which forced ~383 epochs on
  CIFAR-10) with `epochs = args.epochs`, so the `--epochs` CLI argument
  is now honoured.
- **BUG-6 — fixed.**
  `language_model_pretraining/train.py`: removed both top-level
  `import wandb` lines, added `_C.use_wandb = False` to
  `config.py`, and now only import + initialise wandb when
  `config.use_wandb` is True (and only on rank 0). The per-iteration
  `run.log(postfix)` call is now guarded by `if run is not None`.
- **BUG-7 — fixed.**
  `image_classification/vision_transformer/train.py`: uncommented the
  VeLO constructor in the `--opt velo` branch
  (`optimizer = VeLO(model.parameters(), lr=args.lr, num_steps=MAX_STEPS,
  weight_decay=args.weight_decay)`). Combined with BUG-1 (`VeLO` alias),
  `python train.py --opt velo ...` from the README now actually
  constructs an optimizer.
- **BUG-8 — fixed.**
  `image_classification/vision_transformer/train.py`: wrapped the
  stray mid-training `wandb.log({...})` call in
  `if args.log_wandb and has_wandb:` so it can no longer fire before
  (or instead of) `wandb.init()`.

Verification: `python train.py --help` for the CIFAR-10 example succeeds
and now shows the new `--wandb` flag; `ast.parse` succeeds for the LM
and ViT training scripts; `utils.term_width` imports cleanly in a
headless shell. Full example runs were not re-executed here (they
require GPU + datasets) — only static/import-level checks.

Recommendations for Authors
Fix BUG-1 immediately: Add VeLO = VeLO_CUDA to pylo/optim/__init__.py. This one-line fix unblocks everything. *(Done in this branch — see "Fixes Applied" above.)*
Pin dependencies: Provide a requirements.txt or Dockerfile with exact PyTorch and CUDA versions used for the paper's experiments.
Document TF32: Note that torch.set_float32_matmul_precision('high') is required to reproduce forward/backward times.
Investigate VeLO CUDA kernel divergence pattern: With gradient isolation (identical gradients fed to both implementations), VeLO CUDA diverges from the naive implementation by 0.10–0.73 (ParamAbsDiff) over 2000 steps — comparable in magnitude to AdafacLO (0.03–0.35) — but with a qualitatively different pattern: volatile, non-monotonic drift with frequent negative increments (std exceeding mean), unlike AdafacLO/MuLO's steady monotonic drift. This suggests velo_kernel.cu computes a meaningfully different optimization trajectory. No optimizer passes torch.allclose at float32 defaults — the "numerically equivalent" claim needs qualification (likely attributable to --use_fast_math).
Make wandb optional: Guard all wandb imports and calls so examples work without it.
Fix dead code: Uncomment the VeLO constructor in the ViT example's --opt velo branch.
List all dependencies: Add tiktoken, h5py to appropriate requirements files.