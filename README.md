<div align="center">
    
# PyLO: Towards Accessible Learned Optimizers in PyTorch [MLSys 2026]

[![arXiv](https://img.shields.io/badge/arXiv-2410.06511-b31b1b.svg)](https://arxiv.org/abs/2506.10315)
[![forum](https://img.shields.io/badge/PyLO-Docs-green.svg)](https://belilovsky-lab.github.io/pylo/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-lightgrey.svg)](./LICENSE)

</div>

PyLO provides efficient PyTorch implementations of cutting-edge learned optimizers. These optimizers work as drop-in replacements to standard PyTorch optimizers, while potentially delivering improved performance with no hyperparameter tuning. With its huggingface integration, PyLO allows users to download their own optimizers from the huggingface Hub and take advantage of our high-performance kernels for training new optimizees. 

## Key Features

- **Drop-in replacement** for standard PyTorch optimizers
- **CUDA-accelerated** kernels for efficient learned optimizer inference
- **PyTorch-native API** designed for simplicity and familiarity
- **Hugging Face integration** for sharing and loading meta-models

# Installation

### Via URL (slow, no Kernels)
```bash
pip install git+https://github.com/Belilovsky-Lab/pylo
```


### Build from source (Fast, with custom CUDA kernels)

The CUDA Toolkit must be installed and visible through `CUDA_HOME`. The
CUDA version of your PyTorch wheel must match the `nvcc` version on
`PATH`. The paper's benchmarks used Python 3.11, PyTorch 2.6.0+cu118
and CUDA 11.8 on an A100-SXM4-80GB; a matching pinned environment is
provided in `requirements.txt`.

```bash
git clone https://github.com/Belilovsky-Lab/pylo
cd pylo

# (Recommended) install the pinned environment that matches the paper:
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

# Then install pylo with the fused CUDA kernels. PYLO_CUDA=1 is the
# supported way to request the CUDA build; the legacy
# `python setup.py install --cuda` invocation is deprecated by modern
# setuptools and may silently fall through to easy_install without
# compiling the kernels.
PYLO_CUDA=1 pip install --no-build-isolation .
```

If you omit `--no-build-isolation`, pip will build in an isolated
environment that does not have your system PyTorch installed, and the
CUDA extension build will fail.

#### Installation of MuP patch (After installing library)

```bash
python -m pylo.util.patch_mup
```

#### Troubleshooting

- `pybind11.h: No such file or directory` — install `pybind11` (now
  listed in `requirements.txt` / `install_requires`), or export
  `CPLUS_INCLUDE_PATH` to include `$(python -m pybind11 --includes | sed 's/-I//g')`.
- `libc10.so: cannot open shared object file` when importing the CUDA
  extensions — make sure `import torch` runs before any
  `import pylo` / `import velo_cuda_kernel` / `import cuda_lo`, so the
  PyTorch runtime libraries are loaded first. `pylo` itself already
  imports torch at package load; the caveat only matters if you import
  the raw extension modules directly.

## Quick Start

```python
import torch
from pylo.optim import VeLO_CUDA

# Initialize a model
model = torch.nn.Linear(10, 2)

# Create a learned optimizer instance
optimizer = VeLO_CUDA(model.parameters())

# Use it like any PyTorch optimizer
for epoch in range(10):
    optimizer.zero_grad()
    loss = loss_fn(model(input), target)
    loss.backward()
    optimizer.step(loss) # pass the loss 
```

## Sharing Learned Optimizers

PyLO integrates with Hugging Face Hub for sharing meta-trained optimizers:

```python
# Login to Hugging Face
from huggingface_hub import login
login()  # Or use huggingface-cli login from command line

# Push your meta-model to Hugging Face Hub
meta_model.push_to_hub("username/model-name")

# Load a learned optimizer from Hugging Face Hub
from pylo import MetaMLP
meta_model = MetaMLP.from_pretrained("username/model-name")
```

## Reproducing paper benchmarks

The step-time numbers in Table 2 (ViT-B/16, batch size 32) assume TF32
matmul is enabled. Without it, the forward/backward path on A100 is
roughly 5× slower than reported. Enable TF32 at the top of any
benchmarking script:

```python
import torch
torch.set_float32_matmul_precision("high")
# Optional: raise the TorchDynamo cache limit to silence VeLO's
# per-parameter-shape recompilation warnings. pylo.optim.velo_cuda
# does this automatically on import, but you can also set it yourself:
torch._dynamo.config.cache_size_limit = 64
```

## Examples

Examples of using Pylo for language modeling and image classification are available here [pylo_examples](https://github.com/Belilovsky-Lab/pylo_examples).

## Documentation

For detailed documentation and examples, visit [our documentation site](https://pylo.readthedocs.io).

## Contributing

We welcome contributions to PyLO! Please see our [contributing guide](CONTRIBUTING.md) for more information.


## Citation

If you use PyLO in your research, please consider citing our work:

```bibtex
@software{pylo2025,
  author = {Paul Janson, Benjamin Therien, Quentin Anthony, Xialong Huang, Abhinav Moudgil and Eugene Belilovsky},
  title = {PyLO: Towards Accessible Learned Optimizers in Pytorch},
  year = {2025},
  url = {https://github.com/Belilovsky-Lab/pylo}
}
```

## License


PyLO is released under the [BSD License](LICENSE).


