<div align="center">
    
# PyLO: Learned Optimization for PyTorch

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
Cuda must be installed for the build to succeed. Users must set the `CUDA_HOME` environment variable before installing the kernels. CUDA version of pytorch should match the nvcc version.
```bash
git clone https://github.com/Belilovsky-Lab/pylo
cd pylo
pip install .
python setup.py install --cuda # or try pip install --no-build-isolation --config-settings="--build-option=--cuda" .
```

#### Installation of MuP patch (After installing library)

```bash
python -m pylo.util.patch_mup
```

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


