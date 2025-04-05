# PyLO: Learned Optimization for PyTorch

PyLO provides efficient PyTorch implementations of cutting-edge learned optimizers that seamlessly integrate with any PyTorch project. These optimizers can replace standard PyTorch optimizers while potentially delivering improved performance with no hyperparameter tuning.

## Key Features

- **Drop-in replacement** for standard PyTorch optimizers
- **CUDA-accelerated** kernels for efficient application of learned optimizers
- **Production-ready** implementation supporting training of large-scale models
- **PyTorch-native API** designed for simplicity and familiarity
- **Hugging Face integration** for sharing and loading meta-models

## Why PyLO?

Learned optimizers have shown promising results in research but have faced adoption barriers in real-world applications. PyLO bridges this gap by providing a practical, efficient implementation that makes these advanced optimization techniques accessible to the broader PyTorch community.

## Installation

### Prerequisites
- Python 3.7+
- PyTorch 1.9+
- CUDA toolkit (for GPU acceleration)
- Set `CUDA_HOME` environment variable before installing the kernels

### Installation Options

#### Installation without custom CUDA kernels
```bash
pip install .
```

#### Installation with custom CUDA kernels
```bash
pip install . --config-settings="--build-option=--cuda"
```

#### Installation of MuP patch (After installing library)
```bash
python -m pylo.util.patch_mup
```

## Quick Start

```python
import torch
from pylo.optim import VeLO

# Initialize a model
model = torch.nn.Linear(10, 2)

# Create a learned optimizer instance
optimizer = VeLO(model.parameters())

# Use it like any PyTorch optimizer
for epoch in range(10):
    optimizer.zero_grad()
    loss = loss_fn(model(input), target)
    loss.backward()
    optimizer.step(loss) # pass the loss 
```

## Sharing Meta Models

PyLO integrates with Hugging Face Hub for sharing trained meta-models:

```python
# Login to Hugging Face
from huggingface_hub import login
login()  # Or use huggingface-cli login from command line

# Push your meta-model to Hugging Face Hub
meta_model.push_to_hub("username/model-name")

# Load a meta-model from Hugging Face Hub
from pylo import MetaMLP
meta_model = MetaMLP.from_pretrained("username/model-name")
```

## Documentation

For detailed documentation and examples, visit [our documentation site](https://pylo.readthedocs.io).

## Contributing

We welcome contributions to PyLO! Please see our [contributing guide](CONTRIBUTING.md) for more information.

## Citation

If you use PyLO in your research, please consider citing our work:

```bibtex
@software{pylo2025,
  author = {Paul Janson, Benjamin Therien, Xialong Huang, and Eugene Belilovsky},
  title = {PyLo: A PyTorch Library for Learned Optimizers},
  year = {2025},
  url = {https://github.com/Belilovsky-Lab/pylo}
}
```

## License

PyLO is released under the [Apache 2.0 License](LICENSE).