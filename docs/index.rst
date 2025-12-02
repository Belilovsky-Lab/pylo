Welcome to PyLO !
================================

.. image:: ../assets/overview.png
   :alt: Overview
   :align: center

|

.. image:: https://img.shields.io/badge/GitHub-Belilovsky--Lab%2Fpylo-blue?logo=github
   :target: https://github.com/Belilovsky-Lab/pylo
   :alt: GitHub Repository

.. image:: https://img.shields.io/github/stars/Belilovsky-Lab/pylo?style=social
   :target: https://github.com/Belilovsky-Lab/pylo/stargazers
   :alt: GitHub stars

.. image:: https://img.shields.io/github/license/Belilovsky-Lab/pylo
   :target: https://github.com/Belilovsky-Lab/pylo/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/badge/arXiv-2506.10315-b31b1b.svg
   :target: https://arxiv.org/abs/2506.10315
   :alt: arXiv Paper

PyLo is a PyTorch-based learned optimizer library that enables researchers and practitioners to implement, experiment with, and share learned optimizers. 
It bridges the gap found in the research of learned optimizers and using it for actual practical scenarios.

Checkout our paper here: `arXiv <https://arxiv.org/abs/2506.10315>`_

.. note::
   New to PyLo? Check out our :doc:`usage` guide and explore complete training examples at `pylo_examples <https://github.com/Belilovsky-Lab/pylo_examples>`_.

Key Features
-----------

* Pre-trained learned optimizers ready for production use
* Seamless integration with PyTorch optim library and training loops
* Comprehensive benchmarking utilities against standard optimizers
* Supports sharing model weights through Hugging Face Hub

Quick Example
------------

.. code-block:: python

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

More Examples
-------------

Looking for complete, runnable examples? Check out the `pylo_examples repository <https://github.com/Belilovsky-Lab/pylo_examples>`_ which includes:

* **Image Classification** - Training Vision Transformers (ViT) and ResNets on ImageNet and CIFAR-10
* **Language Modeling** - Training GPT-2 models
* **Distributed Training** - Multi-GPU examples with FSDP and DDP

Each example includes detailed setup instructions, training scripts, and configuration files to help you get started quickly.

Documentation
============

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:
   
   installation
   background
   .. basic_concepts

.. toctree::
   :maxdepth: 2
   :caption: User Guide:
   
   usage
   how_it_works
   concepts
   .. tutorials/index
   .. examples/index
   .. benchmarks

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development:

   changelog

.. Benchmarks
.. =========

.. PyLo has been benchmarked against standard optimizers like Adam, SGD, and RMSProp across various tasks:

.. .. image:: _static/benchmark_plot.png
..    :alt: Benchmark results comparing PyLo to standard optimizers

.. *See the detailed :doc:`benchmarks` page for more information.*

How to Cite
==========

If you use PyLo in your research, please cite:

.. code-block:: bibtex

   @article{pylo,
   title={PyLO: Towards Accessible Learned Optimizers in PyTorch},
   author={Janson, Paul and Therien, Benjamin and Anthony, Quentin and Huang, Xiaolong and Moudgil, Abhinav and Belilovsky, Eugene},
   journal={arXiv preprint arXiv:2506.10315},
   year={2025}
   }

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`