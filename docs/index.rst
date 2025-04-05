Welcome to PyLO's documentation!
================================

.. .. image:: https://img.shields.io/pypi/v/pylo.svg
..    :target: https://pypi.python.org/pypi/pylo
..    :alt: PyPI Version

.. .. image:: https://img.shields.io/github/license/yourusername/pylo.svg
..    :target: https://github.com/yourusername/pylo/blob/main/LICENSE
..    :alt: License

.. .. image:: https://img.shields.io/github/stars/yourusername/pylo.svg
..    :target: https://github.com/yourusername/pylo/stargazers
..    :alt: GitHub stars

PyLo is a PyTorch-based learned optimizer library that enables researchers and practitioners to implement, experiment with, and share learned optimizers. 
It bridges the gap found in the research of learned optimizers and using it for actual practical scenarios.

.. note::
   New to PyLo? Check out our :doc:`usage` guide.

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

Documentation
============

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:
   
   installation
   .. usage
   .. basic_concepts

.. toctree::
   :maxdepth: 2
   :caption: User Guide:
   
   usage
   .. tutorials/index
   .. examples/index
   .. benchmarks

.. toctree::
   :maxdepth: 2
   :caption: API Reference:
   
   api/index

.. .. toctree::
..    :maxdepth: 1
..    :caption: Development:
   
..    contributing
..    changelog

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

    @software{pylo2025,
      author = {Paul Janson, Benjamin Therien, Xialong Huang, and Eugene Belilovsky},
      title = {PyLo: A PyTorch Library for Learned Optimizers},
      year = {2025},
      url = {https://github.com/Belilovsky-Lab/pylo}
    }

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`