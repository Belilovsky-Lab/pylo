Changelog
=========

This page contains the release history of PyLO.

Version 0.2.0
-------------

**Release Date:** November 2025

New Features
~~~~~~~~~~~~

* Added CUDA-accelerated implementation of VeLO optimizer (:class:`pylo.optim.VeLO_CUDA`)

  * Provides significant speedup for training on CUDA-enabled GPUs
  * Fully compatible with the existing VeLO optimizer API
  * Requires CUDA toolkit for installation

Improvements
~~~~~~~~~~~~

* Enhanced documentation with API reference
* Improved installation instructions

Version 0.1.0
-------------

**Release Date:** May 2025

* Initial release of PyLO
* Core learned optimizer implementations
* Integration with PyTorch optim library
* Hugging Face Hub support for sharing models
* Basic benchmarking utilities
