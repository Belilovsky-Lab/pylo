Installation
===========

Prerequisites
------------
* Python 3.7+
* PyTorch 1.9+
* CUDA toolkit (for GPU acceleration)
* Set ``CUDA_HOME`` environment variable before installing the kernels

Installation Options
------------------

Installation without custom CUDA kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install .

Installation with custom CUDA kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install . --config-settings="--build-option=--cuda"

Install patch for MuP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python -m pylo.util.patch_mup