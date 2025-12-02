Key Concepts and Glossary
=========================

This page provides definitions and explanations of key terms used in learned optimization and PyLO.

Core Concepts
------------

Learned Optimizer
~~~~~~~~~~~~~~~~

A neural network (typically a small MLP) that predicts parameter updates during training. Instead of using hand-designed update rules (like Adam's momentum and adaptive learning rates), learned optimizers are trained via meta-learning to generate good updates across many tasks.

**Example:** small_fc_lopt uses a 2-3 layer MLP with ~32-64 hidden units to predict updates for each parameter.

Meta-Learning (Learning to Learn)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The process of training an optimizer by optimizing its performance across many different tasks. The meta-objective is to find optimizer parameters that lead to good performance when the optimizer is applied to new, unseen tasks.

**Key idea:** Instead of hand-designing optimization rules, we learn them from data.

**Meta-training process:**

1. Sample a random task (e.g., train a CNN on CIFAR-10)
2. Apply the learned optimizer for K steps
3. Compute meta-loss (e.g., final validation loss,sum of training loss)
4. Update the optimizer's weights via gradient estimation strategies such as Evolution Strategies.
5. Repeat for huge number of tasks

Optimizee
~~~~~~~~

The model being optimized. In the context of learned optimization:

* **Optimizee:** The neural network you're training (e.g., a ResNet, GPT model)
* **Optimizer:** The learned optimizer doing the training

**Example:** When you train a Vision Transformer with VeLO, the ViT is the optimizee and VeLO is the optimizer.

Accumulator
~~~~~~~~~~

State variables maintained by the optimizer across training steps. These store historical information about gradients and parameters.

**Common accumulators:**

* **Momentum (m):** Exponential moving average of gradients
* **Second moment (v):** Exponential moving average of squared gradients
* **Row/column factors (r, c):** Factored second moment estimates (Adafactor-style)

**Why they matter:** Learned optimizers construct features from these accumulators to make informed update decisions.

Features
~~~~~~~

Input values fed to the learned optimizer's neural network. Features are constructed from:

* Current gradients
* Parameter values
* Accumulator states
* Derived quantities (e.g., normalized momentum)
* Time information

**VeLO uses 29 features, small_fc_lopt uses 39 features.**

**Example features:**

* ``gradient`` -  gradient
* ``1 / sqrt(second_moment)`` - Rsqrt of second moment 
* ``tanh(step / 1000)`` - Temporal feature indicating training progress

Feature Normalization
~~~~~~~~~~~~~~~~~~~~

The process of scaling features to have unit variance before feeding them to the MLP. This helps the learned optimizer generalize across different parameter scales.

**Normalization formula:**

.. code-block:: text

    normalized_feature = feature / sqrt(mean(feature²) + ε)

Computed separately for each feature dimension across all parameters.

Training Horizon
~~~~~~~~~~~~~~~

The number of optimization steps the learned optimizer is expected to run for. Different learned optimizers are meta-trained for different horizons:

* **VeLO:** Long horizons (150K steps) - suitable for large-scale pre-training
* **small_fc_lopt:** Short-medium horizons (1K-10K steps) 

**Why it matters:** Using an optimizer outside its meta-training horizon can lead to instability or divergence.

Optimizer Components
-------------------

MLP (Multi-Layer Perceptron)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The neural network inside a learned optimizer that predicts parameter updates. Typically:

* **Architecture:** 2-3 hidden layers with 32-64 units
* **Activations:** ReLU
* **Output:** 2 values per parameter (direction and magnitude)

**Size:** Small (~100 parameters) compared to the models being optimized (millions to billions).

Direction and Magnitude
~~~~~~~~~~~~~~~~~~~~~~

The two outputs of a learned optimizer's MLP for each parameter:

* **Direction:** Suggests the sign and relative direction of the update
* **Magnitude:** Suggests the scale of the update

**Update formula:**

.. code-block:: text

    update = direction × exp(magnitude × α) × β
    parameter = parameter - update

Where α and β are fixed hyperparameters (typically 0.01).

Momentum
~~~~~~~

An exponential moving average of past gradients, helping optimization "build up speed" in consistent directions.

**Update rule:**

.. code-block:: text

    momentum_t = β × momentum_{t-1} + (1 - β) × gradient_t

**Typical value:** β ≈ 0.9-0.99

Used by Adam, RMSprop, SGD with momentum, and learned optimizers.

Second Moment
~~~~~~~~~~~~

An exponential moving average of squared gradients, used for adaptive learning rates.

**Update rule:**

.. code-block:: text

    second_moment_t = β × second_moment_{t-1} + (1 - β) × gradient_t²

**Typical value:** β ≈ 0.999

Used by Adam, RMSprop, and learned optimizers for gradient normalization.

Factored Second Moment (Adafactor-style)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An efficient approximation of the full second moment matrix using row and column factors. Instead of storing an m×n matrix, store:

* **Row factors (r):** Vector of length m
* **Column factors (c):** Vector of length n

**Memory savings:** O(m×n) → O(m + n)

**Approximation:** ``second_moment[i,j] ≈ row_factor[i] × col_factor[j]``

Used by Adafactor and learned optimizers for memory efficiency.

Performance Concepts
-------------------

CUDA Kernel
~~~~~~~~~~

A function that runs on the GPU. Each kernel launch has overhead, so reducing kernel count improves performance.


Kernel Fusion
~~~~~~~~~~~~

Combining multiple operations into a single CUDA kernel to reduce memory traffic and launch overhead.

**Example:** PyLO fuses feature construction, normalization, MLP evaluation, and parameter update into one kernel.

Memory Hierarchy
~~~~~~~~~~~~~~~

The different levels of memory on a GPU, with varying speeds and sizes:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Memory Type
     - Size
     - Latency
     - Usage in PyLO
   * - Registers
     - ~256 KB/SM
     - 1 cycle
     - Features, MLP activations
   * - Shared Memory
     - 48-164 KB/SM
     - ~20 cycles
     - Normalization stats
   * - L2 Cache
     - 6-40 MB
     - ~200 cycles
     - MLP weights
   * - Global Memory
     - 40-80 GB
     - ~400 cycles
     - Parameters, gradients

**Optimization goal:** Keep frequently accessed data in fast memory (registers, shared memory).

Memory Bandwidth
~~~~~~~~~~~~~~~

The rate at which data can be transferred to/from GPU memory. Often the bottleneck for learned optimizers.

~~~~~~~~

The ratio of active warps to maximum possible warps on a GPU. Higher occupancy generally means better GPU utilization.


Grid-Stride Loop
~~~~~~~~~~~~~~~

A CUDA programming pattern that allows a kernel to process more elements than there are threads.

.. code-block:: cuda

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < total_elements; 
         i += blockDim.x * gridDim.x)
    {
        process_element(i);
    }

**Benefits:** Flexible thread configuration, better instruction-level parallelism.

PyLO-Specific Terms
------------------

VeLO (Versatile Learned Optimizer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A learned optimizer meta-trained by Google for 4000 TPU-months on diverse tasks. The most robust learned optimizer currently available.

**Key properties:**

* 29 input features
* Meta-trained for long horizons (150K steps)
* Works well without hyperparameter tuning

**Paper:** `VeLO: Training Versatile Learned Optimizers by Scaling Up <https://arxiv.org/abs/2211.09760>`_

small_fc_lopt 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A small mlp based learned optimizer 

**Key properties:**

* 39 input features
* Meta-trained for short horizons
* More memory-efficient during meta-training

**Paper:** `Practical tradeoffs between memory, compute, and performance in learned optimizers <https://arxiv.org/abs/2203.11860>`_

Distributed Optimizer Step
~~~~~~~~~~~~~~~~~~~~~~~~~

An optimization for multi-GPU training that distributes optimizer computation across devices instead of redundantly computing on all devices.

**How it works:**

1. Reduce-scatter gradients (instead of all-reduce)
2. Each GPU computes optimizer step for its shard
3. All-gather updated parameters


Decoupled Weight Decay
~~~~~~~~~~~~~~~~~~~~~

Weight decay applied directly to parameters rather than through gradients. This separates regularization from gradient-based optimization.

**Standard weight decay (L2 regularization):**

.. code-block:: text

    gradient = gradient + λ × parameter

**Decoupled weight decay:**

.. code-block:: text

    parameter = (1 - λ) × parameter - learning_rate × gradient

**Why it matters:** Decoupled weight decay interacts better with adaptive learning rates and learned optimizers.

Training Concepts
----------------

Meta-Training vs Training
~~~~~~~~~~~~~~~~~~~~~~~~

* **Meta-training:** Training the optimizer itself on many tasks (done once, expensive)
* **Training (or optimization):** Using the optimizer to train a model (done for each new model)

Hyperparameter Tuning
~~~~~~~~~~~~~~~~~~~~

The process of searching for good hyperparameters (learning rate, weight decay, etc.) for a given task.

**Traditional optimizers:** Require extensive hyperparameter tuning

**Learned optimizers:** Work well with default settings (one of their key benefits!)


Technical Jargon
---------------

Warp
~~~

A group of 32 threads on NVIDIA GPUs that execute in lockstep. The fundamental unit of execution.

**Warp shuffle:** Communication between threads in a warp without using shared memory.

Thread Block (or Block)
~~~~~~~~~~~~~~~~~~~~~~

A group of threads (typically 128-1024) that can cooperate via shared memory.

Streaming Multiprocessor (SM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A GPU compute unit. Modern GPUs have 40-140 SMs.

**Each SM has:**

* Registers (~256 KB)
* Shared memory (48-164 KB)
* L1 cache
* Tensor cores (on recent GPUs)

Atomic Operation
~~~~~~~~~~~~~~~

An operation that completes without interruption. Used for thread-safe updates to shared variables.

**Example:** ``atomicAdd(&global_sum, thread_value)``