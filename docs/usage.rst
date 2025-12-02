Usage Guide
===========

This guide shows how to use PyLO's learned optimizers in various training scenarios.

Quick Start
-----------

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

Basic Training Loop
-------------------

Here's a complete example with a typical PyTorch training loop:

.. code-block:: python

    import torch
    import torch.nn as nn
    from pylo.optim import VeLO_CUDA
    from torch.utils.data import DataLoader

    # Setup model and data
    model = YourModel()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    # Create optimizer
    optimizer = VeLO_CUDA(model.parameters())

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data, target = data.cuda(), target.cuda()

            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # Backward pass
            loss.backward()

            # Optimizer step (pass the loss)
            optimizer.step(loss)

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

Using Weight Decay
------------------

PyLO optimizers support decoupled weight decay, which applies regularization directly to parameters rather than through gradients. This is similar to AdamW.

.. code-block:: python

    from pylo.optim import VeLO_CUDA

    # Initialize optimizer with weight decay
    optimizer = VeLO_CUDA(
        model.parameters(),
        weight_decay=0.01  # Typical values: 0.01 to 0.1
    )

    # Training loop remains the same
    for epoch in range(num_epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step(loss)


Using Learning Rate Schedulers
-------------------------------

PyLO optimizers work seamlessly with PyTorch's learning rate schedulers. The scheduler modulates the optimizer's internal learning rate parameter.

Basic Scheduler Example
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pylo.optim import VeLO_CUDA
    from torch.optim.lr_scheduler import CosineAnnealingLR

    # Create optimizer
    optimizer = VeLO_CUDA(model.parameters())

    # Create scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop
    for epoch in range(num_epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step(loss)

        # Step the scheduler after each epoch
        scheduler.step()

        # Print current learning rate
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch}, LR: {current_lr:.6f}')


Complete Training Examples
---------------------------

For more comprehensive, real-world examples, visit the `pylo_examples repository <https://github.com/Belilovsky-Lab/pylo_examples>`_.

The examples repository provides production-ready training scripts for Image Classification and Language Modeling tasks