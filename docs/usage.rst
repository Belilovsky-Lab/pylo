Quick Start
==========

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