import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from pylo import VeLO, AdafacLO_CUDA


class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


@pytest.mark.parametrize("optimizer_class", [VeLO, AdafacLO_CUDA])
def test_optimizer_resume(optimizer_class, tmp_path):
    """Test that optimizer correctly resumes from a saved state."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create model and move to device
    model = SimpleLinearModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create optimizer
    optimizer = optimizer_class(model.parameters(), lr=1)

    # Generate some dummy data
    batch_size = 32
    x = torch.randn(batch_size, 10, device=device)
    y = torch.randn(batch_size, 1, device=device)

    # Train for first 5 steps
    n_steps = 5
    losses = []

    for step in range(n_steps):
        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, y)
        loss.backward()
        optimizer.step(loss)
        losses.append(loss.item())

    # Save checkpoint after 5 steps
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": n_steps,
    }

    checkpoint_path = tmp_path / "optimizer_test_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    # Continue training for 5 more steps with original model/optimizer
    original_continuation_losses = []
    original_params_history = []

    for step in range(n_steps, n_steps * 2):
        # Record parameters before step
        current_params = {}
        for name, param in model.named_parameters():
            current_params[name] = param.clone().detach()
        original_params_history.append(current_params)

        # Training step
        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, y)
        loss.backward()
        optimizer.step(loss)

        # Record loss
        current_loss = loss.item()
        original_continuation_losses.append(current_loss)

    # Final parameters after all 10 steps
    original_final_params = {}
    for name, param in model.named_parameters():
        original_final_params[name] = param.clone().detach()

    # Now, load the checkpoint and train from step 5 again

    # Create a new model and optimizer instance (simulating a restart)
    loaded_model = SimpleLinearModel().to(device)
    loaded_optimizer = optimizer_class(loaded_model.parameters(), lr=1)

    # Load the checkpoint (state after 5 steps)
    checkpoint = torch.load(checkpoint_path)
    loaded_model.load_state_dict(checkpoint["model"])
    loaded_optimizer.load_state_dict(checkpoint["optimizer"])
    start_step = checkpoint["step"]

    # Verify model parameters match the original after 5 steps
    orig_params_at_checkpoint = original_params_history[0]
    for name, loaded_param in loaded_model.named_parameters():
        assert torch.allclose(
            loaded_param, orig_params_at_checkpoint[name]
        ), f"Model parameter {name} doesn't match at checkpoint"

    # Continue training for 5 more steps with loaded model/optimizer
    loaded_continuation_losses = []
    loaded_params_history = []
    for step in range(start_step, start_step * 2):
        # Record parameters before step
        current_params = {}
        for name, param in loaded_model.named_parameters():
            current_params[name] = param.clone().detach()
        loaded_params_history.append(current_params)

        # Training step
        loaded_optimizer.zero_grad()
        output = loaded_model(x)
        loss = F.mse_loss(output, y)
        loss.backward()
        loaded_optimizer.step(loss)

        # Record loss
        current_loss = loss.item()
        loaded_continuation_losses.append(current_loss)

    # Final parameters after all 10 steps with loaded model
    loaded_final_params = {}
    for name, param in loaded_model.named_parameters():
        loaded_final_params[name] = param.clone().detach()

    # Compare losses during the continuation
    for i, (orig_loss, loaded_loss) in enumerate(
        zip(original_continuation_losses, loaded_continuation_losses)
    ):
        step_num = n_steps + i + 1
        loss_diff = abs(orig_loss - loaded_loss)
        assert (
            loss_diff < 1e-6
        ), f"Loss at step {step_num} differs too much: {loss_diff}"

    # Compare parameters at each step
    for i in range(len(original_params_history)):
        step_num = n_steps + i + 1
        orig_params = original_params_history[i]
        loaded_params = loaded_params_history[i]

        for name in orig_params:
            param_diff = torch.max(
                torch.abs(orig_params[name] - loaded_params[name])
            ).item()
            assert (
                param_diff < 1e-5
            ), f"Parameter {name} at step {step_num} differs too much: {param_diff}"

    # Compare final parameters
    for name in original_final_params:
        param_diff = torch.max(
            torch.abs(original_final_params[name] - loaded_final_params[name])
        ).item()
        assert (
            param_diff < 1e-5
        ), f"Final parameter {name} differs too much: {param_diff}"
