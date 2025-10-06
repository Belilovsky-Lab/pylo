import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pylo.optim import Velo
from pylo.optim.velo_cuda import VeLO_CUDA
from pylo.optim.Velo import VeLO
from muon import SingleDeviceMuonWithAuxAdam as MuonWithAuxAdam

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Create a simple neural network
class SimpleNet(nn.Module):
    def __init__(self, input_size=10, hidden_size=512, output_size=3, depth=2, use_bias=True):
        super(SimpleNet, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size, bias=use_bias))
        self.layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(depth - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size, bias=use_bias))
            self.layers.append(nn.ReLU())

        # Output layer
        self.layers.append(nn.Linear(hidden_size, output_size, bias=use_bias))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Configuration
use_bias = True  # Set to False to disable biases in all linear layers
depth = 10  # Number of hidden layers
optimizer_type = "muon"  # "velo" or "muon"
print(f"Bias enabled: {use_bias}")
print(f"Depth: {depth}")
print(f"Optimizer: {optimizer_type}")
# Generate synthetic dataset
n_samples = 1000
input_size = 10
output_size = 3
batch_size = 32

X = torch.randn(n_samples, input_size)
y = torch.randint(0, output_size, (n_samples,))

# Create DataLoader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model

# Training loop
num_epochs = 25
model = SimpleNet(input_size=input_size, output_size=output_size, depth=depth, use_bias=use_bias).to(
    device
)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Initialize optimizer
if optimizer_type == "velo":
    optimizer = VeLO_CUDA(
        model.parameters(), lr=1.0, num_steps=num_epochs * len(dataloader), legacy=False
    )
elif optimizer_type == "velo_legacy":
    optimizer = VeLO(
        model.parameters(), lr=1.0, num_steps=num_epochs * len(dataloader)
    )
elif optimizer_type == "muon":
    # Separate parameters by dimensionality for Muon
    hidden_weights = [p for p in model.layers.parameters() if p.ndim >= 2]
    hidden_gains_biases = [p for p in model.layers.parameters() if p.ndim < 2]

    param_groups = [
        dict(params=hidden_weights, use_muon=True,
             lr=0.02, weight_decay=0.01),
        dict(params=hidden_gains_biases, use_muon=False,
             lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
    ]
    optimizer = MuonWithAuxAdam(param_groups)
else:
    raise ValueError(f"Unknown optimizer type: {optimizer_type}")

# Loss function
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    total_loss = 0.0
    correct = 0
    total = 0
    optimizer_time = 0.0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Optimizer step with timing
        if device.type == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            if optimizer_type == "velo" or optimizer_type == "velo_legacy":
                optimizer.step(loss)
            else:
                optimizer.step()
            end_event.record()
            torch.cuda.synchronize()
            optimizer_time += start_event.elapsed_time(end_event)  # milliseconds
        else:
            import time
            start_time = time.perf_counter()
            if optimizer_type == "velo":
                optimizer.step(loss)
            else:
                optimizer.step()
            optimizer_time += (time.perf_counter() - start_time) * 1000  # convert to ms

        # Track statistics
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    # Print epoch statistics
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    avg_optimizer_time = optimizer_time / len(dataloader)
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, "
        f"Avg Optimizer Time: {avg_optimizer_time:.3f}ms"
    )

print("\nTraining completed!")
