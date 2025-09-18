import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pylo.optim.velo_cuda import VeLO_CUDA

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create a simple neural network
class SimpleNet(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, output_size=3):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

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
model = SimpleNet(input_size=input_size, output_size=output_size).to(device)

# Initialize VeLO_CUDA optimizer
optimizer = VeLO_CUDA(model.parameters(), lr=1.0,num_steps=num_epochs * len(dataloader),legacy=False)

# Loss function
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Optimizer step
        optimizer.step(loss)

        # Track statistics
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    # Print epoch statistics
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

print("\nTraining completed!")