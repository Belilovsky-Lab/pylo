import torch
import torchvision
from pylo.optim import AdafacLO_naive
import os
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

os.environ["DATA_PATH"] = "/speed-scratch/p_janso/data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.nn.Linear(784, 10).to(device)
optimizer = AdafacLO_naive(model.parameters())

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = DataLoader(MNIST(root=os.environ["DATA_PATH"], download=True,transform=train_transforms), batch_size=32)
test_loader = DataLoader(MNIST(root=os.environ["DATA_PATH"], download=True, train=False,transform=train_transforms), batch_size=32)

for epoch in range(10):
    for x, y in train_loader:
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        out = model(x.view(-1, 784))
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} loss: {loss.item()}")