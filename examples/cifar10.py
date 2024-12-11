import torch
import torchvision
import torch.nn as nn

# import torch.optim as optim
import pylo.optim as optim
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import os


def get_model(num_classes=10, pretrained=True):
    model = resnet50(pretrained=pretrained)
    # Modify first conv layer for CIFAR10
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool

    # Modify final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained = False
model = get_model(num_classes=10, pretrained=pretrained).to(device)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.AdafacLO_naive(model.parameters())
epochs = 30
batch_size = 256
no_of_workers = 8
# CIFAR10 transforms
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)

# Data loaders
train_dataset = CIFAR10(
    root=os.environ["DATA_PATH"], train=True, download=True, transform=transform
)
test_dataset = CIFAR10(
    root=os.environ["DATA_PATH"], train=False, download=True, transform=transform
)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=no_of_workers
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=no_of_workers
)

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(
                f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], "
                f"Loss: {running_loss/100:.4f}"
            )
            running_loss = 0.0

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Epoch [{epoch+1}/{epochs}] Accuracy: {100 * correct / total:.2f}%")

# Save model
# torch.save(model.state_dict(), 'resnet50_cifar10.pth')
