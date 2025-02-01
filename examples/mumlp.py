from pickletools import optimize
import time
import os
import numpy as np
from torchvision import datasets, transforms
import torch
from torch import nn
import argparse
import math
from mup import MuSGD, get_shapes, set_base_shapes, make_base_shapes, MuReadout

import torch.nn.functional as F
import torch.optim as optim


class MuMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        width=128,
        num_classes=10,
        nonlin=F.relu,
        bias=True,
        output_mult=1.0,
        input_mult=1.0,
        init_std=1.0,
    ):
        super(MuMLP, self).__init__()
        self.nonlin = nonlin
        self.input_mult = input_mult
        self.output_mult = output_mult
        self.init_std = init_std
        self.fc_1 = nn.Linear(in_channels, width, bias=bias)
        self.fc_2 = nn.Linear(width, width, bias=bias)
        self.fc_3 = nn.Linear(width, width, bias=bias)
        self.fc_4 = MuReadout(width, num_classes, bias=bias, output_mult=output_mult)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc_1.weight, a=1, mode="fan_in")
        self.fc_1.weight.data /= self.input_mult**0.5
        self.fc_1.weight.data *= self.init_std
        nn.init.kaiming_normal_(self.fc_2.weight, a=1, mode="fan_in")
        self.fc_2.weight.data *= self.init_std
        nn.init.kaiming_normal_(self.fc_3.weight, a=1, mode="fan_in")
        self.fc_3.weight.data *= self.init_std
        nn.init.zeros_(self.fc_4.weight)

    def forward(self, x):
        out = self.nonlin(self.fc_1(x.flatten(1)) * self.input_mult**0.5)
        out = self.nonlin(self.fc_2(out))
        out = self.nonlin(self.fc_3(out))
        return self.fc_4(out)


def train(
    args,
    model,
    device,
    train_loader,
    optimizer,
    epoch,
    scheduler=None,
    criterion=F.cross_entropy,
):
    model.train()
    train_loss = 0
    correct = 0
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.view(data.size(0), -1))

        loss = criterion(output, target)
        loss.backward()
        train_loss += loss.item() * data.shape[0]  # sum up batch loss
        pred = output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            elapsed = time.time() - start_time
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} | ms/batch {:5.2f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    elapsed * 1000 / args.log_interval,
                )
            )
            start_time = time.time()
        if scheduler is not None:
            scheduler.step()
    train_loss /= len(train_loader.dataset)
    train_acc = correct / len(train_loader.dataset)
    print(
        "\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            train_loss,
            correct,
            len(train_loader.dataset),
            100.0 * correct / len(train_loader.dataset),
        )
    )
    return train_loss, train_acc

def test(
    args, model, device, test_loader, evalmode=True, criterion=F.cross_entropy
):
    if evalmode:
        model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.view(data.size(0), -1))
            test_loss += criterion(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return test_loss, correct / len(test_loader.dataset)


width = 128
logs = [] 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mynet = MuMLP(3072,width=width).to(device)
class args(object):
    pass
args.load_base_shapes = "examples/base-shapes/width64.bsh"
args.lr = 1
args.momentum = 0.9
args.data_dir = os.environ["DATA_PATH"]
args.batch_size = 128
args.no_shuffle = False
args.epochs = 100
args.log_interval = 1
if args.load_base_shapes:
    print(f'loading base shapes from {args.load_base_shapes}')
    set_base_shapes(mynet, args.load_base_shapes)
    print('done')
else:
    print(f'using own shapes')
    set_base_shapes(mynet, None)
    print('done')

# optimizer = MuSGD(mynet.parameters(), lr=args.lr, momentum=args.momentum)
from pylo.optim import MuLO_naive,MuLO_CUDA
# optimizer = MuLO_naive(mynet.parameters(), lr=args.lr)
optimizer = MuLO_CUDA(mynet.parameters(), lr=args.lr)

kwargs = {'num_workers': 1, 'pin_memory': True}

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root=args.data_dir, train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                        shuffle=not args.no_shuffle, num_workers=2)

testset = datasets.CIFAR10(root=args.data_dir, train=False,
                                    download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                        shuffle=False, num_workers=2)

criterion = F.cross_entropy
nonlin = F.relu
for epoch in range(1, args.epochs+1):
    train_loss, train_acc, = train(args, mynet, device, train_loader, optimizer, epoch, criterion=criterion)
    test_loss, test_acc = test(args, mynet, device, test_loader)
    logs.append(dict(
        epoch=epoch,
        train_loss=train_loss,
        train_acc=train_acc,
        test_loss=test_loss,
        test_acc=test_acc,
        width=width,
        nonlin=nonlin.__name__,
        criterion='xent' if criterion.__name__=='cross_entropy' else 'mse',
    ))
    if math.isnan(train_loss):
        break
