# centralized/train_centralized.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.mnist_cnn import CentralizedMNISTCNN


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloaders(batch_size: int = 64):
    """
    Centralized train/test loaders.
    Uses MNIST right now but can be swapped for any handwritten dataset
    (just change the datasets.MNIST lines).
    """
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device, epoch: int):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    print(f"Epoch [{epoch}] - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, prefix="Test"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)

        running_loss += loss.item() * y.size(0)
        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()

    loss_avg = running_loss / total
    acc = 100.0 * correct / total
    print(f"{prefix} Loss: {loss_avg:.4f}, {prefix} Acc: {acc:.2f}%")
    return loss_avg, acc


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, test_loader = get_dataloaders(batch_size=64)

    model = CentralizedMNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    best_acc = 0.0
    num_epochs = 10  # keep at 10 for fast, consistent runs

    for epoch in range(1, num_epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        _, test_acc = evaluate(model, test_loader, criterion, device, prefix="Test")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "centralized_mnist_cnn.pt")
            print(f"  New best model saved with Test Acc: {best_acc:.2f}%")

    print(f"Training finished. Best Test Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
