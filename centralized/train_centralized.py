# centralized/train_centralized.py
#
# Robust centralized training for handwritten digits:
# - Strong augmentation
# - Label smoothing
# - LR scheduler
#
# Same architecture (CentralizedMNISTCNN), same output filename
# (centralized_mnist_cnn.pt), but better generalization to
# GAN / other handwritten styles.

import sys
import os

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.mnist_cnn import CentralizedMNISTCNN


MODEL_SAVE_PATH = "centralized_mnist_cnn.pt"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # We keep CuDNN fast but not fully deterministic
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_dataloaders(batch_size: int = 64):
    """
    Centralized MNIST loaders with stronger augmentation so that
    the model is less sensitive to style / distortions and
    generalizes better to unseen handwritten digits (e.g., GAN).
    """
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    # Robust training transform:
    # - rotate, translate, scale, shear
    # - perspective warp
    # - contrast / sharpness changes
    # - random erasing (simulates occlusion / artifacts)
    train_transform = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomRotation(degrees=25)
        ], p=0.9),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=0,
                translate=(0.2, 0.2),
                scale=(0.8, 1.25),
                shear=12,
            )
        ], p=0.9),
        transforms.RandomPerspective(distortion_scale=0.35, p=0.5),
        transforms.RandomAutocontrast(p=0.5),
        transforms.RandomAdjustSharpness(sharpness_factor=1.8, p=0.5),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(
            p=0.4,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3),
            value=0.0,
            inplace=False,
        ),
    ])

    # For test / evaluation: keep it clean
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
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
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

    # Label smoothing: reduces extreme over-confidence -> better OOD behaviour
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    # Smooth cosine LR schedule for ~12 epochs
    num_epochs = 12
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-5,
    )

    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        _, test_acc = evaluate(model, test_loader, criterion, device, prefix="Test")

        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  New best robust model saved with Test Acc: {best_acc:.2f}%")

    print(f"\nRobust centralized training finished. Best Test Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
