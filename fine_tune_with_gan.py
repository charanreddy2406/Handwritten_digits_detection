import os
import sys

# Make project root + models importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "models")))

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, random_split
from torchvision import datasets, transforms

from models.mnist_cnn import CentralizedMNISTCNN


BASE_MODEL_PATH = "centralized_mnist_cnn.pt"          # strong MNIST model
GAN_ROOT = "gan_labeled"                              # your labeled GAN tiles
OUTPUT_MODEL_PATH = "centralized_mnist_cnn_gan.pt"    # GAN-adapted model


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_mnist_test_loader(batch_size: int = 256):
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=test_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return test_loader


def get_combined_loader(batch_size: int = 64):
    """
    Combined training dataset:
      - MNIST train (full)
      - GAN-labeled images (small, but oversampled)
    We also split GAN-labeled into train/val to measure GAN accuracy.
    """
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    mnist_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize,
    ])

    gan_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.RandomRotation(25),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.15, 0.15),
            scale=(0.85, 1.2),
            shear=10,
        ),
        transforms.RandomAutocontrast(p=0.5),
        transforms.ToTensor(),
        normalize,
    ])

    mnist_train = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=mnist_transform,
    )

    if not os.path.exists(GAN_ROOT):
        raise FileNotFoundError(
            f"GAN labeled folder '{GAN_ROOT}' not found. "
            f"Create it and put digit folders 0..9 with images."
        )

    full_gan_dataset = datasets.ImageFolder(
        root=GAN_ROOT,
        transform=gan_transform,
    )

    num_gan = len(full_gan_dataset)
    if num_gan == 0:
        raise RuntimeError(
            f"No images found under '{GAN_ROOT}'. "
            "Make sure you moved labeled tiles into 0..9 folders."
        )

    # Small val split for GAN (e.g., 20% of GAN set)
    val_size = max(1, int(0.2 * num_gan))
    train_size = num_gan - val_size
    gan_train, gan_val = random_split(full_gan_dataset, [train_size, val_size])

    print(f"MNIST train size: {len(mnist_train)}")
    print(f"GAN train size:   {len(gan_train)}")
    print(f"GAN val size:     {len(gan_val)}")

    combined_train = ConcatDataset([mnist_train, gan_train])

    # Oversample GAN samples so they matter, but not too crazy
    num_mnist = len(mnist_train)
    num_gan_train = len(gan_train)

    weights = [1.0] * num_mnist + [50.0] * num_gan_train
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(combined_train),
        replacement=True
    )

    train_loader = DataLoader(
        combined_train,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )

    gan_val_loader = DataLoader(
        gan_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, gan_val_loader


@torch.no_grad()
def evaluate_loader(model, loader, device, prefix="Eval"):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()
    acc = 100.0 * correct / total
    print(f"{prefix} accuracy: {acc:.2f}%  (on {total} samples)")
    return acc


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if not os.path.exists(BASE_MODEL_PATH):
        raise FileNotFoundError(
            f"Base model '{BASE_MODEL_PATH}' not found. "
            f"Train centralized model first."
        )

        # 1) Load strong MNIST model
    model = CentralizedMNISTCNN().to(device)
    state = torch.load(BASE_MODEL_PATH, map_location=device)
    model.load_state_dict(state)

    # Do NOT freeze anything – fine-tune the whole model with a tiny LR.
    for _, param in model.named_parameters():
        param.requires_grad = True

    # 2) Build combined loader + GAN val + MNIST test loader
    train_loader, gan_val_loader = get_combined_loader(batch_size=64)
    mnist_test_loader = get_mnist_test_loader(batch_size=256)

    # 3) Optimizer on all parameters, but small LR to preserve MNIST performance
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.Adam(trainable_params, lr=5e-5, weight_decay=1e-4)

    num_epochs = 6  # small but focused adaptation
    best_combined_score = -1.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for x, y in train_loader:
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

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        print(f"\nEpoch [{epoch}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # Evaluate on MNIST test and GAN val
        mnist_acc = evaluate_loader(model, mnist_test_loader, device, prefix="MNIST Test")
        gan_acc = evaluate_loader(model, gan_val_loader, device, prefix="GAN Val")

        # Combined score: give more weight to GAN, but require MNIST >= 99
        if mnist_acc >= 99.0:
            combined_score = gan_acc + 0.3 * mnist_acc
        else:
            combined_score = gan_acc  # don't reward low MNIST

        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_state = model.state_dict()
            print(f"  -> New best combined score: {best_combined_score:.2f} (MNIST {mnist_acc:.2f}%, GAN {gan_acc:.2f}%)")

    if best_state is not None:
        torch.save(best_state, OUTPUT_MODEL_PATH)
        print(f"\nSaved GAN-adapted model to '{OUTPUT_MODEL_PATH}' "
              f"(best combined MNIST+GAN score = {best_combined_score:.2f})")
    else:
        print("\nNo improvement found; not saving.")


if __name__ == "__main__":
    main()
