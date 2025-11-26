import os
import sys

# Make project root + models importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "models")))

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from models.mnist_cnn import CentralizedMNISTCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "centralized_mnist_cnn_gan.pt"   # GAN-adapted model
GAN_ROOT = "gan_labeled"                      # your labeled folders 0..9


def main():
    print("Using device:", DEVICE)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model '{MODEL_PATH}' not found.")

    # Load model
    model = CentralizedMNISTCNN().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # Same preprocessing as test
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if not os.path.exists(GAN_ROOT):
        raise FileNotFoundError(f"GAN labeled folder '{GAN_ROOT}' not found.")

    gan_dataset = datasets.ImageFolder(
        root=GAN_ROOT,
        transform=transform
    )

    gan_loader = DataLoader(
        gan_dataset,
        batch_size=32,
        shuffle=False
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in gan_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            _, preds = logits.max(1)
            total += y.size(0)
            correct += preds.eq(y).sum().item()

    acc = 100.0 * correct / total
    print(f"GAN labeled accuracy: {acc:.2f}% on {total} samples")


if __name__ == "__main__":
    main()
