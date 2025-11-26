import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.mnist_cnn import CentralizedMNISTCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Same normalization as in train_centralized.py
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST test set
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Load trained model
model = CentralizedMNISTCNN().to(device)
model.load_state_dict(torch.load("centralized_mnist_cnn.pt", map_location=device))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        _, preds = logits.max(1)
        total += y.size(0)
        correct += preds.eq(y).sum().item()

test_acc = 100.0 * correct / total
print(f"MNIST Test Accuracy (re-check): {test_acc:.2f}%")

# Also print first 10 predictions vs labels
x0, y0 = next(iter(test_loader))
x0, y0 = x0.to(device), y0.to(device)
with torch.no_grad():
    logits0 = model(x0)
    _, preds0 = logits0.max(1)

print("\nFirst 10 samples (label -> pred):")
for i in range(10):
    print(f"{i}: {int(y0[i].item())} -> {int(preds0[i].item())}")
