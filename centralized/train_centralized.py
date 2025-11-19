import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# ==========================
# 1. Hyperparameters
# ==========================
batch_size = 64
num_epochs = 10
learning_rate = 1e-3
model_save_path = "centralized_mnist_cnn.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================
# 2. CNN Model Definition
# ==========================
class CentralizedMNISTCNN(nn.Module):
    def __init__(self):
        super(CentralizedMNISTCNN, self).__init__()
        # Input: 1 x 28 x 28
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 32 x 28 x 28
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # 32 x 14 x 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),# 64 x 14 x 14
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                          # 64 x 7 x 7
        )

        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)  # 10 digits (0–9)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x


# ==========================
# 3. Dataset & Dataloaders
# ==========================
def get_dataloaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # standard MNIST mean/std
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


# ==========================
# 4. Training & Evaluation
# ==========================
def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    print(f"Epoch [{epoch}] - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, test_loader, criterion, prefix="Test"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    print(f"{prefix} Loss: {epoch_loss:.4f}, {prefix} Acc: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc


# ==========================
# 5. Save & Load Utilities
# ==========================
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(path, device):
    model = CentralizedMNISTCNN().to(device)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {path}")
    return model


# ==========================
# 6. Main Training Loop
# ==========================
def main():
    train_loader, test_loader = get_dataloaders(batch_size)

    model = CentralizedMNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        _, test_acc = evaluate(model, test_loader, criterion, prefix="Test")

        # Save the best model (highest test accuracy)
        if test_acc > best_acc:
            best_acc = test_acc
            save_model(model, model_save_path)

    print(f"Training finished. Best Test Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
