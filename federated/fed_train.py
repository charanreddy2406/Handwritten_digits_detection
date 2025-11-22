import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from models.mnist_cnn import CentralizedMNISTCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Hyperparameters
# -----------------------------
NUM_CLIENTS = 10
ROUNDS = 10           # number of communication rounds
LOCAL_EPOCHS = 1      # local epochs per round
BATCH_SIZE = 64
LR = 1e-3
MODEL_SAVE_PATH = "federated_global_mnist_cnn.pt"


# -----------------------------
# Dataset & Federated Split
# -----------------------------
def get_federated_dataloaders(
    num_clients: int,
    batch_size: int
) -> Tuple[List[DataLoader], DataLoader]:
    """
    For now: create federated split by partitioning the standard MNIST train set
    into `num_clients` disjoint subsets.

    Later: you can replace this function with the instructor's provided
    federated dataloader while keeping the rest of the FL logic unchanged.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
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

    # Split train_dataset indices into num_clients parts
    n = len(train_dataset)
    lengths = [n // num_clients] * num_clients
    for i in range(n % num_clients):
        lengths[i] += 1

    subsets: List[Subset] = list(random_split(train_dataset, lengths))

    client_loaders: List[DataLoader] = []
    for subset in subsets:
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True
        )
        client_loaders.append(loader)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return client_loaders, test_loader


# -----------------------------
# Local client update
# -----------------------------
def client_update(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    lr: float
) -> Tuple[nn.Module, int]:
    """
    Train a copy of the global model on a single client's data.
    Returns the updated model and number of samples used.
    """
    model = copy.deepcopy(model)
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_samples = 0

    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            num_samples += y.size(0)

    return model, num_samples


# -----------------------------
# Server-side aggregation (FedAvg)
# -----------------------------
def fed_avg(
    global_model: nn.Module,
    client_models: List[nn.Module],
    client_sizes: List[int]
) -> nn.Module:
    """
    global_param = sum_k (n_k / N) * param_k
    """
    global_dict = global_model.state_dict()

    # initialize with zeros
    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key])

    total_samples = sum(client_sizes)

    for client_model, n_k in zip(client_models, client_sizes):
        client_state = client_model.state_dict()
        weight = n_k / total_samples
        for key in global_dict.keys():
            global_dict[key] += client_state[key] * weight

    global_model.load_state_dict(global_dict)
    return global_model


# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, test_loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)

        total_loss += loss.item() * y.size(0)
        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


# -----------------------------
# Main federated training loop
# -----------------------------
def main():
    client_loaders, test_loader = get_federated_dataloaders(NUM_CLIENTS, BATCH_SIZE)

    global_model = CentralizedMNISTCNN().to(device)
    best_acc = 0.0

    for round_idx in range(1, ROUNDS + 1):
        print(f"\n--- Round {round_idx}/{ROUNDS} ---")

        client_models = []
        client_sizes = []

        # each client trains locally
        for client_id, train_loader in enumerate(client_loaders):
            local_model, n_k = client_update(
                global_model,
                train_loader,
                epochs=LOCAL_EPOCHS,
                lr=LR
            )
            client_models.append(local_model)
            client_sizes.append(n_k)
            print(f" Client {client_id}: trained on {n_k} samples")

        # server aggregates
        global_model = fed_avg(global_model, client_models, client_sizes)

        # evaluate global model
        test_loss, test_acc = evaluate(global_model, test_loader)
        print(f" Round {round_idx}: Test Loss = {test_loss:.4f}, Test Acc = {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(global_model.state_dict(), MODEL_SAVE_PATH)
            print(f"  New best global model saved with acc = {best_acc:.2f}%")

    print(f"\nTraining finished. Best Global Test Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
