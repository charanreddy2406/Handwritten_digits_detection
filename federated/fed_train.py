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

from models.mnist_cnn import CentralizedMNISTCNN   # IMPORTANT: must match file


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Hyperparameters (LEVEL-2 REQUIRED)
# -----------------------------
NUM_CLIENTS = 10          # as required by professor
ROUNDS = 10               # communication rounds
LOCAL_EPOCHS = 1          # each client trains 1 epoch per round
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4       # robust generalization
MODEL_SAVE_PATH = "federated_global_mnist_cnn.pt"


# -----------------------------
# Federated Dataset Split
# -----------------------------
def get_federated_dataloaders(num_clients: int, batch_size: int):
    """
    Splits MNIST training set into num_clients disjoint partitions.
    Each partition = 1 client.
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

    n = len(train_dataset)
    lengths = [n // num_clients] * num_clients
    for i in range(n % num_clients):
        lengths[i] += 1

    subsets = random_split(train_dataset, lengths)

    client_loaders = [
        DataLoader(subset, batch_size=batch_size, shuffle=True)
        for subset in subsets
    ]

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return client_loaders, test_loader


# -----------------------------
# Local Training on Client
# -----------------------------
def client_update(model, train_loader, epochs, lr, weight_decay):
    """
    Each client trains a *copy* of the global model.
    """
    model = copy.deepcopy(model)
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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
# FedAvg Server Aggregation
# -----------------------------
def fed_avg(global_model, client_models, client_sizes):
    """
    Classic FedAvg:
        w_global = Σ (n_k / N_total) * w_k
    """
    global_dict = global_model.state_dict()

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
def evaluate(model, test_loader):
    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)

        total_loss += loss.item() * y.size(0)
        _, pred = out.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / total
    acc = 100 * correct / total
    return avg_loss, acc


# -----------------------------
# Main Federated Training Loop
# -----------------------------
def main():

    print("Preparing federated dataloaders (10 clients)...")
    client_loaders, test_loader = get_federated_dataloaders(NUM_CLIENTS, BATCH_SIZE)

    print("Initializing global model...\n")
    global_model = CentralizedMNISTCNN().to(device)

    best_acc = 0

    for round_idx in range(1, ROUNDS + 1):
        print(f"\n==================== Round {round_idx}/{ROUNDS} ====================")

        client_models = []
        client_sizes = []

        # ---- CLIENT TRAINING ----
        for client_id, train_loader in enumerate(client_loaders):
            local_model, n_k = client_update(
                global_model,
                train_loader,
                epochs=LOCAL_EPOCHS,
                lr=LR,
                weight_decay=WEIGHT_DECAY
            )
            client_models.append(local_model)
            client_sizes.append(n_k)
            print(f" Client {client_id} finished training on {n_k} samples")

        # ---- SERVER AGGREGATION ----
        global_model = fed_avg(global_model, client_models, client_sizes)

        # ---- EVALUATE GLOBAL MODEL ----
        loss, acc = evaluate(global_model, test_loader)
        print(f" Round {round_idx}: Test Loss = {loss:.4f}, Test Acc = {acc:.2f}%")

        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(global_model.state_dict(), MODEL_SAVE_PATH)
            print(f"  >>> Best model updated! New Accuracy = {best_acc:.2f}%")

    print(f"\nTraining Finished! Best Global Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
