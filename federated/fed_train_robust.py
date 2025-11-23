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


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Hyperparameters (LEVEL 3)
# -----------------------------
NUM_CLIENTS = 10
ROUNDS = 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
MODEL_SAVE_PATH = "federated_global_mnist_cnn_robust.pt"

# Attack configuration
ATTACKER_ID = 0              # which client is malicious
ENABLE_ATTACK = True         # turn attack on/off
ATTACK_TYPE = "scaling"      # "scaling" or "random"
ATTACK_SCALE = 10.0          # factor for scaling attack

# Defense configuration
ENABLE_DETECTION = True
NORM_THRESHOLD_MULTIPLIER = 3.0   # > median_norm * this => suspicious


# -----------------------------
# Federated dataloaders
# -----------------------------
def get_federated_dataloaders(num_clients: int, batch_size: int):
    """
    Split MNIST train set into `num_clients` disjoint subsets (one per client).
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
# Local client training
# -----------------------------
def client_update(model, train_loader, epochs, lr, weight_decay):
    """
    Train a copy of the global model on a single client's local data.
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
# Attack simulation
# -----------------------------
def apply_attack(global_model, local_model):
    """
    Modify the local_model to simulate a malicious client update.
    """
    if not ENABLE_ATTACK:
        return local_model

    if ATTACK_TYPE == "scaling":
        # Scale the update relative to global weights
        global_state = global_model.state_dict()
        local_state = local_model.state_dict()
        attacked_state = {}

        for key in global_state.keys():
            delta = local_state[key] - global_state[key]
            attacked_state[key] = global_state[key] + ATTACK_SCALE * delta

        local_model.load_state_dict(attacked_state)
        return local_model

    elif ATTACK_TYPE == "random":
        # Replace weights with random noise
        local_state = local_model.state_dict()
        for key in local_state.keys():
            noise = torch.randn_like(local_state[key])
            local_state[key] = noise
        local_model.load_state_dict(local_state)
        return local_model

    # no-op if unknown type
    return local_model


# -----------------------------
# Utility: compute update norm
# -----------------------------
def compute_update_norm(global_model, client_model) -> float:
    """
    Compute L2 norm of client's update (difference from global model).
    """
    g_state = global_model.state_dict()
    c_state = client_model.state_dict()
    sq_sum = 0.0

    for key in g_state.keys():
        delta = (c_state[key] - g_state[key]).float()
        sq_sum += torch.sum(delta * delta).item()

    return sq_sum ** 0.5


# -----------------------------
# Standard FedAvg
# -----------------------------
def fed_avg(global_model, client_models, client_sizes):
    """
    FedAvg that ignores non-floating-point parameters.
    Some parameters (e.g., BatchNorm counters) are int64 and should NOT be averaged.
    """
    global_dict = global_model.state_dict()

    # initialize aggregation buffers
    for key in global_dict.keys():
        # Only aggregate float/half parameters
        if global_dict[key].dtype in [torch.float32, torch.float64, torch.float16]:
            global_dict[key] = torch.zeros_like(global_dict[key])
        # Otherwise leave them as-is (keep global values)
        else:
            global_dict[key] = global_dict[key]

    total_samples = sum(client_sizes)

    for client_model, n_k in zip(client_models, client_sizes):
        client_state = client_model.state_dict()
        weight = n_k / total_samples

        for key in global_dict.keys():
            # Only aggregate floating-point tensors
            if global_dict[key].dtype in [torch.float32, torch.float64, torch.float16]:
                global_dict[key] += client_state[key] * weight

    global_model.load_state_dict(global_dict)
    return global_model



# -----------------------------
# Robust aggregation with detection
# -----------------------------
def robust_fed_avg(global_model, client_models, client_sizes):
    """
    Detect suspicious clients based on update norm, and aggregate only benign ones.
    """
    if not ENABLE_DETECTION:
        # No detection, just FedAvg
        return fed_avg(global_model, client_models, client_sizes), list(range(len(client_models)))

    norms = [compute_update_norm(global_model, cm) for cm in client_models]
    norms_tensor = torch.tensor(norms)
    median_norm = torch.median(norms_tensor).item()

    benign_indices = []
    suspicious_indices = []

    for idx, norm in enumerate(norms):
        if norm > median_norm * NORM_THRESHOLD_MULTIPLIER:
            suspicious_indices.append(idx)
        else:
            benign_indices.append(idx)

    # If everything is flagged suspicious by mistake, fall back to using all
    if len(benign_indices) == 0:
        benign_indices = list(range(len(client_models)))
        suspicious_indices = []

    print(f"  Detection: benign clients = {benign_indices}, suspicious clients = {suspicious_indices}")

    benign_models = [client_models[i] for i in benign_indices]
    benign_sizes = [client_sizes[i] for i in benign_indices]

    new_global = fed_avg(global_model, benign_models, benign_sizes)
    return new_global, benign_indices


# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def evaluate(model, test_loader):
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
        correct += pred.eq(y).sum().item()
        total += y.size(0)

    return total_loss / total, 100.0 * correct / total


# -----------------------------
# Main: Robust Federated Training
# -----------------------------
def main():
    print("Preparing federated dataloaders (10 clients)...")
    client_loaders, test_loader = get_federated_dataloaders(NUM_CLIENTS, BATCH_SIZE)

    print("Initializing global model for robust FL...")
    global_model = CentralizedMNISTCNN().to(device)

    best_acc = 0.0

    for round_idx in range(1, ROUNDS + 1):
        print(f"\n=== Robust Federated Round {round_idx}/{ROUNDS} ===")

        client_models = []
        client_sizes = []

        # ----- Local training -----
        for client_id, train_loader in enumerate(client_loaders):
            local_model, n_k = client_update(
                global_model,
                train_loader,
                epochs=LOCAL_EPOCHS,
                lr=LR,
                weight_decay=WEIGHT_DECAY
            )

            # Simulate attack on attacker client
            if ENABLE_ATTACK and client_id == ATTACKER_ID:
                print(f"  >> Applying ATTACK ({ATTACK_TYPE}) to client {client_id}")
                local_model = apply_attack(global_model, local_model)

            client_models.append(local_model)
            client_sizes.append(n_k)
            print(f"  Client {client_id}: trained on {n_k} samples")

        # ----- Robust aggregation with detection -----
        global_model, benign_indices = robust_fed_avg(global_model, client_models, client_sizes)

        # ----- Evaluate global model -----
        test_loss, test_acc = evaluate(global_model, test_loader)
        print(f"  Round {round_idx}: Test Loss = {test_loss:.4f}, Test Acc = {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(global_model.state_dict(), MODEL_SAVE_PATH)
            print(f"  >>> Saved best robust global model with acc = {best_acc:.2f}%")

    print(f"\nRobust training finished. Best Global Test Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
