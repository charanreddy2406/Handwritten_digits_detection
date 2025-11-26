import sys
import os

# Allow importing from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "models")))

import torch
from PIL import Image
from torchvision import transforms
from collections import Counter

from models.mnist_cnn import CentralizedMNISTCNN

# ---- CONFIG ----
GRID_IMAGE_PATH = "gan_50_grid.png"   # rename if needed
OUTPUT_DIR = "gan_eval_images"        # where cropped tiles will be saved
MODEL_PATH = "centralized_mnist_cnn_gan.pt"


def split_grid_to_tiles(
    grid_path: str,
    output_dir: str,
    rows: int = 8,
    cols: int = 8,
    max_tiles: int = 50,
):
    """
    Split a grid image containing rows x cols digit cells into separate tiles.
    This does NOT assume tiles are 28x28. It computes cell size from the image.
    """
    os.makedirs(output_dir, exist_ok=True)

    img = Image.open(grid_path).convert("L")  # grayscale
    w, h = img.size

    cell_w = w // cols
    cell_h = h // rows

    print(f"Grid size: {w}x{h}")
    print(f"Detected rows={rows}, cols={cols}, cell_w={cell_w}, cell_h={cell_h}")

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= max_tiles:
                break

            left = c * cell_w
            upper = r * cell_h
            right = left + cell_w
            lower = upper + cell_h

            tile = img.crop((left, upper, right, lower))
            fname = os.path.join(output_dir, f"gan_tile_{idx:02d}.png")
            tile.save(fname)
            idx += 1

        if idx >= max_tiles:
            break

    print(f"Saved {idx} tiles to '{output_dir}'")
    return idx


def load_model(model_path: str, device: torch.device):
    """
    Load your trained centralized CNN model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. Train Level 1 first."
        )

    model = CentralizedMNISTCNN()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def evaluate_tiles(model, tiles_dir: str, device: torch.device):
    """
    Evaluate each GAN tile using the trained CNN.
    """
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    files = sorted(
        [f for f in os.listdir(tiles_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )

    print(f"Found {len(files)} tile images in '{tiles_dir}'")

    preds = []
    softmax = torch.nn.Softmax(dim=1)

    for fname in files:
        path = os.path.join(tiles_dir, fname)
        img = Image.open(path)
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            prob = softmax(logits)
            conf, pred = prob.max(1)

        pred_digit = pred.item()
        conf_val = conf.item()
        preds.append(pred_digit)

        print(f"{fname}: predicted = {pred_digit}, confidence = {conf_val:.4f}")

    # Histogram of predictions
    counts = Counter(preds)
    print("\nPrediction histogram:")
    for d in range(10):
        print(f"Digit {d}: {counts[d]}")

    return counts


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- Step 1: split grid ----
    num_tiles = split_grid_to_tiles(
        GRID_IMAGE_PATH,
        OUTPUT_DIR,
        rows=8,    # your grid is 8 x 8
        cols=8,
        max_tiles=50
    )

    # ---- Step 2: load model ----
    model = load_model(MODEL_PATH, device)

    # ---- Step 3: evaluate ----
    print(f"\nEvaluating model on {num_tiles} GAN-style tiles...")
    evaluate_tiles(model, OUTPUT_DIR, device)


if __name__ == "__main__":
    main()
