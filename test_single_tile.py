from PIL import Image
import torch
from torchvision import transforms
from models.mnist_cnn import CentralizedMNISTCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = CentralizedMNISTCNN()
model.load_state_dict(torch.load("centralized_mnist_cnn.pt", map_location=device))
model.to(device)
model.eval()

# Standard MNIST normalization + resize
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# CHANGE THIS to the tile you want to test
img_path = "gan_eval_images/gan_tile_12.png"

img = Image.open(img_path)
x = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(x)
    prob = torch.softmax(logits, dim=1)
    conf, pred = prob.max(1)

print(f"Testing: {img_path}")
print("Predicted digit:", pred.item())
print("Confidence:", conf.item())
