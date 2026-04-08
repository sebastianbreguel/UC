"""Neural Network - PyTorch Version.

Same architecture as pure Python/NumPy versions: 784 -> 40 (ReLU) -> 10 (Softmax).
Uses nn.Module, optim.SGD, DataLoader, and torchvision for MNIST.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# === Configuration ===
N_X = 28 * 28
N_Y = 40
N_Z = 10
LRATE = 0.1
BATCH_SIZE = 64
EPOCHS = 20
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mnist_torch")
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "weights_torch")


# === Data ===
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset


# === Model ===
class SimpleNN(nn.Module):
    """Same architecture: 784 -> 40 (ReLU) -> 10."""

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(N_X, N_Y)
        self.relu = nn.ReLU()
        self.output = nn.Linear(N_Y, N_Z)

    def forward(self, x):
        x = x.view(-1, N_X)  # Flatten 28x28 -> 784
        x = self.relu(self.hidden(x))
        x = self.output(x)
        return x  # Raw logits (CrossEntropyLoss applies softmax)


# === Visualization ===
def show_sample_digits(dataset, n=10):
    fig, axes = plt.subplots(2, n // 2, figsize=(12, 5))
    indices = torch.randperm(len(dataset))[:n]

    for ax, idx in zip(axes.flat, indices):
        image, label = dataset[idx]
        ax.imshow(image.squeeze(), cmap="gray")
        ax.set_title(f"Label: {label}", fontsize=10)
        ax.axis("off")

    fig.suptitle("Sample Digits from MNIST (PyTorch)", fontsize=14)
    plt.tight_layout()
    plt.savefig("torch_sample_digits.png", dpi=100)
    plt.close()
    print("Saved: torch_sample_digits.png")


def plot_training_history(losses, accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(losses, color="red", linewidth=1.5)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot([a * 100 for a in accuracies], color="blue", linewidth=1.5)
    ax2.set_title("Training Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("torch_training_curves.png", dpi=100)
    plt.close()
    print("Saved: torch_training_curves.png")


def plot_confusion_matrix(model, test_loader, device):
    model.eval()
    matrix = np.zeros((N_Z, N_Z), dtype=int)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            for true, pred in zip(labels.cpu(), preds.cpu()):
                matrix[true][pred] += 1

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(N_Z))
    ax.set_yticks(range(N_Z))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (PyTorch)")

    for i in range(N_Z):
        for j in range(N_Z):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color, fontsize=8)

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("torch_confusion_matrix.png", dpi=100)
    plt.close()
    print("Saved: torch_confusion_matrix.png")


def show_predictions(model, test_dataset, device, n=10):
    model.eval()
    indices = torch.randperm(len(test_dataset))[:n]
    fig, axes = plt.subplots(2, n // 2, figsize=(12, 5))

    with torch.no_grad():
        for ax, idx in zip(axes.flat, indices):
            image, true_label = test_dataset[idx]
            output = model(image.unsqueeze(0).to(device))
            pred = output.argmax(dim=1).item()
            color = "green" if pred == true_label else "red"

            ax.imshow(image.squeeze(), cmap="gray")
            ax.set_title(f"Pred: {pred} (True: {true_label})", color=color, fontsize=10)
            ax.axis("off")

    fig.suptitle("Sample Predictions (PyTorch)", fontsize=14)
    plt.tight_layout()
    plt.savefig("torch_predictions.png", dpi=100)
    plt.close()
    print("Saved: torch_predictions.png")


def visualize_weights(model):
    weights = model.hidden.weight.data.cpu().numpy()
    fig, axes = plt.subplots(5, 8, figsize=(14, 9))

    for i, ax in enumerate(axes.flat):
        ax.imshow(weights[i].reshape(28, 28), cmap="coolwarm", vmin=-0.5, vmax=0.5)
        ax.set_title(f"N{i}", fontsize=7)
        ax.axis("off")

    fig.suptitle("Hidden Layer Weights (40 neurons as 28x28 images)", fontsize=14)
    plt.tight_layout()
    plt.savefig("torch_weights.png", dpi=100)
    plt.close()
    print("Saved: torch_weights.png")


# === Training ===
def train(model, train_loader, device, epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LRATE)

    print(f"\nTraining for {epochs} epochs with batch size {BATCH_SIZE}...")
    print(f"Architecture: {N_X} -> {N_Y} (ReLU) -> {N_Z}")
    print(f"Optimizer: SGD (lr={LRATE})\n")

    losses = []
    accuracies = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

        avg_loss = total_loss / total
        acc = correct / total
        losses.append(avg_loss)
        accuracies.append(acc)

        bar = "=" * int(acc * 40)
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Accuracy: {acc:.4f} [{bar:<40}]")

    return losses, accuracies


# === Evaluation ===
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    acc = correct / total
    print(f"\nTest accuracy: {acc:.4f} ({correct}/{total})")
    return acc


# === Main ===
if __name__ == "__main__":
    print("=" * 60)
    print("  Neural Network - PyTorch Version")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    train_loader, test_loader, train_dataset, test_dataset = load_data()
    print(f"Train: {len(train_dataset)} samples | Test: {len(test_dataset)} samples")

    # Show sample digits
    show_sample_digits(train_dataset)

    # Create and train model
    model = SimpleNN().to(device)
    print(f"\nModel:\n{model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    losses, accuracies = train(model, train_loader, device)

    # Visualize training
    plot_training_history(losses, accuracies)

    # Evaluate
    evaluate(model, test_loader, device)

    # Visualize results
    plot_confusion_matrix(model, test_loader, device)
    show_predictions(model, test_dataset, device)
    visualize_weights(model)

    # Save
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "model.pt"))
    print(f"Model saved to {WEIGHTS_DIR}/model.pt")
