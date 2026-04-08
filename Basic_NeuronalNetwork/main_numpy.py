"""Neural Network - NumPy Version.

Same architecture as pure Python/PyTorch versions: 784 -> 40 (ReLU) -> 10 (Softmax).
Downloads MNIST directly and includes matplotlib visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import struct
import gzip
import urllib.request

# === Configuration ===
N_X = 28 * 28
N_Y = 40
N_Z = 10
LRATE = 0.1
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mnist")
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "weights_numpy")

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


# === Data Loading ===
def download_mnist(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    paths = {}
    for name, url in MNIST_URLS.items():
        filepath = os.path.join(data_dir, f"{name}.gz")
        if not os.path.exists(filepath):
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, filepath)
        paths[name] = filepath
    return paths


def load_mnist_images(filepath):
    with gzip.open(filepath, "rb") as f:
        _, n_images, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n_images, rows * cols).astype(np.float32) / 255.0


def load_mnist_labels(filepath):
    with gzip.open(filepath, "rb") as f:
        _, n_labels = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def one_hot(labels, n_classes=10):
    result = np.zeros((len(labels), n_classes))
    result[np.arange(len(labels)), labels] = 1.0
    return result


def load_data():
    paths = download_mnist(DATA_DIR)
    train_x = load_mnist_images(paths["train_images"])
    train_y = load_mnist_labels(paths["train_labels"])
    test_x = load_mnist_images(paths["test_images"])
    test_y = load_mnist_labels(paths["test_labels"])
    return train_x, train_y, test_x, test_y


# === Network ===
def create_network():
    w1 = np.random.randn(N_Y, N_X) * np.sqrt(2.0 / N_X)
    b1 = np.zeros(N_Y)
    w2 = np.random.randn(N_Z, N_Y) * np.sqrt(2.0 / N_Y)
    b2 = np.zeros(N_Z)
    return w1, b1, w2, b2


def relu(z):
    return np.maximum(z, 0)


def softmax(z):
    shifted = z - np.max(z, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)


def forward(x, w1, b1, w2, b2):
    z1 = x @ w1.T + b1
    a1 = relu(z1)
    z2 = a1 @ w2.T + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2


# === Backpropagation ===
def backward(z1, a1, a2, w2, x, y_onehot):
    dz2 = a2 - y_onehot
    dw2 = np.outer(dz2, a1)
    db2 = dz2

    dz1 = (w2.T @ dz2) * (z1 > 0).astype(float)
    dw1 = np.outer(dz1, x)
    db1 = dz1

    return dw1, db1, dw2, db2


def update_weights(w1, b1, w2, b2, dw1, db1, dw2, db2, lr, batch_size):
    scale = lr / batch_size
    w1 -= scale * dw1
    b1 -= scale * db1
    w2 -= scale * dw2
    b2 -= scale * db2
    return w1, b1, w2, b2


def cross_entropy_loss(a2, y_onehot):
    return -np.log(np.clip(a2[np.argmax(y_onehot)], 1e-10, 1.0))


# === Visualization ===
def show_sample_digits(images, labels, n=10):
    fig, axes = plt.subplots(2, n // 2, figsize=(12, 5))
    indices = np.random.choice(len(images), n, replace=False)
    for ax, idx in zip(axes.flat, indices):
        ax.imshow(images[idx].reshape(28, 28), cmap="gray")
        ax.set_title(f"Label: {labels[idx]}", fontsize=10)
        ax.axis("off")
    fig.suptitle("Sample Digits from MNIST", fontsize=14)
    plt.tight_layout()
    plt.savefig("numpy_sample_digits.png", dpi=100)
    plt.close()
    print("Saved: numpy_sample_digits.png")


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
    plt.savefig("numpy_training_curves.png", dpi=100)
    plt.close()
    print("Saved: numpy_training_curves.png")


def plot_confusion_matrix(test_x, test_y, w1, b1, w2, b2):
    _, _, _, a2 = forward(test_x, w1, b1, w2, b2)
    preds = np.argmax(a2, axis=1)

    matrix = np.zeros((N_Z, N_Z), dtype=int)
    for true, pred in zip(test_y, preds):
        matrix[true][pred] += 1

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(N_Z))
    ax.set_yticks(range(N_Z))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(N_Z):
        for j in range(N_Z):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color, fontsize=8)

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("numpy_confusion_matrix.png", dpi=100)
    plt.close()
    print("Saved: numpy_confusion_matrix.png")


def show_predictions(test_x, test_y, w1, b1, w2, b2, n=10):
    indices = np.random.choice(len(test_x), n, replace=False)
    fig, axes = plt.subplots(2, n // 2, figsize=(12, 5))

    for ax, idx in zip(axes.flat, indices):
        _, _, _, a2 = forward(test_x[idx], w1, b1, w2, b2)
        pred = np.argmax(a2)
        true = test_y[idx]
        color = "green" if pred == true else "red"

        ax.imshow(test_x[idx].reshape(28, 28), cmap="gray")
        ax.set_title(f"Pred: {pred} (True: {true})", color=color, fontsize=10)
        ax.axis("off")

    fig.suptitle("Sample Predictions", fontsize=14)
    plt.tight_layout()
    plt.savefig("numpy_predictions.png", dpi=100)
    plt.close()
    print("Saved: numpy_predictions.png")


# === Training ===
def train(train_x, train_y_onehot, w1, b1, w2, b2, epochs=100, train_size=20000):
    print(f"\nTraining for {epochs} epochs on {train_size} samples...")
    print(f"Architecture: {N_X} -> {N_Y} (ReLU) -> {N_Z} (Softmax)")
    print(f"Learning rate: {LRATE}\n")

    losses = []
    accuracies = []

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0

        for n in range(train_size):
            x = train_x[n]
            y = train_y_onehot[n]

            z1, a1, z2, a2 = forward(x, w1, b1, w2, b2)
            dw1, db1, dw2, db2 = backward(z1, a1, a2, w2, x, y)
            w1, b1, w2, b2 = update_weights(w1, b1, w2, b2, dw1, db1, dw2, db2, LRATE, train_size)

            total_loss += cross_entropy_loss(a2, y)
            if np.argmax(a2) == np.argmax(y):
                correct += 1

        avg_loss = total_loss / train_size
        acc = correct / train_size
        losses.append(avg_loss)
        accuracies.append(acc)

        if epoch % 10 == 0:
            bar = "=" * int(acc * 40)
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Accuracy: {acc:.4f} [{bar:<40}]")

    return w1, b1, w2, b2, losses, accuracies


# === Evaluation ===
def evaluate(test_x, test_y, w1, b1, w2, b2):
    _, _, _, a2 = forward(test_x, w1, b1, w2, b2)
    preds = np.argmax(a2, axis=1)
    correct = np.sum(preds == test_y)
    acc = correct / len(test_y)
    print(f"\nTest accuracy: {acc:.4f} ({correct}/{len(test_y)})")
    return acc


# === Weight I/O ===
def save_weights(w1, b1, w2, b2, directory):
    os.makedirs(directory, exist_ok=True)
    np.save(os.path.join(directory, "w1.npy"), w1)
    np.save(os.path.join(directory, "b1.npy"), b1)
    np.save(os.path.join(directory, "w2.npy"), w2)
    np.save(os.path.join(directory, "b2.npy"), b2)
    print(f"Weights saved to {directory}/")


# === Main ===
if __name__ == "__main__":
    print("=" * 60)
    print("  Neural Network - NumPy Version")
    print("=" * 60)

    # Load data
    train_x, train_y, test_x, test_y = load_data()
    train_y_onehot = one_hot(train_y)
    print(f"Train: {len(train_x)} samples | Test: {len(test_x)} samples")

    # Show sample digits
    show_sample_digits(train_x, train_y)

    # Create and train
    w1, b1, w2, b2 = create_network()
    w1, b1, w2, b2, losses, accuracies = train(
        train_x, train_y_onehot, w1, b1, w2, b2, epochs=100, train_size=20000
    )

    # Visualize training
    plot_training_history(losses, accuracies)

    # Evaluate
    evaluate(test_x, test_y, w1, b1, w2, b2)

    # Visualize results
    plot_confusion_matrix(test_x, test_y, w1, b1, w2, b2)
    show_predictions(test_x, test_y, w1, b1, w2, b2)

    # Save
    save_weights(w1, b1, w2, b2, WEIGHTS_DIR)
