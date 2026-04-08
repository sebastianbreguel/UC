"""Neural Network - Pure Python (no external libraries).

Same architecture as NumPy/PyTorch versions: 784 -> 40 (ReLU) -> 10 (Softmax).
Uses only stdlib: random, math, os.
"""

import random
import math
import os

# === Configuration ===
N_X = 28 * 28   # Input: 28x28 pixels
N_Y = 40         # Hidden layer neurons
N_Z = 10         # Output: digits 0-9
LRATE = 0.1
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "original", "oneline.txt")
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "weights")


# === Data Loading ===
def load_data(path):
    with open(path) as f:
        lines = f.readlines()

    n_nums = int(lines.pop(0).strip())
    _n_rows = int(lines.pop(0).strip())
    _n_cols = int(lines.pop(0).strip())

    data = []
    labels = []
    for _ in range(n_nums):
        label_str = lines.pop(0).strip()
        pixel_str = lines.pop(0).strip()
        labels.append(one_hot(int(label_str)))
        data.append([int(c) for c in pixel_str])

    return data, labels


def one_hot(digit):
    vec = [0.0] * N_Z
    vec[digit] = 1.0
    return vec


# === Network Creation ===
def create_network():
    w1 = [[random.uniform(-0.5, 0.5) for _ in range(N_X)] for _ in range(N_Y)]
    b1 = [random.uniform(-0.5, 0.5) for _ in range(N_Y)]
    w2 = [[random.uniform(-0.5, 0.5) for _ in range(N_Y)] for _ in range(N_Z)]
    b2 = [random.uniform(-0.5, 0.5) for _ in range(N_Z)]
    return w1, b1, w2, b2


# === Forward Propagation ===
def forward(x, w1, b1, w2, b2):
    z1, a1 = forward_hidden(x, w1, b1)
    z2, a2 = forward_output(a1, w2, b2)
    return z1, a1, z2, a2


def forward_hidden(x, w1, b1):
    z1 = []
    a1 = []
    for y in range(N_Y):
        activation = sum(x[i] * w1[y][i] for i in range(N_X)) + b1[y]
        z1.append(activation)
        a1.append(max(activation, 0.0))  # ReLU
    return z1, a1


def forward_output(a1, w2, b2):
    z2 = []
    for z in range(N_Z):
        activation = sum(a1[i] * w2[z][i] for i in range(N_Y)) + b2[z]
        z2.append(activation)
    a2 = softmax(z2)
    return z2, a2


def softmax(z):
    max_z = max(z)
    exps = [math.exp(zi - max_z) for zi in z]
    total = sum(exps)
    return [e / total for e in exps]


# === Backpropagation ===
def backward(z1, a1, a2, w2, x, y):
    # Output layer gradients
    dz2 = [a2[z] - y[z] for z in range(N_Z)]
    dw2 = [[dz2[z] * a1[j] for j in range(N_Y)] for z in range(N_Z)]
    db2 = dz2[:]

    # Hidden layer gradients
    dz1 = [0.0] * N_Y
    for j in range(N_Y):
        for z in range(N_Z):
            dz1[j] += w2[z][j] * dz2[z]
        if z1[j] <= 0:
            dz1[j] = 0.0  # ReLU derivative

    dw1 = [[dz1[j] * x[i] for i in range(N_X)] for j in range(N_Y)]
    db1 = dz1[:]

    return dw1, db1, dw2, db2


# === Weight Update ===
def update_weights(w1, b1, w2, b2, dw1, db1, dw2, db2, lr, batch_size):
    scale = lr / batch_size
    for y in range(N_Y):
        for x in range(N_X):
            w1[y][x] -= scale * dw1[y][x]
        b1[y] -= scale * db1[y]

    for z in range(N_Z):
        for y in range(N_Y):
            w2[z][y] -= scale * dw2[z][y]
        b2[z] -= scale * db2[z]

    return w1, b1, w2, b2


# === Loss ===
def cross_entropy_loss(a2, y):
    true_idx = y.index(1.0)
    return -math.log(max(a2[true_idx], 1e-10))


def get_prediction(a2):
    return a2.index(max(a2))


# === Visualization ===
def print_digit(pixels):
    for row in range(28):
        line = ""
        for col in range(28):
            line += "##" if pixels[row * 28 + col] > 0 else "  "
        print(line)


def print_sample_digits(data, labels, n=3):
    print(f"\n{'=' * 60}")
    print(f"  {n} Sample Digits")
    print(f"{'=' * 60}")
    indices = random.sample(range(len(data)), n)
    for idx in indices:
        true_label = labels[idx].index(1.0)
        print(f"\nLabel: {int(true_label)}")
        print_digit(data[idx])


# === Training ===
def train(data, labels, w1, b1, w2, b2, epochs=100, train_size=20000):
    print(f"\nTraining for {epochs} epochs on {train_size} samples...")
    print(f"Architecture: {N_X} -> {N_Y} (ReLU) -> {N_Z} (Softmax)")
    print(f"Learning rate: {LRATE}\n")

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0

        for n in range(train_size):
            x = data[n]
            y = labels[n]

            z1, a1, z2, a2 = forward(x, w1, b1, w2, b2)
            dw1, db1, dw2, db2 = backward(z1, a1, a2, w2, x, y)
            w1, b1, w2, b2 = update_weights(w1, b1, w2, b2, dw1, db1, dw2, db2, LRATE, train_size)

            if epoch % 10 == 0:
                total_loss += cross_entropy_loss(a2, y)
                if get_prediction(a2) == y.index(1.0):
                    correct += 1

        if epoch % 10 == 0:
            acc = correct / train_size
            avg_loss = total_loss / train_size
            bar = "=" * int(acc * 40)
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Accuracy: {acc:.4f} [{bar:<40}]")

    return w1, b1, w2, b2


# === Evaluation ===
def evaluate(data, labels, w1, b1, w2, b2, start=42000):
    test_data = data[start:]
    test_labels = labels[start:]
    correct = 0
    for i in range(len(test_data)):
        _, _, _, a2 = forward(test_data[i], w1, b1, w2, b2)
        if get_prediction(a2) == test_labels[i].index(1.0):
            correct += 1
    acc = correct / len(test_data)
    print(f"\nTest accuracy: {acc:.4f} ({correct}/{len(test_data)})")
    return acc


# === Weight I/O ===
def save_weights(w1, b1, w2, b2, directory):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "w1.txt"), "w") as f:
        for row in w1:
            f.write(",".join(str(v) for v in row) + "\n")
    with open(os.path.join(directory, "w2.txt"), "w") as f:
        for row in w2:
            f.write(",".join(str(v) for v in row) + "\n")
    with open(os.path.join(directory, "b1.txt"), "w") as f:
        f.write(",".join(str(v) for v in b1))
    with open(os.path.join(directory, "b2.txt"), "w") as f:
        f.write(",".join(str(v) for v in b2))
    print(f"Weights saved to {directory}/")


def load_weights(directory):
    w1 = []
    with open(os.path.join(directory, "w1.txt")) as f:
        for line in f:
            w1.append([float(v) for v in line.strip().split(",")])
    w2 = []
    with open(os.path.join(directory, "w2.txt")) as f:
        for line in f:
            w2.append([float(v) for v in line.strip().split(",")])
    with open(os.path.join(directory, "b1.txt")) as f:
        b1 = [float(v) for v in f.read().strip().split(",")]
    with open(os.path.join(directory, "b2.txt")) as f:
        b2 = [float(v) for v in f.read().strip().split(",")]
    return w1, b1, w2, b2


# === Main ===
if __name__ == "__main__":
    print("=" * 60)
    print("  Neural Network - Pure Python (no external libraries)")
    print("=" * 60)

    # Load data
    data, labels = load_data(DATA_PATH)
    print(f"Loaded {len(data)} samples ({N_X} pixels each)")

    # Show sample digits
    print_sample_digits(data, labels, n=3)

    # Create and train network
    w1, b1, w2, b2 = create_network()
    w1, b1, w2, b2 = train(data, labels, w1, b1, w2, b2, epochs=100, train_size=20000)

    # Evaluate on test set
    evaluate(data, labels, w1, b1, w2, b2)

    # Save weights
    save_weights(w1, b1, w2, b2, WEIGHTS_DIR)
