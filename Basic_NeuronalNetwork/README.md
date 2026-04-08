# Basic Neural Network

Three implementations of the **same neural network** for MNIST digit recognition, showing progressive complexity:

| Version | File | Dependencies | Speed |
|---------|------|-------------|-------|
| Pure Python | `main_pure.py` | None (stdlib only) | ~minutes/epoch |
| NumPy | `main_numpy.py` | numpy, matplotlib | ~seconds/epoch |
| PyTorch | `main_torch.py` | torch, torchvision, matplotlib | ~ms/epoch |

**Architecture (all three):** 784 (input) → 40 (ReLU) → 10 (Softmax)

## Setup

Requires [uv](https://docs.astral.sh/uv/):

```bash
# Pure Python (no dependencies)
uv run python main_pure.py

# NumPy version
uv run --extra numpy python main_numpy.py

# PyTorch version
uv run --extra torch python main_torch.py
```

## Dataset

- **Pure Python:** uses pre-processed `data/original/oneline.txt` (70k binary pixel strings)
- **NumPy:** downloads MNIST IDX files automatically
- **PyTorch:** downloads via `torchvision.datasets.MNIST`

## Results

All three versions achieve ~87% accuracy with the same architecture (40 hidden neurons).
Increasing `N_Y` or adding layers would improve accuracy further.
