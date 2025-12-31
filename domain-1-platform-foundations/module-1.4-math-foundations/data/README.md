# Data Directory - Module 1.4: Mathematics for Deep Learning

This module primarily uses synthetic/generated data for demonstrations. No external datasets are required.

## Generated Data

The notebooks generate the following data on-the-fly:

### 1. XOR Dataset (Lab 1.4.1)
- Simple 4-sample binary classification
- Used to test backpropagation implementation
- Generated inline: `X = [[0,0], [0,1], [1,0], [1,1]]`, `y = [0, 1, 1, 0]`

### 2. Two Moons Dataset (Lab 1.4.3)
- 2D classification dataset with curved decision boundary
- Generated using custom function `create_moons_dataset()`
- Typical size: 200 samples

### 3. Rosenbrock Function (Lab 1.4.2, 1.4.3)
- Classic optimization test function
- `f(x,y) = (1-x)² + 100(y-x²)²`
- Global minimum at (1, 1)

### 4. Weight Matrices (Lab 1.4.4)
- Random matrices simulating neural network weights
- Generated with controlled rank for LoRA demonstrations
- Typical size: 768×768 (like BERT-base)

## Data Generation Scripts

If you need to pre-generate data:

```python
# XOR data
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y = np.array([[0], [1], [1], [0]], dtype=np.float64)

# Two moons
def create_moons_dataset(n_samples=200, noise=0.1):
    n_per_class = n_samples // 2
    theta1 = np.linspace(0, np.pi, n_per_class)
    X1 = np.column_stack([np.cos(theta1), np.sin(theta1)])
    theta2 = np.linspace(0, np.pi, n_per_class)
    X2 = np.column_stack([1 - np.cos(theta2), 1 - np.sin(theta2) - 0.5])
    X = np.vstack([X1, X2]) + np.random.randn(n_samples, 2) * noise
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y

# Low-rank weight matrix (simulating trained weights)
def create_low_rank_matrix(d=768, true_rank=64, noise_level=0.01):
    A = np.random.randn(d, true_rank) / np.sqrt(true_rank)
    B = np.random.randn(true_rank, d) / np.sqrt(true_rank)
    noise = np.random.randn(d, d) * noise_level
    return A @ B + noise
```

## Memory Considerations

On DGX Spark with 128GB unified memory, you can easily work with:
- Weight matrices up to 10,000 × 10,000 (~800MB)
- Datasets with millions of samples
- Full SVD decompositions of large matrices

The exercises in this module use small data for clarity, but the concepts scale.
