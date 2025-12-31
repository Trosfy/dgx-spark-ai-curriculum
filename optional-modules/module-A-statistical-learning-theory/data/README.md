# Data Directory - Module A: Statistical Learning Theory

## Overview

This module primarily uses **synthetically generated data** to demonstrate learning theory concepts. This approach is intentional:

1. **Controlled experiments**: We know the true underlying function, so we can measure bias exactly
2. **Reproducibility**: Same random seeds produce identical datasets
3. **Pedagogical clarity**: Focus on concepts without data preprocessing complexity

## Data Generation

All data is generated programmatically in the notebooks using:

### Regression Data (`generate_regression_data`)
```python
def true_function(x):
    return np.sin(2 * x) + 0.5 * np.cos(4 * x)

X = np.random.uniform(0, 4, n_samples)
y = true_function(X) + np.random.normal(0, noise_std, n_samples)
```

**Parameters**:
- `n_samples`: Number of data points (typically 100-500)
- `noise_std`: Standard deviation of Gaussian noise (typically 0.1-0.6)
- `x_min`, `x_max`: Range of x values (default 0 to 4)

### Classification Data (`generate_classification_data`)
```python
X = np.random.randn(n_samples, n_features)
true_weights = np.random.randn(n_features)
logits = X @ true_weights
probs = sigmoid(3 * logits)
y = (random() < probs).astype(int)
```

**Parameters**:
- `n_samples`: Number of data points
- `n_features`: Dimensionality of feature space
- `noise`: Label flip probability (0-1)

## External Datasets (Optional)

For Exercise 2 in Lab A.3, the notebooks optionally use:

### MNIST
- **Source**: `sklearn.datasets.fetch_openml('mnist_784')`
- **Size**: 70,000 images (28x28 = 784 features)
- **Classes**: 10 (digits 0-9)
- **Cache location**: `~/.cache/sklearn`

If MNIST is unavailable, the notebook falls back to synthetic classification data.

## Usage in Notebooks

Each notebook imports data generation functions from either:
- Inline definitions in the notebook
- `scripts/learning_theory_utils.py`

Example:
```python
from scripts.learning_theory_utils import generate_regression_data

# Generate 100 noisy samples
X, y = generate_regression_data(
    n_samples=100,
    true_function=lambda x: np.sin(2*x),
    noise_std=0.3,
    seed=42
)
```

## Random Seeds

All experiments use controlled random seeds for reproducibility:
- Main seed: 42
- Bootstrap iterations: Use iteration index as seed
- Multiple trials: Use trial number as seed

## Memory Considerations (DGX Spark)

With 128GB unified memory, you can easily:
- Generate millions of synthetic samples
- Load full MNIST (140MB)
- Run 1000+ bootstrap iterations

No special memory management needed for this module.

## Adding Custom Datasets

If you want to add custom datasets for experimentation:

1. Place CSV/NPY files in this directory
2. Load using standard methods:
   ```python
   import numpy as np
   import pandas as pd

   # NumPy
   data = np.load('data/custom_data.npy')

   # Pandas
   df = pd.read_csv('data/custom_data.csv')
   X = df.drop('target', axis=1).values
   y = df['target'].values
   ```

3. Ensure data is normalized before using with learning theory experiments
