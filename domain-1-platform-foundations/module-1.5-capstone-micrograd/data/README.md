# Data Directory

This directory contains data files for the MicroGrad+ capstone project.

## MNIST Dataset

The MNIST dataset will be automatically downloaded when you run the MNIST example notebook (notebook 05). The downloaded files are:

| File | Description | Size |
|------|-------------|------|
| `train-images-idx3-ubyte.gz` | Training images (60,000) | ~9.5 MB |
| `train-labels-idx1-ubyte.gz` | Training labels (60,000) | ~29 KB |
| `t10k-images-idx3-ubyte.gz` | Test images (10,000) | ~1.6 MB |
| `t10k-labels-idx1-ubyte.gz` | Test labels (10,000) | ~5 KB |

### Manual Download

If automatic download fails, you can manually download from these sources:

1. **PyTorch mirror (recommended)**: https://ossci-datasets.s3.amazonaws.com/mnist/
2. **Original source**: http://yann.lecun.com/exdb/mnist/

Download all four files and place them in this `data/` directory:
```bash
# Example using wget (from PyTorch mirror)
wget https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
```

### Alternative: Using scikit-learn or Keras

If you have scikit-learn or TensorFlow/Keras installed, you can load MNIST directly:

```python
# Using scikit-learn
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'].values, mnist['target'].values.astype(int)

# Using Keras
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

### Data Format

- **Images**: 28x28 grayscale images, pixel values 0-255
- **Labels**: Integer class labels 0-9

After loading and preprocessing:
- Images are flattened to 784 features and normalized to [0, 1]
- Labels remain as integer class indices

## Generating Synthetic Data

For testing purposes, you can also generate synthetic datasets:

```python
import numpy as np

def make_spiral(n_points=100, n_classes=3):
    """Generate spiral dataset for classification."""
    X = np.zeros((n_points * n_classes, 2), dtype=np.float32)
    y = np.zeros(n_points * n_classes, dtype=np.int32)

    for c in range(n_classes):
        ix = range(n_points * c, n_points * (c + 1))
        r = np.linspace(0.0, 1, n_points)
        t = np.linspace(c * 4, (c + 1) * 4, n_points) + np.random.randn(n_points) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = c

    return X, y

def make_regression(n_samples=100, n_features=10, noise=0.1):
    """Generate linear regression dataset."""
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    true_weights = np.random.randn(n_features, 1).astype(np.float32)
    y = X @ true_weights + np.random.randn(n_samples, 1).astype(np.float32) * noise
    return X, y
```

## Storage Notes

- MNIST data is ~12 MB total
- Data files are gitignored to avoid bloating the repository
- Each run will cache downloaded data for future use
