# Data Files for Module 4: Neural Network Fundamentals

This directory stores dataset files used in the module notebooks.

## MNIST Dataset

The primary dataset used in this module is **MNIST** (Modified National Institute of Standards and Technology), a classic benchmark for image classification.

### Dataset Details

| Property | Value |
|----------|-------|
| Training samples | 60,000 |
| Test samples | 10,000 |
| Image size | 28x28 pixels |
| Classes | 10 (digits 0-9) |
| File format | IDX (compressed with gzip) |

### Files

When you run the notebooks, the following files will be automatically downloaded:

```
data/
├── train-images-idx3-ubyte.gz    # Training images (~9.9 MB)
├── train-labels-idx1-ubyte.gz    # Training labels (~29 KB)
├── t10k-images-idx3-ubyte.gz     # Test images (~1.6 MB)
├── t10k-labels-idx1-ubyte.gz     # Test labels (~5 KB)
└── README.md                      # This file
```

### Source

MNIST is automatically downloaded from:
- http://yann.lecun.com/exdb/mnist/

### Manual Download

If automatic download fails, you can manually download the files:

```bash
cd data
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
```

### Data Format

The data is loaded and preprocessed as follows:

```python
# Images: Normalized to [0, 1], flattened to 784 dimensions
X_train.shape = (60000, 784)  # float32
X_test.shape = (10000, 784)   # float32

# Labels: Integer class indices
y_train.shape = (60000,)      # uint8, values 0-9
y_test.shape = (10000,)       # uint8, values 0-9
```

### Sample Images

Each image is a 28x28 grayscale handwritten digit:

```
     ████████████
   ████████████████
  ████████    ████████
  ██████        ██████
  ██████        ██████
                ██████
              ████████
            ████████
          ████████
        ████████
      ████████
    ████████
  ████████████████████
  ████████████████████

        (Example: "7")
```

## Memory Considerations (DGX Spark)

The MNIST dataset is small enough to fit entirely in memory:
- Full dataset: ~50 MB uncompressed
- DGX Spark's 128GB: Can hold ~2,500 copies of MNIST!

For this module, memory is not a concern. Later modules will work with larger datasets where DGX Spark's unified memory becomes essential.

## Citation

If using MNIST in research:

```bibtex
@article{lecun1998mnist,
  title={The MNIST database of handwritten digits},
  author={LeCun, Yann and Cortes, Corinna and Burges, Christopher JC},
  year={1998},
  url={http://yann.lecun.com/exdb/mnist/}
}
```

---

## Other Datasets (Optional)

### Fashion-MNIST
A drop-in replacement for MNIST with clothing items (same format):
- Source: https://github.com/zalandoresearch/fashion-mnist

### CIFAR-10
Color images (32x32x3), 10 classes, more challenging:
- Source: https://www.cs.toronto.edu/~kriz/cifar.html
- Note: Requires CNN architecture (covered in Module 7)
