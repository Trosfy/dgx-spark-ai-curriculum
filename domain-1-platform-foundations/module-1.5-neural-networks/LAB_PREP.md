# Lab Preparation - Module 1.5: Neural Network Fundamentals

**Module:** 1.5 - Neural Network Fundamentals
**Domain:** 1 - Platform Foundations

---

## Quick Setup Checklist

- [ ] DGX Spark powered on and accessible
- [ ] NGC container ready (for Lab 6)
- [ ] Jupyter Lab accessible
- [ ] Internet connection for MNIST download

---

## Environment Setup

### For Labs 1.5.1 - 1.5.5 (NumPy Only)

These labs use only NumPy and Matplotlib. You can run them in any Python environment:

```bash
# Verify Python and NumPy
python -c "import numpy as np; print(f'NumPy: {np.__version__}')"
python -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"
```

**Expected output:**
```
NumPy: 1.24.0 or higher
Matplotlib: 3.7.0 or higher
```

### For Lab 1.5.6 (GPU Acceleration)

This lab requires PyTorch with CUDA support. Use the NGC container:

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

**Verify GPU access inside container:**
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

**Expected output:**
```
PyTorch: 2.x.x
CUDA available: True
Device: NVIDIA Graphics Device
```

---

## Data Preparation

### MNIST Dataset

The labs will automatically download MNIST on first run. To pre-download:

```python
import urllib.request
import os

os.makedirs('../data', exist_ok=True)
base_url = 'http://yann.lecun.com/exdb/mnist/'
files = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz'
]
for f in files:
    if not os.path.exists(f'../data/{f}'):
        print(f'Downloading {f}...')
        urllib.request.urlretrieve(f'{base_url}{f}', f'../data/{f}')
print('MNIST ready!')
```

---

## Directory Structure

Ensure your workspace has this structure:

```
module-1.5-neural-networks/
├── README.md
├── QUICKSTART.md
├── STUDY_GUIDE.md
├── ELI5.md
├── QUICK_REFERENCE.md
├── PREREQUISITES.md
├── LAB_PREP.md          ← You are here
├── TROUBLESHOOTING.md
├── labs/
│   ├── lab-1.5.1-numpy-neural-network.ipynb
│   ├── lab-1.5.2-activation-function-study.ipynb
│   ├── lab-1.5.3-regularization-experiments.ipynb
│   ├── lab-1.5.4-normalization-comparison.ipynb
│   ├── lab-1.5.5-training-diagnostics-lab.ipynb
│   └── lab-1.5.6-gpu-acceleration.ipynb
├── scripts/
│   ├── nn_layers.py
│   ├── optimizers.py
│   └── normalization.py
└── data/                 ← Created automatically
    └── [MNIST files]
```

---

## Memory Requirements

| Lab | Minimum RAM | Recommended |
|-----|-------------|-------------|
| 1.5.1 | 4 GB | 8 GB |
| 1.5.2 | 2 GB | 4 GB |
| 1.5.3 | 4 GB | 8 GB |
| 1.5.4 | 4 GB | 8 GB |
| 1.5.5 | 4 GB | 8 GB |
| 1.5.6 | 8 GB + GPU | 16 GB + GPU |

DGX Spark's 128GB unified memory exceeds all requirements.

---

## Pre-Lab Verification Commands

Run these before starting each lab:

### Lab 1.5.1 - NumPy Neural Network
```python
import numpy as np
import matplotlib.pyplot as plt
print(f"NumPy ready: {np.__version__}")
```

### Lab 1.5.6 - GPU Acceleration
```python
import torch
assert torch.cuda.is_available(), "CUDA not available!"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

## Cleanup Between Labs

To free memory between labs:

```python
import gc

# Clear variables
# del model, X_train, y_train  # Uncomment as needed

# Force garbage collection
gc.collect()

# For GPU labs
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

---

## Common Setup Issues

See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for detailed solutions to:

- `torch.cuda.is_available()` returns `False`
- MNIST download fails
- Jupyter kernel crashes
- Import errors

---

*Lab preparation verified for DGX Spark AI Curriculum v2.0*
