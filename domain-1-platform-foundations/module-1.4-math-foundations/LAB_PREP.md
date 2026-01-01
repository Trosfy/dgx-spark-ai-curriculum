# Module 1.4: Mathematics for Deep Learning - Lab Preparation

## Environment Setup

### Required Environment

This module runs in the NGC PyTorch container:

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

> **Note:** This module focuses on mathematical foundations using NumPy and basic PyTorch.
> GPU acceleration is minimal—we're building intuition, not training large models.

---

## Package Verification

All required packages are pre-installed in the NGC container. Verify with:

```python
import numpy as np
import torch
import matplotlib.pyplot as plt

print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

**Expected output:**
```
NumPy version: 1.x.x
PyTorch version: 2.x.x
CUDA available: True
```

### Optional Package (Lab 1.4.3)

Lab 1.4.3 (Loss Landscape Visualization) uses scikit-learn for PCA:

```python
try:
    from sklearn.decomposition import PCA
    print("scikit-learn available")
except ImportError:
    print("Optional: pip install scikit-learn")
```

---

## Directory Structure

Navigate to the module directory:

```bash
cd /workspace/domain-1-platform-foundations/module-1.4-math-foundations
```

Verify structure:
```
module-1.4-math-foundations/
├── labs/
│   ├── lab-1.4.1-manual-backpropagation.ipynb
│   ├── lab-1.4.2-optimizer-implementation.ipynb
│   ├── lab-1.4.3-loss-landscape-visualization.ipynb
│   ├── lab-1.4.4-svd-for-lora.ipynb
│   └── lab-1.4.5-probability-distributions.ipynb
├── solutions/
│   └── [solution notebooks]
├── scripts/
│   ├── math_utils.py
│   └── visualization_utils.py
└── data/
    └── README.md
```

---

## Pre-Lab Checklist

### Before Lab 1.4.1 (Manual Backpropagation)

- [ ] Understand basic derivatives (power rule, chain rule)
- [ ] Can explain what a gradient represents
- [ ] Familiar with NumPy matrix operations

**Quick test:**
```python
# Can you predict the output?
import numpy as np
x = np.array([1.0, 2.0, 3.0])
print(2 * x)  # Should be [2.0, 4.0, 6.0]
```

### Before Lab 1.4.2 (Optimizer Implementation)

- [ ] Completed Lab 1.4.1
- [ ] Understand gradient descent concept

### Before Lab 1.4.3 (Loss Landscape Visualization)

- [ ] Completed Labs 1.4.1-1.4.2
- [ ] scikit-learn available (optional but recommended)

### Before Lab 1.4.4 (SVD for LoRA)

- [ ] Completed Labs 1.4.1-1.4.3
- [ ] Basic understanding of matrix decomposition

### Before Lab 1.4.5 (Probability Distributions)

- [ ] Completed Labs 1.4.1-1.4.4
- [ ] Basic probability knowledge (distributions, likelihood)

---

## Scripts Usage

The module includes production-ready implementations:

### math_utils.py

```python
# Import from the scripts directory
import sys
sys.path.insert(0, '/workspace/domain-1-platform-foundations/module-1.4-math-foundations/scripts')

from math_utils import (
    sigmoid, relu, softmax,           # Activation functions
    mse_loss, cross_entropy_loss,     # Loss functions
    SGD, Adam, AdamW,                 # Optimizers
    numerical_gradient                 # Gradient checking
)
```

### visualization_utils.py

```python
from visualization_utils import (
    plot_loss_landscape,
    plot_training_curve,
    plot_svd_analysis
)
```

---

## Memory Considerations

On DGX Spark with 128GB unified memory:

| Operation | Typical Memory | Notes |
|-----------|---------------|-------|
| Lab 1.4.1 (Backprop) | < 1 GB | Small networks |
| Lab 1.4.2 (Optimizers) | < 1 GB | 2D optimization |
| Lab 1.4.3 (Landscapes) | 1-2 GB | Grid computation |
| Lab 1.4.4 (SVD) | 2-5 GB | 768×768 matrices |
| Lab 1.4.5 (Probability) | < 1 GB | Distribution plots |

No memory concerns for this module on DGX Spark.

---

## Cleanup Between Labs

Each notebook includes cleanup cells, but you can also run:

```python
import gc
import torch

if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
```

---

## Troubleshooting Setup

### Import Error for scripts

If you see `ModuleNotFoundError: No module named 'scripts'`:

```python
import sys
# Use absolute path
sys.path.insert(0, '/workspace/domain-1-platform-foundations/module-1.4-math-foundations/scripts')
```

### Matplotlib Display Issues

If plots don't display in JupyterLab:

```python
%matplotlib inline
import matplotlib.pyplot as plt
```

### scikit-learn Not Found (Lab 1.4.3)

```bash
pip install scikit-learn
```

---

## Ready to Start?

1. Open JupyterLab in your browser
2. Navigate to `labs/lab-1.4.1-manual-backpropagation.ipynb`
3. Begin with Part 1: The Math Behind Backpropagation
