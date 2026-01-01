# Module 1.2: Python for AI/ML - Lab Preparation Guide

## ⏱️ Time Estimates

| Lab | Preparation | Execution | Total |
|-----|-------------|-----------|-------|
| Lab 1.2.1: NumPy Broadcasting | 10 min | 110 min | 2 hr |
| Lab 1.2.2: Dataset Preprocessing | 15 min | 105 min | 2 hr |
| Lab 1.2.3: Visualization Dashboard | 10 min | 110 min | 2 hr |
| Lab 1.2.4: Einsum Mastery | 10 min | 110 min | 2 hr |
| Lab 1.2.5: Profiling Exercise | 15 min | 105 min | 2 hr |

**Total module time**: ~10 hours

## 📦 Required Packages

All packages are included in the NGC PyTorch container. No additional installs needed!

| Package | Purpose | NGC Container |
|---------|---------|---------------|
| NumPy | Array operations | ✅ Included |
| Pandas | Data manipulation | ✅ Included |
| Matplotlib | Visualization | ✅ Included |
| Seaborn | Statistical plots | ✅ Included |
| scikit-learn | Preprocessing utilities | ✅ Included |

### Optional (for enhanced profiling)
```bash
# Only if you want detailed line profiling
pip install line_profiler memory_profiler

# Or use conda (more reliable on ARM64)
conda install -c conda-forge line_profiler
```

## 🔧 Environment Setup

### 1. Start NGC Container
```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

### 2. Verify Imports
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print("✅ All imports successful")
```

### 3. Create Workspace
```bash
mkdir -p $HOME/workspace/module-1.2/{data,outputs,scripts}
```

## ✅ Pre-Lab Checklists

### Lab 1.2.1: NumPy Broadcasting
- [ ] NumPy imported successfully
- [ ] Understand basic array indexing
- [ ] Have scratch notebook ready for experiments
- [ ] Reviewed broadcasting rules in QUICK_REFERENCE.md

### Lab 1.2.2: Dataset Preprocessing
- [ ] Pandas imported successfully
- [ ] Download sample dataset:
  ```python
  # Titanic dataset (built into seaborn)
  df = sns.load_dataset('titanic')
  ```
- [ ] Familiar with basic DataFrame operations
- [ ] Understand train/test splitting concept

### Lab 1.2.3: Visualization Dashboard
- [ ] Matplotlib and Seaborn imported
- [ ] `%matplotlib inline` set in Jupyter
- [ ] Sample data prepared:
  ```python
  # Generate sample training curves
  train_loss = np.exp(-np.linspace(0, 2, 100)) + 0.1 * np.random.randn(100) * 0.1
  val_loss = np.exp(-np.linspace(0, 2, 100)) + 0.15 * np.random.randn(100) * 0.1
  ```

### Lab 1.2.4: Einsum Mastery
- [ ] Reviewed einsum notation in QUICK_REFERENCE.md
- [ ] Understand matrix multiplication conceptually
- [ ] Have attention mechanism diagram handy
- [ ] Ready to implement: `softmax(QK^T / sqrt(d)) @ V`

### Lab 1.2.5: Profiling Exercise
- [ ] cProfile works (stdlib, always available)
- [ ] Optional: line_profiler installed
- [ ] Slow sample function ready to optimize
- [ ] Understand what "bottleneck" means

## 🚫 Common Setup Mistakes

| Mistake | Consequence | Prevention |
|---------|-------------|------------|
| Running outside container | Missing packages or wrong versions | Always use NGC container |
| Forgot `%matplotlib inline` | Plots don't display in Jupyter | Add to first cell |
| Using float64 by default | 2x memory usage | Specify `dtype=np.float32` |
| Not clearing large arrays | Memory pressure | `del arr; gc.collect()` |

## 📁 Expected File Structure

After preparation:
```
$HOME/workspace/module-1.2/
├── data/
│   └── (datasets will go here)
├── outputs/
│   └── (figures and results)
├── scripts/
│   └── preprocessing_pipeline.py  # Created in Lab 2
└── notebooks/
    └── (lab notebooks)
```

## ⚡ Quick Start Commands

```bash
# Create workspace
mkdir -p $HOME/workspace/module-1.2/{data,outputs,scripts,notebooks}

# Start container with Jupyter
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser

# Click the URL shown to open Jupyter
```

## 📊 Memory Requirements

| Lab | Peak Memory | Notes |
|-----|-------------|-------|
| Lab 1.2.1 | ~500 MB | Large array operations |
| Lab 1.2.2 | ~200 MB | Dataset loading |
| Lab 1.2.3 | ~300 MB | Multiple figures |
| Lab 1.2.4 | ~400 MB | Batch attention matrices |
| Lab 1.2.5 | ~500 MB | Profiling overhead |

**Total recommended**: 2 GB RAM minimum

## 🔄 Between Labs

Clean up memory between labs:
```python
import gc

# Delete large variables
# del large_array

# Force garbage collection
gc.collect()

# In Jupyter: restart kernel if needed
# Kernel → Restart Kernel
```

## 📋 Sample Data for Labs

### Lab 1.2.2 Dataset Options
```python
# Option 1: Seaborn built-in (no download)
df = sns.load_dataset('titanic')

# Option 2: sklearn datasets
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
df = housing.frame
```

### Lab 1.2.5 Slow Function to Optimize
```python
def slow_pairwise_distances(X):
    """Compute pairwise distances - intentionally slow."""
    n = len(X)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = X[i] - X[j]
            distances[i, j] = np.sqrt(np.sum(diff ** 2))
    return distances

# This will be ~1000x slower than vectorized version
```
