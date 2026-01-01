# Troubleshooting - Module 1.5: Neural Network Fundamentals

**Module:** 1.5 - Neural Network Fundamentals
**Domain:** 1 - Platform Foundations

---

## Quick Fixes

| Symptom | Quick Fix |
|---------|-----------|
| `torch.cuda.is_available()` returns `False` | Use NGC container with `--gpus all` |
| Kernel crashes on large batch | Reduce batch size or clear memory |
| MNIST download fails | Check internet connection, try manual download |
| `ModuleNotFoundError` for scripts | Verify `sys.path.insert(0, str(scripts_dir))` |
| Loss is `NaN` | Lower learning rate, check for division by zero |

---

## Detailed Solutions

### Issue 1: CUDA Not Available

**Symptom:**
```python
>>> torch.cuda.is_available()
False
```

**Cause:** Running outside NGC container or missing `--gpus all` flag.

**Solution:**

1. **Use the NGC container:**
```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

2. **Verify GPU access:**
```bash
nvidia-smi  # Should show GPU info
```

3. **Never pip install PyTorch on DGX Spark** - use the NGC container which has ARM64-optimized PyTorch pre-installed.

---

### Issue 2: Kernel Crashes During Training

**Symptom:**
- Jupyter kernel dies unexpectedly
- "Kernel Restarting" message
- System becomes unresponsive

**Cause:** Out of memory (OOM) error.

**Solution:**

1. **Reduce batch size:**
```python
BATCH_SIZE = 32  # Instead of 64 or 128
```

2. **Clear memory between experiments:**
```python
import gc

# Delete large objects
del model, X_train, y_train

# Force garbage collection
gc.collect()

# For GPU
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

3. **Clear buffer cache (system-wide):**
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

---

### Issue 3: MNIST Download Fails

**Symptom:**
```
URLError: <urlopen error [Errno -3] Temporary failure in name resolution>
```

**Cause:** Network issues or firewall blocking.

**Solution:**

1. **Check internet connection:**
```bash
ping google.com
```

2. **Manual download:**
```bash
cd module-1.5-neural-networks/data
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
```

3. **Alternative source (if primary fails):**
```python
# Use torchvision (in NGC container)
from torchvision import datasets
datasets.MNIST('../data', train=True, download=True)
```

---

### Issue 4: Import Errors for Scripts

**Symptom:**
```
ModuleNotFoundError: No module named 'nn_layers'
```

**Cause:** Scripts directory not in Python path.

**Solution:**

The notebooks include this code, but verify it's running:
```python
import sys
from pathlib import Path

notebook_dir = Path().resolve()
if notebook_dir.name == 'labs':
    scripts_dir = notebook_dir.parent / 'scripts'
else:
    scripts_dir = notebook_dir / 'scripts'
    if not scripts_dir.exists():
        scripts_dir = notebook_dir.parent / 'scripts'

if scripts_dir.exists():
    sys.path.insert(0, str(scripts_dir))
    print(f"Scripts directory added: {scripts_dir}")
else:
    print(f"WARNING: Scripts directory not found!")
```

---

### Issue 5: Loss is NaN or Infinity

**Symptom:**
```
Epoch 1: Loss = nan
```

**Cause:** Numerical instability, often from:
- Learning rate too high
- Division by zero
- Exploding gradients

**Solution:**

1. **Lower learning rate:**
```python
lr = 0.01  # Instead of 0.1 or 1.0
```

2. **Add epsilon to divisions:**
```python
# Bad
normalized = x / std

# Good
epsilon = 1e-8
normalized = x / (std + epsilon)
```

3. **Clip gradients:**
```python
max_grad_norm = 1.0
for layer in model.layers:
    if 'weights' in layer.gradients:
        grad = layer.gradients['weights']
        grad_norm = np.linalg.norm(grad)
        if grad_norm > max_grad_norm:
            layer.gradients['weights'] = grad * max_grad_norm / grad_norm
```

4. **Check for NaN in data:**
```python
assert not np.isnan(X_train).any(), "NaN in training data!"
assert not np.isnan(y_train).any(), "NaN in labels!"
```

---

### Issue 6: Training is Extremely Slow

**Symptom:**
- Each epoch takes minutes (should be seconds)
- CPU at 100%, GPU idle

**Cause:** Not using GPU or inefficient operations.

**Solution:**

1. **Verify GPU usage:**
```python
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")
model = model.to(device)
```

2. **Move data to GPU:**
```python
X_batch = X_batch.to(device)
y_batch = y_batch.to(device)
```

3. **Use DataLoader with workers (requires --ipc=host):**
```python
train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2,  # Parallel data loading
    pin_memory=True  # Faster GPU transfer
)
```

---

### Issue 7: Vanishing Gradients

**Symptom:**
- Loss barely decreases
- Early layer gradients are nearly zero
- Model doesn't learn

**Cause:** Using sigmoid/tanh in deep networks.

**Solution:**

1. **Use ReLU activation:**
```python
class ReLU:
    def forward(self, x):
        return np.maximum(0, x)
```

2. **Use He initialization (for ReLU):**
```python
W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
```

3. **Add batch/layer normalization:**
```python
from scripts.normalization import LayerNorm
ln = LayerNorm(num_features)
x = ln(x)
```

---

### Issue 8: PyTorch DataLoader Errors

**Symptom:**
```
RuntimeError: DataLoader worker (pid XXXXX) is killed by signal: Bus error
```

**Cause:** Missing `--ipc=host` Docker flag.

**Solution:**

Include `--ipc=host` in your Docker command:
```bash
docker run --gpus all -it --rm \
    --ipc=host \  # <- This flag is required!
    -v $HOME/workspace:/workspace \
    nvcr.io/nvidia/pytorch:25.11-py3
```

Or set `num_workers=0` (slower but works without the flag):
```python
train_loader = DataLoader(dataset, batch_size=64, num_workers=0)
```

---

### Issue 9: Plots Not Displaying

**Symptom:**
- `plt.show()` does nothing
- Plots don't appear inline

**Solution:**

1. **Enable inline plotting:**
```python
%matplotlib inline
```

2. **Use explicit display:**
```python
from IPython.display import display
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3])
display(fig)
plt.close()
```

---

### Issue 10: Random Results Not Reproducible

**Symptom:**
- Different results each run
- Can't reproduce specific outcomes

**Solution:**

Set all random seeds:
```python
import numpy as np
import random

# Set seeds
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# For PyTorch
import torch
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## Still Having Issues?

1. **Check the FAQ:** [FAQ.md](./FAQ.md)
2. **Review prerequisites:** [PREREQUISITES.md](./PREREQUISITES.md)
3. **Restart with clean environment:** Restart kernel and clear outputs
4. **Report issue:** https://github.com/anthropics/claude-code/issues

---

*Troubleshooting guide for DGX Spark AI Curriculum v2.0*
