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

## ❓ Frequently Asked Questions

### General Questions

**Q: Why do we implement neural networks from scratch in NumPy first?**

Building neural networks from scratch gives you deep understanding of:

1. **Forward propagation**: How data flows through layers
2. **Backpropagation**: How gradients flow backward (it's just the chain rule!)
3. **Weight updates**: How optimizers modify parameters
4. **Numerical stability**: Why we need epsilon values and careful initialization

This knowledge is invaluable when debugging PyTorch/TensorFlow models. When something goes wrong, you'll understand what's happening under the hood.

---

**Q: Why can't I pip install PyTorch on DGX Spark?**

DGX Spark uses ARM64 (aarch64) architecture, not x86. The standard PyTorch pip wheels are compiled for x86 and won't work.

**Solution:** Use the NGC container which has PyTorch pre-compiled for ARM64:

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    nvcr.io/nvidia/pytorch:25.11-py3
```

---

**Q: What's the difference between He and Xavier initialization?**

Both maintain stable gradients, but for different activations:

| Initialization | Formula | Best For |
|----------------|---------|----------|
| **He** | `W = randn() * sqrt(2/fan_in)` | ReLU, Leaky ReLU |
| **Xavier/Glorot** | `W = randn() * sqrt(2/(fan_in + fan_out))` | Tanh, Sigmoid |

**Why it matters:** Wrong initialization can cause vanishing or exploding gradients. He initialization accounts for ReLU zeroing out half the activations.

---

**Q: When should I use BatchNorm vs LayerNorm vs RMSNorm?**

| Method | Normalizes Over | Best For | Used In |
|--------|-----------------|----------|---------|
| **BatchNorm** | Batch dimension | CNNs, vision | ResNet, EfficientNet |
| **LayerNorm** | Feature dimension | Transformers, RNNs | BERT, GPT |
| **RMSNorm** | Feature (no mean) | Modern LLMs | LLaMA, Mistral |

**Key differences:**
- BatchNorm needs batch size > 1 (different behavior train vs inference)
- LayerNorm works with any batch size (same behavior always)
- RMSNorm is faster than LayerNorm (no mean computation)

---

**Q: What learning rate should I start with?**

Common starting points:

| Optimizer | Starting LR |
|-----------|-------------|
| SGD | 0.1 |
| SGD + Momentum | 0.01 - 0.1 |
| Adam | 0.001 |
| AdamW | 0.001 |

**Tuning strategy:**
1. Start with default
2. If loss explodes → decrease by 10x
3. If loss barely moves → increase by 10x
4. Use learning rate finder for fine-tuning

---

**Q: What's the "overfit one batch" trick?**

Before training on full data, verify your model can memorize a single batch:

```python
# Take one small batch
X_batch = X_train[:32]
y_batch = y_train[:32]

# Train for many iterations on just this batch
for i in range(200):
    loss = train_step(model, X_batch, y_batch)

# Should achieve ~100% accuracy on this batch
```

**If it fails:** There's a bug in your forward or backward pass. This is the most important debugging technique!

---

### DGX Spark Specific Questions

**Q: Why does DGX Spark have unified memory?**

Traditional systems have separate CPU RAM and GPU VRAM, requiring slow data transfers. DGX Spark's 128GB unified memory means:

1. **No transfers needed**: CPU and GPU share the same memory
2. **Larger models**: 70B+ parameter models fit entirely in memory
3. **Faster development**: No need to optimize memory transfers
4. **Simpler code**: Less memory management complexity

This is possible because of the NVIDIA Grace-Blackwell architecture.

---

**Q: What's special about the GB10 Blackwell chip?**

Key features:

| Feature | Specification |
|---------|---------------|
| CUDA Cores | 6,144 |
| Tensor Cores | 192 (5th generation) |
| NVFP4 Performance | 1 PFLOP |
| FP8 Performance | ~209 TFLOPS |
| BF16 Performance | ~100 TFLOPS |

**Why it matters for AI:**
- Native FP8 and NVFP4 support for efficient inference
- Tensor Cores accelerate matrix operations
- 5th gen Tensor Cores are 2x faster than previous generation

---

**Q: What batch size should I use on DGX Spark?**

DGX Spark's 128GB unified memory allows larger batch sizes than typical GPUs:

| Model Size | Recommended Batch Size |
|------------|----------------------|
| Small MLP | 256-2048 |
| Medium CNN | 128-512 |
| Large Transformer | 32-128 |
| LLM (7B) | 8-32 |
| LLM (70B) | 1-4 |

**Finding optimal:**
```python
# Start large and reduce if OOM
for batch_size in [2048, 1024, 512, 256, 128]:
    try:
        train_epoch(model, batch_size)
        print(f"Batch size {batch_size} works!")
        break
    except RuntimeError:
        print(f"Batch size {batch_size} OOM")
```

---

### Training Questions

**Q: My loss is stuck and not decreasing. What should I check?**

Debugging checklist:

1. **Learning rate too low?** Try increasing by 10x
2. **Vanishing gradients?** Check gradient magnitudes, switch to ReLU
3. **Data issue?** Visualize samples, check for NaN values
4. **Labels shuffled?** Verify X and y correspond correctly
5. **Stuck in local minimum?** Add momentum or use Adam

---

**Q: What's the difference between train and validation loss behavior?**

| Pattern | Diagnosis | Fix |
|---------|-----------|-----|
| Both high, not moving | Underfitting | Bigger model, more training |
| Train ↓, Val ↓ | Good training | Keep going! |
| Train ↓, Val ↑ | Overfitting | Add regularization, more data |
| Both oscillating | LR too high | Reduce learning rate |

---

**Q: Should I use dropout or L2 regularization?**

Both prevent overfitting but work differently:

| Technique | How It Works | When to Use |
|-----------|--------------|-------------|
| **Dropout** | Randomly zeros neurons during training | Large models, dense layers |
| **L2 (Weight Decay)** | Penalizes large weights | Always reasonable default |
| **Both** | Complementary effects | Very large models |

**Typical values:**
- Dropout: 0.1-0.5 (start with 0.2)
- L2/Weight Decay: 0.01-0.0001 (start with 0.01)

---

### Architecture Questions

**Q: How many layers/neurons should my network have?**

Rules of thumb for MLPs:

| Problem Complexity | Layers | Neurons per Layer |
|-------------------|--------|------------------|
| Simple (XOR) | 1-2 | 4-16 |
| Medium (MNIST) | 2-3 | 128-512 |
| Complex (ImageNet) | Use CNNs | - |

**Start simple and add complexity only if needed.** An overfit simple model is easier to regularize than an underfit complex one.

---

**Q: Why do modern LLMs use RMSNorm instead of LayerNorm?**

RMSNorm is simpler and faster:

```python
# LayerNorm: center then scale
x_norm = (x - mean) / std * gamma + beta

# RMSNorm: just scale (no centering)
x_norm = x / rms * gamma
```

**Benefits:**
- ~10% faster (no mean computation)
- Empirically works just as well for transformers
- Used in LLaMA, Mistral, and most modern LLMs

---

## Still Having Issues?

1. **Review prerequisites:** [PREREQUISITES.md](./PREREQUISITES.md)
2. **Restart with clean environment:** Restart kernel and clear outputs
3. **Report issue:** https://github.com/anthropics/claude-code/issues

---

*Troubleshooting guide for DGX Spark AI Curriculum v2.0*
