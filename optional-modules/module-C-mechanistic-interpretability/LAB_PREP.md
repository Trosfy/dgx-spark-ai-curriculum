# Module C: Mechanistic Interpretability - Lab Preparation Guide

## ⏱️ Time Estimates

| Lab | Preparation | Execution | Total |
|-----|-------------|-----------|-------|
| Lab 1: TransformerLens Setup | 15 min | 1.5 hr | 1.75 hr |
| Lab 2: Activation Patching (IOI) | 10 min | 2 hr | 2.25 hr |
| Lab 3: Induction Head Analysis | 10 min | 2 hr | 2.25 hr |
| Lab 4: SAE Feature Extraction | 20 min | 2.5 hr | 2.75 hr |

**Total preparation time**: ~55 minutes (mostly Lab 1 setup)

---

## 📦 Required Downloads

### Python Packages

```bash
# Core package
pip install transformer-lens

# Visualization
pip install plotly
pip install circuitsvis  # Optional: Neel Nanda's visualization tools

# For SAE lab
pip install sae-lens  # If using pre-trained SAEs

# Verify installation
python -c "from transformer_lens import HookedTransformer; print('TransformerLens OK')"
```

### Models (Auto-download on First Use)

Models download automatically but you can pre-download:

```python
from transformer_lens import HookedTransformer

# Pre-download GPT-2 Small (124M) - Required for all labs
model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
del model  # Free memory

# Optional: Pre-download larger models for advanced exploration
# model = HookedTransformer.from_pretrained("gpt2-medium", device="cpu")
```

**Model sizes:**
| Model | Parameters | Download | GPU Memory |
|-------|------------|----------|------------|
| gpt2-small | 124M | ~500 MB | ~2 GB |
| gpt2-medium | 355M | ~1.4 GB | ~4 GB |
| gpt2-large | 774M | ~3 GB | ~6 GB |
| gpt2-xl | 1.5B | ~6 GB | ~10 GB |

### Pre-trained SAEs (Lab 4)

```python
# Option 1: Use Neuronpedia's pre-trained SAEs
# Download manually from: https://www.neuronpedia.org/

# Option 2: Use SAE Lens
from sae_lens import SparseAutoencoder
# Models download automatically
```

---

## 🔧 Environment Setup

### 1. Start Container

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    nvcr.io/nvidia/pytorch:25.11-py3
```

### 2. Install Packages

```bash
pip install transformer-lens plotly nbformat
pip install circuitsvis  # Optional but helpful
```

### 3. Verify GPU Access

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
```

**Expected output:**
```
CUDA available: True
Device: NVIDIA GH200 480GB  # or similar
Memory: 128.0 GB
```

### 4. Test TransformerLens

```python
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
tokens = model.to_tokens("Hello, world!")
logits, cache = model.run_with_cache(tokens)

print(f"Model loaded: {model.cfg.model_name}")
print(f"Cache size: {len(cache)} tensors")
print(f"Prediction: {model.tokenizer.decode(logits[0, -1].argmax())}")
```

**Expected output:**
```
Model loaded: gpt2-small
Cache size: 156 tensors
Prediction:  I
```

---

## ✅ Pre-Lab Checklists

### Lab 1: TransformerLens Setup & Exploration

**Prerequisites:**
- [ ] TransformerLens installed
- [ ] Plotly installed for visualizations
- [ ] Understanding of transformer architecture (Module 2.3)
- [ ] At least 4 GB GPU memory free

**Concepts to review:**
- Attention mechanism (Q, K, V)
- Residual connections
- LayerNorm placement

**Quick test:**
```python
# This should run without errors
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
logits = model("The quick brown")
print(f"Next token: {model.tokenizer.decode(logits[0, -1].argmax())}")  # Should be " fox" or similar
```

---

### Lab 2: Activation Patching (IOI)

**Prerequisites:**
- [ ] Completed Lab 1
- [ ] Understand attention patterns
- [ ] Clear GPU memory (clean restart)

**Concepts to review:**
- Activation patching concept (see ELI5.md)
- What the IOI task is: "John and Mary... John gave to [Mary]"

**Dataset prep:**
```python
# IOI dataset is generated in the notebook, but understand the format:
# Clean: "John and Mary went to the store. John gave a bottle to" → Mary
# Corrupted: "John and Mary went to the store. Mary gave a bottle to" → John

# The task: Which components are responsible for picking the correct indirect object?
```

---

### Lab 3: Induction Head Analysis

**Prerequisites:**
- [ ] Completed Labs 1-2
- [ ] Understand induction heads conceptually (see ELI5.md)

**Concepts to review:**
- Pattern: [A][B]...[A] → [B]
- How attention patterns reveal induction behavior

**Test your intuition:**
```python
# What should happen here?
prompt = "abc def ghi abc"
# If model has induction heads, it should predict: " def"
# Because it saw "abc" → " def" earlier!
```

---

### Lab 4: SAE Feature Extraction

**Prerequisites:**
- [ ] Completed Labs 1-3
- [ ] Pre-trained SAE downloaded (or plan to train one)
- [ ] At least 8 GB GPU memory free

**Additional downloads:**
```bash
# If using SAE Lens
pip install sae-lens

# If using Neuronpedia SAEs, download from their website
```

**Concepts to review:**
- What superposition is (ELI5.md)
- Why we need SAEs to extract features
- Sparse coding / dictionary learning basics

---

## 🚫 Common Setup Mistakes

| Mistake | Consequence | Prevention |
|---------|-------------|------------|
| Using CPU instead of GPU | 10x slower, may OOM | Always pass `device="cuda"` |
| Not clearing memory between experiments | OOM errors | Run cleanup before each lab |
| Wrong TransformerLens version | Import errors | `pip install transformer-lens --upgrade` |
| Running full cache on large models | OOM | Use selective caching or gpt2-small |
| Forgetting to set random seeds | Inconsistent results | `torch.manual_seed(42)` |

---

## 📁 Expected File Structure

```
/workspace/
├── module-C-mech-interp/
│   ├── notebooks/
│   │   ├── 01-transformerlens-setup.ipynb
│   │   ├── 02-activation-patching-ioi.ipynb
│   │   ├── 03-induction-head-analysis.ipynb
│   │   └── 04-feature-extraction-saes.ipynb
│   ├── outputs/
│   │   ├── attention_patterns/
│   │   ├── patching_results/
│   │   └── sae_features/
│   └── checkpoints/
│       └── [saved SAE weights if training]
```

---

## ⚡ Quick Setup Script

Copy-paste this to set up everything:

```bash
# Navigate to workspace
cd /workspace
mkdir -p module-C-mech-interp/{notebooks,outputs/{attention_patterns,patching_results,sae_features},checkpoints}
cd module-C-mech-interp

# Install packages
pip install transformer-lens plotly circuitsvis

# Verify installation
python -c "
from transformer_lens import HookedTransformer
import torch

print('Testing TransformerLens...')
model = HookedTransformer.from_pretrained('gpt2-small', device='cuda')
logits, cache = model.run_with_cache('Test prompt')
print(f'✓ Model loaded, cache has {len(cache)} tensors')

# Cleanup
del model, logits, cache
torch.cuda.empty_cache()
print('✓ Setup complete!')
"
```

---

## 🔬 Memory Management Tips

TransformerLens caches are LARGE. Use these patterns:

```python
# Pattern 1: Clear between experiments
import gc
torch.cuda.empty_cache()
gc.collect()

# Pattern 2: Selective caching
logits, cache = model.run_with_cache(
    tokens,
    names_filter=lambda name: "pattern" in name  # Only attention patterns
)

# Pattern 3: Immediate CPU transfer in hooks
def memory_efficient_hook(activation, hook):
    results[hook.name] = activation.cpu().clone()
    return activation

# Pattern 4: Delete cache after use
analysis_result = analyze(cache)
del cache
torch.cuda.empty_cache()

# Pattern 5: Check memory usage
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

---

## 📊 Expected Memory Usage

| Operation | Memory Required |
|-----------|-----------------|
| Load GPT-2 Small | ~1.5 GB |
| Full cache (1 prompt) | ~0.5 GB |
| Full cache (batch of 10) | ~5 GB |
| Patching experiment (2 caches) | ~1.5 GB |
| SAE training | ~4-8 GB |

**DGX Spark has 128 GB** - you have plenty of headroom, but still clean up!

---

## 🎯 Ready to Start?

- [ ] All packages installed
- [ ] GPU accessible and tested
- [ ] TransformerLens working
- [ ] Understand the conceptual foundations (ELI5.md)
- [ ] Memory management patterns memorized

**Start with Lab 1: TransformerLens Setup!**
