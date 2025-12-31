# Module 2.3: NLP & Transformers - Lab Preparation Guide

## Time Estimates

| Lab | Preparation | Execution | Total |
|-----|-------------|-----------|-------|
| 2.3.1 Attention from Scratch | 5 min | 2 hr | ~2 hr |
| 2.3.2 Transformer Block | 5 min | 2 hr | ~2 hr |
| 2.3.3 Positional Encoding | 5 min | 2 hr | ~2 hr |
| 2.3.4 Tokenizer Training | 10 min | 3 hr | ~3 hr |
| 2.3.5 BERT Fine-tuning | 15 min | 2 hr | ~2.5 hr |
| 2.3.6 GPT Text Generation | 10 min | 2 hr | ~2 hr |

**Total**: ~12-15 hours

---

## Required Downloads

### Datasets (Auto-download)

```python
# IMDb for sentiment (Lab 2.3.5) - ~85 MB
from datasets import load_dataset
dataset = load_dataset("imdb")

# WikiText for tokenization (Lab 2.3.4) - ~180 MB
dataset = load_dataset("wikitext", "wikitext-103-v1")
```

### Pre-trained Models (Auto-download)

```python
# BERT (Lab 2.3.5) - ~440 MB
from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")

# GPT-2 (Lab 2.3.6) - ~550 MB
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")
```

### Additional Packages

```bash
# Already in NGC container, but verify/install
pip install transformers datasets tokenizers
```

---

## Environment Setup

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

### 2. Verify Setup

```python
import torch
from transformers import AutoModel, AutoTokenizer

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Test loading a model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
print(f"BERT loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
```

### 3. Test Attention Components

```python
import torch
import math

# Quick attention test
Q = K = V = torch.randn(1, 8, 64)  # (batch, seq_len, dim)
scores = Q @ K.transpose(-2, -1) / math.sqrt(64)
attn = torch.softmax(scores, dim=-1) @ V
print(f"Attention output shape: {attn.shape}")
```

---

## Pre-Lab Checklist

### Lab 2.3.1: Attention from Scratch
- [ ] Container running with GPU
- [ ] Understand matrix multiplication (A @ B)
- [ ] Know what softmax does

### Lab 2.3.2: Transformer Block
- [ ] Completed Lab 2.3.1
- [ ] Understand residual connections
- [ ] Know LayerNorm vs BatchNorm

### Lab 2.3.3: Positional Encoding
- [ ] Completed Lab 2.3.2
- [ ] Basic understanding of sin/cos functions
- [ ] Know why order matters in language

### Lab 2.3.4: Tokenizer Training
- [ ] WikiText dataset accessible
- [ ] Understand why subword tokenization exists
- [ ] Python string manipulation comfort

### Lab 2.3.5: BERT Fine-tuning
- [ ] transformers library installed
- [ ] datasets library installed
- [ ] IMDb dataset accessible
- [ ] Understand classification task

### Lab 2.3.6: GPT Text Generation
- [ ] GPT-2 model accessible
- [ ] Understand autoregressive generation
- [ ] Know what temperature means in softmax

---

## Common Setup Mistakes

| Mistake | Consequence | Prevention |
|---------|-------------|------------|
| Wrong transformers version | API differences | `pip install transformers>=4.40.0` |
| Not mounting HuggingFace cache | Re-downloads every time | `-v $HOME/.cache/huggingface:/root/.cache/huggingface` |
| Forgetting datasets library | Can't load IMDb | `pip install datasets` |
| Memory issues with large models | OOM errors | Use smaller model variants (bert-base, gpt2) |

---

## Expected File Structure

```
/workspace/
├── domain-2-deep-learning-frameworks/
│   └── module-2.3-nlp-transformers/
│       ├── README.md
│       ├── QUICKSTART.md
│       ├── PREREQUISITES.md
│       ├── STUDY_GUIDE.md
│       ├── QUICK_REFERENCE.md
│       ├── ELI5.md
│       ├── LAB_PREP.md
│       ├── TROUBLESHOOTING.md
│       ├── labs/
│       │   ├── lab-2.3.1-attention-from-scratch.ipynb
│       │   ├── lab-2.3.2-transformer-block.ipynb
│       │   ├── lab-2.3.3-positional-encoding-study.ipynb
│       │   ├── lab-2.3.4-tokenizer-training.ipynb
│       │   ├── lab-2.3.5-bert-fine-tuning.ipynb
│       │   └── lab-2.3.6-gpt-text-generation.ipynb
│       ├── scripts/
│       │   ├── attention.py
│       │   ├── transformer.py
│       │   ├── positional_encoding.py
│       │   ├── tokenizer_utils.py
│       │   └── generation.py
│       └── solutions/
```

---

## Resource Requirements by Lab

| Lab | GPU Memory | Disk Space | Notes |
|-----|------------|------------|-------|
| 2.3.1 | ~1 GB | Minimal | From scratch, small tensors |
| 2.3.2 | ~2 GB | Minimal | Building blocks |
| 2.3.3 | ~1 GB | Minimal | Position encodings |
| 2.3.4 | ~1 GB | ~180 MB | WikiText dataset |
| 2.3.5 | ~4 GB | ~500 MB | BERT base + IMDb |
| 2.3.6 | ~4 GB | ~600 MB | GPT-2 |

All labs fit easily within DGX Spark's 128GB unified memory.

---

## Quick Start Commands

```bash
# Inside NGC container
cd /workspace/domain-2-deep-learning-frameworks/module-2.3-nlp-transformers

# Install/verify packages
pip install transformers datasets tokenizers --quiet

# Pre-download models and datasets (optional but recommended)
python -c "
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

print('Downloading BERT...')
AutoModel.from_pretrained('bert-base-uncased')
AutoTokenizer.from_pretrained('bert-base-uncased')

print('Downloading GPT-2...')
AutoModel.from_pretrained('gpt2')
AutoTokenizer.from_pretrained('gpt2')

print('Downloading IMDb...')
load_dataset('imdb')

print('All downloads complete!')
"
```

---

## Using Scripts Directory

The `scripts/` folder contains reusable implementations:

```python
import sys
from pathlib import Path

# Add scripts to path (when running from labs/)
scripts_path = Path.cwd().parent / "scripts"
if str(scripts_path) not in sys.path:
    sys.path.insert(0, str(scripts_path.parent))

# Now import works
from scripts import (
    MultiHeadAttention,
    TransformerEncoder,
    SinusoidalPositionalEncoding,
    RoPE,
    top_p_sampling,
    SimpleBPE
)
```

Or from module root:
```python
from scripts import MultiHeadAttention, TransformerEncoder
```

---

## DGX Spark Advantages

| Feature | Benefit for This Module |
|---------|------------------------|
| 128GB Memory | Load larger models (GPT-2 Large, BERT Large) |
| BF16 Support | Faster training without accuracy loss |
| Fast Memory | Larger batch sizes for faster training |
| Tensor Cores | Accelerated attention computation |

Example with BF16:
```python
from torch.amp import autocast

with autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(**inputs)
```
