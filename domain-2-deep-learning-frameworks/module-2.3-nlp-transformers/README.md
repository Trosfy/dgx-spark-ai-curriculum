# Module 2.3: Natural Language Processing & Transformers

**Domain:** 2 - Deep Learning Frameworks  
**Duration:** Weeks 11-12 (12-15 hours)  
**Prerequisites:** Module 6 (PyTorch)

---

## Overview

This module provides a deep dive into the Transformer architecture—the foundation of modern NLP and LLMs. You'll implement attention mechanisms from scratch, understand positional encodings, and fine-tune pre-trained language models.

---

## Learning Outcomes

By the end of this module, you will be able to:

- ✅ Implement the Transformer architecture from scratch
- ✅ Explain attention mechanisms and their variations
- ✅ Apply tokenization strategies for different tasks
- ✅ Fine-tune language models for downstream tasks

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 2.3.1 | Implement multi-head self-attention from scratch | Apply |
| 2.3.2 | Explain positional encoding strategies (sinusoidal, RoPE) | Understand |
| 2.3.3 | Tokenize text using BPE and SentencePiece | Apply |
| 2.3.4 | Fine-tune BERT for classification tasks | Apply |

---

## Topics

### 2.3.1 Attention Mechanisms
- Scaled dot-product attention
- Multi-head attention
- Cross-attention
- Attention visualization

### 2.3.2 Transformer Architecture
- Encoder and decoder blocks
- Layer normalization (Pre-LN vs Post-LN)
- Feed-forward networks
- Residual connections

### 2.3.3 Positional Encodings
- Sinusoidal encoding
- Learned embeddings
- Rotary Position Embeddings (RoPE)
- ALiBi

### 2.3.4 Tokenization
- Word-level vs subword
- BPE algorithm
- SentencePiece

### 2.3.5 Pre-trained Models
- BERT (masked LM)
- GPT (causal LM)
- T5 (encoder-decoder)

---

## Labs

| # | Task | Time | Deliverable |
|---|------|------|-------------|
| 2.3.1 | Attention from Scratch | 2h | Scaled dot-product and multi-head attention with visualization |
| 2.3.2 | Transformer Block | 3h | Complete encoder block, stack 6 layers |
| 2.3.3 | Positional Encoding Study | 2h | Sinusoidal and RoPE implementation |
| 2.3.4 | Tokenization Lab | 2h | Train BPE tokenizer, compare with GPT-2/LLaMA |
| 2.3.5 | BERT Fine-tuning | 2h | Sentiment classification with evaluation |
| 2.3.6 | GPT Text Generation | 2h | Implement decoding strategies (greedy, beam, sampling) |

---

## Guidance

### Attention Formula

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, heads, seq_len, d_k)
    K: (batch, heads, seq_len, d_k)
    V: (batch, heads, seq_len, d_v)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attention = F.softmax(scores, dim=-1)
    return torch.matmul(attention, V), attention
```

### RoPE Implementation

```python
def rotary_embedding(x, seq_len, dim):
    """Apply rotary position embedding"""
    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs)
    
    cos = freqs.cos()
    sin = freqs.sin()
    
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
```

### Decoding Strategies

```python
# Greedy
next_token = logits.argmax(dim=-1)

# Top-k sampling
top_k_logits, top_k_indices = logits.topk(k=50)
probs = F.softmax(top_k_logits, dim=-1)
next_token = top_k_indices[torch.multinomial(probs, 1)]

# Top-p (nucleus) sampling
sorted_logits, sorted_indices = logits.sort(descending=True)
cumsum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
mask = cumsum <= p
# ... sample from masked distribution
```

---

## Milestone Checklist

- [ ] Multi-head attention implementation complete
- [ ] Full Transformer encoder working
- [ ] Both positional encoding types implemented
- [ ] Custom BPE tokenizer trained
- [ ] BERT fine-tuning achieving good accuracy
- [ ] Text generation with multiple decoding strategies

---

## DGX Spark Setup

For optimal performance on DGX Spark, use the NGC PyTorch container:

```bash
# Start the NGC container with all required flags
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

**Important flags:**
- `--gpus all` - Enable GPU access
- `--ipc=host` - Required for DataLoader multiprocessing
- `-p 8888:8888` - Expose Jupyter port
- `-v huggingface` - Share model cache between runs

**DGX Spark Advantages (128GB Unified Memory):**
- Larger batch sizes (64 vs typical 16-32)
- Full datasets in memory
- Fine-tune larger models without gradient checkpointing
- Use bfloat16 for native Blackwell optimization

---

## Using the Scripts Module

The `scripts/` folder contains reusable implementations. There are several ways to import them:

### Option 1: From Notebook Directory (Recommended)

When running notebooks from the `labs/` directory:

```python
import sys
from pathlib import Path

# Add scripts directory to path
scripts_path = Path.cwd().parent / "scripts"
if str(scripts_path) not in sys.path:
    sys.path.insert(0, str(scripts_path.parent))

from scripts import (
    MultiHeadAttention,
    TransformerEncoder,
    SinusoidalPositionalEncoding,
    RoPE,
    top_p_sampling,
    SimpleBPE
)
```

### Option 2: From Module Root

When running from the module root directory (`module-2.3-nlp-transformers/`):

```python
from scripts import MultiHeadAttention, TransformerEncoder
```

### Option 3: Direct Import

For standalone scripts, you can also use direct file imports:

```python
from scripts.attention import MultiHeadAttention, scaled_dot_product_attention
from scripts.transformer import TransformerEncoder, TransformerEncoderBlock
from scripts.positional_encoding import SinusoidalPositionalEncoding, RoPE
from scripts.generation import greedy_decode, top_k_sampling, top_p_sampling
from scripts.tokenizer_utils import SimpleBPE, estimate_token_cost
```

---

## Resources

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoPE Paper](https://arxiv.org/abs/2104.09864)
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course)
