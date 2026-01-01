# Module 2.3: Natural Language Processing & Transformers

**Domain:** 2 - Deep Learning Frameworks
**Duration:** Week 12 (12-15 hours)
**Prerequisites:** Module 2.2 (Computer Vision)
**Priority:** P2 Expanded (Tokenizer Training)

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
| 2.3.2 | Train custom tokenizers from scratch [P2 Expansion] | Apply |
| 2.3.3 | Explain positional encoding strategies (sinusoidal, RoPE, ALiBi) | Understand |
| 2.3.4 | Fine-tune BERT and GPT models for downstream tasks | Apply |

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

### 2.3.4 Tokenization [P2 Expansion]
- Word-level vs subword tokenization
- BPE algorithm implementation from scratch
- SentencePiece and HuggingFace tokenizers
- Training tokenizers on custom corpora
- Vocabulary size tradeoffs

### 2.3.5 Pre-trained Models
- BERT (masked LM)
- GPT (causal LM)
- T5 (encoder-decoder)

---

## Labs

| # | Task | Time | Deliverable |
|---|------|------|-------------|
| 2.3.1 | Attention from Scratch | 2h | Scaled dot-product and multi-head attention with visualization |
| 2.3.2 | Transformer Block | 2h | Complete encoder block, stack 6 layers |
| 2.3.3 | Positional Encoding Study | 2h | Sinusoidal and RoPE implementation |
| 2.3.4 | Tokenizer Training from Scratch | 3h | Implement BPE algorithm, train on custom corpus |
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

## Study Materials

| Document | Purpose |
|----------|---------|
| [QUICKSTART.md](./QUICKSTART.md) | Run your first attention mechanism in 5 minutes |
| [PREREQUISITES.md](./PREREQUISITES.md) | Skills self-check before starting |
| [STUDY_GUIDE.md](./STUDY_GUIDE.md) | Learning objectives and module roadmap |
| [ELI5.md](./ELI5.md) | Intuitive explanations of attention, Q/K/V, positional encoding |
| [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) | Commands, patterns, and code snippets |
| [LAB_PREP.md](./LAB_PREP.md) | Environment setup and lab preparation |
| [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) | Common errors and solutions |
| [FAQ.md](./FAQ.md) | Frequently asked questions about transformers and NLP |

---

## Milestone Checklist

- [ ] Multi-head attention implementation complete
- [ ] Full Transformer encoder working
- [ ] Both positional encoding types implemented
- [ ] Custom BPE tokenizer trained from scratch
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
- `-it` - Interactive terminal
- `--rm` - Cleanup container on exit
- `--ipc=host` - Required for DataLoader multiprocessing
- `-p 8888:8888` - Expose Jupyter port
- `-v workspace` - Persist your work
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
from scripts.transformer import TransformerEncoder, TransformerEncoderLayer
from scripts.positional_encoding import SinusoidalPositionalEncoding, RoPE
from scripts.generation import greedy_decode, top_k_sampling, top_p_sampling
from scripts.tokenizer_utils import SimpleBPE, estimate_token_cost
```

---

## Next Steps

After completing this module:
1. ✅ Verify all milestones are checked
2. 📁 Save reusable implementations to `scripts/`
3. ➡️ Proceed to [Module 2.4: Efficient Architectures](../module-2.4-efficient-architectures/)

---

## Module Navigation

| Previous | Current | Next |
|----------|---------|------|
| [Module 2.2: Computer Vision](../module-2.2-computer-vision/) | **Module 2.3: NLP & Transformers** | [Module 2.4: Efficient Architectures](../module-2.4-efficient-architectures/) |

---

## Resources

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoPE Paper](https://arxiv.org/abs/2104.09864)
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course)
- [Tokenizers Library](https://huggingface.co/docs/tokenizers)
- [SentencePiece](https://github.com/google/sentencepiece)
