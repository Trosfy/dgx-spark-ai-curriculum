# Module 8: Natural Language Processing & Transformers

**Phase:** 2 - Intermediate  
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
| 8.1 | Implement multi-head self-attention from scratch | Apply |
| 8.2 | Explain positional encoding strategies (sinusoidal, RoPE) | Understand |
| 8.3 | Tokenize text using BPE and SentencePiece | Apply |
| 8.4 | Fine-tune BERT for classification tasks | Apply |

---

## Topics

### 8.1 Attention Mechanisms
- Scaled dot-product attention
- Multi-head attention
- Cross-attention
- Attention visualization

### 8.2 Transformer Architecture
- Encoder and decoder blocks
- Layer normalization (Pre-LN vs Post-LN)
- Feed-forward networks
- Residual connections

### 8.3 Positional Encodings
- Sinusoidal encoding
- Learned embeddings
- Rotary Position Embeddings (RoPE)
- ALiBi

### 8.4 Tokenization
- Word-level vs subword
- BPE algorithm
- SentencePiece

### 8.5 Pre-trained Models
- BERT (masked LM)
- GPT (causal LM)
- T5 (encoder-decoder)

---

## Tasks

| # | Task | Time | Deliverable |
|---|------|------|-------------|
| 8.1 | Attention from Scratch | 2h | Scaled dot-product and multi-head attention with visualization |
| 8.2 | Transformer Block | 3h | Complete encoder block, stack 6 layers |
| 8.3 | Positional Encoding Study | 2h | Sinusoidal and RoPE implementation |
| 8.4 | Tokenization Lab | 2h | Train BPE tokenizer, compare with GPT-2/LLaMA |
| 8.5 | BERT Fine-tuning | 2h | Sentiment classification with evaluation |
| 8.6 | GPT Text Generation | 2h | Implement decoding strategies (greedy, beam, sampling) |

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

## Resources

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoPE Paper](https://arxiv.org/abs/2104.09864)
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course)
