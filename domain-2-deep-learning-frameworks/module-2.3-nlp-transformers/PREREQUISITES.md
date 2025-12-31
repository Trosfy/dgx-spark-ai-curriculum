# Module 2.3: NLP & Transformers - Prerequisites Check

## Purpose
This module introduces attention mechanisms and transformers - the foundation of modern AI. Use this self-check to ensure you're ready.

## Estimated Time
- **If all prerequisites met**: Start with [QUICKSTART.md](./QUICKSTART.md)
- **If 1-2 gaps**: ~2-3 hours of review
- **If multiple gaps**: Complete Modules 2.1-2.2 first

---

## Required Skills

### 1. PyTorch: Matrix Operations

**Can you do this without looking anything up?**
```python
import torch

# Given x of shape (batch, seq_len, dim), compute:
# 1. x @ W where W is (dim, dim)
# 2. Transpose last two dimensions of x
# 3. Apply softmax along the last dimension
```

<details>
<summary>Check your answer</summary>

```python
import torch
import torch.nn.functional as F

x = torch.randn(2, 5, 8)  # (batch, seq_len, dim)
W = torch.randn(8, 8)      # (dim, dim)

# 1. Matrix multiplication
out = x @ W  # or torch.matmul(x, W)
# Shape: (2, 5, 8) @ (8, 8) = (2, 5, 8)

# 2. Transpose last two dimensions
x_T = x.transpose(-2, -1)  # or x.permute(0, 2, 1)
# Shape: (2, 8, 5)

# 3. Softmax along last dimension
probs = F.softmax(x, dim=-1)
# Each row sums to 1
```

**Key points**:
- `@` is matrix multiplication (same as `torch.matmul`)
- `transpose(-2, -1)` swaps the last two dimensions
- `dim=-1` means last dimension

</details>

**Not ready?** Review: Module 2.1, tensor operations

---

### 2. Linear Algebra: Attention Intuition

**Can you answer this?**
> What does Q @ K^T compute, and why do we divide by sqrt(d_k)?

<details>
<summary>Check your answer</summary>

**Q @ K^T computes similarity scores**:
- Q is "what am I looking for" (query)
- K is "what do I have to offer" (key)
- Q @ K^T gives a score for each query-key pair
- Higher score = more similar = higher attention

**Why divide by sqrt(d_k)?**
- Dot products grow with dimension (sum of d_k products)
- Large values → softmax becomes very peaked (all attention on one position)
- Dividing by sqrt(d_k) keeps variance stable regardless of dimension
- This is "scaled dot-product attention"

**Formula**: `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) @ V`

</details>

**Not ready?** Quick read: [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

---

### 3. PyTorch: nn.Module Composition

**Can you write this?**
```python
# Create a module with:
# - Two linear layers
# - LayerNorm
# - Residual connection
class TransformerBlock(nn.Module):
    # Your implementation
    pass
```

<details>
<summary>Check your answer</summary>

```python
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Feed-forward with residual
        residual = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = x + residual  # Residual connection
        x = self.norm(x)  # LayerNorm
        return x
```

**Key points**:
- LayerNorm normalizes across features (not batch)
- Residual connection adds input to output
- FFN typically expands then contracts (4x is common)

</details>

**Not ready?** Review: Module 2.1, Lab 2.1.1

---

### 4. Sequences: Understanding Tokens

**Do you understand?**
> What is a token? Why do we use subword tokenization instead of word-level?

<details>
<summary>Check your answer</summary>

**What is a token?**
- A token is the basic unit the model processes
- Could be a word, part of a word, or character
- Each token maps to an embedding vector

**Why subword (BPE, WordPiece)?**

Word-level problems:
- Fixed vocabulary (can't handle new words)
- Vocabulary explosion (millions of words)
- "running" and "run" have no connection

Character-level problems:
- Too many tokens per sentence (slow)
- Hard to learn word meanings

Subword (best of both):
- "running" → ["run", "ning"] - shares "run" with "run"
- Can handle any word (even made-up ones)
- Reasonable vocabulary size (32K-100K)
- "ChatGPT" → ["Chat", "G", "PT"] - can handle novel words

</details>

**Not ready?** Quick read: [Hugging Face Tokenizers Guide](https://huggingface.co/docs/tokenizers)

---

### 5. Math: Softmax Understanding

**Can you explain?**
> What does softmax do, and why is it used in attention?

<details>
<summary>Check your answer</summary>

**Softmax converts scores to probabilities**:
```python
softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
```

Properties:
- Output values are between 0 and 1
- Output values sum to 1 (valid probability distribution)
- Larger inputs get exponentially larger outputs (amplifies differences)

**In attention**:
- Raw scores can be any value (-∞ to +∞)
- Softmax converts to "how much to attend to each position"
- Values sum to 1, so it's a weighted average
- High score → high attention weight

</details>

**Not ready?** Review: Module 1.4, activation functions

---

## Terminology Check

Do you know these terms?

| Term | Your Definition |
|------|-----------------|
| Token | |
| Embedding | |
| Attention | |
| Query, Key, Value | |
| Positional encoding | |
| Encoder vs Decoder | |

<details>
<summary>Check definitions</summary>

| Term | Definition |
|------|------------|
| **Token** | Basic unit processed by model (word, subword, or character) |
| **Embedding** | Dense vector representation of a token (learned) |
| **Attention** | Mechanism to compute weighted combinations based on relevance |
| **Query, Key, Value** | Q asks questions, K provides identifiers, V provides content |
| **Positional encoding** | Information about position added since attention has no inherent order |
| **Encoder** | Bidirectional (sees full context); **Decoder** | Autoregressive (sees only past) |

</details>

---

## Optional But Helpful

### Linear Algebra: Matrix Multiplication
**Why it helps**: Attention is fundamentally matrix operations
**Key insight**: (A @ B)[i,j] = sum of A[i,:] * B[:,j]

### Understanding Skip Connections
**Why it helps**: Transformers use residual connections heavily
**Review**: Module 2.2, ResNet section

---

## Ready Checklist

- [ ] I can do PyTorch matrix operations confidently
- [ ] I understand what attention computes conceptually
- [ ] I can write nn.Module with LayerNorm and residuals
- [ ] I understand tokens and why subword tokenization exists
- [ ] I know what softmax does
- [ ] My environment is set up (see [LAB_PREP.md](./LAB_PREP.md))

**All boxes checked?** Start with [QUICKSTART.md](./QUICKSTART.md)!

**Some gaps?** The Illustrated Transformer is an excellent 30-minute read that will make everything click.
