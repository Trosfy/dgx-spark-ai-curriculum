# Module 2.3: NLP & Transformers - Quick Reference

## Essential Formulas

### Scaled Dot-Product Attention
```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

Where:
- Q: Queries (batch, seq_len, d_k)
- K: Keys (batch, seq_len, d_k)
- V: Values (batch, seq_len, d_v)
- d_k: Key dimension (for scaling)
```

### Multi-Head Attention
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O

Where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
```

---

## Key Patterns

### Pattern: Scaled Dot-Product Attention

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, heads, seq_len, d_k)
    mask: (batch, 1, seq_len, seq_len) or broadcastable
    """
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attention_weights = F.softmax(scores, dim=-1)
    output = attention_weights @ V

    return output, attention_weights
```

### Pattern: Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear projections and reshape for heads
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(output)
```

### Pattern: Transformer Encoder Layer

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # Pre-LN variant (more stable)
        attn_out = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + attn_out

        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out

        return x
```

### Pattern: Sinusoidal Positional Encoding

```python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

### Pattern: RoPE (Rotary Position Embedding)

```python
def rotary_embedding(x, seq_len, dim):
    """Apply rotary position embedding"""
    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, freqs)  # (seq_len, dim/2)

    cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim/2)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)

    # Split even/odd dimensions
    x1, x2 = x[..., ::2], x[..., 1::2]

    # Apply rotation
    rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)

    return rotated
```

### Pattern: Causal Mask

```python
def create_causal_mask(seq_len, device='cuda'):
    """Creates mask where position i can only attend to positions <= i"""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

# Usage in attention:
# scores = scores + causal_mask  # Add -inf to future positions
```

### Pattern: Decoding Strategies

```python
# Greedy decoding
next_token = logits.argmax(dim=-1)

# Top-k sampling
top_k_logits, top_k_indices = logits.topk(k=50)
probs = F.softmax(top_k_logits / temperature, dim=-1)
idx = torch.multinomial(probs, 1)
next_token = top_k_indices.gather(-1, idx)

# Top-p (nucleus) sampling
sorted_logits, sorted_indices = logits.sort(descending=True)
cumsum = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
mask = cumsum <= p
mask[..., 0] = True  # Always keep at least one token
sorted_logits[~mask] = float('-inf')
probs = F.softmax(sorted_logits / temperature, dim=-1)
idx = torch.multinomial(probs, 1)
next_token = sorted_indices.gather(-1, idx)
```

### Pattern: Simple BPE Tokenizer

```python
class SimpleBPE:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}

    def train(self, texts):
        # Start with character-level vocabulary
        words = [list(word) + ['</w>'] for text in texts for word in text.split()]

        for i in range(self.vocab_size - 256):  # Minus base characters
            # Count pairs
            pairs = {}
            for word in words:
                for j in range(len(word) - 1):
                    pair = (word[j], word[j+1])
                    pairs[pair] = pairs.get(pair, 0) + 1

            if not pairs:
                break

            # Merge most frequent pair
            best_pair = max(pairs, key=pairs.get)
            self.merges[best_pair] = ''.join(best_pair)

            # Apply merge
            new_words = []
            for word in words:
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i+1]) == best_pair:
                        new_word.append(self.merges[best_pair])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_words.append(new_word)
            words = new_words

    def tokenize(self, text):
        # Apply learned merges
        tokens = list(text)
        for pair, merged in self.merges.items():
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                    tokens[i:i+2] = [merged]
                else:
                    i += 1
        return tokens
```

---

## Key Values to Remember

| What | Value | Notes |
|------|-------|-------|
| BERT base hidden size | 768 | d_model |
| BERT base num heads | 12 | |
| GPT-2 hidden size | 768 | small variant |
| Typical d_ff | 4 × d_model | Feed-forward expansion |
| Learning rate for fine-tuning | 2e-5 to 5e-5 | BERT recommendation |
| Vocabulary size (typical) | 32K-100K | BPE/WordPiece |

---

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Wrong attention shape | Q, K must have same d_k; V can differ |
| Forgetting scaling | Always divide by sqrt(d_k) |
| Wrong mask shape | Must broadcast correctly with attention scores |
| Pre-LN vs Post-LN confusion | Pre-LN: norm before attention; more stable |
| Not using causal mask for GPT | Autoregressive models need causal mask |
| Temperature = 0 | Results in division by zero; use very small value or argmax |

---

## Quick Links

**Module Docs:**
- [FAQ.md](./FAQ.md) - Common questions answered
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - Error solutions

**External Resources:**
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoPE Paper](https://arxiv.org/abs/2104.09864)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Tokenizers Documentation](https://huggingface.co/docs/tokenizers)
