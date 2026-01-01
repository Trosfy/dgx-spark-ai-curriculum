# Module 2.3: NLP & Transformers - Quickstart

## Time: ~5 minutes

## What You'll Build
Self-attention from scratch - the core mechanism powering ChatGPT and modern AI.

## Before You Start
- [ ] DGX Spark container running
- [ ] Module 2.2 concepts understood (comfortable with PyTorch)

## Let's Go!

### Step 1: Create Query, Key, Value
```python
import torch
import torch.nn.functional as F
import math

# Simulated input: batch=1, sequence_length=5, dimension=8
x = torch.randn(1, 5, 8)

# Linear projections for Q, K, V
d_k = 8
W_q = torch.randn(8, d_k)
W_k = torch.randn(8, d_k)
W_v = torch.randn(8, d_k)

Q = x @ W_q  # Query: "What am I looking for?"
K = x @ W_k  # Key: "What do I have to offer?"
V = x @ W_v  # Value: "What's my actual content?"

print(f"Q, K, V shapes: {Q.shape}")
```

### Step 2: Compute Attention Scores
```python
# Attention = softmax(Q @ K^T / sqrt(d_k)) @ V
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
print(f"Attention scores shape: {scores.shape}")  # (1, 5, 5)

attention_weights = F.softmax(scores, dim=-1)
print(f"Attention weights (first position attends to all):\n{attention_weights[0, 0]}")
```

### Step 3: Apply Attention
```python
# Weighted sum of values
output = attention_weights @ V
print(f"Output shape: {output.shape}")  # Same as input: (1, 5, 8)
```

### Step 4: Visualize What Attention "Sees"
```python
import matplotlib.pyplot as plt

# Each row shows how much position i attends to all positions j
plt.figure(figsize=(6, 5))
plt.imshow(attention_weights[0].detach().numpy(), cmap='Blues')
plt.xlabel('Key positions (what we attend to)')
plt.ylabel('Query positions (who is attending)')
plt.title('Self-Attention Pattern')
plt.colorbar(label='Attention weight')
plt.show()
```

## You Did It!

You just:
- Created Query, Key, Value projections (the heart of attention)
- Computed scaled dot-product attention
- Visualized how positions attend to each other
- Understood why attention is O(n²) - every position attends to every other!

In the full module, you'll learn:
- Multi-head attention (parallel attention "heads")
- Complete Transformer encoder block
- Positional encodings (sinusoidal, RoPE)
- Tokenization (BPE from scratch)
- BERT/GPT fine-tuning

## Next Steps
1. **Understand multi-head**: Start with [Lab 2.3.1](./labs/lab-2.3.1-attention-from-scratch.ipynb)
2. **Try masking**: Add a causal mask for autoregressive attention
3. **Full setup**: See [LAB_PREP.md](./LAB_PREP.md) for complete environment
