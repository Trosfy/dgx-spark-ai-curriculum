# Module C: Mechanistic Interpretability - Quick Reference

## 🔧 TransformerLens Essentials

### Loading Models

```python
from transformer_lens import HookedTransformer

# Available models (that fit on DGX Spark)
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")   # 124M
model = HookedTransformer.from_pretrained("gpt2-medium", device="cuda")  # 355M
model = HookedTransformer.from_pretrained("gpt2-large", device="cuda")   # 774M
model = HookedTransformer.from_pretrained("gpt2-xl", device="cuda")      # 1.5B
model = HookedTransformer.from_pretrained("pythia-1.4b", device="cuda")  # 1.4B
model = HookedTransformer.from_pretrained("pythia-2.8b", device="cuda")  # 2.8B
```

### Running with Activation Cache

```python
# Basic run with caching
logits, cache = model.run_with_cache(tokens)

# Selective caching (saves memory)
logits, cache = model.run_with_cache(
    tokens,
    names_filter=lambda name: "pattern" in name  # Only attention patterns
)
```

### Accessing Cached Activations

```python
# Cache uses (name, layer) tuples
residual = cache["resid_post", 5]        # After layer 5
attention_patterns = cache["pattern", 3] # Layer 3 attention [batch, head, q, k]
mlp_output = cache["mlp_out", 7]         # Layer 7 MLP output
keys = cache["k", 4]                     # Layer 4 keys
values = cache["v", 4]                   # Layer 4 values
queries = cache["q", 4]                  # Layer 4 queries

# Shape guide
# residual: [batch, seq, d_model]
# pattern: [batch, n_heads, seq, seq]
# mlp_out: [batch, seq, d_model]
```

---

## 📊 Common Activation Names

| Name | Shape | Description |
|------|-------|-------------|
| `resid_pre` | [B, S, D] | Input to layer |
| `resid_post` | [B, S, D] | Output of layer |
| `resid_mid` | [B, S, D] | After attention, before MLP |
| `attn_out` | [B, S, D] | Attention block output |
| `mlp_out` | [B, S, D] | MLP block output |
| `pattern` | [B, H, S, S] | Attention weights |
| `q` | [B, S, H, D_h] | Queries |
| `k` | [B, S, H, D_h] | Keys |
| `v` | [B, S, H, D_h] | Values |
| `z` | [B, S, H, D_h] | Attention output (pre-projection) |

Where: B=batch, S=seq_len, D=d_model, H=n_heads, D_h=d_head

---

## 🎯 Key Patterns

### Visualize Attention

```python
import plotly.express as px

def plot_attention(cache, layer, head, tokens):
    """Interactive attention heatmap."""
    pattern = cache["pattern", layer][0, head].cpu()
    token_strs = model.to_str_tokens(tokens[0])

    fig = px.imshow(
        pattern,
        x=token_strs,
        y=token_strs,
        labels=dict(x="Key", y="Query", color="Attention"),
        title=f"Layer {layer}, Head {head}"
    )
    return fig

# Usage
fig = plot_attention(cache, layer=5, head=1, tokens=tokens)
fig.show()
```

### Activation Patching

```python
def activation_patch(model, clean, corrupted, patch_layer, patch_pos):
    """
    Run clean prompt but patch in corrupted activation at specific location.
    """
    clean_tokens = model.to_tokens(clean)
    corrupted_tokens = model.to_tokens(corrupted)

    # Get corrupted cache
    _, corrupted_cache = model.run_with_cache(corrupted_tokens)

    def patch_hook(activation, hook):
        # Replace activation at position
        activation[:, patch_pos] = corrupted_cache[hook.name][:, patch_pos]
        return activation

    # Run with hook
    patched_logits = model.run_with_hooks(
        clean_tokens,
        fwd_hooks=[(f"blocks.{patch_layer}.hook_resid_post", patch_hook)]
    )

    return patched_logits

# Measure effect
clean_logits = model(clean_tokens)
patched_logits = activation_patch(model, clean, corrupted, layer=5, pos=-1)
effect = (clean_logits[0,-1] - patched_logits[0,-1]).norm()
```

### Find Induction Heads

```python
def find_induction_heads(model, seq_len=50):
    """Identify heads that implement induction (copying patterns)."""
    # Create repeated sequence: ABCD...ABCD...
    half = seq_len // 2
    random_tokens = torch.randint(1000, 10000, (1, half), device="cuda")
    tokens = torch.cat([random_tokens, random_tokens], dim=1)

    _, cache = model.run_with_cache(tokens)

    scores = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            pattern = cache["pattern", layer][0, head]
            # Induction: position i attends to i-half+1
            score = sum(pattern[i, i-half+1].item()
                       for i in range(half, seq_len)) / half
            scores.append((layer, head, score))

    return sorted(scores, key=lambda x: -x[2])[:10]

# Find top induction heads
top_heads = find_induction_heads(model)
for l, h, s in top_heads:
    print(f"L{l}H{h}: induction score = {s:.3f}")
```

---

## 🔍 The Residual Stream View

```
Input tokens
     │
     ▼
┌─────────────────────────────────────────────┐
│           Embedding + Positional            │
└────────────────────┬────────────────────────┘
                     │
     ┌───────────────┼───────────────┐
     │               │               │
     │               ▼               │
     │   ┌───────────────────────┐   │
     │   │    Attention (L0)     │───┼──► Reads from stream
     │   └───────────────────────┘   │
     │               │               │
     └───────────────┴───────────────┘
                     │
                     + (residual)     ◄── Writes to stream
                     │
                     ▼
              [Repeat for L1..L11]
                     │
                     ▼
┌─────────────────────────────────────────────┐
│              Final LayerNorm                │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│            Unembedding → Logits             │
└─────────────────────────────────────────────┘
```

**Key insight:** Each layer ADDS to the residual stream. Information accumulates, not replaces.

---

## 📐 Useful Computations

### Logit Lens (What does each layer "think"?)

```python
def logit_lens(model, cache, layer, position=-1):
    """What would the model predict after this layer?"""
    residual = cache["resid_post", layer][0, position]
    # Apply final layernorm + unembed
    normed = model.ln_final(residual)
    logits = normed @ model.W_U
    return model.tokenizer.decode(logits.argmax())

# Check predictions at each layer
for layer in range(model.cfg.n_layers):
    pred = logit_lens(model, cache, layer)
    print(f"After layer {layer}: predicts '{pred}'")
```

### OV Circuit Analysis

```python
def get_ov_circuit(model, layer, head):
    """Get the OV matrix for a head (what information it moves)."""
    W_O = model.blocks[layer].attn.W_O[head]  # [d_head, d_model]
    W_V = model.blocks[layer].attn.W_V[head]  # [d_model, d_head]
    OV = W_V @ W_O  # [d_model, d_model]
    return OV

# What does this head "copy"?
ov = get_ov_circuit(model, layer=5, head=1)
# High-magnitude entries = what info gets moved
```

### QK Circuit Analysis

```python
def get_qk_circuit(model, layer, head):
    """Get the QK matrix for a head (what attends to what)."""
    W_Q = model.blocks[layer].attn.W_Q[head]  # [d_model, d_head]
    W_K = model.blocks[layer].attn.W_K[head]  # [d_model, d_head]
    QK = W_Q @ W_K.T  # [d_model, d_model]
    return QK

# What tokens attend to what?
qk = get_qk_circuit(model, layer=5, head=1)
```

---

## 🧠 Common Head Types

| Head Type | Behavior | How to Identify |
|-----------|----------|-----------------|
| **Previous Token** | Attends to position i-1 | Check pattern diagonal |
| **Induction** | Copies after previous occurrence | Repeated sequence test |
| **Duplicate Token** | Attends to same token | Check pattern vs token equality |
| **Name Mover** | Moves names to prediction position | IOI task patching |
| **Backup Name Mover** | Activates when primary is ablated | Ablation experiments |

---

## ⚡ DGX Spark Memory Tips

```python
# Clear cache between experiments
import gc
torch.cuda.empty_cache()
gc.collect()

# Use selective caching for large models
logits, cache = model.run_with_cache(
    tokens,
    names_filter=lambda name: name in ["pattern", "resid_post"]
)

# Check memory usage
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")

# For large sequences, use hooks instead of full cache
results = {}
def save_attention(activation, hook):
    results[hook.name] = activation[:, :, -1, :].cpu()  # Only last query
    return activation

model.run_with_hooks(tokens, fwd_hooks=[
    (f"blocks.{layer}.attn.hook_pattern", save_attention)
    for layer in range(model.cfg.n_layers)
])
```

---

## 🔗 Quick Links

- Notebook 01: TransformerLens Setup
- Notebook 02: Activation Patching (IOI)
- Notebook 03: Induction Head Analysis
- Notebook 04: Sparse Autoencoders
- [TransformerLens Docs](https://neelnanda-io.github.io/TransformerLens/)
- [Transformer Circuits Thread](https://transformer-circuits.pub/)
