# Module C: Mechanistic Interpretability - Quickstart

## ⏱️ Time: ~5 minutes

## 🎯 What You'll Discover

You'll peek inside GPT-2's brain and see what it "thinks about" when predicting the next word.

## ✅ Before You Start

- [ ] DGX Spark container running
- [ ] `pip install transformer-lens` completed

## 🚀 Let's Go!

### Step 1: Load GPT-2 with TransformerLens

```python
from transformer_lens import HookedTransformer

# Load GPT-2 Small (124M parameters)
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
print(f"Loaded model with {model.cfg.n_layers} layers, {model.cfg.n_heads} heads per layer")
```

**Expected output:**
```
Loaded model with 12 layers, 12 heads per layer
```

### Step 2: Run Inference and Cache Activations

```python
prompt = "The capital of France is"
tokens = model.to_tokens(prompt)

# Run model AND cache every intermediate activation
logits, cache = model.run_with_cache(tokens)

print(f"Cached {len(cache)} activation tensors!")
print(f"Top prediction: {model.tokenizer.decode(logits[0, -1].argmax())}")
```

**Expected output:**
```
Cached 156 activation tensors!
Top prediction:  Paris
```

### Step 3: See What the Model Attends To

```python
import matplotlib.pyplot as plt

# Get attention pattern from layer 5, head 1
attention = cache["pattern", 5][0, 1].cpu()  # [seq_len, seq_len]
tokens_str = model.to_str_tokens(tokens[0])

plt.figure(figsize=(8, 6))
plt.imshow(attention, cmap='Blues')
plt.xticks(range(len(tokens_str)), tokens_str, rotation=45)
plt.yticks(range(len(tokens_str)), tokens_str)
plt.xlabel("Key (attends to)")
plt.ylabel("Query (from)")
plt.title("What does Layer 5, Head 1 attend to?")
plt.colorbar(label="Attention weight")
plt.tight_layout()
plt.show()
```

### Step 4: Find Where "France" Gets Processed

```python
# Which layers/heads pay attention to "France"?
france_idx = tokens_str.index("France")

for layer in range(model.cfg.n_layers):
    for head in range(model.cfg.n_heads):
        attn = cache["pattern", layer][0, head]
        # Attention from last position to "France"
        france_attention = attn[-1, france_idx].item()
        if france_attention > 0.3:  # High attention
            print(f"L{layer}H{head}: {france_attention:.2%} attention to 'France'")
```

**Expected output:**
```
L9H9: 42.31% attention to 'France'
L10H0: 38.17% attention to 'France'
...
```

## 🎉 You Did It!

You just:
1. **Loaded** a language model with access to all its internals
2. **Cached** every computation the model made
3. **Visualized** what the model "pays attention to"
4. **Identified** which specific components process "France"

This is the foundation of mechanistic interpretability - understanding HOW the model computes its answers!

## ▶️ Next Steps

1. **Learn the residual stream view**: See Notebook 01
2. **Try activation patching**: Causally verify which components matter (Notebook 02)
3. **Discover circuits**: Find the induction heads (Notebook 03)
4. **Extract features**: Use sparse autoencoders (Notebook 04)

---

## 💡 The Key Insight

> **We can see INSIDE neural networks.**
>
> Unlike a black box, we can trace every computation:
> - Which words the model attends to
> - What information flows through each layer
> - Which components are causally responsible for predictions
>
> This is how we verify if models are reasoning correctly - or just pattern matching!
