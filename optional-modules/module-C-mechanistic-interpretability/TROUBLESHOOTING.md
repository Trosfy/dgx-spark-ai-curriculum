# Module C: Mechanistic Interpretability - Troubleshooting & FAQ

## 🔍 Quick Diagnostic

**Before diving into specific errors:**
1. Check GPU memory: `nvidia-smi` (TransformerLens caches are large!)
2. Verify TransformerLens version: `import transformer_lens; print(transformer_lens.__version__)`
3. Confirm model loaded correctly: `print(model.cfg)`

---

## 🚨 Error Categories

### Installation Issues

#### Error: `ModuleNotFoundError: No module named 'transformer_lens'`

**Solution:**
```bash
pip install transformer-lens

# If issues persist, try specific version
pip install transformer-lens==1.12.0
```

---

#### Error: `ImportError: cannot import name 'HookedTransformer'`

**Symptoms:** Old API or version conflict.

**Solution:**
```bash
# Upgrade to latest
pip install --upgrade transformer-lens

# Verify
python -c "from transformer_lens import HookedTransformer; print('OK')"
```

---

### Memory Issues

#### Error: `CUDA out of memory` when caching

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GiB
```

**Cause:** Full activation cache is HUGE. GPT-2 Small caches ~1GB per forward pass.

**Solutions:**
```python
# Solution 1: Selective caching
logits, cache = model.run_with_cache(
    tokens,
    names_filter=lambda name: "pattern" in name  # Only attention
)

# Solution 2: Use hooks instead of cache
results = {}
def save_hook(activation, hook):
    results[hook.name] = activation.cpu()  # Move to CPU immediately
    return activation

model.run_with_hooks(
    tokens,
    fwd_hooks=[(f"blocks.5.attn.hook_pattern", save_hook)]
)

# Solution 3: Process in smaller chunks
# For long sequences, process and save incrementally

# Solution 4: Clear memory between experiments
import gc
torch.cuda.empty_cache()
gc.collect()

# Solution 5: Use smaller model
model = HookedTransformer.from_pretrained("gpt2-small")  # Not gpt2-xl
```

---

#### Error: Running out of memory during patching experiments

**Cause:** Multiple forward passes accumulate memory.

**Solution:**
```python
# Clear gradients (not needed for inference)
with torch.no_grad():
    logits = model(tokens)

# Clear cache between experiments
def run_experiment(model, tokens):
    torch.cuda.empty_cache()
    gc.collect()

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)

    result = analyze(cache)

    del cache  # Explicitly delete
    torch.cuda.empty_cache()

    return result
```

---

### Patching Issues

#### Issue: Patching results are inconsistent

**Symptoms:** Same experiment gives different results on different runs.

**Causes and Solutions:**

1. **Random seeds not set**
   ```python
   import torch
   torch.manual_seed(42)
   torch.cuda.manual_seed(42)
   ```

2. **Backup circuits activating**
   ```python
   # Models have redundancy - multiple heads may do similar things
   # Solution: Use mean ablation instead of zero ablation

   def mean_ablation_hook(activation, hook):
       # Replace with mean activation over many prompts
       return mean_activation  # Pre-computed

   # Better: Use path patching to isolate specific information flow
   ```

3. **Measurement sensitivity**
   ```python
   # Use logit difference, not absolute probability
   def logit_diff(logits, correct_token, incorrect_token):
       return logits[0, -1, correct_token] - logits[0, -1, incorrect_token]
   ```

---

#### Issue: Activation patching shows all layers are important

**Symptoms:** Every layer seems to contribute significantly.

**Explanation:** This is often correct! Information flows through many layers.

**Better approach:**
```python
# Use path patching to isolate specific components

# 1. Patch at attention output level (not residual)
def patch_attn_out(activation, hook, corrupted_cache):
    return corrupted_cache[hook.name]

# 2. Patch specific heads, not whole layers
def patch_single_head(activation, hook, corrupted_cache, head_idx):
    activation[:, :, head_idx] = corrupted_cache[hook.name][:, :, head_idx]
    return activation

# 3. Use causal scrubbing for rigorous analysis
# (More advanced - see ARENA curriculum)
```

---

### Visualization Issues

#### Issue: Attention patterns look uniform (all ~equal values)

**Symptoms:** Heatmap shows no clear patterns.

**Causes and Solutions:**

1. **Looking at wrong head/layer**
   ```python
   # Not all heads do interpretable things!
   # Try multiple layers and heads
   for layer in range(model.cfg.n_layers):
       for head in range(model.cfg.n_heads):
           pattern = cache["pattern", layer][0, head]
           if pattern.max() > 0.5:  # Look for strong attention
               print(f"Interesting: L{layer}H{head}")
   ```

2. **Attention is position-based, not content-based**
   ```python
   # Some heads attend by position (e.g., "previous token head")
   # This is valid! Just different from content-based attention
   ```

3. **Model using different mechanism**
   ```python
   # MLPs also store and retrieve information
   # Check MLP activations too
   mlp_out = cache["mlp_out", 5]
   ```

---

### Model Loading Issues

#### Error: Model not supported in TransformerLens

**Symptoms:**
```
ValueError: Model 'xxx' not found in model_name_to_class
```

**Solution:**
```python
# Check supported models
from transformer_lens import loading_from_pretrained
print(loading_from_pretrained.get_pretrained_model_names())

# For unsupported models, use from_pretrained with config
# Or load via HookedTransformerConfig (advanced)
```

---

## ❓ Frequently Asked Questions

### Conceptual Questions

#### Q: What's the difference between interpretability and explainability?

**A:**

| Mechanistic Interpretability | Explainability (XAI) |
|------------------------------|---------------------|
| Understand HOW model computes | Explain WHAT model does |
| Reverse-engineer circuits | Generate explanations for users |
| Causal analysis | Often correlational |
| Scientific understanding | Practical deployment |
| "What algorithm is the model implementing?" | "Why did the model make this prediction?" |

---

#### Q: Why study GPT-2 when we have GPT-4?

**A:**
1. **GPT-2 is tractable** - We can actually understand its 124M parameters
2. **Circuits transfer** - Induction heads exist in all transformers
3. **Tools work** - TransformerLens supports GPT-2 well
4. **Scientific method** - Start simple, build understanding

**Industry analogy:** We studied neurons in worms (302 neurons) before humans (86 billion).

---

#### Q: What are induction heads and why do they matter?

**A:** Induction heads implement in-context learning:

```
Pattern: [A][B]...[A] → [B]

Example: "Harry Potter... Harry" → "Potter"
```

They matter because:
1. First discovered circuit that explains a general capability
2. Present in ALL transformers we've checked
3. May be foundation of more complex abilities
4. Show models learn interpretable algorithms

---

#### Q: What's the residual stream view?

**A:** Think of transformers as information flowing through a "stream":

```
Standard view:        Residual stream view:
Layer 1 → Layer 2     Stream: ═══════════════════════
                              │      │      │
                              │Attn  │MLP   │Attn
                              │writes│writes│reads
```

**Key insights:**
- Layers ADD to the stream, not replace
- Each component READS from and WRITES to the stream
- Information from early layers persists to late layers

---

### Practical Questions

#### Q: How do I know if my circuit analysis is correct?

**A:** Use these validation techniques:

1. **Ablation** - Remove component, check behavior changes
2. **Path patching** - Trace information flow
3. **Faithfulness** - Circuit alone reproduces behavior
4. **Minimality** - No unnecessary components

```python
# Example: Verify induction head
def verify_induction(model, head_layer, head_idx):
    # 1. Create induction task
    tokens = torch.tensor([[1000, 2000, 3000, 1000]])  # [A][B][C][A] → should predict B

    # 2. Normal prediction
    logits = model(tokens)
    normal_pred = logits[0, -1].argmax()

    # 3. Ablate the head
    def ablate_hook(activation, hook):
        activation[:, :, head_idx] = 0
        return activation

    ablated_logits = model.run_with_hooks(
        tokens,
        fwd_hooks=[(f"blocks.{head_layer}.attn.hook_z", ablate_hook)]
    )
    ablated_pred = ablated_logits[0, -1].argmax()

    print(f"Normal: {normal_pred}, Ablated: {ablated_pred}")
    print(f"Behavior changed: {normal_pred != ablated_pred}")
```

---

#### Q: Why can't I just use attention weights for interpretability?

**A:** Attention alone is misleading:

1. **Attention ≠ importance** - High attention doesn't mean high influence on output
2. **OV circuit matters** - What information is MOVED, not just attended to
3. **Residual stream** - Information can bypass attention entirely
4. **Superposition** - Multiple features share same neurons

**Better:** Use activation patching to establish CAUSAL relationships.

---

#### Q: How do I find interesting circuits to study?

**A:** Start with:

1. **Known behaviors** - IOI, greater-than, simple syntax
2. **Model failures** - Where does it make mistakes?
3. **Surprising capabilities** - Things you wouldn't expect

```python
# Example: Find heads that attend to specific token types
for layer in range(model.cfg.n_layers):
    for head in range(model.cfg.n_heads):
        # Check if head attends to quotes, numbers, names, etc.
        pattern = cache["pattern", layer][0, head]
        # Analyze pattern relative to token types
```

---

### Beyond the Basics

#### Q: How does this connect to AI safety?

**A:**

| Safety Goal | How Mech Interp Helps |
|-------------|----------------------|
| Detect deception | Find circuits that produce different internal/external answers |
| Verify reasoning | Check if model uses valid logic circuits |
| Remove harmful knowledge | Identify and ablate specific circuits |
| Alignment verification | Understand what model actually optimizes for |

---

#### Q: What are the current limitations?

**A:**

| Limitation | Current Status |
|------------|----------------|
| Scale | Works on small models, struggles with large |
| Superposition | Features share neurons, hard to disentangle |
| Automation | Still mostly manual analysis |
| Completeness | Can't fully explain any model yet |
| Training dynamics | Mostly study trained models, not training |

---

## 🔄 Reset Procedures

### Clear TransformerLens Cache

```python
# Clear activation cache
del cache
import gc
torch.cuda.empty_cache()
gc.collect()

# Reset model hooks
model.reset_hooks()
```

### Restart from Clean State

```python
# If things are corrupted, reload model
del model
torch.cuda.empty_cache()
gc.collect()

model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
```

---

## 📞 Still Stuck?

1. **Check TransformerLens demos** - Many examples in their docs
2. **ARENA curriculum** - Interactive tutorials
3. **Neel Nanda's videos** - Visual explanations
4. **200 Concrete Problems** - List of research directions
