# Optional Module C: Mechanistic Interpretability

**Category:** Optional - AI Safety Research
**Duration:** 6-8 hours
**Prerequisites:** Module 2.3 (NLP & Transformers), Module 4.2 (AI Safety)
**Priority:** P3 (Optional - Research Frontier)

---

## Overview

What's actually happening inside a neural network? Mechanistic interpretability (mech interp) reverse-engineers neural networks to understand *how* they compute answers, not just *what* answers they give. This is one of the most active areas of AI safety research, aiming to make AI systems transparent and trustworthy.

**Why This Matters:** As AI systems become more powerful, understanding their internals becomes critical for safety. Can we verify that a model reasons correctly rather than exploiting spurious correlations? Can we detect deceptive behavior before deployment? Mechanistic interpretability provides tools to answer these questions.

### The Kitchen Table Explanation

Imagine you have a calculator that always gives the right answer to multiplication problems. But is it *actually* multiplying, or has it just memorized a lookup table? You can't tell from the outputs alone. Mechanistic interpretability is like opening the calculator to see the gears inside - understanding not just that it works, but *how* it works. For neural networks, we're finding that they develop surprisingly human-interpretable "circuits" that implement specific computations.

---

## Learning Outcomes

By the end of this module, you will be able to:

- ✅ Understand how transformers represent and process information
- ✅ Use activation patching to identify causal mechanisms
- ✅ Discover and analyze circuits in small language models
- ✅ Apply interpretability tools to real models

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| C.1 | Explain the residual stream view of transformers | Understand |
| C.2 | Perform activation patching to isolate causal components | Apply |
| C.3 | Identify and analyze circuits for specific behaviors | Analyze |
| C.4 | Use TransformerLens for interpretability research | Apply |

---

## Topics

### C.1 The Residual Stream View

- **Transformers as Residual Streams**
  - Information flows through residual connections
  - Attention and MLP blocks read/write to the stream
  - Each layer adds to, rather than replaces, information

- **Attention Head Roles**
  - Induction heads (copy patterns)
  - Previous token heads
  - Positional heads
  - Specialized computation heads

- **MLP Knowledge Storage**
  - Key-value memories in MLPs
  - Superposition: more features than neurons
  - Polysemanticity challenges

### C.2 Activation Patching

- **Causal Intervention Basics**
  - Replace activations from clean to corrupted runs
  - Measure change in output
  - Identify components that matter

- **Patching Techniques**
  - Full activation patching
  - Attention pattern patching
  - Path patching
  - Causal scrubbing

- **Interpreting Results**
  - Direct effects vs indirect effects
  - Backup behavior and redundancy
  - When patching misleads

### C.3 Circuit Discovery

- **What is a Circuit?**
  - Subgraph of model that implements a behavior
  - Minimal sufficient explanation
  - Examples: Indirect Object Identification, Greater-Than

- **Finding Circuits**
  - Start with behavior of interest
  - Use patching to narrow down components
  - Trace information flow through attention patterns
  - Verify with ablations

- **Circuit Analysis**
  - Composition of attention heads
  - Q, K, V decomposition
  - Virtual weights and OV circuits

### C.4 Advanced Topics

- **Superposition and Features**
  - Linear representation hypothesis
  - Sparse autoencoders for feature extraction
  - Dictionary learning in activations

- **Probing and Interventions**
  - Linear probes for concepts
  - Activation steering
  - Editing factual associations

- **Scaling Challenges**
  - Tools that work on GPT-2 may fail on larger models
  - Automated interpretability
  - Current limitations

---

## Labs

### Lab C.1: TransformerLens Setup and Exploration
**Time:** 1.5 hours

Get familiar with TransformerLens and explore GPT-2 internals.

**Instructions:**
1. Install TransformerLens and set up environment
2. Load GPT-2 Small (124M parameters)
3. Run inference and cache all activations
4. Visualize attention patterns for sample prompts
5. Examine residual stream norms across layers
6. Identify which layers attend to which positions

**Deliverable:** Notebook with attention visualizations and layer analysis

---

### Lab C.2: Activation Patching on IOI
**Time:** 2 hours

Replicate the Indirect Object Identification (IOI) circuit discovery.

**Instructions:**
1. Create IOI dataset: "John and Mary went to the store. John gave a bottle to" → "Mary"
2. Implement activation patching at each layer
3. Identify which layers are crucial for IOI task
4. Patch individual attention heads to find key contributors
5. Create heatmap of head importance
6. Compare to published IOI circuit

**Deliverable:** Notebook identifying key IOI components

---

### Lab C.3: Induction Head Analysis
**Time:** 2 hours

Study induction heads - a fundamental circuit for in-context learning.

**Instructions:**
1. Create dataset with repeated patterns: "[A][B]...[A]" → "[B]"
2. Identify candidate induction heads via attention patterns
3. Use path patching to verify composition
4. Show that previous-token head composes with induction head
5. Ablate induction heads and measure loss degradation
6. Compare induction strength across model sizes

**Deliverable:** Notebook demonstrating induction head mechanism

---

### Lab C.4: Feature Extraction with SAEs
**Time:** 2.5 hours

Use Sparse Autoencoders to extract interpretable features.

**Instructions:**
1. Load pre-trained SAE for GPT-2 (from Anthropic or community)
2. Run model on diverse prompts and extract SAE features
3. Identify interpretable features (e.g., "Python code", "questions")
4. Visualize feature activation across tokens
5. Attempt activation steering using found features
6. Document most interpretable features discovered

**Deliverable:** Notebook with interpretable feature analysis

---

## Guidance

### Setting Up TransformerLens

```python
# Install TransformerLens (works on DGX Spark ARM64)
# pip install transformer-lens

import torch
from transformer_lens import HookedTransformer

# Load model (GPT-2 Small fits easily in DGX Spark memory)
model = HookedTransformer.from_pretrained(
    "gpt2-small",
    device="cuda"  # Use GPU
)

# Basic inference with activation caching
prompt = "The capital of France is"
tokens = model.to_tokens(prompt)

# Run and cache ALL activations
logits, cache = model.run_with_cache(tokens)

# Access cached activations
residual_stream = cache["resid_post", 5]  # After layer 5
attention_patterns = cache["pattern", 3]   # Layer 3 attention
mlp_output = cache["mlp_out", 7]          # Layer 7 MLP output

print(f"Residual stream shape: {residual_stream.shape}")
print(f"Attention patterns shape: {attention_patterns.shape}")
```

### Visualizing Attention Patterns

```python
import plotly.express as px
import pandas as pd

def visualize_attention(cache, layer, head, tokens):
    """
    Create attention heatmap for a specific head.
    """
    pattern = cache["pattern", layer][0, head].cpu().numpy()  # [seq, seq]
    token_strs = model.to_str_tokens(tokens)

    fig = px.imshow(
        pattern,
        labels=dict(x="Key Position", y="Query Position", color="Attention"),
        x=token_strs,
        y=token_strs,
        title=f"Layer {layer}, Head {head}"
    )
    fig.show()

# Visualize different heads
for layer in [0, 5, 11]:
    for head in [0, 5]:
        visualize_attention(cache, layer, head, tokens)
```

### Activation Patching

```python
from transformer_lens import patching

def activation_patch_experiment(model, clean_prompt, corrupted_prompt, answer_token):
    """
    Patch activations from corrupted to clean run,
    measure effect on predicting answer_token.
    """
    # Tokenize
    clean_tokens = model.to_tokens(clean_prompt)
    corrupted_tokens = model.to_tokens(corrupted_prompt)

    # Get clean logits
    clean_logits = model(clean_tokens)
    clean_prob = clean_logits[0, -1].softmax(dim=-1)[answer_token].item()

    # Get corrupted logits
    corrupted_logits = model(corrupted_tokens)
    corrupted_prob = corrupted_logits[0, -1].softmax(dim=-1)[answer_token].item()

    # Patch each layer's residual stream
    results = []
    for layer in range(model.cfg.n_layers):
        def patch_hook(activation, hook):
            # Replace with corrupted activation
            _, corrupted_cache = model.run_with_cache(corrupted_tokens)
            return corrupted_cache[hook.name]

        patched_logits = model.run_with_hooks(
            clean_tokens,
            fwd_hooks=[(f"blocks.{layer}.hook_resid_post", patch_hook)]
        )
        patched_prob = patched_logits[0, -1].softmax(dim=-1)[answer_token].item()

        # How much did patching hurt?
        effect = (clean_prob - patched_prob) / (clean_prob - corrupted_prob + 1e-8)
        results.append({"layer": layer, "effect": effect})

    return pd.DataFrame(results)

# Example: IOI task
clean = "John and Mary went to the store. John gave a book to"
corrupted = "John and Mary went to the store. Mary gave a book to"
answer = model.to_tokens(" Mary")[0, 0]

df = activation_patch_experiment(model, clean, corrupted, answer)
px.bar(df, x="layer", y="effect", title="Activation Patching: IOI Task").show()
```

### Finding Induction Heads

```python
def find_induction_heads(model, seq_len=100):
    """
    Identify induction heads by their distinctive attention pattern:
    attending to tokens that follow previous occurrences of current token.

    Induction heads: high attention to position (i-1) when token[i] == token[j] for j < i
    """
    # Create random repeated sequence: [A][B][C]...[A][B][C]...
    half_len = seq_len // 2
    random_tokens = torch.randint(1000, 10000, (1, half_len), device=model.cfg.device)
    repeated_tokens = torch.cat([random_tokens, random_tokens], dim=1)

    # Get attention patterns
    _, cache = model.run_with_cache(repeated_tokens)

    induction_scores = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            pattern = cache["pattern", layer][0, head]  # [seq, seq]

            # Induction score: attention to position (i - half_len + 1) from position i
            # This is where the "next token after previous occurrence" lives
            score = 0
            count = 0
            for i in range(half_len, seq_len):
                target_pos = i - half_len + 1  # Position after previous occurrence
                if target_pos > 0:
                    score += pattern[i, target_pos].item()
                    count += 1

            induction_scores.append({
                "layer": layer,
                "head": head,
                "induction_score": score / count if count > 0 else 0
            })

    df = pd.DataFrame(induction_scores)
    top_heads = df.nlargest(10, "induction_score")
    return top_heads

induction_heads = find_induction_heads(model)
print("Top Induction Heads:")
print(induction_heads)
```

### Sparse Autoencoder Feature Analysis

```python
# Using SAE features (assuming pre-trained SAE loaded)
# Community SAEs available at: https://www.neuronpedia.org/

class SparseAutoencoder(torch.nn.Module):
    """Simplified SAE for feature extraction."""

    def __init__(self, d_model, n_features):
        super().__init__()
        self.encoder = torch.nn.Linear(d_model, n_features)
        self.decoder = torch.nn.Linear(n_features, d_model)

    def encode(self, x):
        return torch.relu(self.encoder(x))  # Sparse activations

    def forward(self, x):
        features = self.encode(x)
        reconstruction = self.decoder(features)
        return reconstruction, features

def analyze_features(sae, model, prompts):
    """
    Find which SAE features activate for different prompts.
    """
    results = []
    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens)

        # Get activations at layer 6 (middle layer)
        activations = cache["resid_post", 6][0]  # [seq, d_model]

        # Encode to features
        features = sae.encode(activations)  # [seq, n_features]

        # Find top activating features
        max_features = features.max(dim=0).values
        top_k = torch.topk(max_features, k=10)

        results.append({
            "prompt": prompt,
            "top_features": top_k.indices.tolist(),
            "activations": top_k.values.tolist()
        })

    return results
```

### DGX Spark Advantages

> **DGX Spark Tip:** Mechanistic interpretability is memory-intensive - you're caching every activation in the model. With 128GB unified memory, you can:
> - Analyze full attention patterns for long sequences (4K+ tokens)
> - Cache activations for multiple runs simultaneously
> - Train SAEs on larger datasets without OOM errors
> - Work with medium-sized models (GPT-2 XL, Pythia-2.8B) comfortably

---

## 📖 Study Materials

| Document | Purpose |
|----------|---------|
| [QUICKSTART.md](./QUICKSTART.md) | Peek inside GPT-2 in 5 minutes |
| [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) | TransformerLens cache access and patching patterns |
| [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) | Debug memory issues and patching inconsistencies |
| [ELI5.md](./ELI5.md) | Intuitive explanations of circuits and features |
| [LAB_PREP.md](./LAB_PREP.md) | Environment setup and pre-lab checklist |

---

## Milestone Checklist

- [ ] TransformerLens environment working
- [ ] Attention patterns visualized and understood
- [ ] Activation patching experiment completed
- [ ] IOI or induction head circuit analyzed
- [ ] SAE features explored
- [ ] Can explain residual stream view

---

## Common Issues

| Issue | Solution |
|-------|----------|
| CUDA OOM with large models | Use smaller batch size, clear cache frequently |
| Attention patterns hard to interpret | Start with clear tasks (IOI), compare to known circuits |
| Patching results inconsistent | Check for backup circuits, use mean ablation |
| TransformerLens version conflicts | Pin version, check model compatibility |

---

## Why This Module is Optional

Mechanistic interpretability is cutting-edge research, not standard practice. Most AI practitioners don't need to reverse-engineer model internals. However, this skill is valuable for:

1. **AI Safety Research** - Active hiring area at leading labs
2. **Debugging Models** - Understand why models fail unexpectedly
3. **Red Teaming** - Find and exploit model weaknesses
4. **Research Contributions** - Many open problems, accessible to newcomers

---

## Next Steps

After completing this module:
1. Read the Anthropic Transformer Circuits papers
2. Contribute to open-source interpretability research
3. Apply interpretability to your capstone's safety evaluation
4. Consider AI safety research career paths

---

## Resources

- [TransformerLens Documentation](https://neelnanda-io.github.io/TransformerLens/)
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) - SAE features
- [Neuronpedia](https://www.neuronpedia.org/) - Community SAE feature explorer
- [ARENA Curriculum](https://arena3-chapter1-transformer-interp.streamlit.app/) - Interactive tutorials
- [200 Concrete Open Problems in Mechanistic Interpretability](https://www.alignmentforum.org/posts/LbrPTJ4fmABEdEnLf/200-concrete-open-problems-in-mechanistic-interpretability)

