# Module 2.4: Efficient Architectures - Study Guide

## Learning Objectives

By the end of this module, you will be able to:

1. **Explain Mamba's selective state space mechanism** and why it achieves O(n) complexity
2. **Understand MoE architecture** with routing, expert selection, and load balancing
3. **Compare architectures** (Transformer vs Mamba vs MoE) on memory, speed, and quality
4. **Run and benchmark** efficient architectures on DGX Spark
5. **Fine-tune Mamba** using LoRA for custom tasks

---

## Module Roadmap

| # | Lab | Focus | Time | Key Outcome |
|---|-----|-------|------|-------------|
| 1 | lab-2.4.1-mamba-inference.ipynb | Mamba vs Transformer | ~2 hr | Benchmark comparison |
| 2 | lab-2.4.2-mamba-architecture.ipynb | Selective scan | ~2 hr | Mechanism visualization |
| 3 | lab-2.4.3-moe-exploration.ipynb | MoE basics | ~2 hr | Expert activation analysis |
| 4 | lab-2.4.4-moe-router.ipynb | Routing deep-dive | ~2 hr | Router visualization |
| 5 | lab-2.4.5-architecture-comparison.ipynb | Systematic comparison | ~2 hr | Comparison report |
| 6 | lab-2.4.6-mamba-finetuning.ipynb | Mamba + LoRA | ~2 hr | Fine-tuned Mamba |

**Total time**: ~10-12 hours

---

## Core Concepts

### Quadratic Attention Problem
**What**: Transformer attention computes all pairs of tokens → O(n²) complexity
**Why it matters**: 32K context = 1 billion attention scores per layer; limits practical context length
**First appears in**: Background for understanding why alternatives exist

### Selective State Space (Mamba)
**What**: A recurrent mechanism that processes one token at a time with compressed state
**Why it matters**: O(n) complexity and constant memory per token enables 100K+ contexts
**First appears in**: Lab 2.4.1, Lab 2.4.2

### Mixture of Experts (MoE)
**What**: Multiple "expert" networks where only a subset is activated per token
**Why it matters**: 200B total params with only 20B active = efficient large models
**First appears in**: Lab 2.4.3

### Expert Routing
**What**: A learned network that decides which experts process each token
**Why it matters**: Good routing = specialized experts; bad routing = collapsed/unused experts
**First appears in**: Lab 2.4.4

---

## How This Module Connects

```
Previous                    This Module                 Next
─────────────────────────────────────────────────────────────
Module 2.3              ►   Module 2.4              ►  Module 2.5
NLP & Transformers          Efficient Arch              Hugging Face
(attention mechanism)       (alternatives)              (ecosystem)
```

**Builds on**:
- **Attention mechanism** from Module 2.3 (now understand its limitations)
- **Transformer architecture** from Module 2.3 (now see alternatives)
- **Model loading patterns** from Modules 2.2-2.3

**Prepares for**:
- **Module 2.5** will use HuggingFace to load these models easily
- **Module 3.1-3.3** will fine-tune and deploy LLMs (some are Mamba/MoE)
- **Module 3.4** (Test-Time Compute) benefits from efficient architectures

---

## Architecture Comparison

```
                 Memory      Speed        Quality     Best For
─────────────────────────────────────────────────────────────────
Transformer      O(n²)       Fast         SOTA        Most tasks
Mamba            O(n)        Fast         Near-SOTA   Long context
MoE              O(params)   Fast*        SOTA        Large scale
Jamba (Hybrid)   Mixed       Fast         High        Best of both
─────────────────────────────────────────────────────────────────
* MoE: Only activates fraction of params per token
```

---

## Recommended Approach

### Standard Path (10-12 hours)
1. **Start with Lab 2.4.1** - See the practical difference (Mamba vs Transformer)
2. **Dive into Lab 2.4.2** - Understand WHY Mamba works (selective scan)
3. **Explore Lab 2.4.3** - MoE is used in many production LLMs (Mixtral, DeepSeek)
4. **Understand Lab 2.4.4** - Routing is the key to MoE success
5. **Compare in Lab 2.4.5** - Build intuition for when to use each
6. **Apply Lab 2.4.6** - Fine-tuning proves these work for real tasks

### Quick Path (6-8 hours)
1. Lab 2.4.1 - Practical benchmark (what, not how)
2. Lab 2.4.3 - MoE is more common in production
3. Lab 2.4.5 - Summary comparison
4. Lab 2.4.6 - Fine-tuning hands-on

### Deep-Dive Path (15+ hours)
1. Implement simplified selective scan from scratch
2. Train a tiny MoE from scratch
3. Analyze expert specialization patterns in detail
4. Compare all architectures on multiple benchmarks

---

## DGX Spark Advantages

| Feature | Transformer | Mamba | MoE |
|---------|-------------|-------|-----|
| Max context (BF16) | ~64K tokens | 100K+ tokens | Varies |
| Max model (BF16) | ~50B params | ~50B params | ~200B total (sparse) |
| Memory pressure | High (KV cache) | Low (constant state) | High (all experts loaded) |

DGX Spark's 128GB memory enables:
- **Mamba**: Process 100K+ token documents without memory issues
- **MoE**: Load full Mixtral 8x7B without quantization
- **Comparison**: Run models side-by-side for proper benchmarking

---

## When to Use Each

| Use Case | Best Architecture | Why |
|----------|-------------------|-----|
| Long documents (>32K tokens) | Mamba | Linear memory |
| Streaming/real-time | Mamba | No KV cache growth |
| General chat/instruction | Transformer | Mature, highest quality |
| Multi-domain tasks | MoE | Expert specialization |
| Cost-sensitive at scale | MoE | Lower compute per token |
| Maximum quality | Transformer or Hybrid | Still SOTA overall |

---

## Before You Start

- See [PREREQUISITES.md](./PREREQUISITES.md) for skill self-check
- See [LAB_PREP.md](./LAB_PREP.md) for environment setup
- See [QUICKSTART.md](./QUICKSTART.md) for 5-minute first comparison
- See [ELI5.md](./ELI5.md) for intuitive explanations

---

## Common Challenges

| Challenge | Solution |
|-----------|----------|
| "Mamba not found in transformers" | Upgrade: `pip install transformers>=4.46.0` |
| "MoE OOM error" | Use 8-bit quantization: `load_in_8bit=True` |
| "Mamba slow first token" | Expected - initialization overhead; subsequent tokens fast |
| "Can't access router weights" | Check model structure: `model.model.layers[N].block_sparse_moe.gate.weight` |
| "Benchmark results inconsistent" | Warm up with 2-3 runs before measuring |

---

## Success Metrics

You've mastered this module when you can:

- [ ] Explain why Mamba is O(n) instead of O(n²)
- [ ] Describe what the "selective" in selective state space means
- [ ] Explain how MoE activates only a fraction of parameters
- [ ] Choose the right architecture for a given use case
- [ ] Fine-tune Mamba on a custom task
