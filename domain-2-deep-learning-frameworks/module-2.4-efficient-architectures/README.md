# Module 2.4: Efficient Architectures

**Domain:** 2 - Deep Learning Frameworks
**Duration:** Week 13 (10-12 hours)
**Prerequisites:** Module 2.3 (NLP & Transformers)
**Priority:** P1 High

---

## Overview

Transformers revolutionized AI, but their O(n²) attention complexity limits sequence length. This module explores the cutting edge: **Mamba** (State Space Models) with linear complexity and **Mixture of Experts (MoE)** that activates only a fraction of parameters per token.

On DGX Spark's 128GB unified memory, these architectures unlock capabilities impossible on consumer hardware—process 100K+ token contexts with Mamba, or run 200B+ MoE models that only activate 20B parameters per forward pass.

---

## Learning Outcomes

By the end of this module, you will be able to:

- ✅ Explain Mamba (State Space Models) and their advantages over transformers
- ✅ Understand Mixture of Experts (MoE) architecture and routing mechanisms
- ✅ Compare transformer vs alternative architectures on memory, speed, and quality
- ✅ Run and fine-tune Mamba and MoE models on DGX Spark

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 2.4.1 | Explain the selective state space mechanism in Mamba | Understand |
| 2.4.2 | Describe MoE architecture with gating and load balancing | Understand |
| 2.4.3 | Compare memory/compute tradeoffs of different architectures | Analyze |
| 2.4.4 | Run inference and fine-tune Mamba/MoE models | Apply |

---

## Topics

### 2.4.1 Limitations of Transformers

- **Quadratic Complexity**
  - O(n²) attention computation
  - Memory grows with sequence length squared
  - KV cache bloats for long contexts

- **Computational Bottlenecks**
  - 32K context = 1B attention computations per layer
  - GPU memory limits practical context length
  - Why transformers struggle with 100K+ tokens

### 2.4.2 Mamba (State Space Models)

- **Linear Time Complexity**
  - O(n) instead of O(n²)—transformative for long sequences
  - Processes one token at a time, maintaining compressed state
  - No KV cache (constant memory per token!)

- **Selective State Space Mechanism**
  - Input-dependent state transitions (the "selective" part)
  - Hardware-aware parallel scan algorithm
  - Why it rivals transformers on quality

- **Mamba Architecture**
  - S4/S6 building blocks
  - Gated linear units + convolutions
  - Mamba-2 improvements (8× faster training)

- **When to Use Mamba**
  - Long document processing (100K+ tokens)
  - Audio/time-series (1M+ datapoints)
  - Streaming inference scenarios

### 2.4.3 Mixture of Experts (MoE)

- **Sparse Activation**
  - Only 10-20% of parameters active per token
  - 200B total params, 20B active = 10× efficiency
  - Memory for all params, compute for few

- **Router/Gating Mechanisms**
  - Top-k routing (typically k=2)
  - Token-to-expert assignment
  - Learned routing networks

- **Load Balancing**
  - Auxiliary loss for balanced expert usage
  - Preventing expert collapse
  - Capacity factors

- **MoE Architectures**
  - Mixtral 8x7B (45B total, 12B active)
  - DeepSeekMoE-16B (16B total, 2.5B active)
  - Qwen-MoE, Switch Transformers

### 2.4.4 Hybrid Architectures

- **Jamba** (Mamba + Attention)
  - Best of both worlds
  - Attention for precision, Mamba for efficiency
  - IBM Granite 4.0 architecture

- **Efficient Attention Variants**
  - Flash Attention (memory-efficient standard attention)
  - Grouped Query Attention (GQA)
  - Multi-Query Attention (MQA)
  - Sliding window attention

---

## Labs

### Lab 2.4.1: Mamba Inference
**Time:** 2 hours

Run Mamba on DGX Spark and compare with transformers.

**Instructions:**
1. Open `labs/lab-2.4.1-mamba-inference.ipynb`
2. Load Mamba-2.8B using HuggingFace transformers (>=4.46.0)
3. Load comparable transformer model (similar parameters)
4. Benchmark: tok/s (decode speed), memory usage, time-to-first-token
5. Test with increasing context lengths (4K, 8K, 16K, 32K)
6. Document where Mamba's linear scaling shines

**Deliverable:** Benchmark notebook showing Mamba's advantages on long contexts

---

### Lab 2.4.2: Mamba Architecture Study
**Time:** 2 hours

Understand Mamba's selective scan algorithm.

**Instructions:**
1. Open `labs/lab-2.4.2-mamba-architecture.ipynb`
2. Implement simplified selective scan in PyTorch
3. Visualize state evolution across a sequence
4. Compare with attention patterns (what does each "look at"?)
5. Analyze how selectivity enables learning

**Deliverable:** Notebook with selective scan visualization and analysis

---

### Lab 2.4.3: MoE Exploration
**Time:** 2 hours

Run MoE models and analyze expert activation.

**Instructions:**
1. Open `labs/lab-2.4.3-moe-exploration.ipynb`
2. Load DeepSeekMoE-16B on DGX Spark
3. Run inference on diverse prompts (code, math, creative writing)
4. Log which experts activate for each token
5. Analyze: Do certain experts specialize in topics?
6. Document total memory vs active memory

**Deliverable:** Analysis notebook showing expert specialization patterns

---

### Lab 2.4.4: MoE Router Analysis
**Time:** 2 hours

Deep dive into expert selection and load balancing.

**Instructions:**
1. Open `labs/lab-2.4.4-moe-router.ipynb`
2. Extract router weights from an MoE model
3. Visualize expert selection distribution across 1000 tokens
4. Identify any load imbalance
5. Explain the auxiliary loss mechanism
6. Document how top-k routing works

**Deliverable:** Router visualization notebook with load analysis

---

### Lab 2.4.5: Architecture Comparison
**Time:** 2 hours

Systematic benchmark of Mamba vs Transformer vs MoE.

**Instructions:**
1. Open `labs/lab-2.4.5-architecture-comparison.ipynb`
2. Select three models: Mamba-3B, Llama-3B, MoE-7B (similar active)
3. Benchmark on perplexity (wikitext)
4. Benchmark on generation speed
5. Benchmark on memory usage
6. Create comparison report with recommendations

**Deliverable:** Comprehensive comparison report with charts

---

### Lab 2.4.6: Mamba Fine-tuning
**Time:** 2 hours

Fine-tune a Mamba model for a custom task.

**Instructions:**
1. Open `labs/lab-2.4.6-mamba-finetuning.ipynb`
2. Load a small Mamba model (1.4B or 2.8B)
3. Prepare a custom instruction dataset
4. Apply LoRA (yes, LoRA works on Mamba!)
5. Fine-tune and evaluate
6. Compare with transformer fine-tuning memory

**Deliverable:** Fine-tuned Mamba model with evaluation

---

## Guidance

### Why Mamba on DGX Spark?

```
┌────────────────────────────────────────────────────────────────┐
│                    CONTEXT LENGTH LIMITS                        │
├────────────────────────────────────────────────────────────────┤
│ Architecture │ 24GB GPU    │ 128GB DGX Spark                   │
├──────────────┼─────────────┼───────────────────────────────────┤
│ Transformer  │ ~16K tokens │ ~64K tokens (KV cache grows!)     │
│ Mamba        │ ~64K tokens │ 100K+ tokens (constant memory!)   │
└──────────────┴─────────────┴───────────────────────────────────┘
```

### Loading Mamba with HuggingFace

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Requires transformers >= 4.46.0 for Mamba support
model = AutoModelForCausalLM.from_pretrained(
    "state-spaces/mamba-2.8b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")

# Inference is identical to transformers!
inputs = tokenizer("The key insight of Mamba is", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

### MoE Memory Math

```python
# DeepSeekMoE-16B Example
total_params = 16_000_000_000  # 16B total
active_params = 2_500_000_000  # 2.5B active per token
num_experts = 64
experts_per_token = 6  # top-6 routing

# Memory: Need to load ALL 16B params
# Compute: Only process 2.5B params per forward pass
# = 6.4× inference efficiency vs dense 16B model!

# On DGX Spark: Easy to load full model
memory_fp16 = total_params * 2 / 1e9  # ~32 GB
# Plenty of room for batching and long contexts
```

### When to Use Each Architecture

| Use Case | Best Choice | Why |
|----------|-------------|-----|
| Long documents (>32K) | Mamba | Linear memory |
| Real-time streaming | Mamba | No KV cache |
| General chat/instruction | Transformer | Mature ecosystem |
| Maximum quality | Transformer | Still SOTA for most tasks |
| Cost-effective large scale | MoE | Sparse activation |
| Multi-domain tasks | MoE | Expert specialization |

### Hybrid Approach: Jamba

```python
# Jamba: Mamba layers for efficiency, Attention layers for precision
# Architecture: Attention every 8 Mamba layers
# Result: 256K context, competitive quality

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "ai21labs/Jamba-v0.1",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
# On DGX Spark: Fits comfortably with 128GB
```

---

## Milestone Checklist

Use this checklist to track your progress:

- [ ] Mamba inference running with benchmark data
- [ ] Mamba's selective scan mechanism understood and visualized
- [ ] MoE expert activation patterns analyzed
- [ ] Router and load balancing explained
- [ ] Architecture comparison report complete
- [ ] Mamba fine-tuning demonstrated
- [ ] Can explain when to use each architecture type

---

## Common Issues

| Issue | Solution |
|-------|----------|
| `ImportError: Mamba not found` | Upgrade: `pip install transformers>=4.46.0` |
| MoE out of memory | Load with `load_in_8bit=True` or reduce batch size |
| Mamba slow first token | Expected—initialization overhead. Subsequent tokens fast. |
| Router weights not accessible | Use `model.model.layers[N].block_sparse_moe.gate.weight` |
| Jamba requires too much memory | Use 8-bit: `load_in_8bit=True` |

---

## Next Steps

After completing this module:
1. ✅ Verify all milestones are checked
2. 📁 Save benchmark notebooks and analysis
3. ➡️ Proceed to [Module 2.5: Hugging Face Ecosystem](../module-2.5-huggingface/)

---

## Resources

- [Mamba Paper](https://arxiv.org/abs/2312.00752) - "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
- [Mamba-2 Paper](https://arxiv.org/abs/2405.21060) - 8× faster training
- [Mixtral Paper](https://arxiv.org/abs/2401.04088) - Mixture of Experts Made Easy
- [DeepSeekMoE Paper](https://arxiv.org/abs/2401.06066) - Efficient MoE architecture
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135) - Memory-efficient attention
- [Jamba Paper](https://arxiv.org/abs/2403.19887) - Hybrid Mamba-Attention
- [HuggingFace Mamba Guide](https://huggingface.co/docs/transformers/model_doc/mamba)
