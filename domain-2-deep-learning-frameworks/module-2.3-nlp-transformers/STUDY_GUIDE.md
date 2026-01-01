# Module 2.3: NLP & Transformers - Study Guide

## Learning Objectives

By the end of this module, you will be able to:

1. **Implement self-attention** from scratch with visualization
2. **Build a Transformer encoder** with all components (attention, FFN, LayerNorm)
3. **Understand positional encodings** (sinusoidal, RoPE, ALiBi)
4. **Train a BPE tokenizer** from scratch on custom data
5. **Fine-tune BERT** for text classification
6. **Implement decoding strategies** (greedy, beam search, sampling)

---

## Module Roadmap

| # | Lab | Focus | Time | Key Outcome |
|---|-----|-------|------|-------------|
| 1 | lab-2.3.1-attention-from-scratch.ipynb | Attention mechanisms | ~2 hr | Multi-head attention implementation |
| 2 | lab-2.3.2-transformer-block.ipynb | Transformer architecture | ~2 hr | Complete encoder block |
| 3 | lab-2.3.3-positional-encoding-study.ipynb | Position representations | ~2 hr | Sinusoidal and RoPE |
| 4 | lab-2.3.4-tokenizer-training.ipynb | Tokenization | ~3 hr | BPE tokenizer from scratch |
| 5 | lab-2.3.5-bert-fine-tuning.ipynb | BERT fine-tuning | ~2 hr | Sentiment classifier |
| 6 | lab-2.3.6-gpt-text-generation.ipynb | Text generation | ~2 hr | Multiple decoding strategies |

**Total time**: ~12-15 hours

---

## Core Concepts

### Self-Attention
**What**: Each position attends to all positions to compute a weighted representation
**Why it matters**: Enables capturing long-range dependencies (unlike RNNs that struggle with distance)
**First appears in**: Lab 2.3.1

### Multi-Head Attention
**What**: Multiple attention "heads" operating in parallel, each learning different patterns
**Why it matters**: One head might learn syntax, another semantics, another coreference
**First appears in**: Lab 2.3.1

### Positional Encoding
**What**: Information about token position added to embeddings
**Why it matters**: Attention is permutation-invariant; without position info, "dog bites man" = "man bites dog"
**First appears in**: Lab 2.3.3

### Tokenization (BPE)
**What**: Breaking text into subword units using learned vocabulary
**Why it matters**: Balances vocabulary size with sequence length; handles unknown words
**First appears in**: Lab 2.3.4

### Pre-trained Models (BERT, GPT)
**What**: Models trained on massive text corpora, ready for fine-tuning
**Why it matters**: Transfer learning for NLP - don't train from scratch
**First appears in**: Lab 2.3.5

---

## How This Module Connects

```
Previous                    This Module                 Next
─────────────────────────────────────────────────────────────
Module 2.2              ►   Module 2.3              ►  Module 2.4
Computer Vision             NLP & Transformers          Efficient Arch
(ViT attention)             (full transformer)          (Mamba, MoE)
```

**Builds on**:
- **Vision Transformer (ViT)** from Module 2.2 - you've seen attention for images
- **nn.Module patterns** from Module 2.1 - now build more complex modules
- **Skip connections** from Module 2.2 - transformers use them extensively

**Prepares for**:
- **Module 2.4** will explore alternatives to quadratic attention (Mamba)
- **Module 2.5** will use HF Transformers library (you'll know what's under the hood)
- **Module 3.1** will fine-tune LLMs (builds directly on this)
- **Module 3.5** will use transformers for RAG systems

---

## Recommended Approach

### Standard Path (12-15 hours)
1. **Start with Lab 2.3.1** - Attention is THE core concept; understand it deeply
2. **Build Lab 2.3.2** - See how attention fits into full transformer
3. **Complete Lab 2.3.3** - Positional encoding often confuses people; master it
4. **Work through Lab 2.3.4** - Tokenization is practical and important
5. **Apply Lab 2.3.5** - BERT fine-tuning is a key practical skill
6. **Finish Lab 2.3.6** - Generation strategies for LLM applications

### Quick Path (6-8 hours, if comfortable with attention)
1. Skim Lab 2.3.1 - Focus on multi-head implementation
2. Focus on Lab 2.3.3 - RoPE is used in modern LLMs (Llama, etc.)
3. Complete Lab 2.3.4 - Tokenization is often overlooked but crucial
4. Apply Lab 2.3.5 - Practical BERT fine-tuning

### Deep-Dive Path (18+ hours)
1. Implement all attention variants in Lab 2.3.1 (cross-attention, causal)
2. Build encoder AND decoder in Lab 2.3.2
3. Implement ALiBi in Lab 2.3.3
4. Train tokenizer on custom corpus in Lab 2.3.4
5. Fine-tune multiple models in Lab 2.3.5
6. Implement beam search with constraints in Lab 2.3.6

---

## Lab-by-Lab Summary

### Lab 2.3.1: Attention from Scratch
**Goal**: Implement and visualize attention mechanisms
**Key skills**:
- Scaled dot-product attention formula
- Multi-head attention implementation
- Attention masking (causal, padding)
- Attention pattern visualization

### Lab 2.3.2: Transformer Block
**Goal**: Build complete encoder block
**Key skills**:
- Layer normalization (Pre-LN vs Post-LN)
- Feed-forward network (expand → activate → contract)
- Residual connections
- Stacking multiple layers

### Lab 2.3.3: Positional Encoding Study
**Goal**: Understand and implement position representations
**Key skills**:
- Sinusoidal encoding (original transformer)
- Learned embeddings (BERT style)
- RoPE (Rotary Position Embeddings - modern)
- Extrapolation to longer sequences

### Lab 2.3.4: Tokenizer Training
**Goal**: Train BPE tokenizer from scratch
**Key skills**:
- Byte-Pair Encoding algorithm
- Vocabulary construction
- Handling special tokens
- Vocabulary size tradeoffs

### Lab 2.3.5: BERT Fine-tuning
**Goal**: Fine-tune BERT for classification
**Key skills**:
- Loading pre-trained models
- Adding classification head
- Training with HuggingFace
- Evaluation metrics

### Lab 2.3.6: GPT Text Generation
**Goal**: Implement and compare decoding strategies
**Key skills**:
- Greedy decoding
- Beam search
- Top-k and Top-p (nucleus) sampling
- Temperature effects

---

## Architecture Overview

```
Input Tokens: ["The", "cat", "sat"]
       ↓
┌──────────────────────────────────────┐
│ Token Embedding + Positional Encoding │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│         Transformer Block × N         │
│  ┌────────────────────────────────┐  │
│  │ Multi-Head Self-Attention      │  │
│  │ + Residual + LayerNorm         │  │
│  └────────────────────────────────┘  │
│  ┌────────────────────────────────┐  │
│  │ Feed-Forward Network           │  │
│  │ + Residual + LayerNorm         │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
       ↓
   Output (classification head or next token prediction)
```

---

## Before You Start

- See [PREREQUISITES.md](./PREREQUISITES.md) for skill self-check
- See [LAB_PREP.md](./LAB_PREP.md) for environment setup
- See [QUICKSTART.md](./QUICKSTART.md) for 5-minute attention demo
- See [ELI5.md](./ELI5.md) for intuitive explanations
- See [FAQ.md](./FAQ.md) for common questions and answers

---

## Common Challenges

| Challenge | Solution |
|-----------|----------|
| "Attention math is confusing" | Start with small examples (3x3 matrices), visualize everything |
| "Multi-head shapes are wrong" | Draw out dimensions at each step; reshape carefully |
| "Position encoding doesn't seem to matter" | Try shuffling inputs - see how predictions change |
| "BERT fine-tuning accuracy low" | Check learning rate (2e-5 to 5e-5 typical), more epochs |
| "Generation is repetitive" | Increase temperature, use top-p sampling |

---

## Success Metrics

You've mastered this module when you can:

- [ ] Explain attention with a whiteboard (no code)
- [ ] Implement multi-head attention from scratch
- [ ] Explain why transformers need positional encoding
- [ ] Train a tokenizer on new data
- [ ] Fine-tune BERT on a classification task
- [ ] Generate coherent text with proper sampling
