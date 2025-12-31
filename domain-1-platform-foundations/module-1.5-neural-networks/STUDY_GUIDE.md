# Module 1.5: Neural Network Fundamentals - Study Guide

## 🎯 Learning Objectives
By the end of this module, you will be able to:
1. **Build** neural networks from scratch using only NumPy
2. **Explain** the purpose of each neural network component
3. **Train** networks on real datasets and diagnose common issues
4. **Implement** regularization and normalization techniques

## 🗺️ Module Roadmap

| # | Lab | Focus | Time | Key Outcome |
|---|-----|-------|------|-------------|
| 1 | NumPy Neural Network | Full implementation | ~4 hr | >95% MNIST accuracy |
| 2 | Activation Function Study | Compare activations | ~2 hr | Understand vanishing gradients |
| 3 | Regularization Experiments | L2, Dropout | ~2 hr | Prevent overfitting |
| 4 | Normalization Comparison | BatchNorm vs LayerNorm | ~2 hr | Faster, stable training |
| 5 | Training Diagnostics | Debug training | ~2 hr | Troubleshooting skills |
| 6 | GPU Acceleration | CPU vs GPU | ~2 hr | Measure DGX Spark speedup |

**Total time**: ~14 hours

## 🔑 Core Concepts

### Activation Functions
**What**: Nonlinear functions applied after each layer (ReLU, GELU, etc.).
**Why it matters**: Without nonlinearity, stacking layers gives no benefit—network would just be one linear transformation.
**First appears in**: Lab 2

### Vanishing/Exploding Gradients
**What**: Gradients become extremely small (vanish) or large (explode) in deep networks.
**Why it matters**: Prevents learning. ReLU and proper initialization solve vanishing; gradient clipping solves exploding.
**First appears in**: Lab 2, Lab 5

### Regularization
**What**: Techniques to prevent overfitting (L2 penalty, Dropout).
**Why it matters**: Without it, networks memorize training data instead of learning patterns.
**First appears in**: Lab 3

### Normalization
**What**: Normalizing activations to have consistent statistics.
**Why it matters**: Faster, more stable training. BatchNorm for CNNs, LayerNorm for Transformers.
**First appears in**: Lab 4

## 🔗 How This Module Connects

```
    Module 1.4              Module 1.5                Module 1.6
    ───────────────────────────────────────────────────────────────
    Math for DL        ──►   Neural Networks    ──►   Classical ML

    Gradients                Build from scratch        Compare approaches
    Optimizers               Training loop             When to use each
    Loss functions           Regularization            Baselines
```

**Builds on**:
- Module 1.2: NumPy for implementation
- Module 1.4: Gradients and optimization

**Prepares for**:
- **Module 1.6**: Understand when neural networks vs classical ML
- **Module 1.7**: Build complete autograd library
- **Domain 2**: PyTorch makes this easier, but you'll understand it deeply

## 📊 Key Comparisons

### Activation Functions
| Function | Pros | Cons |
|----------|------|------|
| Sigmoid | Smooth, bounded | Vanishing gradients |
| ReLU | Fast, no vanishing | Dead neurons |
| GELU | Smooth ReLU | Slightly slower |
| SiLU/Swish | Self-gated | Slightly slower |

### Normalization
| Type | Normalizes Over | Best For |
|------|-----------------|----------|
| BatchNorm | Batch dimension | CNNs, large batches |
| LayerNorm | Feature dimension | Transformers, RNNs |
| RMSNorm | Features (no mean) | Modern LLMs |

### Initialization
| Method | Use When |
|--------|----------|
| Xavier | Sigmoid, Tanh activations |
| He | ReLU activations |

## 📖 Recommended Approach

**Standard path** (14 hours):
1. Lab 1: Essential—build everything from scratch
2. Lab 2: Compare activations empirically
3. Lab 3: Understand regularization
4. Lab 4: Implement normalization
5. Lab 5: Debug training issues
6. Lab 6: See GPU speedup

**Quick path** (if experienced, 7-8 hours):
1. Complete Lab 1 (skip if confident)
2. Skim Lab 2, focus on GELU/SiLU (modern)
3. Focus on Lab 3 Dropout implementation
4. Complete Lab 4 normalization
5. Review Lab 5 diagnostics checklist
6. Quick Lab 6 GPU comparison

## 📋 Before You Start
→ See [QUICKSTART.md](./QUICKSTART.md) for 5-minute Linear + ReLU demo
→ See [ELI5.md](./ELI5.md) for intuitive explanations
→ See [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) for implementation patterns
