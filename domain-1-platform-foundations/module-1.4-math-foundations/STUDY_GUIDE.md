# Module 1.4: Mathematics for Deep Learning - Study Guide

## 🎯 Learning Objectives
By the end of this module, you will be able to:
1. **Compute** partial derivatives for composite functions using the chain rule
2. **Implement** gradient descent variants (SGD, Momentum, Adam) from scratch
3. **Visualize** and interpret loss landscapes
4. **Perform** matrix calculus for backpropagation derivation

## 🗺️ Module Roadmap

| # | Lab | Focus | Time | Key Outcome |
|---|-----|-------|------|-------------|
| 1 | Manual Backpropagation | Chain rule | ~3 hr | Gradients match autograd within 1e-6 |
| 2 | Optimizer Implementation | SGD/Adam | ~2 hr | Convergence comparison plots |
| 3 | Loss Landscape Visualization | 2D/3D plots | ~2 hr | Interactive visualizations |
| 4 | SVD for LoRA | Linear algebra | ~2 hr | Understand low-rank approximation |
| 5 | Probability Distributions | Statistics | ~2 hr | Derive cross-entropy from MLE |

**Total time**: ~8-10 hours

## 🔑 Core Concepts

### Chain Rule
**What**: For y = f(g(x)), the derivative is dy/dx = dy/dg × dg/dx.
**Why it matters**: This IS backpropagation. Every layer passes gradients backward using chain rule.
**First appears in**: Lab 1

### Gradient Descent
**What**: Update parameters by moving opposite to the gradient: θ = θ - α∇L.
**Why it matters**: The core optimization algorithm for all neural network training.
**First appears in**: Lab 2

### Loss Landscape
**What**: Visualization of loss function as a surface over parameter space.
**Why it matters**: Explains why training can get stuck, why learning rate matters, and why optimization is hard.
**First appears in**: Lab 3

### SVD (Singular Value Decomposition)
**What**: Any matrix W can be decomposed as W = UΣV^T.
**Why it matters**: Foundation for LoRA fine-tuning—approximate large weights with low-rank matrices.
**First appears in**: Lab 4

## 🔗 How This Module Connects

```
    Module 1.3              Module 1.4                Module 1.5
    ───────────────────────────────────────────────────────────────
    CUDA Python        ──►   Math for DL         ──►   Neural Networks

    GPU acceleration         Gradients                 Build from scratch
    CuPy arrays              Optimization              Use your math knowledge
                             Loss landscapes           Training diagnostics
```

**Builds on**:
- Module 1.2: NumPy for computations
- High school calculus (derivatives)

**Prepares for**:
- **Module 1.5**: You'll implement these gradients in actual networks
- **Module 1.7**: MicroGrad+ capstone uses all these concepts
- **Module 3.1**: LoRA fine-tuning uses SVD intuition

## 📊 Key Formulas

### Gradient Descent Variants
```
SGD:        θ = θ - α∇L

Momentum:   v = βv + ∇L
            θ = θ - αv

Adam:       m = β₁m + (1-β₁)∇L           # First moment
            v = β₂v + (1-β₂)(∇L)²        # Second moment
            m̂ = m/(1-β₁ᵗ)                # Bias correction
            v̂ = v/(1-β₂ᵗ)
            θ = θ - α·m̂/(√v̂ + ε)
```

### Backprop for Linear Layer (z = Wx + b)
```
dL/dW = dL/dz @ x.T
dL/dx = W.T @ dL/dz
dL/db = sum(dL/dz, axis=0)
```

### SVD Connection to LoRA
```
Full: W ∈ R^(d×d)          → d² parameters
Low-rank: W ≈ BA           → 2dr parameters
         where B ∈ R^(d×r), A ∈ R^(r×d)

For r << d, this is MUCH smaller!
```

## 📖 Recommended Approach

**Standard path** (8-10 hours):
1. Lab 1: Essential—backprop is foundational
2. Lab 2: Implement all three optimizers
3. Lab 3: Visualize to build intuition
4. Lab 4: SVD is key for understanding LoRA
5. Lab 5: Connect probability to loss functions

**Quick path** (if strong math background, 5-6 hours):
1. Focus on Lab 1 backprop implementation
2. Skim Lab 2, ensure you understand Adam
3. Quick look at Lab 3 visualizations
4. Complete Lab 4 SVD for LoRA understanding
5. Skip Lab 5 if comfortable with MLE

## 📋 Before You Start
- See [QUICKSTART.md](./QUICKSTART.md) for 5-minute gradient verification
- See [PREREQUISITES.md](./PREREQUISITES.md) for skill self-assessment
- See [ELI5.md](./ELI5.md) for intuitive explanations
- See [LAB_PREP.md](./LAB_PREP.md) for environment setup
- See [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) for formulas
