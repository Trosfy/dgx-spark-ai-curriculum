# Module 1.3: Mathematics for Deep Learning

**Domain:** 1 - Platform Foundations
**Duration:** Week 3 (8-10 hours)
**Prerequisites:** Module 2 (NumPy proficiency), High school calculus

### Required Packages

This module uses the following Python packages (all included in NGC PyTorch containers):

| Package | Purpose |
|---------|---------|
| NumPy | Core numerical computing |
| PyTorch | Autograd verification, neural networks |
| Matplotlib | Visualizations |
| scikit-learn | PCA for loss landscape visualization (Lab 1.3.3) |

> **Note:** This module focuses on mathematical foundations using NumPy and basic PyTorch.
> GPU acceleration is not heavily used—we're building intuition, not training large models.
> GPU-accelerated training is covered in Module 4 onwards.

---

## Overview

This module builds the mathematical intuition essential for understanding deep learning. You'll manually compute gradients, implement optimization algorithms from scratch, visualize loss landscapes, and connect mathematical concepts to neural network operations.

The goal isn't to become a mathematician—it's to develop intuition that helps you debug models and understand why things work (or don't).

---

## Learning Outcomes

By the end of this module, you will be able to:

- ✅ Compute gradients for neural network operations manually and verify with autograd
- ✅ Explain and implement common optimization algorithms
- ✅ Interpret loss landscapes and understand convergence behavior
- ✅ Apply linear algebra concepts to neural network operations

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 1.3.1 | Compute partial derivatives for composite functions using chain rule | Apply |
| 1.3.2 | Implement gradient descent variants from scratch | Apply |
| 1.3.3 | Visualize and interpret loss landscapes | Analyze |
| 1.3.4 | Perform matrix calculus for backpropagation derivation | Understand |

---

## Topics

### 1.3.1 Linear Algebra for Neural Networks
- Matrix/vector operations and their gradients
- Eigenvalues/eigenvectors (for PCA, weight initialization)
- Singular Value Decomposition (essential for understanding LoRA)
- Tensor operations and reshaping

### 1.3.2 Calculus for Backpropagation
- Chain rule for composite functions
- Partial derivatives and gradients
- Jacobian and Hessian matrices
- Computational graphs

### 1.3.3 Optimization Theory
- Gradient descent and learning rates
- Momentum and adaptive methods (Adam, AdamW)
- Learning rate schedules
- Loss landscape geometry

### 1.3.4 Probability for ML
- Probability distributions (Gaussian, Categorical)
- Maximum likelihood estimation
- Cross-entropy and KL divergence
- Bayesian basics

---

## Labs

### Lab 1.3.1: Manual Backpropagation
**Time:** 3 hours

Implement forward and backward passes from scratch.

**Instructions:**
1. Create a 3-layer MLP (input → hidden → hidden → output)
2. Implement forward pass manually
3. Derive gradients using chain rule
4. Implement backward pass manually
5. Verify gradients match PyTorch autograd (within 1e-6)

**Deliverable:** Notebook with manual backprop matching autograd

---

### Lab 1.3.2: Optimizer Implementation
**Time:** 2 hours

Implement optimization algorithms from scratch.

**Instructions:**
1. Implement vanilla SGD
2. Implement SGD with momentum
3. Implement Adam optimizer
4. Train on a simple problem (e.g., 2D function minimization)
5. Plot convergence curves for all three
6. Compare convergence speed and stability

**Deliverable:** Optimizer comparison notebook with visualizations

---

### Lab 1.3.3: Loss Landscape Visualization
**Time:** 2 hours

Visualize what neural networks are optimizing.

**Instructions:**
1. Train a small network on simple data
2. Create 2D loss landscape (vary 2 parameters, plot loss as heatmap)
3. Create 3D surface plot of loss landscape
4. Overlay optimization trajectory on landscape
5. Identify local minima, saddle points, flat regions

**Deliverable:** Interactive loss landscape visualizations

---

### Lab 1.3.4: SVD for LoRA Intuition
**Time:** 2 hours

Understand low-rank approximations that power LoRA.

**Instructions:**
1. Create a random weight matrix (e.g., 768 × 768)
2. Compute full SVD: W = UΣV^T
3. Reconstruct with varying ranks (r = 1, 4, 16, 64, 256)
4. Visualize reconstruction error vs rank
5. Explain connection to LoRA's W + BA formulation
6. Calculate memory savings at different ranks

**Deliverable:** SVD analysis notebook with LoRA connection

---

### Lab 1.3.5: Probability Distributions Lab
**Time:** 2 hours

Connect probability to loss functions.

**Instructions:**
1. Implement and visualize: Gaussian, Bernoulli, Categorical
2. Derive MSE loss from Gaussian MLE
3. Derive cross-entropy from Categorical MLE
4. Implement KL divergence
5. Show cross-entropy = entropy + KL divergence

**Deliverable:** Probability and loss function derivation notebook

---

## Guidance

### Gradient Checking

Always verify manual gradients with numerical approximation:

```python
def numerical_gradient(f, x, eps=1e-5):
    """Compute numerical gradient for verification"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

# Verify: should be < 1e-6
diff = np.abs(analytical_grad - numerical_grad).max()
```

### Chain Rule for Backprop

For y = f(g(x)):
```
dy/dx = dy/dg * dg/dx
```

For neural network layer z = W @ x + b:
```
dL/dW = dL/dz @ x.T
dL/dx = W.T @ dL/dz
dL/db = dL/dz.sum(axis=0)
```

### SVD Connection to LoRA

```python
# Full matrix: W ∈ R^(d×d), stores d² parameters
W = U @ S @ V.T

# Low-rank: W ≈ U_r @ S_r @ V_r.T, stores 2dr parameters
# LoRA: W_new = W + B @ A, where B ∈ R^(d×r), A ∈ R^(r×d)
# Only train B and A (2dr parameters instead of d²)
```

### Adam Algorithm

```python
def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * grad           # Update biased first moment
    v = beta2 * v + (1 - beta2) * grad**2        # Update biased second moment
    m_hat = m / (1 - beta1**t)                   # Bias correction
    v_hat = v / (1 - beta2**t)                   # Bias correction
    param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    return param, m, v
```

---

## Milestone Checklist

- [ ] Manual backprop implementation matches autograd within 1e-6
- [ ] Three optimizers (SGD, Momentum, Adam) implemented
- [ ] Convergence comparison plot created
- [ ] 2D and 3D loss landscape visualizations complete
- [ ] SVD decomposition notebook with LoRA explanation
- [ ] Cross-entropy derived from maximum likelihood
- [ ] All concepts connected to neural network applications

---

## Next Steps

After completing this module:
1. ✅ Verify all milestones are checked
2. 📝 Review any concepts that felt unclear
3. ➡️ Proceed to [Module 1.4: Neural Network Fundamentals](../module-1.4-neural-networks/)

---

## Resources

- [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [3Blue1Brown: Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
- [Matrix Calculus for Deep Learning](https://explained.ai/matrix-calculus/)
- [Why Momentum Really Works](https://distill.pub/2017/momentum/)
