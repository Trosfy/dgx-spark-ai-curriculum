# Prerequisites - Module 1.5: Neural Network Fundamentals

**Module:** 1.5 - Neural Network Fundamentals
**Domain:** 1 - Platform Foundations

---

## Required Prerequisites

Before starting this module, ensure you have completed:

### Module 1.2: Python for AI/ML
- [ ] Comfortable with NumPy array operations
- [ ] Can perform matrix multiplication with `@` operator
- [ ] Understand broadcasting rules
- [ ] Can slice and reshape arrays

### Module 1.4: Mathematics for Deep Learning
- [ ] Understand derivatives and chain rule
- [ ] Comfortable with matrix calculus basics
- [ ] Understand probability distributions
- [ ] Know mean, variance, and standard deviation

---

## Self-Assessment Checklist

Test your readiness with these quick checks:

### NumPy Proficiency

<details>
<summary>Can you create a 3x4 matrix of random numbers between 0 and 1?</summary>

```python
import numpy as np
matrix = np.random.rand(3, 4)
# or: np.random.random((3, 4))
```

</details>

<details>
<summary>What is the shape after multiplying a (32, 784) matrix with a (784, 256) matrix?</summary>

The result is **(32, 256)** - batch of 32 samples, each with 256 features.

```python
A = np.random.randn(32, 784)
B = np.random.randn(784, 256)
C = A @ B  # Shape: (32, 256)
```

</details>

<details>
<summary>How do you compute the mean across rows vs columns?</summary>

```python
arr = np.random.randn(32, 10)

# Mean across columns (for each row) - axis=1
row_means = arr.mean(axis=1)  # Shape: (32,)

# Mean across rows (for each column) - axis=0
col_means = arr.mean(axis=0)  # Shape: (10,)
```

</details>

### Calculus Understanding

<details>
<summary>What is the derivative of f(x) = x²?</summary>

**f'(x) = 2x**

This is the power rule: d/dx(x^n) = n * x^(n-1)

</details>

<details>
<summary>If y = f(g(x)), what is dy/dx using the chain rule?</summary>

**dy/dx = f'(g(x)) · g'(x)**

Example: If y = (x² + 1)³, then:
- g(x) = x² + 1, g'(x) = 2x
- f(u) = u³, f'(u) = 3u²
- dy/dx = 3(x² + 1)² · 2x = 6x(x² + 1)²

</details>

<details>
<summary>Why is the chain rule essential for neural networks?</summary>

**Backpropagation IS the chain rule!**

A neural network is a composition of functions:
- y = f₃(f₂(f₁(x)))

To find how changing x affects y, we need:
- dy/dx = f₃'(·) · f₂'(·) · f₁'(x)

This is exactly what backpropagation computes!

</details>

---

## Technical Requirements

### Software Environment
- Python 3.10+
- NumPy 1.24+
- Matplotlib 3.7+
- NGC PyTorch container for Lab 1.5.6

### Hardware
- DGX Spark (recommended)
- Or any system with 8GB+ RAM
- GPU optional for Labs 1-5, required for Lab 6

---

## Recommended Preparation

If you feel uncertain about any prerequisites:

1. **NumPy refresher**: Review Module 1.2 notebooks on array operations
2. **Calculus review**: Watch [3Blue1Brown's Essence of Calculus](https://www.3blue1brown.com/topics/calculus) series
3. **Linear algebra**: Review matrix multiplication rules

---

## Module Connections

```
Module 1.2: Python for AI/ML
         ↓
Module 1.4: Mathematics for Deep Learning
         ↓
    [YOU ARE HERE]
    Module 1.5: Neural Network Fundamentals
         ↓
Module 1.6: Classical ML Foundations
         ↓
Module 2.1: Deep Learning with PyTorch
```

---

## Time Estimate

- **Total module time:** 8-10 hours
- **Lab 1:** 2 hours (NumPy neural network)
- **Lab 2:** 1.5 hours (Activation functions)
- **Lab 3:** 2 hours (Regularization)
- **Lab 4:** 2 hours (Normalization)
- **Lab 5:** 2 hours (Training diagnostics)
- **Lab 6:** 2 hours (GPU acceleration)

---

*Prerequisites verified for DGX Spark AI Curriculum v2.0*
