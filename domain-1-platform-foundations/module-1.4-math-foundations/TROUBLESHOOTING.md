# Module 1.4: Mathematics for Deep Learning - Troubleshooting

## Common Issues and Solutions

### Gradient Mismatch Errors

#### Issue: Manual gradients don't match PyTorch autograd

**Symptoms:**
```
❌ Gradients don't match. Check your calculations!
Max difference: 0.01234
```

**Causes and Solutions:**

1. **Wrong chain rule order**
   ```python
   # ❌ Wrong: Incorrect matrix multiplication order
   dL_dW = dz_dW * dL_dz

   # ✅ Correct: Proper order for matrix dimensions
   dL_dW = h_prev.T @ dL_dz
   ```

2. **Missing batch dimension handling**
   ```python
   # ❌ Wrong: Bias gradient without summing over batch
   dL_db = dL_dz  # Shape: (batch_size, hidden_size)

   # ✅ Correct: Sum over batch dimension
   dL_db = np.sum(dL_dz, axis=0, keepdims=True)
   ```

3. **Forgetting to normalize by batch size**
   ```python
   # ❌ Wrong: Not dividing by batch size for MSE
   dL_dy_hat = 2 * (y_hat - y)

   # ✅ Correct: Include batch normalization
   dL_dy_hat = 2 * (y_hat - y) / batch_size
   ```

---

### Numerical Stability Issues

#### Issue: NaN values in loss or gradients

**Symptoms:**
```
Loss: nan
Warning: overflow encountered in exp
```

**Solutions:**

1. **Softmax overflow**
   ```python
   # ❌ Wrong: Direct exponential can overflow
   exp_z = np.exp(z)

   # ✅ Correct: Subtract max for stability
   z_shifted = z - np.max(z, axis=-1, keepdims=True)
   exp_z = np.exp(z_shifted)
   ```

2. **Log of zero in cross-entropy**
   ```python
   # ❌ Wrong: No protection against log(0)
   loss = -np.sum(y * np.log(p))

   # ✅ Correct: Add small epsilon
   eps = 1e-10
   loss = -np.sum(y * np.log(p + eps))
   ```

3. **Sigmoid saturation**
   ```python
   # ❌ Wrong: Large values cause underflow
   sigmoid = 1 / (1 + np.exp(-z))

   # ✅ Correct: Clip input range
   z = np.clip(z, -500, 500)
   sigmoid = 1 / (1 + np.exp(-z))
   ```

---

### Optimizer Convergence Issues

#### Issue: Loss not decreasing

**Symptoms:**
```
Epoch 1000: Loss = 0.693147  # Same as initial!
```

**Solutions:**

1. **Learning rate too small**
   ```python
   # ❌ Wrong: Too conservative
   optimizer = SGD(lr=0.00001)

   # ✅ Try: Larger learning rate
   optimizer = SGD(lr=0.01)  # Start here, adjust
   ```

2. **Learning rate too large (oscillating)**
   ```python
   # ❌ Wrong: Overshooting minimum
   optimizer = SGD(lr=10.0)

   # ✅ Correct: More conservative
   optimizer = SGD(lr=0.001)
   ```

3. **Adam t starting at 0**
   ```python
   # ❌ Wrong: Division by zero in bias correction
   self.t = 0
   m_hat = self.m / (1 - self.beta1 ** self.t)  # Division by 0!

   # ✅ Correct: Start at 1 before first step
   self.t = 0
   # In step():
   self.t += 1  # Now t=1 before division
   m_hat = self.m / (1 - self.beta1 ** self.t)
   ```

---

### Visualization Issues

#### Issue: Plots not displaying

**Symptoms:**
```
<Figure size 1400x500 with 2 Axes>
# But no actual plot shown
```

**Solutions:**

1. **Missing matplotlib inline**
   ```python
   # Add at start of notebook
   %matplotlib inline
   import matplotlib.pyplot as plt
   ```

2. **Missing plt.show()**
   ```python
   # ❌ Wrong: No show call
   plt.plot(losses)

   # ✅ Correct: Explicitly show
   plt.plot(losses)
   plt.show()
   ```

#### Issue: 3D plots look distorted

**Solution:**
```python
# Adjust figure size and viewing angle
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=30, azim=45)  # Adjust angles
```

---

### Import Errors

#### Issue: ModuleNotFoundError for scripts

**Symptoms:**
```
ModuleNotFoundError: No module named 'scripts'
```

**Solution:**
```python
import sys

# Option 1: Relative path (if in labs/ directory)
sys.path.insert(0, '..')

# Option 2: Absolute path (more reliable)
sys.path.insert(0, '/workspace/domain-1-platform-foundations/module-1.4-math-foundations/scripts')

from math_utils import sigmoid, Adam
```

#### Issue: scikit-learn not found

**Symptoms:**
```
ModuleNotFoundError: No module named 'sklearn'
```

**Solution:**
```bash
pip install scikit-learn
```

Or work around it:
```python
# Lab 1.4.3 works without PCA, just with reduced functionality
SKLEARN_AVAILABLE = False  # Skip PCA visualization
```

---

### SVD Issues

#### Issue: SVD taking too long

**Symptoms:**
- Computation hangs on large matrices

**Solutions:**

1. **Use reduced SVD**
   ```python
   # ❌ Slow: Full SVD
   U, S, Vt = np.linalg.svd(W, full_matrices=True)

   # ✅ Fast: Reduced SVD
   U, S, Vt = np.linalg.svd(W, full_matrices=False)
   ```

2. **For very large matrices, use truncated SVD**
   ```python
   from sklearn.decomposition import TruncatedSVD
   svd = TruncatedSVD(n_components=100)
   svd.fit(W)
   ```

---

### Memory Issues

#### Issue: Out of memory during loss landscape computation

**Symptoms:**
```
MemoryError: Unable to allocate X GiB
```

**Solutions:**

1. **Reduce resolution**
   ```python
   # ❌ Too fine
   resolution = 500

   # ✅ Reasonable
   resolution = 50  # Start here
   ```

2. **Clear previous computations**
   ```python
   import gc
   import torch

   if torch.cuda.is_available():
       torch.cuda.empty_cache()
   gc.collect()
   ```

---

### Probability/Loss Function Issues

#### Issue: Cross-entropy loss is negative

**Symptoms:**
```
Loss: -0.523  # Should be positive!
```

**Cause:** Missing negative sign in implementation

```python
# ❌ Wrong: Missing negative
loss = np.sum(y_true * np.log(y_pred))

# ✅ Correct: Include negative
loss = -np.sum(y_true * np.log(y_pred))
```

#### Issue: KL divergence is negative

**Symptoms:**
```
KL Divergence: -0.1  # Should be >= 0!
```

**Cause:** Arguments swapped

```python
# ❌ Wrong: KL(Q||P) instead of KL(P||Q)
kl = np.sum(q * np.log(q / p))

# ✅ Correct: KL(P||Q)
kl = np.sum(p * np.log(p / q))
```

---

## Frequently Asked Questions

### General Questions

**Q: Do I need to be good at math to complete this module?**

You need basic calculus (derivatives, chain rule) and linear algebra (matrix multiplication). If you can compute d/dx[x²] = 2x and multiply two matrices, you have the prerequisites. The module builds intuition—you don't need to prove theorems.

---

**Q: Why implement backpropagation manually when PyTorch does it automatically?**

Three reasons:

1. **Debugging:** When your model doesn't learn, understanding backprop helps you diagnose why (vanishing gradients, wrong loss function, etc.)

2. **Custom layers:** If you need to implement a novel operation, you may need to define its gradient

3. **Intuition:** Knowing what `loss.backward()` actually does makes you a better practitioner

---

**Q: How is this connected to actual LLM training?**

Everything in this module scales directly:

| This Module | LLM Training |
|-------------|--------------|
| 3-layer MLP backprop | Transformer backprop (same chain rule) |
| Adam optimizer | AdamW (same algorithm + weight decay) |
| SVD for LoRA intuition | LoRA fine-tuning of 70B+ models |
| Cross-entropy loss | Next-token prediction loss |

---

### Backpropagation (Lab 1.4.1)

**Q: Why do we multiply matrices in a specific order during backprop?**

Matrix dimensions must align. For a layer z = Wx + b:

- Forward: W is (out_dim, in_dim), x is (in_dim, 1) → z is (out_dim, 1)
- Backward: To get dL/dW with shape (out_dim, in_dim), we compute:
  - dL/dz @ x.T = (out_dim, 1) @ (1, in_dim) = (out_dim, in_dim) ✓

---

**Q: What's the difference between gradient and derivative?**

They're the same concept in different dimensions:

- **Derivative:** Rate of change for a function of ONE variable (f: R → R)
- **Gradient:** Vector of partial derivatives for a function of MULTIPLE variables (f: Rⁿ → R)

In neural networks, we have many parameters, so we use "gradient" (the vector of all partial derivatives).

---

**Q: Why does ReLU work when its derivative is 0 or 1?**

The derivative being exactly 0 for negative inputs means those neurons don't contribute to learning for that input—but they might for other inputs where they're positive. This "dying ReLU" issue is why variants like LeakyReLU exist, but standard ReLU works remarkably well in practice.

---

### Optimizers (Lab 1.4.2)

**Q: Why is Adam the default optimizer for most projects?**

Adam combines two powerful ideas:

1. **Momentum:** Smooths out noisy gradients by averaging
2. **Adaptive learning rates:** Different learning rates per parameter

This means it works well "out of the box" without extensive learning rate tuning. For transformers, AdamW (Adam with decoupled weight decay) is even better.

---

**Q: What learning rate should I use?**

Rules of thumb:

| Optimizer | Starting Learning Rate |
|-----------|----------------------|
| SGD | 0.01 - 0.1 |
| SGD + Momentum | 0.01 - 0.1 |
| Adam/AdamW | 0.001 (1e-3) |

For transformers specifically: 1e-4 to 3e-4 for pretraining, 1e-5 to 3e-5 for fine-tuning.

---

**Q: What's the difference between Adam and AdamW?**

Weight decay implementation:

- **Adam:** Weight decay is added to the gradient (L2 regularization)
- **AdamW:** Weight decay is applied directly to weights (decoupled)

AdamW is generally better for transformers. The difference matters most with adaptive learning rates.

---

### Loss Landscapes (Lab 1.4.3)

**Q: Are real neural network loss landscapes actually this smooth?**

No! Real landscapes are:

- Much higher dimensional (millions of parameters)
- More complex with many local minima
- Surprisingly, often still navigable due to "mode connectivity"

The 2D visualizations are projections that help build intuition but don't capture full complexity.

---

**Q: Why do wider networks have smoother landscapes?**

More parameters = more "directions to escape" local minima. Research shows:

1. Overparameterized networks have more paths to good solutions
2. Skip connections (ResNet) make landscapes dramatically smoother
3. This partly explains why bigger models are easier to train

---

### SVD and LoRA (Lab 1.4.4)

**Q: How does LoRA achieve 96%+ parameter reduction?**

The key insight: weight UPDATES during fine-tuning are low-rank.

- Original weight W: 768×768 = 590,000 parameters
- LoRA: B (768×16) + A (16×768) = 24,576 parameters
- We only train B and A, not W

The math: If the "direction" of adaptation is low-dimensional (rank 16 captures it), we don't need all 590K parameters.

---

**Q: What rank should I use for LoRA?**

Common choices:

| Task | Typical Rank |
|------|--------------|
| Simple adaptation | 4-8 |
| General fine-tuning | 16-32 |
| Complex tasks | 64-128 |

Start with rank=16 and adjust based on results. Higher rank = more capacity but more parameters.

---

**Q: Why initialize B to zero in LoRA?**

So the model starts identical to the pretrained version:

- W_new = W_original + B @ A
- If B = 0, then W_new = W_original
- Training gradually moves away from the pretrained initialization

This is crucial for stable fine-tuning—you don't want random changes at the start.

---

### Probability (Lab 1.4.5)

**Q: Why is cross-entropy better than MSE for classification?**

Gradient properties:

- **MSE gradient:** 2(p - y) — small when p is far from y and wrong!
- **Cross-entropy gradient:** (p - y) / (p(1-p)) — LARGE when p is wrong

Cross-entropy provides stronger learning signal when the model is very wrong, leading to faster initial learning.

---

**Q: What's the connection between temperature and softmax?**

Temperature scales logits before softmax:

```python
probs = softmax(logits / temperature)
```

- Temperature = 1: Normal softmax
- Temperature < 1: Sharper distribution (more confident)
- Temperature > 1: Flatter distribution (more random)

This is used in LLM sampling to control creativity vs. determinism.

---

**Q: Why do we use KL divergence instead of just comparing probabilities?**

KL divergence has information-theoretic meaning:

- It measures the "extra bits" needed to encode data from P using a code optimized for Q
- It's asymmetric: KL(P||Q) ≠ KL(Q||P), which captures that mistaking rare events for common is worse than vice versa
- It's always non-negative (≥ 0), with 0 only when P = Q

---

## Getting Help

If you're still stuck:

1. **Check the solution notebooks** in the `solutions/` directory
2. **Review the ELI5 explanations** in [ELI5.md](./ELI5.md)
3. **Reference the formulas** in [QUICK_REFERENCE.md](./QUICK_REFERENCE.md)
4. **Verify gradients numerically** using the gradient checking function:
   ```python
   from math_utils import numerical_gradient, check_gradient

   num_grad = numerical_gradient(f, x)
   passed, diff = check_gradient(analytical_grad, num_grad)
   print(f"Gradient check: {'PASSED' if passed else 'FAILED'}, diff={diff:.2e}")
   ```
