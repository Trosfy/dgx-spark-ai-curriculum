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
