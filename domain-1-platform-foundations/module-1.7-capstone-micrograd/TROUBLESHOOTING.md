# Module 1.7: MicroGrad+ Troubleshooting Guide

This guide expands on common issues you may encounter while building MicroGrad+ and provides detailed solutions.

---

## Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Gradient explosion | Loss becomes NaN or Inf | Use gradient clipping or lower learning rate |
| Gradients all zero | Model not learning | Check for dying ReLU, verify backward implementation |
| Wrong gradient shape | Shape mismatch errors | Ensure broadcasting handled correctly in backward pass |
| Numerical instability | NaN in softmax/log | Use log-sum-exp trick for numerical stability |
| Memory issues | Out of memory errors | Clear intermediate tensors, reduce batch size |

---

## Detailed Solutions

### Gradient Explosion

**Problem:** During training, loss suddenly becomes `NaN` or `Inf`.

**Cause:** Gradients growing exponentially through deep networks or with large learning rates.

**Solutions:**

1. **Lower learning rate:**
```python
# Instead of
optimizer = Adam(model.parameters(), lr=0.01)

# Try
optimizer = Adam(model.parameters(), lr=0.001)
```

2. **Add gradient clipping:**
```python
def clip_gradients(params, max_norm=1.0):
    """Clip gradients to prevent explosion."""
    total_norm = 0
    for p in params:
        if p.grad is not None:
            total_norm += np.sum(p.grad ** 2)
    total_norm = np.sqrt(total_norm)

    if total_norm > max_norm:
        scale = max_norm / total_norm
        for p in params:
            if p.grad is not None:
                p.grad *= scale
```

3. **Use proper weight initialization:**
```python
# Xavier/Glorot initialization (already in Linear layer)
scale = np.sqrt(2.0 / (in_features + out_features))
self.weight = Tensor(np.random.randn(in_features, out_features) * scale, requires_grad=True)
```

---

### Dying ReLU Problem

**Problem:** Gradients are all zero and model stops learning.

**Cause:** ReLU neurons can "die" when they always output 0 (input always negative).

**Symptoms:**
```python
# Check for dead neurons
activations = model[1](model[0](x))  # After first Linear + ReLU
dead_neurons = np.mean(activations.data == 0, axis=0)
print(f"Dead neurons: {np.sum(dead_neurons > 0.9)}")  # Neurons dead >90% of time
```

**Solutions:**

1. **Lower learning rate** (prevents weights from going too negative)

2. **Use Leaky ReLU instead:**
```python
def leaky_relu(self, alpha=0.01):
    """Leaky ReLU: max(alpha*x, x)"""
    out = Tensor(
        np.where(self.data > 0, self.data, alpha * self.data),
        requires_grad=self.requires_grad,
        _children=(self,),
        _op='leaky_relu'
    )

    def _backward():
        if self.requires_grad:
            self.grad += np.where(self.data > 0, 1, alpha) * out.grad
    out._backward = _backward
    return out
```

3. **Better weight initialization** (keep initial weights small but positive-biased)

---

### Wrong Gradient Shape (Broadcasting Issues)

**Problem:** Shape mismatch errors during backward pass.

**Cause:** When operations broadcast tensors, gradients must be "unbroadcast" back to original shapes.

**Example Problem:**
```python
a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # Shape: (2, 3)
b = Tensor([10, 20, 30], requires_grad=True)  # Shape: (3,)
c = a + b  # Shape: (2, 3) - b was broadcast!

c.sum().backward()
# b.grad should be shape (3,), not (2, 3)!
```

**Solution:** The `_unbroadcast` helper function sums over broadcasted dimensions:
```python
def _unbroadcast(grad, shape):
    """Reverse broadcasting by summing along broadcasted dimensions."""
    # Sum along dimensions that were added (leading dimensions)
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    # Sum along dimensions that were broadcasted (size 1 -> size n)
    for i, (grad_dim, shape_dim) in enumerate(zip(grad.shape, shape)):
        if shape_dim == 1 and grad_dim != 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad
```

---

### Numerical Instability in Softmax

**Problem:** `NaN` values when computing softmax or log-softmax.

**Cause:** Exponentiating large numbers causes overflow.

**Bad Implementation:**
```python
# WRONG - will overflow for large inputs
def softmax_bad(x):
    exp_x = np.exp(x)  # Can overflow!
    return exp_x / np.sum(exp_x)
```

**Good Implementation (Numerically Stable):**
```python
def softmax_stable(x, axis=-1):
    """Numerically stable softmax."""
    # Subtract max for numerical stability
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
```

This is already implemented in MicroGrad+:
```python
# In tensor.py - softmax method
shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
exp_x = np.exp(shifted)
softmax_out = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
```

---

### Memory Issues

**Problem:** Running out of memory during training.

**Solutions:**

1. **Reduce batch size:**
```python
# Instead of
BATCH_SIZE = 256

# Try
BATCH_SIZE = 32
```

2. **Clear variables after use:**
```python
import gc

# After each epoch
del logits, loss
gc.collect()
```

3. **Use the cleanup utility:**
```python
from micrograd_plus.utils import cleanup_notebook
cleanup_notebook(globals())
```

4. **Process in smaller chunks for evaluation:**
```python
# Instead of evaluating entire test set at once
def evaluate_chunked(model, X, y, chunk_size=256):
    correct = 0
    for i in range(0, len(X), chunk_size):
        chunk_x = Tensor(X[i:i+chunk_size])
        chunk_y = y[i:i+chunk_size]
        preds = np.argmax(model(chunk_x).data, axis=1)
        correct += np.sum(preds == chunk_y)
    return correct / len(X)
```

---

### Backward Called on Non-Scalar

**Problem:** RuntimeError when calling `backward()`.

**Symptom:**
```
RuntimeError: Gradient must be specified for non-scalar outputs.
```

**Cause:** `backward()` requires a scalar (single value) to start backpropagation.

**Wrong:**
```python
x = Tensor([1, 2, 3], requires_grad=True)
y = x ** 2  # y is a vector!
y.backward()  # ERROR!
```

**Correct:**
```python
x = Tensor([1, 2, 3], requires_grad=True)
y = x ** 2
loss = y.sum()  # Reduce to scalar
loss.backward()  # Works!
```

Or use `mean()`:
```python
loss = y.mean()
loss.backward()
```

---

### Gradient Accumulation (Forgetting to Zero)

**Problem:** Gradients keep growing with each iteration.

**Cause:** Gradients accumulate by design (useful for some cases), but you need to reset them.

**Wrong:**
```python
for epoch in range(10):
    loss = compute_loss()
    loss.backward()  # Gradients accumulate!
    optimizer.step()
```

**Correct:**
```python
for epoch in range(10):
    optimizer.zero_grad()  # Reset gradients first!
    loss = compute_loss()
    loss.backward()
    optimizer.step()
```

---

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'micrograd_plus'`

**Solution:** Ensure you're in the correct directory or add to path:
```python
import sys
from pathlib import Path

# Add module directory to path
sys.path.insert(0, str(Path.cwd().parent))

from micrograd_plus import Tensor
```

Or run from the module root directory:
```bash
cd module-1.7-capstone-micrograd
python -c "from micrograd_plus import Tensor; print('Success!')"
```

---

## Verification Commands

### Test Tensor Operations
```python
from micrograd_plus import Tensor

# Basic test
a = Tensor([2.0], requires_grad=True)
b = Tensor([3.0], requires_grad=True)
c = a * b + a
c.backward()

assert abs(a.grad[0] - 4.0) < 1e-6, f"Expected 4.0, got {a.grad[0]}"
assert abs(b.grad[0] - 2.0) < 1e-6, f"Expected 2.0, got {b.grad[0]}"
print("Basic autograd test passed!")
```

### Run Test Suite
```bash
cd module-1.7-capstone-micrograd
python -m pytest tests/ -v
```

### Check Gradient Correctness
```python
from micrograd_plus import Tensor
from micrograd_plus.utils import gradient_check

x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
passed, error = gradient_check(lambda t: (t ** 2).sum(), x)
print(f"Gradient check: {'PASS' if passed else 'FAIL'} (error: {error:.2e})")
```

---

## Getting Help

If you're still stuck:

1. **Review the notebooks** - Labs 1.7.1-1.7.6 have detailed explanations
2. **Check the source code** - `micrograd_plus/*.py` files are well-documented
3. **Run the tests** - `python -m pytest tests/ -v` to see working examples
4. **Review Karpathy's micrograd** - [github.com/karpathy/micrograd](https://github.com/karpathy/micrograd)
