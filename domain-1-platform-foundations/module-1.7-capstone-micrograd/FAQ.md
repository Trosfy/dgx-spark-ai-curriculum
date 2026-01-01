# Module 1.7: MicroGrad+ FAQ

Frequently asked questions about the MicroGrad+ capstone project.

---

## General Questions

### Why build an autograd engine from scratch?

**Short answer:** To truly understand how deep learning works under the hood.

**Long answer:** When you call `loss.backward()` in PyTorch, hundreds of things happen automatically. By building MicroGrad+, you'll understand:
- How computation graphs are constructed during the forward pass
- How gradients flow backward through the chain rule
- Why certain operations "break" gradient flow
- What's actually happening when training "diverges" or "explodes"

This knowledge is invaluable for debugging and designing neural networks.

---

### How does MicroGrad+ compare to PyTorch?

| Aspect | MicroGrad+ | PyTorch |
|--------|------------|---------|
| Purpose | Learning | Production |
| Speed | 10-100x slower | Optimized |
| GPU support | No (CPU/NumPy only) | Yes |
| Features | Core autograd only | Full framework |
| Code readability | Very high | More complex |
| Lines of code | ~2,000 | ~2,000,000 |

MicroGrad+ is intentionally simple. PyTorch has the same concepts but adds:
- GPU acceleration via CUDA
- Automatic memory management
- Thousands of operations and layers
- Distributed training support
- Production deployment tools

---

### Is this related to Karpathy's micrograd?

Yes! MicroGrad+ is inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd). Key differences:

| Feature | Original micrograd | MicroGrad+ |
|---------|-------------------|------------|
| Tensors | Scalars only | Full N-D tensors |
| Broadcasting | No | Yes |
| Layers | Basic MLP | Linear, ReLU, Dropout, BatchNorm, etc. |
| Optimizers | SGD only | SGD, Adam, AdamW, RMSprop |
| Loss functions | Basic | MSE, CrossEntropy, BCE, Huber |
| Documentation | Minimal | Extensive (this curriculum) |

We recommend watching [Karpathy's video](https://www.youtube.com/watch?v=VMj-3S1tku0) as a companion resource.

---

## Technical Questions

### Why does backward() require a scalar?

Backpropagation computes how the **loss** affects each parameter. The loss must be a single number (scalar) because:

1. You need a single "starting point" for the backward pass
2. The gradient of a scalar w.r.t. a tensor is well-defined
3. The gradient of a tensor w.r.t. a tensor is more complex (Jacobian)

**Solution:** Always reduce to scalar before calling backward:
```python
# Wrong
y = x ** 2  # Vector
y.backward()  # Error!

# Correct
loss = (x ** 2).sum()  # Scalar
loss.backward()  # Works!
```

---

### Why do gradients accumulate?

Gradients accumulate by design because:

1. **Mini-batch training:** You might want to accumulate gradients over multiple batches before updating
2. **Multiple paths:** When a tensor is used multiple times, gradients from all paths should sum

**Always zero gradients before each iteration:**
```python
for epoch in range(num_epochs):
    optimizer.zero_grad()  # Clear old gradients
    loss = compute_loss()
    loss.backward()
    optimizer.step()
```

---

### What is broadcasting and why is it tricky?

Broadcasting lets you operate on tensors with different shapes:
```python
a = Tensor([[1, 2, 3],
            [4, 5, 6]])  # Shape: (2, 3)
b = Tensor([10, 20, 30])  # Shape: (3,)
c = a + b  # Works! b is "broadcast" to match a
```

**The tricky part:** During backward, we must "unbroadcast" the gradient:
- `c.grad` has shape `(2, 3)`
- `b.grad` should have shape `(3,)` - the original shape

We sum over the broadcasted dimensions to get the correct gradient shape.

---

### Why is my loss NaN?

Common causes:

1. **Gradient explosion:** Learning rate too high
   ```python
   # Try lowering learning rate
   optimizer = Adam(params, lr=0.0001)  # Instead of 0.01
   ```

2. **Log of zero:** In cross-entropy with very wrong predictions
   ```python
   # Add small epsilon for stability
   log_probs = np.log(probs + 1e-10)
   ```

3. **Division by zero:** In normalization layers
   ```python
   # Add epsilon to denominator
   normalized = x / (std + 1e-5)
   ```

4. **Overflow in exp:** Large logits in softmax
   ```python
   # Subtract max for stability (already done in MicroGrad+)
   shifted = logits - np.max(logits, axis=-1, keepdims=True)
   exp_logits = np.exp(shifted)
   ```

---

### Why is training so slow?

MicroGrad+ uses pure NumPy on CPU. It's intentionally simple, not fast.

**Tips to speed up:**
1. Use smaller batch sizes (32-64)
2. Use fewer training samples (10k instead of 60k)
3. Use simpler models (3 layers instead of 10)
4. Train for fewer epochs (10-20)

**For production speed:** Use PyTorch with GPU (covered in Domain 2).

---

### What's the difference between train() and eval() modes?

```python
model.train()  # Training mode
model.eval()   # Evaluation mode
```

| Layer | train() | eval() |
|-------|---------|--------|
| Dropout | Randomly zeros neurons | Does nothing (pass through) |
| BatchNorm | Uses batch statistics | Uses running statistics |

**Always use:**
- `model.train()` before training loops
- `model.eval()` before validation/testing

---

## Conceptual Questions

### What's the difference between SGD and Adam?

**SGD (Stochastic Gradient Descent):**
- Simple: `param = param - lr * gradient`
- Same learning rate for all parameters
- Can get stuck in local minima

**Adam (Adaptive Moment Estimation):**
- Tracks running average of gradients (momentum)
- Tracks running average of squared gradients (adaptive learning rate)
- Each parameter gets its own effective learning rate
- Generally converges faster and more reliably

**When to use which:**
| Scenario | Recommended |
|----------|-------------|
| First try | Adam (lr=0.001) |
| Fine-tuning | SGD with momentum |
| Computer vision | SGD with momentum (often better final accuracy) |
| NLP/Transformers | Adam or AdamW |

---

### Why Xavier/Glorot initialization?

Random weight initialization affects training significantly.

**Too small weights:** Gradients shrink to zero (vanishing gradients)
**Too large weights:** Gradients explode

**Xavier initialization** scales weights based on layer size:
```python
scale = np.sqrt(2.0 / (in_features + out_features))
weights = np.random.randn(in_features, out_features) * scale
```

This keeps gradients roughly the same magnitude across layers.

---

### What's the difference between L1 and L2 loss?

**L1 (Mean Absolute Error):**
```
L1 = mean(|prediction - target|)
```
- Less sensitive to outliers
- Gradient is constant (±1)
- Can have multiple solutions

**L2 (Mean Squared Error):**
```
L2 = mean((prediction - target)²)
```
- More sensitive to outliers (squares the error)
- Gradient proportional to error
- Unique minimum

**When to use:**
| Scenario | Recommended |
|----------|-------------|
| General regression | L2 (MSE) |
| Robust to outliers | L1 or Huber |
| Classification | Cross-Entropy (not L1/L2) |

---

### Why use ReLU instead of sigmoid?

**Sigmoid problems:**
1. Vanishing gradients: For very positive or negative inputs, gradient → 0
2. Not zero-centered: Outputs are always positive (0 to 1)
3. Computationally expensive: Requires exp()

**ReLU advantages:**
1. No vanishing gradient for positive inputs (gradient = 1)
2. Computationally cheap: just max(0, x)
3. Creates sparse representations (many zeros)

**ReLU problems:**
- "Dying ReLU": If neuron outputs always negative, gradient = 0 forever

**Solutions:** Leaky ReLU, ELU, GELU (covered in later modules)

---

## Debugging Questions

### How do I verify my gradients are correct?

Use numerical gradient checking:
```python
from micrograd_plus.utils import gradient_check

x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
passed, error = gradient_check(lambda t: (t ** 2).sum(), x)
print(f"Gradient check: {'PASS' if passed else 'FAIL'}")
print(f"Max error: {error:.2e}")
```

The idea: Compare analytical gradients (from autograd) with numerical gradients (from finite differences).

---

### Why isn't my model learning?

Checklist:
1. **Learning rate:** Too high? Too low? Try 0.001, 0.01, 0.0001
2. **Gradients:** Are they zero? Check `model.parameters()[0].grad`
3. **Mode:** Did you call `model.train()`?
4. **Zero grad:** Are you calling `optimizer.zero_grad()`?
5. **Loss:** Is the loss function appropriate for your task?
6. **Data:** Is your data normalized? Shuffled?

---

### How do I see what's happening during training?

Add verbose output:
```python
for epoch in range(num_epochs):
    train_loss = train_epoch(model, loader, loss_fn, optimizer)

    # Check gradients
    max_grad = max(np.max(np.abs(p.grad)) for p in model.parameters() if p.grad is not None)

    # Check weights
    max_weight = max(np.max(np.abs(p.data)) for p in model.parameters())

    print(f"Epoch {epoch}: loss={train_loss:.4f}, max_grad={max_grad:.4f}, max_weight={max_weight:.4f}")
```

---

## Further Questions?

- Check [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for specific error solutions
- Review the source code in `micrograd_plus/*.py` - it's well documented
- Watch [Karpathy's micrograd video](https://www.youtube.com/watch?v=VMj-3S1tku0)
- Re-read the relevant lab notebook
