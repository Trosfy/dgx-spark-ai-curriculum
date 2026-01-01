# Module 1.4: Mathematics for Deep Learning - Quick Reference

## 📐 Derivatives Cheatsheet

### Basic Rules
```
d/dx [c] = 0                    (constant)
d/dx [x^n] = n·x^(n-1)          (power rule)
d/dx [e^x] = e^x
d/dx [ln(x)] = 1/x
d/dx [sin(x)] = cos(x)
d/dx [cos(x)] = -sin(x)
```

### Chain Rule
```
d/dx [f(g(x))] = f'(g(x)) · g'(x)

Example: d/dx [sin(x²)] = cos(x²) · 2x
```

### Product & Quotient Rules
```
d/dx [f·g] = f'g + fg'          (product)
d/dx [f/g] = (f'g - fg') / g²   (quotient)
```

## 🧮 Backpropagation Formulas

### Linear Layer: z = Wx + b
```python
# Forward
z = W @ x + b

# Backward (given dL/dz)
dL_dW = dL_dz @ x.T
dL_dx = W.T @ dL_dz
dL_db = dL_dz.sum(axis=0)  # or sum over batch
```

### Activation Functions
| Activation | Forward | Backward (dL/dx given dL/dy) |
|------------|---------|------------------------------|
| ReLU | max(0, x) | dL/dy * (x > 0) |
| Sigmoid | σ(x) = 1/(1+e^-x) | dL/dy * σ(x) * (1-σ(x)) |
| Tanh | tanh(x) | dL/dy * (1 - tanh²(x)) |
| Softmax | e^xi / Σe^xj | (complex, see below) |

### Softmax + Cross-Entropy (Combined)
```python
# Forward
probs = softmax(logits)
loss = -sum(y_true * log(probs))

# Backward (elegant combined gradient!)
dL_dlogits = probs - y_true  # That's it!
```

## 🔄 Optimizer Implementations

### SGD
```python
def sgd_step(param, grad, lr=0.01):
    param = param - lr * grad
    return param
```

### SGD with Momentum
```python
def momentum_step(param, grad, velocity, lr=0.01, beta=0.9):
    velocity = beta * velocity + grad
    param = param - lr * velocity
    return param, velocity
```

### Adam
```python
def adam_step(param, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * grad        # First moment
    v = beta2 * v + (1 - beta2) * grad**2     # Second moment

    m_hat = m / (1 - beta1**t)                # Bias correction
    v_hat = v / (1 - beta2**t)

    param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    return param, m, v
```

## 📊 Loss Functions

### MSE (Regression)
```python
# Forward
loss = mean((y_pred - y_true)**2)

# Backward
dL_dy_pred = 2 * (y_pred - y_true) / n
```

### Cross-Entropy (Classification)
```python
# Forward (with softmax built in)
loss = -sum(y_true * log(softmax(logits)))

# Backward
dL_dlogits = softmax(logits) - y_true
```

### Binary Cross-Entropy
```python
# Forward
loss = -mean(y_true * log(p) + (1-y_true) * log(1-p))

# Backward (w.r.t. p)
dL_dp = (p - y_true) / (p * (1-p))
```

## 🔢 Linear Algebra

### SVD Decomposition
```python
import numpy as np

# Full SVD
U, S, Vt = np.linalg.svd(W, full_matrices=True)

# Reconstruct: W = U @ diag(S) @ Vt
W_reconstructed = U @ np.diag(S) @ Vt

# Low-rank approximation (rank r)
W_approx = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
```

### LoRA Connection
```python
# Original: d×d matrix = d² parameters
W_original = np.random.randn(d, d)

# LoRA: d×r + r×d = 2dr parameters
r = 16  # rank
B = np.random.randn(d, r) * 0.01
A = np.random.randn(r, d)

# Effective weight
W_effective = W_original + B @ A
```

### Memory Savings
| d | Full (d²) | LoRA r=16 (2dr) | Savings |
|---|-----------|-----------------|---------|
| 768 | 590K | 25K | 96% |
| 4096 | 16.8M | 131K | 99.2% |
| 8192 | 67.1M | 262K | 99.6% |

## 📈 Gradient Checking

```python
def numerical_gradient(f, x, eps=1e-5):
    """Finite difference gradient check."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        old = x[idx]

        x[idx] = old + eps
        f_plus = f(x)

        x[idx] = old - eps
        f_minus = f(x)

        grad[idx] = (f_plus - f_minus) / (2 * eps)
        x[idx] = old
        it.iternext()

    return grad

# Usage: Verify analytical gradient
analytical = compute_gradient(x)
numerical = numerical_gradient(f, x)
diff = np.abs(analytical - numerical).max()
assert diff < 1e-5, f"Gradient check failed: {diff}"
```

## 📊 Probability Formulas

### Gaussian Distribution
```python
p(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))

# MLE → MSE loss
log p(x) ∝ -(x-μ)²  →  Minimize MSE
```

### Categorical Distribution
```python
p(x=k) = softmax(logits)[k]

# MLE → Cross-entropy loss
log p(x=k) = log(softmax(logits)[k])  →  Cross-entropy
```

### KL Divergence
```python
KL(P || Q) = sum(P * log(P/Q))
           = sum(P * log(P)) - sum(P * log(Q))
           = -H(P) + H(P,Q)  # Negative entropy + cross-entropy
```

## ⚠️ Common Mistakes

| Mistake | Fix |
|---------|-----|
| Wrong axis in sum | Check batch vs feature dimension |
| Forgetting bias gradient | dL/db = sum(dL/dz) |
| Division by batch size | Be consistent: loss and grads |
| Numerical instability in softmax | Subtract max before exp |
| Adam t=0 causes division by zero | Start t at 1, not 0 |

## 🔗 Quick Links

### Module Labs
- [Lab 1.4.1: Manual Backpropagation](./labs/lab-1.4.1-manual-backpropagation.ipynb) - Chain rule, gradients
- [Lab 1.4.2: Optimizer Implementation](./labs/lab-1.4.2-optimizer-implementation.ipynb) - SGD, Momentum, Adam
- [Lab 1.4.3: Loss Landscape Visualization](./labs/lab-1.4.3-loss-landscape-visualization.ipynb) - 2D/3D plots
- [Lab 1.4.4: SVD for LoRA](./labs/lab-1.4.4-svd-for-lora.ipynb) - Low-rank approximation
- [Lab 1.4.5: Probability Distributions](./labs/lab-1.4.5-probability-distributions.ipynb) - MLE, cross-entropy

### External Resources
- [3Blue1Brown: Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
- [3Blue1Brown: Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [Matrix Calculus for DL](https://explained.ai/matrix-calculus/)
- [Why Momentum Works](https://distill.pub/2017/momentum/)
