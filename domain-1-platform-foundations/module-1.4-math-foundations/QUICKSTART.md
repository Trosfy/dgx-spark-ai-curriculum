# Module 1.4: Mathematics for Deep Learning - Quickstart

## ⏱️ Time: ~5 minutes

## 🎯 What You'll Do
Verify gradients you compute manually match PyTorch's autograd.

## ✅ Before You Start
- [ ] NGC PyTorch container running
- [ ] Basic calculus knowledge (derivatives)

## 🚀 Let's Go!

### Step 1: Define a Simple Function
```python
import torch
import numpy as np

# f(x) = x^2 + 2x + 1
# df/dx = 2x + 2

def f(x):
    return x**2 + 2*x + 1

def df_dx(x):
    return 2*x + 2  # Analytical gradient
```

### Step 2: Compute Gradient Manually
```python
x = 3.0

# Manual derivative
manual_grad = df_dx(x)
print(f"Manual gradient at x={x}: {manual_grad}")
```

**Expected:** `Manual gradient at x=3: 8.0`

### Step 3: Verify with PyTorch Autograd
```python
# PyTorch autograd
x_tensor = torch.tensor(3.0, requires_grad=True)
y = x_tensor**2 + 2*x_tensor + 1
y.backward()

print(f"PyTorch gradient at x=3: {x_tensor.grad.item()}")
print(f"Match: {np.isclose(manual_grad, x_tensor.grad.item())}")
```

**Expected output:**
```
Manual gradient at x=3: 8.0
PyTorch gradient at x=3: 8.0
Match: True
```

### Step 4: Chain Rule Preview
```python
# Nested function: g(f(x)) where f(x)=x^2, g(u)=sin(u)
# dg/dx = dg/df * df/dx = cos(f(x)) * 2x

x = torch.tensor(1.0, requires_grad=True)
y = torch.sin(x**2)
y.backward()

manual = torch.cos(torch.tensor(1.0)**2) * 2 * 1.0  # cos(1) * 2
print(f"Chain rule gradient: {x.grad.item():.4f}")
print(f"Manual calculation: {manual.item():.4f}")
```

**Expected:** Both ~1.0806

## 🎉 You Did It!

You just:
- ✅ Computed derivatives manually
- ✅ Verified with PyTorch autograd
- ✅ Applied the chain rule
- ✅ Saw the foundation of backpropagation

In the full module, you'll learn:
- Manual backprop for entire neural networks
- Optimizer implementations (SGD, Adam)
- Loss landscape visualization
- SVD and its connection to LoRA

## ▶️ Next Steps
1. **Understand the math**: Read [ELI5.md](./ELI5.md)
2. **See all formulas**: Check [QUICK_REFERENCE.md](./QUICK_REFERENCE.md)
3. **Start Lab 1**: Open `labs/lab-1.4.1-manual-backpropagation.ipynb`
