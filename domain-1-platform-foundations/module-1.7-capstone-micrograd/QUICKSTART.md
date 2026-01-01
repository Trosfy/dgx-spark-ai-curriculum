# Module 1.7: Capstone - MicroGrad+ - Quickstart

## ⏱️ Time: ~5 minutes

## 🎯 What You'll Build
A tiny autograd engine that computes gradients automatically.

## ✅ Before You Start
- [ ] Completed Modules 1.1-1.6
- [ ] Understand backpropagation (Module 1.4)
- [ ] NGC PyTorch container running

## 🚀 Let's Go!

### Step 1: Create Value Class (Mini Autograd)
```python
import numpy as np

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
```

### Step 2: Implement Multiplication
```python
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

# Add these methods to Value class
Value.__mul__ = __mul__
Value.__add__ = __add__
```

### Step 3: Implement Backward
```python
def backward(self):
    # Topological sort
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)

    # Go backwards
    self.grad = 1.0
    for v in reversed(topo):
        v._backward()

Value.backward = backward
```

### Step 4: Test It!
```python
# f(a, b) = a * b + a
a = Value(2.0)
b = Value(3.0)
c = a * b + a  # c = 2*3 + 2 = 8

c.backward()

print(f"c = {c}")
print(f"a.grad = {a.grad}")  # dc/da = b + 1 = 4
print(f"b.grad = {b.grad}")  # dc/db = a = 2
```

**Expected output:**
```
c = Value(data=8.0000, grad=1.0000)
a.grad = 4.0
b.grad = 2.0
```

### Step 5: Verify with PyTorch
```python
import torch

a_t = torch.tensor(2.0, requires_grad=True)
b_t = torch.tensor(3.0, requires_grad=True)
c_t = a_t * b_t + a_t
c_t.backward()

print(f"PyTorch a.grad: {a_t.grad.item()}")
print(f"PyTorch b.grad: {b_t.grad.item()}")
print(f"Match: {a.grad == a_t.grad.item() and b.grad == b_t.grad.item()}")
```

**Expected:** `Match: True`

## 🎉 You Did It!

You just:
- ✅ Built automatic differentiation from scratch
- ✅ Implemented chain rule via _backward
- ✅ Used topological sort for correct gradient flow
- ✅ Verified gradients match PyTorch

In the full module, you'll:
- Add more operations (matmul, relu, softmax)
- Implement layers (Linear, ReLU, Dropout)
- Build optimizers (SGD, Adam)
- Train on MNIST to >95% accuracy

## ▶️ Next Steps
1. **Understand the design**: Read [STUDY_GUIDE.md](./STUDY_GUIDE.md)
2. **See full implementation**: Check [QUICK_REFERENCE.md](./QUICK_REFERENCE.md)
3. **Start Lab 1**: Open `labs/lab-1.7.1-core-tensor-implementation.ipynb`
