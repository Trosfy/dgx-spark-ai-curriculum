# Module 2.1: Deep Learning with PyTorch - Prerequisites Check

## Purpose
This module assumes you've completed Domain 1 (especially the Capstone). Use this self-check to verify you're ready.

## Estimated Time
- **If all prerequisites met**: Start with [QUICKSTART.md](./QUICKSTART.md)
- **If 1-2 gaps**: ~2-4 hours of review
- **If multiple gaps**: Complete Domain 1 first

---

## Required Skills

### 1. Python: Object-Oriented Programming

**Can you write this without looking anything up?**
```python
# Create a class with __init__, a method, and inheritance
class Animal:
    # Your implementation here
    pass

class Dog(Animal):
    # Your implementation here
    pass
```

<details>
<summary>Check your answer</summary>

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)
        self.breed = breed

    def speak(self):
        return f"{self.name} says woof!"
```

**Key points**:
- `__init__` initializes instance attributes
- `super().__init__()` calls parent constructor
- Methods take `self` as first argument

</details>

**Not ready?** Review: Module 1.2, Python OOP section

---

### 2. NumPy: Array Operations

**Can you do this?**
```python
import numpy as np
# Create a 3x4 matrix of random values
# Compute the mean of each row
# Find the index of the maximum value
```

<details>
<summary>Check your answer</summary>

```python
import numpy as np

matrix = np.random.randn(3, 4)
row_means = matrix.mean(axis=1)  # Mean along columns = row means
max_idx = matrix.argmax()        # Flat index
max_idx_2d = np.unravel_index(max_idx, matrix.shape)  # (row, col)
```

**Key points**:
- `axis=1` operates along columns (row-wise)
- `argmax()` returns flat index by default
- Broadcasting rules apply to shape compatibility

</details>

**Not ready?** Review: Module 1.2, NumPy section

---

### 3. Neural Networks: Forward and Backward Pass

**Can you explain this?**
> What happens during backpropagation? Why do we need gradients?

<details>
<summary>Check your answer</summary>

**Backpropagation**:
1. Forward pass computes output and loss
2. Backward pass computes gradient of loss w.r.t. each parameter
3. Gradients flow backward through the network using chain rule
4. Optimizer uses gradients to update parameters

**Why gradients?**
- Gradients tell us how to change each parameter to reduce loss
- Positive gradient → decrease parameter
- Negative gradient → increase parameter
- Magnitude indicates sensitivity

**The chain rule**: If `L = f(g(x))`, then `dL/dx = dL/dg * dg/dx`

</details>

**Not ready?** Review: Module 1.5, Backpropagation section + Module 1.7 Capstone

---

### 4. GPU Basics: Memory and Compute

**Do you know?**
- What is VRAM/GPU memory?
- Why is GPU faster than CPU for neural networks?
- What happens when you run out of GPU memory?

<details>
<summary>Check your answer</summary>

**GPU Memory (VRAM)**:
- Dedicated memory on the GPU (DGX Spark: 128GB unified)
- Stores model parameters, activations, gradients
- Faster than system RAM for GPU operations

**Why GPU is faster**:
- Thousands of cores for parallel computation
- Matrix operations (the core of deep learning) are highly parallelizable
- Tensor Cores accelerate specific operations (matmul, convolutions)

**Out of Memory (OOM)**:
- Training crashes with `CUDA out of memory`
- Solutions: reduce batch size, use mixed precision, gradient checkpointing

</details>

**Not ready?** Review: Module 1.3, CUDA Basics

---

### 5. MicroGrad: Building from Scratch

**Can you answer?**
> In your MicroGrad implementation, how did you implement the `backward()` function for multiplication?

<details>
<summary>Check your answer</summary>

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
        # d(a*b)/da = b
        # d(a*b)/db = a
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad

    out._backward = _backward
    return out
```

**Key insight**: The gradient of multiplication is the other operand, times the incoming gradient (chain rule).

</details>

**Not ready?** Complete: Module 1.7 Capstone (MicroGrad+)

---

### 6. Basic PyTorch Tensor Operations

**Can you do this?**
```python
import torch
# Create a tensor on GPU
# Perform matrix multiplication
# Move result back to CPU
```

<details>
<summary>Check your answer</summary>

```python
import torch

# Create tensors
a = torch.randn(3, 4, device='cuda')
b = torch.randn(4, 5, device='cuda')

# Matrix multiplication (multiple ways)
c = torch.matmul(a, b)  # or a @ b or torch.mm(a, b)

# Move to CPU
c_cpu = c.cpu()
# or
c_cpu = c.to('cpu')
```

**Key points**:
- `device='cuda'` creates tensor on GPU
- `.cpu()` and `.cuda()` move tensors
- `@` operator is matrix multiplication

</details>

**Not ready?** Quick review: [PyTorch 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

---

## Terminology Check

Do you know these terms?

| Term | Your Definition |
|------|-----------------|
| Tensor | |
| Gradient | |
| Epoch | |
| Batch size | |
| Learning rate | |
| Loss function | |

<details>
<summary>Check definitions</summary>

| Term | Definition |
|------|------------|
| **Tensor** | Multi-dimensional array (generalization of matrices) |
| **Gradient** | Derivative of loss w.r.t. a parameter; direction to update |
| **Epoch** | One complete pass through the entire training dataset |
| **Batch size** | Number of samples processed before parameter update |
| **Learning rate** | Step size for parameter updates (how much to change) |
| **Loss function** | Measures how wrong the model's predictions are |

</details>

---

## Optional But Helpful

These aren't required but will accelerate your learning:

### Debugging Experience
**Why it helps**: PyTorch errors can be cryptic; debugging skills save hours
**Quick primer**: Learn to read stack traces, use `print(tensor.shape)` liberally

### Basic Statistics
**Why it helps**: Understanding distributions helps with initialization and normalization
**Key concepts**: Mean, variance, normal distribution

---

## Ready Checklist

- [ ] I can write Python classes with inheritance
- [ ] I understand NumPy array operations and broadcasting
- [ ] I can explain backpropagation and why we need gradients
- [ ] I completed MicroGrad (Module 1.7)
- [ ] I know basic PyTorch tensor operations
- [ ] My environment is set up (see [LAB_PREP.md](./LAB_PREP.md))

**All boxes checked?** Start with [QUICKSTART.md](./QUICKSTART.md)!

**Some gaps?** Review the linked materials first - it's worth the time investment.
