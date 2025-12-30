# Module 5: Phase 1 Capstone — MicroGrad+

**Phase:** 1 - Foundations  
**Duration:** Week 6 (8-10 hours)  
**Prerequisites:** Modules 1-4

---

## Overview

This capstone project consolidates everything you've learned in Phase 1. You'll build **MicroGrad+**, an extended version of Andrej Karpathy's micrograd—a tiny autograd engine that can train neural networks. This project demonstrates your understanding of automatic differentiation, neural network components, and software engineering practices.

---

## Learning Outcomes

By the end of this module, you will be able to:

- ✅ Design and implement a modular neural network library
- ✅ Create reusable, well-documented code following software engineering practices
- ✅ Benchmark your implementation against PyTorch

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 5.1 | Design a modular architecture for neural network components | Create |
| 5.2 | Implement automatic differentiation for common operations | Apply |
| 5.3 | Write comprehensive unit tests for neural network operations | Apply |
| 5.4 | Document code with docstrings and usage examples | Create |

---

## Project: MicroGrad+

### Required Components

Build a Python package with the following:

#### 1. Tensor Class with Autograd
```python
class Tensor:
    def __init__(self, data, requires_grad=False):
        ...
    
    def backward(self):
        """Compute gradients via reverse-mode autodiff"""
        ...
    
    # Operations: +, -, *, /, @, sum, mean, reshape, etc.
```

#### 2. Layer Classes
```python
class Linear:
    """Fully connected layer: y = Wx + b"""
    
class ReLU:
    """ReLU activation"""
    
class Softmax:
    """Softmax activation"""
    
class Dropout:
    """Dropout regularization"""

class BatchNorm:  # Bonus
    """Batch normalization"""
```

#### 3. Loss Functions
```python
class MSELoss:
    """Mean Squared Error for regression"""

class CrossEntropyLoss:
    """Cross-entropy for classification"""
```

#### 4. Optimizers
```python
class SGD:
    """Stochastic Gradient Descent with optional momentum"""

class Adam:
    """Adam optimizer"""
```

#### 5. Training Utilities
```python
class Sequential:
    """Container for stacking layers"""

def train_epoch(model, dataloader, loss_fn, optimizer):
    """Train for one epoch"""

def evaluate(model, dataloader, loss_fn):
    """Evaluate model on dataset"""
```

---

## Project Structure

```
micrograd_plus/
├── __init__.py
├── tensor.py          # Tensor class with autograd
├── layers.py          # Neural network layers
├── losses.py          # Loss functions
├── optimizers.py      # SGD, Adam
├── nn.py              # Sequential, training utilities
└── utils.py           # Helper functions

tests/
├── test_tensor.py     # Tensor operation tests
├── test_layers.py     # Layer tests
├── test_autograd.py   # Gradient verification tests
└── test_training.py   # End-to-end training tests

examples/
├── mnist_example.ipynb
└── cifar10_example.ipynb

docs/
├── API.md             # API documentation
└── TUTORIAL.md        # Getting started guide
```

---

## Tasks

### Task 5.1: Core Tensor Implementation
**Time:** 3 hours

Implement the Tensor class with automatic differentiation.

**Requirements:**
- Store data as numpy array
- Track computational graph for backward pass
- Implement operations: +, -, *, /, matmul, sum, mean
- Implement `backward()` using reverse-mode autodiff

**Verification:**
```python
import numpy as np
a = Tensor([2.0], requires_grad=True)
b = Tensor([3.0], requires_grad=True)
c = a * b + a
c.sum().backward()  # Reduce to scalar before backward
assert np.allclose(a.grad, 4.0)  # dc/da = b + 1 = 4
assert np.allclose(b.grad, 2.0)  # dc/db = a = 2
```

---

### Task 5.2: Layer Implementation
**Time:** 2 hours

Implement neural network layers.

**Requirements:**
- Linear layer with weight initialization
- ReLU, Sigmoid, Softmax activations
- Dropout with training/eval modes
- All layers work with Tensor autograd

---

### Task 5.3: Loss and Optimizers
**Time:** 1.5 hours

Implement loss functions and optimizers.

**Requirements:**
- MSELoss and CrossEntropyLoss
- SGD with momentum option
- Adam optimizer
- Optimizers update parameters in-place

---

### Task 5.4: Testing Suite
**Time:** 1.5 hours

Write comprehensive tests.

**Requirements:**
- Test each Tensor operation
- Gradient check against numerical gradients
- Test layer forward/backward
- Test full training loop
- Achieve >80% code coverage

```python
# Example gradient check
def test_matmul_gradient():
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    b = Tensor(np.random.randn(4, 5), requires_grad=True)
    c = a @ b
    c.sum().backward()
    
    # Numerical gradient
    eps = 1e-5
    numerical_grad = numerical_gradient(lambda x: (x @ b.data).sum(), a.data)
    
    assert np.allclose(a.grad, numerical_grad, atol=1e-5)
```

---

### Task 5.5: MNIST Example
**Time:** 1.5 hours

Train on MNIST using your library.

**Requirements:**
- Load MNIST data
- Build MLP: 784 → 256 → 128 → 10
- Train with CrossEntropyLoss and Adam
- Achieve >95% accuracy
- Plot training curves

---

### Task 5.6: Documentation
**Time:** 1 hour

Document your library.

**Requirements:**
- Docstrings for all public functions/classes
- API.md with complete reference
- TUTORIAL.md with getting started guide
- README.md with installation and examples

---

## Guidance

### Autograd Implementation Tips

```python
class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, 
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='*')
        
        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad
        out._backward = _backward
        
        return out
    
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
        
        # Backprop
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()
```

### Benchmarking Against PyTorch

```python
import time
import torch
from micrograd_plus import Tensor

# Your implementation
start = time.time()
# ... training loop ...
your_time = time.time() - start

# PyTorch
start = time.time()
# ... same training loop with PyTorch ...
pytorch_time = time.time() - start

print(f"MicroGrad+: {your_time:.2f}s")
print(f"PyTorch: {pytorch_time:.2f}s")
print(f"Ratio: {your_time/pytorch_time:.1f}x slower")
# Expected: 10-100x slower than PyTorch (that's okay!)
```

---

## Deliverables

| Deliverable | Description |
|-------------|-------------|
| `micrograd_plus/` | Python package with all components |
| `tests/` | Unit tests with >80% coverage |
| `examples/mnist_example.ipynb` | MNIST training achieving >95% |
| `examples/cifar10_example.ipynb` | CIFAR-10 training (bonus) |
| `docs/API.md` | API documentation |
| `docs/TUTORIAL.md` | Getting started guide |
| Benchmark notebook | Comparison with PyTorch |

---

## Milestone Checklist

- [ ] Tensor class with autograd implemented
- [ ] All gradients verified against numerical gradients
- [ ] Linear, ReLU, Softmax, Dropout layers working
- [ ] MSELoss and CrossEntropyLoss implemented
- [ ] SGD and Adam optimizers implemented
- [ ] Unit tests passing with >80% coverage
- [ ] MNIST example achieving >95% accuracy
- [ ] Documentation complete (API.md, TUTORIAL.md)
- [ ] Benchmark notebook comparing with PyTorch

---

## Grading Rubric (Self-Assessment)

| Criteria | Points |
|----------|--------|
| Tensor autograd correctness | 25 |
| Layer implementations | 20 |
| Loss functions and optimizers | 15 |
| Test coverage and quality | 15 |
| MNIST accuracy >95% | 10 |
| Documentation quality | 10 |
| Code organization and style | 5 |
| **Total** | **100** |

---

## Next Steps

Congratulations on completing Phase 1! 🎉

You now have:
- Deep understanding of neural network internals
- Your own autograd engine
- Solid Python/NumPy foundation

Next: [Phase 2: Intermediate](../../phase-2-intermediate/) where you'll use PyTorch for more advanced architectures.

---


---

## DGX Spark Notes

On DGX Spark, dependencies like NumPy and matplotlib are typically pre-installed
in NGC containers. Use the PyTorch NGC container for the best experience:

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache:/root/.cache \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

This container includes NumPy, matplotlib, and all other common dependencies.
The `--ipc=host` flag enables shared memory for DataLoader workers.

## Optional Dependencies

For full functionality, you may want to install these optional packages:

```bash
pip install psutil>=5.9.0      # For memory monitoring (get_memory_usage)
pip install matplotlib>=3.5.0  # For visualization (plot_training_history)
pip install graphviz>=0.20     # For computation graphs (visualize_computation_graph)
pip install pytest>=7.0.0      # For running the test suite
```

These are optional - core functionality works without them.

---

## Resources

- [Karpathy's micrograd](https://github.com/karpathy/micrograd)
- [Karpathy's micrograd video](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [PyTorch Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
