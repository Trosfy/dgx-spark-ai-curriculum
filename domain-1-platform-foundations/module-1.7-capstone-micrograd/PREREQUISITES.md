# Module 1.7: MicroGrad+ Prerequisites

This capstone module integrates everything from Domain 1. Before starting, verify you have the required skills.

---

## Required Prerequisites

### From Module 1.1: DGX Spark Platform Mastery

| Skill | How to Verify |
|-------|---------------|
| Can launch NGC containers | `docker run --gpus all -it nvcr.io/nvidia/pytorch:25.11-py3 python -c "print('Ready')"` |
| Understand container volumes | Know why `-v $HOME/workspace:/workspace` is used |
| Can run Jupyter notebooks | Successfully opened and run a .ipynb file |

<details>
<summary>Self-Check Answer</summary>

The volume mount `-v $HOME/workspace:/workspace` persists your work outside the container.
Without it, all files created inside the container are lost when it stops.
</details>

---

### From Module 1.2: Python for AI/ML

| Skill | How to Verify |
|-------|---------------|
| Python classes | Can define `__init__`, `__repr__`, `__add__` methods |
| NumPy array operations | Can multiply, sum, reshape arrays |
| List comprehensions | Can write `[x**2 for x in range(10)]` |
| Lambda functions | Can write `lambda x: x**2` |

<details>
<summary>Self-Check: NumPy Test</summary>

```python
import numpy as np

# Can you predict the output?
a = np.array([[1, 2], [3, 4]])
b = np.array([10, 20])
result = a + b

# Answer: [[11, 22], [13, 24]] due to broadcasting
```
</details>

---

### From Module 1.3: CUDA Python & GPU Programming

| Skill | How to Verify |
|-------|---------------|
| Understand CPU vs GPU tradeoffs | Know when GPU helps (parallelizable ops) |
| Basic memory concepts | Understand unified memory on DGX Spark |

<details>
<summary>Self-Check Answer</summary>

MicroGrad+ is CPU-based (NumPy), but understanding GPU concepts helps appreciate:
- Why PyTorch is faster (GPU parallelism)
- Why we use NGC containers (optimized libraries)
- Memory layout for efficient operations
</details>

---

### From Module 1.4: Mathematics for Deep Learning

| Skill | How to Verify |
|-------|---------------|
| Chain rule | Can compute d/dx of f(g(x)) |
| Partial derivatives | Can compute ∂f/∂x for f(x, y) |
| Matrix multiplication | Can compute A @ B dimensions and values |
| Gradients | Understand gradient as direction of steepest ascent |

<details>
<summary>Self-Check: Chain Rule</summary>

**Problem:** If y = (3x + 2)², what is dy/dx at x = 2?

**Solution:**
- Let u = 3x + 2, so y = u²
- dy/du = 2u = 2(3x + 2)
- du/dx = 3
- dy/dx = dy/du × du/dx = 2(3x + 2) × 3 = 6(3x + 2)
- At x = 2: dy/dx = 6(3×2 + 2) = 6 × 8 = **48**
</details>

---

### From Module 1.5: Neural Network Fundamentals

| Skill | How to Verify |
|-------|---------------|
| Forward pass | Understand y = σ(Wx + b) |
| Backpropagation | Know how gradients flow backward |
| Activation functions | Know ReLU, sigmoid, softmax |
| Loss functions | Understand MSE, cross-entropy |

<details>
<summary>Self-Check: Backprop Concept</summary>

**Question:** In a network with layers L1 → L2 → L3, if we compute ∂Loss/∂L3,
how do we get ∂Loss/∂L1?

**Answer:** Chain rule: ∂Loss/∂L1 = ∂Loss/∂L3 × ∂L3/∂L2 × ∂L2/∂L1

This is "reverse-mode autodiff" - we compute gradients by walking backward through the graph.
</details>

---

### From Module 1.6: Classical ML Foundations

| Skill | How to Verify |
|-------|---------------|
| Train/test split | Know why we need separate sets |
| Optimization basics | Understand gradient descent |
| Evaluation metrics | Can compute accuracy |

<details>
<summary>Self-Check Answer</summary>

We split data to detect overfitting - high train accuracy but low test accuracy
means the model memorized training data instead of learning general patterns.
</details>

---

## Technical Requirements

### Software (All included in NGC container)

- Python 3.10+
- NumPy 1.24+
- Matplotlib 3.7+ (for visualization)
- pytest 7.0+ (for running tests)

### Environment Setup

```bash
# Launch NGC container with required mounts
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

### Verify Setup

```python
# Run this to verify prerequisites
import numpy as np

# NumPy operations
a = np.array([[1, 2], [3, 4]], dtype=np.float32)
b = np.array([[5], [6]], dtype=np.float32)
c = a @ b  # Matrix multiplication
assert c.shape == (2, 1)
print(f"Matrix mult: {a.shape} @ {b.shape} = {c.shape} ✓")

# Broadcasting
d = a + np.array([10, 20])
assert d.shape == (2, 2)
print(f"Broadcasting works ✓")

# Class with special methods
class TestValue:
    def __init__(self, x):
        self.x = x
    def __add__(self, other):
        return TestValue(self.x + other.x)
    def __repr__(self):
        return f"TestValue({self.x})"

v1 = TestValue(3)
v2 = TestValue(4)
v3 = v1 + v2
assert v3.x == 7
print(f"Class methods work: {v3} ✓")

print("\n✅ All prerequisites verified!")
```

---

## Estimated Time

| Prior Experience | Estimated Module Time |
|------------------|----------------------|
| Strong Python, calculus background | 8 hours |
| Completed all Domain 1 modules | 10 hours |
| Some gaps in prerequisites | 12-15 hours (with review) |

---

## If You Have Gaps

| Gap | Recommended Action |
|-----|-------------------|
| Python classes | Review Module 1.2, practice with simple classes |
| NumPy operations | Work through NumPy tutorial in Module 1.2 |
| Chain rule/calculus | Review Module 1.4 math foundations |
| Backpropagation concept | Re-watch Karpathy's micrograd video (first 30 min) |

---

## Ready to Start?

If you can:
- ✅ Define Python classes with `__add__` and other special methods
- ✅ Perform NumPy matrix operations and understand broadcasting
- ✅ Apply the chain rule to compute derivatives
- ✅ Explain how backpropagation flows gradients backward

Then you're ready for **Lab 1.7.1: Core Tensor Implementation**!

→ Start with [QUICKSTART.md](./QUICKSTART.md) for a 5-minute introduction.
