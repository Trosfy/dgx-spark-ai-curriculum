# Optional Modules - Prerequisites Check

## 🎯 Purpose

Before diving into optional modules, verify you have the foundational skills. Use this self-assessment to identify gaps.

---

## ⏱️ Time Estimates

| If prerequisites are... | Action | Time |
|------------------------|--------|------|
| All met | Jump to your chosen module! | 0 hours |
| 1-2 gaps | Quick review recommended | 1-2 hours |
| Multiple gaps | Complete core modules first | 4-8 hours |

---

## 🔧 Universal Prerequisites (All Modules)

### 1. PyTorch Fundamentals

**Can you do this without looking anything up?**

```python
import torch
import torch.nn as nn

# Create a simple neural network
class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # TODO: Add two linear layers with ReLU activation
        pass

    def forward(self, x):
        # TODO: Implement forward pass
        pass

# Create model, loss, optimizer, and run one training step
# TODO: Can you write this from memory?
```

<details>
<summary>✅ Check your answer</summary>

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training step
model = SimpleNet(10, 64, 2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

x = torch.randn(32, 10)  # batch of 32
y = torch.randint(0, 2, (32,))

optimizer.zero_grad()
output = model(x)
loss = criterion(output, y)
loss.backward()
optimizer.step()
```

**Key points:**
- `super().__init__()` is required
- `nn.Linear` for fully connected layers
- `optimizer.zero_grad()` before `loss.backward()`

</details>

**Not ready?** Review: Module 2.1 (PyTorch Fundamentals)

---

### 2. Neural Network Concepts

**Can you explain these in simple terms?**

| Concept | Your explanation |
|---------|------------------|
| Gradient descent | |
| Backpropagation | |
| Overfitting vs underfitting | |
| Learning rate | |

<details>
<summary>✅ Check your understanding</summary>

| Concept | Simple explanation |
|---------|-------------------|
| Gradient descent | Find the minimum of a loss function by taking steps in the direction of steepest descent |
| Backpropagation | Compute gradients for each weight by applying chain rule backwards through the network |
| Overfitting | Model memorizes training data, fails on new data (low train error, high test error) |
| Underfitting | Model is too simple to capture patterns (high train error, high test error) |
| Learning rate | Step size for gradient updates - too big = unstable, too small = slow |

</details>

**Not ready?** Review: Module 1.5 (Neural Networks)

---

### 3. Python Proficiency

**Can you read this code and predict the output?**

```python
import numpy as np

def mystery(data, func, k=3):
    return [func(data[i:i+k]) for i in range(len(data)-k+1)]

result = mystery([1, 4, 2, 8, 5, 7], np.mean, k=3)
print(len(result), result[0], result[-1])
```

<details>
<summary>✅ Check your answer</summary>

Output: `4 2.3333333333333335 6.666666666666667`

**Explanation:**
- Creates sliding windows of size k=3
- Applies np.mean to each window
- Windows: [1,4,2], [4,2,8], [2,8,5], [8,5,7]
- Means: 2.33, 4.67, 5.0, 6.67
- Length 4, first is 2.33, last is 6.67

</details>

---

## 📚 Module-Specific Prerequisites

### Module A: Statistical Learning Theory

**Additional requirement:** Module 1.4 (Math Foundations)

**Can you compute this?**

Given:
- Mean squared error: MSE = (1/n) Σ(yᵢ - ŷᵢ)²
- Bias: E[ŷ] - y_true
- Variance: E[(ŷ - E[ŷ])²]

If a model has Bias = 2 and Variance = 3, what's the lower bound on MSE?

<details>
<summary>✅ Check your answer</summary>

MSE ≥ Bias² + Variance = 4 + 3 = 7

(Equality when there's no irreducible noise)

</details>

**Comfortable with:**
- [ ] Probability and expectation
- [ ] Basic calculus (derivatives)
- [ ] Linear algebra (matrix operations)

**Not ready?** Review: Module 1.4 (Math Foundations)

---

### Module B: Recommender Systems

**Additional requirement:** Module 2.3 (NLP & Transformers)

**Do you understand embeddings?**

```python
# What does this do and why is it useful?
embedding = nn.Embedding(10000, 128)
user_ids = torch.tensor([42, 123, 7])
user_vectors = embedding(user_ids)
print(user_vectors.shape)  # ?
```

<details>
<summary>✅ Check your answer</summary>

Output: `torch.Size([3, 128])`

**Explanation:**
- `nn.Embedding(10000, 128)` creates a lookup table of 10,000 vectors, each 128-dimensional
- Given user IDs [42, 123, 7], it returns their corresponding 128-d vectors
- Embeddings are learned representations that capture similarity

</details>

**Comfortable with:**
- [ ] Embedding layers
- [ ] Matrix operations (dot products)
- [ ] Loss functions for classification

---

### Module C: Mechanistic Interpretability

**Additional requirements:** Module 2.3 (Transformers) + Module 4.2 (AI Safety)

**Can you explain the transformer architecture?**

| Component | What it does |
|-----------|--------------|
| Self-attention | |
| Query, Key, Value | |
| Residual connections | |
| Layer normalization | |

<details>
<summary>✅ Check your understanding</summary>

| Component | What it does |
|-----------|--------------|
| Self-attention | Allows each token to attend to all other tokens, computing weighted combinations |
| Query, Key, Value | Q and K compute attention weights, V contains the information to aggregate |
| Residual connections | Add input to output of each block, enabling gradient flow and incremental updates |
| Layer normalization | Normalize activations to stabilize training |

</details>

**Understanding required:**
- [ ] Transformer architecture details
- [ ] Why interpretability matters for AI safety
- [ ] What "mechanistic" means (understanding *how*, not just *what*)

---

### Module D: Reinforcement Learning

**Core prerequisites only** - no additional requirements!

**Basic intuition check:**

> An agent in a maze gets +10 for finding the exit and -1 for each step taken.
> If the agent can see the exit but takes 100 steps to reach it, what's wrong?

<details>
<summary>✅ Check your answer</summary>

The agent is **exploring too much** (high epsilon) or not learning efficiently.

With -1 per step and +10 at exit:
- Direct path (10 steps): reward = 10 - 10 = 0
- 100 steps: reward = 10 - 100 = -90

The agent should learn to minimize steps. If it's not, check:
- Exploration rate (ε) should decay
- Learning rate might be too low
- Discount factor (γ) might be too low to propagate future rewards

</details>

---

### Module E: Graph Neural Networks

**Core prerequisites only** - no additional requirements!

**Graph intuition check:**

> In a social network, how would you predict if two people might become friends?

<details>
<summary>✅ Check your answer</summary>

Good answers include:
- **Common neighbors**: People with many mutual friends are likely to connect
- **Similar features**: People with similar interests/demographics
- **Network distance**: People 2 hops apart more likely than 4 hops

This is exactly what GNNs learn! They aggregate neighbor information to create node representations that capture both node features and network structure.

</details>

---

## ✅ Final Checklist

### Ready for any optional module:
- [ ] Can write PyTorch training loop from memory
- [ ] Understand gradient descent and backpropagation
- [ ] Comfortable with NumPy and basic Python

### Module A (Learning Theory):
- [ ] All above +
- [ ] Comfortable with probability and statistics
- [ ] Can work with mathematical notation

### Module B (Recommender Systems):
- [ ] All above +
- [ ] Understand embeddings
- [ ] Familiar with transformers (Module 2.3)

### Module C (Mechanistic Interpretability):
- [ ] All above +
- [ ] Deep understanding of transformer architecture
- [ ] Completed Module 4.2 (AI Safety)

### Module D (Reinforcement Learning):
- [ ] Core prerequisites only!

### Module E (Graph Neural Networks):
- [ ] Core prerequisites only!

---

## 🚀 Ready?

**All boxes checked?** → Start with your chosen module's QUICKSTART.md!

**Some gaps?** → No shame! Review the linked materials first. These optional modules assume solid foundations.
