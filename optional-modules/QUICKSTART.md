# Optional Modules - Quickstart Guide

## ⏱️ Time: ~10 minutes to choose your path

## 🎯 What This Guide Does

Helps you decide which optional module(s) to pursue based on your interests and career goals.

---

## 🔍 Quick Assessment

Answer these questions to find your path:

### 1. What interests you most?

| If you want to... | Start with |
|-------------------|------------|
| Understand *why* ML models generalize | **Module A: Learning Theory** |
| Build personalization systems (Netflix, Spotify) | **Module B: Recommender Systems** |
| Reverse-engineer how neural networks think | **Module C: Mechanistic Interpretability** |
| Understand RLHF behind ChatGPT | **Module D: Reinforcement Learning** |
| Work with molecules, networks, knowledge graphs | **Module E: Graph Neural Networks** |

### 2. What's your career direction?

| Career Path | Recommended Modules | Why |
|-------------|--------------------|----|
| **AI Safety Research** | C → A | Interpretability is core skill |
| **Industry ML Engineer** | B → E | Practical, high-demand skills |
| **LLM/Foundation Models** | D → C | Understand training and internals |
| **Drug Discovery/BioML** | E → A | Molecular graphs + theory |
| **Academic Research** | A → D | Strong theoretical foundation |

---

## 🚀 5-Minute First Success Per Module

### Module A: Learning Theory
```python
# Can a line separate 3 points? Let's check VC dimension!
import numpy as np
from sklearn.svm import SVC

points = np.array([[0, 0], [1, 0], [0.5, 1]])
# Try all 8 labelings for 3 points
for labels in [[0,0,0], [0,0,1], [0,1,0], [0,1,1],
               [1,0,0], [1,0,1], [1,1,0], [1,1,1]]:
    clf = SVC(kernel='linear', C=1e10)
    clf.fit(points, labels)
    if np.all(clf.predict(points) == labels):
        print(f"✓ Can separate: {labels}")
# All 8 work! → VC dimension ≥ 3
```

### Module B: Recommender Systems
```python
import torch
import torch.nn as nn

# Simplest recommender: matrix factorization
class MF(nn.Module):
    def __init__(self, n_users, n_items, dim=32):
        super().__init__()
        self.users = nn.Embedding(n_users, dim)
        self.items = nn.Embedding(n_items, dim)

    def forward(self, user, item):
        return (self.users(user) * self.items(item)).sum(-1)

model = MF(1000, 5000)
print(f"Predicted rating: {model(torch.tensor([42]), torch.tensor([123])).item():.2f}")
```

### Module C: Mechanistic Interpretability
```python
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
prompt = "The capital of France is"
logits, cache = model.run_with_cache(prompt)

# See what the model attends to!
print(f"Top prediction: {model.tokenizer.decode(logits[0, -1].argmax())}")
print(f"Cached {len(cache)} activation tensors for analysis")
```

### Module D: Reinforcement Learning
```python
import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=False)
Q = np.zeros((16, 4))  # 16 states, 4 actions

# One Q-learning update
state, _ = env.reset()
action = 1  # right
next_state, reward, done, _, _ = env.step(action)
Q[state, action] += 0.1 * (reward + 0.99 * Q[next_state].max() - Q[state, action])
print(f"Q-value updated: Q[{state}, {action}] = {Q[state, action]:.4f}")
```

### Module E: Graph Neural Networks
```python
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

print(f"Cora citation network:")
print(f"  {data.num_nodes} papers (nodes)")
print(f"  {data.num_edges} citations (edges)")
print(f"  {data.num_features} word features per paper")
print(f"  {dataset.num_classes} research topics to predict")
```

---

## ⏱️ Time Investment

| Module | Duration | Difficulty | Payoff |
|--------|----------|------------|--------|
| A: Learning Theory | 4-6 hrs | ⭐⭐⭐⭐ | Deep understanding |
| B: Recommenders | 6-8 hrs | ⭐⭐⭐ | Industry skills |
| C: Mech Interp | 6-8 hrs | ⭐⭐⭐⭐ | Research frontier |
| D: RL | 8-10 hrs | ⭐⭐⭐ | RLHF foundation |
| E: GNNs | 6-8 hrs | ⭐⭐⭐ | New data types |

---

## ✅ Prerequisites Check

All modules require:
- [x] Module 1.5: Neural Networks ✓
- [x] Module 2.1: PyTorch Fundamentals ✓

Additional requirements:

| Module | Also Needs |
|--------|-----------|
| A | Module 1.4 (Math Foundations) |
| B | Module 2.3 (NLP & Transformers) |
| C | Module 2.3 + Module 4.2 (AI Safety) |
| D | Just core prereqs |
| E | Just core prereqs |

→ See [PREREQUISITES.md](./PREREQUISITES.md) for detailed skill checks

---

## 🗺️ Recommended Learning Paths

### Path 1: The Theorist (10-14 hours)
```
Module A ──► Module D
Learning      RL uses
Theory        similar math
```

### Path 2: The Practitioner (12-16 hours)
```
Module B ──► Module E
Recommenders  GNNs extend
              retrieval skills
```

### Path 3: The Safety Researcher (12-16 hours)
```
Module C ──► Module A
Mech Interp   Theory explains
              what you observe
```

### Path 4: The LLM Specialist (14-18 hours)
```
Module D ──► Module C
RL for        Understand
RLHF          model internals
```

---

## 🎯 Next Steps

1. **Choose your path** based on interests above
2. **Check prerequisites** in [PREREQUISITES.md](./PREREQUISITES.md)
3. **Dive into your chosen module's QUICKSTART.md**

Remember: These are **optional** - the core curriculum (Domains 1-4) is complete without them!
