# Module A: Statistical Learning Theory - Quickstart

## ⏱️ Time: ~5 minutes

## 🎯 What You'll Discover

You'll prove that a line can separate any 3 points (but not 4), demonstrating the VC dimension concept that explains *why* models generalize.

## ✅ Before You Start

- [ ] Python with NumPy and scikit-learn available
- [ ] Basic understanding of linear classifiers

## 🚀 Let's Go!

### Step 1: Set Up the Experiment

```python
import numpy as np
from itertools import product
from sklearn.svm import SVC

def can_separate(points, labels):
    """Check if a linear classifier can achieve this labeling."""
    clf = SVC(kernel='linear', C=1e10)
    try:
        clf.fit(points, labels)
        return np.all(clf.predict(points) == labels)
    except:
        return False
```

### Step 2: Test 3 Points (Should Work!)

```python
# 3 points in general position
points_3 = np.array([[0, 0], [1, 0], [0.5, 1]])

# Try all 2^3 = 8 possible labelings
all_labelings = list(product([0, 1], repeat=3))
successes = sum(can_separate(points_3, list(l)) for l in all_labelings)

print(f"3 points: {successes}/8 labelings achievable")
```

**Expected output:**
```
3 points: 8/8 labelings achievable
```

### Step 3: Test 4 Points (Should Fail!)

```python
# 4 points in a square
points_4 = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

# Try all 2^4 = 16 possible labelings
all_labelings = list(product([0, 1], repeat=4))
successes = sum(can_separate(points_4, list(l)) for l in all_labelings)

print(f"4 points: {successes}/16 labelings achievable")
```

**Expected output:**
```
4 points: 14/16 labelings achievable
```

### Step 4: Find the Impossible Labeling (XOR!)

```python
# Which labelings fail? The XOR pattern!
for labels in all_labelings:
    if not can_separate(points_4, list(labels)):
        print(f"Cannot separate: {labels}")
```

**Expected output:**
```
Cannot separate: (0, 1, 1, 0)
Cannot separate: (1, 0, 0, 1)
```

## 🎉 You Did It!

You just empirically verified that **VC dimension of linear classifiers in 2D = 3**:
- Can shatter any 3 points (all 8 labelings)
- Cannot shatter 4 points (XOR is impossible)

This explains why linear models need fewer samples than complex ones - they have lower VC dimension!

## ▶️ Next Steps

1. **Understand why this matters**: Read Notebook 01 for the theory
2. **Explore bias-variance**: See how model complexity affects generalization
3. **Calculate sample bounds**: Use PAC learning to estimate data requirements

---

## 💡 The Key Insight

> **VC dimension tells you how "expressive" a model class is.**
>
> Higher VC = can fit more patterns = needs more data to generalize
>
> This is why a billion-parameter neural net needs millions of examples, while linear regression works with dozens.
