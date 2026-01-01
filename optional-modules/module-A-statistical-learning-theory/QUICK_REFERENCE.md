# Module A: Statistical Learning Theory - Quick Reference

## 🔑 Key Formulas

### VC Dimension
```
VC(H) = largest n such that H can shatter n points
```

| Hypothesis Class | VC Dimension |
|------------------|--------------|
| Linear classifiers in ℝᵈ | d + 1 |
| Axis-aligned rectangles in ℝ² | 4 |
| Circles in ℝ² | 3 |
| k-nearest neighbors | ∞ |
| Decision trees (unlimited depth) | ∞ |
| Neural networks | O(W log W) where W = # weights |

### Generalization Bound

```
P(|R(h) - R̂(h)| > ε) ≤ 4 · m_H(2n) · exp(-ε²n/8)
```

**In words:** With high probability, test error ≈ training error when n >> VC dimension.

### PAC Sample Complexity

```python
def pac_sample_complexity(vc_dim, epsilon, delta):
    """
    How many samples for (ε, δ)-PAC learning?

    Args:
        vc_dim: VC dimension of hypothesis class
        epsilon: Desired accuracy (smaller = more samples)
        delta: Allowed failure probability (smaller = more samples)
    """
    import numpy as np
    return int(np.ceil(
        (8 / epsilon) * (vc_dim * np.log(16 / epsilon) + np.log(2 / delta))
    ))

# Example: Linear classifier in 100D, 5% error, 95% confidence
samples = pac_sample_complexity(101, 0.05, 0.05)
print(f"Need at least {samples:,} samples")  # ~40,000
```

### Bias-Variance Decomposition

```
Expected Error = Bias² + Variance + Irreducible Noise
```

| Term | What it measures | How to reduce |
|------|------------------|---------------|
| Bias² | Underfitting | More complex model |
| Variance | Overfitting | More data, regularization, ensembles |
| Noise | Data quality | Better data collection |

---

## 📊 Code Patterns

### Check if Points Can Be Shattered

```python
import numpy as np
from itertools import product
from sklearn.svm import SVC

def check_shattering(points, classifier_class=SVC):
    """Check if classifier can shatter these points."""
    n = len(points)
    all_labelings = list(product([0, 1], repeat=n))

    for labels in all_labelings:
        clf = classifier_class(kernel='linear', C=1e10)
        try:
            clf.fit(points, labels)
            if not np.all(clf.predict(points) == labels):
                return False
        except:
            return False
    return True

# Test: 3 points in 2D should be shattered by lines
points = np.array([[0, 0], [1, 0], [0.5, 1]])
print(f"Shattered: {check_shattering(points)}")  # True
```

### Bias-Variance via Bootstrap

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def compute_bias_variance(X_train_sets, y_train_sets, X_test, y_true, degree):
    """
    Compute bias and variance through bootstrap sampling.

    Returns: (bias², variance)
    """
    predictions = []

    for X_train, y_train in zip(X_train_sets, y_train_sets):
        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(X_train.reshape(-1, 1))
        X_test_poly = poly.transform(X_test.reshape(-1, 1))

        model = LinearRegression()
        model.fit(X_poly, y_train)
        predictions.append(model.predict(X_test_poly))

    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)

    bias_squared = np.mean((mean_pred - y_true) ** 2)
    variance = np.mean(np.var(predictions, axis=0))

    return bias_squared, variance

# Generate bootstrap samples
def generate_bootstrap_samples(X, y, n_samples=100):
    X_sets, y_sets = [], []
    n = len(X)
    for _ in range(n_samples):
        indices = np.random.choice(n, n, replace=True)
        X_sets.append(X[indices])
        y_sets.append(y[indices])
    return X_sets, y_sets
```

### PAC Learning Bounds Calculator

```python
import numpy as np

def analyze_sample_requirements(vc_dim, target_accuracy=0.95, confidence=0.95):
    """
    Print sample complexity for various scenarios.
    """
    epsilon = 1 - target_accuracy  # e.g., 0.05 for 95% accuracy
    delta = 1 - confidence

    # Standard bound
    m = (8 / epsilon) * (vc_dim * np.log(16 / epsilon) + np.log(2 / delta))

    print(f"VC Dimension: {vc_dim}")
    print(f"Target: {target_accuracy:.0%} accuracy with {confidence:.0%} confidence")
    print(f"Sample complexity: {int(np.ceil(m)):,}")

    # Sensitivity analysis
    print("\nHow samples change with requirements:")
    for acc in [0.90, 0.95, 0.99]:
        for conf in [0.90, 0.95, 0.99]:
            eps = 1 - acc
            delt = 1 - conf
            m = (8 / eps) * (vc_dim * np.log(16 / eps) + np.log(2 / delt))
            print(f"  {acc:.0%} acc, {conf:.0%} conf: {int(np.ceil(m)):,} samples")

# Example: neural network with 10,000 weights
analyze_sample_requirements(vc_dim=10000)
```

---

## ⚡ Key Concepts at a Glance

### The Learning Theory Hierarchy

```
                        Can we learn?
                             │
              ┌──────────────┴──────────────┐
              │                             │
         Finite VC                      Infinite VC
    (PAC learnable)                    (Not learnable
                                        in worst case)
              │
     ┌────────┴────────┐
     │                 │
  Low VC            High VC
(few samples)    (many samples)
```

### When Theory vs Practice Diverge

| Theory Says | Practice Shows | Why? |
|-------------|----------------|------|
| Need O(VC/ε²) samples | Deep nets need fewer | Implicit regularization |
| High VC = overfitting | GPT generalizes | Inductive bias matters |
| PAC bounds are tight | Bounds very loose | Worst-case vs average-case |

---

## 📚 Quick Reference Table

| Concept | Definition | Practical Use |
|---------|------------|---------------|
| **VC Dimension** | Max points that can be shattered | Estimate model complexity |
| **Shattering** | Achieving all possible labelings | Theoretical capability |
| **PAC Learning** | Probably Approximately Correct | Sample size bounds |
| **Bias** | E[model] - truth | Measures underfitting |
| **Variance** | Spread of model predictions | Measures overfitting |
| **ERM** | Empirical Risk Minimization | Training = minimize train error |
| **Generalization** | Test - Train error gap | What we care about |

---

## 🔗 Quick Links

- Notebook 01: VC Dimension Exploration
- Notebook 02: Bias-Variance Decomposition
- Notebook 03: PAC Learning Bounds
- [Understanding Machine Learning (free textbook)](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/)
