# Module 1.2: Python for AI/ML - Quick Reference

## 🚀 NumPy Essentials

### Array Creation
```python
import numpy as np

# From list
a = np.array([1, 2, 3])

# Zeros/ones
zeros = np.zeros((3, 4))
ones = np.ones((3, 4), dtype=np.float32)

# Range
sequence = np.arange(0, 10, 0.5)  # start, stop, step
linear = np.linspace(0, 1, 100)   # start, stop, num points

# Random
uniform = np.random.rand(3, 4)     # [0, 1)
normal = np.random.randn(3, 4)     # standard normal
integers = np.random.randint(0, 10, (3, 4))
```

### Broadcasting Rules
```python
# Shapes align from the RIGHT
# Dimensions are compatible if: equal OR one is 1

(5, 4, 3) + (4, 3)      # ✅ → (5, 4, 3)
(5, 4, 3) + (4, 1)      # ✅ → (5, 4, 3)
(5, 4, 3) + (5, 1, 3)   # ✅ → (5, 4, 3)
(5, 4, 3) + (2, 3)      # ❌ 4 ≠ 2

# Add dimensions with None/np.newaxis
a = np.array([1, 2, 3])        # shape (3,)
a[:, None]                      # shape (3, 1)
a[None, :]                      # shape (1, 3)
```

### Matrix Operations
```python
# Element-wise
c = a * b          # Hadamard product
c = a + b          # Addition

# Matrix multiplication
c = a @ b          # Preferred syntax
c = np.matmul(a, b)
c = np.dot(a, b)   # Only for 1D/2D

# Transpose
a.T                # Swap last two dims
a.transpose(0, 2, 1)  # Custom axis order
```

## 📐 Einsum Notation

### Basic Operations
```python
# Matrix multiply: (M,K) @ (K,N) → (M,N)
np.einsum('mk,kn->mn', A, B)

# Batch matrix multiply: (B,M,K) @ (B,K,N) → (B,M,N)
np.einsum('bmk,bkn->bmn', A, B)

# Outer product: (M,) × (N,) → (M,N)
np.einsum('i,j->ij', a, b)

# Inner product (dot): (N,) · (N,) → scalar
np.einsum('i,i->', a, b)

# Trace: (N,N) → scalar
np.einsum('ii->', A)

# Diagonal: (N,N) → (N,)
np.einsum('ii->i', A)

# Sum along axis
np.einsum('ijk->ik', A)  # Sum over j
```

### Attention Pattern
```python
# Attention scores: Q @ K.T for each batch and head
# Q: (batch, heads, seq_q, dim)
# K: (batch, heads, seq_k, dim)
# Result: (batch, heads, seq_q, seq_k)
scores = np.einsum('bhsd,bhtd->bhst', Q, K)
```

## 📊 Pandas Quick Reference

### Data Loading
```python
import pandas as pd

df = pd.read_csv('data.csv')
df = pd.read_parquet('data.parquet')  # Faster, smaller
df = pd.read_json('data.json')
```

### Data Inspection
```python
df.head()           # First 5 rows
df.info()           # Dtypes and null counts
df.describe()       # Statistics
df.shape            # (rows, cols)
df.columns          # Column names
df['col'].value_counts()  # Frequency
```

### Missing Data
```python
# Check
df.isnull().sum()

# Fill
df['col'].fillna(0)
df['col'].fillna(df['col'].mean())
df['col'].fillna(method='ffill')  # Forward fill

# Drop
df.dropna()
df.dropna(subset=['col1', 'col2'])
```

### Transformations
```python
# Select columns
df[['col1', 'col2']]

# Filter rows
df[df['col'] > 0]
df.query('col > 0 and other < 10')

# Create new column
df['new'] = df['a'] + df['b']
df['category'] = df['value'].apply(lambda x: 'high' if x > 50 else 'low')

# Group and aggregate
df.groupby('category')['value'].mean()
df.groupby('category').agg({'value': 'mean', 'count': 'sum'})
```

### Encoding
```python
# One-hot encoding
pd.get_dummies(df, columns=['category'])

# Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['encoded'] = le.fit_transform(df['category'])
```

## 📈 Matplotlib/Seaborn Patterns

### Basic Figure
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, label='line')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_title('Title')
ax.legend()
plt.tight_layout()
plt.savefig('figure.png', dpi=150)
```

### Multi-Panel Figure
```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(train_loss, label='Train')
axes[0, 0].plot(val_loss, label='Val')
axes[0, 0].set_title('Loss Curves')
axes[0, 0].legend()

axes[0, 1].imshow(confusion_matrix, cmap='Blues')
axes[0, 1].set_title('Confusion Matrix')

axes[1, 0].bar(names, importance)
axes[1, 0].set_title('Feature Importance')

axes[1, 1].hist(predictions, bins=50)
axes[1, 1].set_title('Prediction Distribution')

plt.tight_layout()
```

### Seaborn Plots
```python
import seaborn as sns

# Heatmap (attention, confusion matrix)
sns.heatmap(matrix, annot=True, cmap='viridis')

# Distribution
sns.histplot(data=df, x='value', hue='category')
sns.kdeplot(data=df, x='value')

# Scatter with regression
sns.regplot(x='x', y='y', data=df)

# Pair plot (EDA)
sns.pairplot(df, hue='category')
```

## 🔧 Profiling Commands

### cProfile (Always Available)
```python
import cProfile
import pstats

# Profile a function
cProfile.run('my_function()', 'output.prof')

# View results
stats = pstats.Stats('output.prof')
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20
```

### line_profiler (Install Separately)
```python
# In Jupyter
%load_ext line_profiler
%lprun -f my_function my_function(args)
```

### Memory Profiling
```python
# Using tracemalloc (stdlib)
import tracemalloc

tracemalloc.start()
# ... your code ...
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1e6:.1f} MB")
print(f"Peak: {peak / 1e6:.1f} MB")
tracemalloc.stop()
```

### Timing
```python
import time

# Simple timing
start = time.time()
result = my_function()
print(f"Took: {time.time() - start:.3f}s")

# Jupyter magic
%timeit my_function()  # Multiple runs, statistics
%time my_function()    # Single run
```

## ⚠️ Common Mistakes

| Mistake | Fix |
|---------|-----|
| Python loop over arrays | Use NumPy vectorization |
| `x = x + 1` creates copy | Use `x += 1` for in-place |
| Using float64 by default | Specify `dtype=np.float32` |
| Forgot `.copy()` on slice | Arrays share memory by default |
| Pandas `apply` with loop | Use vectorized operations |

## 📊 Memory Efficiency

```python
# Check array size
print(f"Size: {arr.nbytes / 1e6:.1f} MB")

# Use smaller dtype
arr = np.array(data, dtype=np.float32)  # vs float64

# Check if contiguous (affects speed)
print(arr.flags['C_CONTIGUOUS'])

# Make contiguous if needed
arr = np.ascontiguousarray(arr)
```

## 🔗 Quick Links
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)
- [Einsum Tutorial](https://ajcr.net/Basic-guide-to-einsum/)
