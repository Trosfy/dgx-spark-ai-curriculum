# Module 1.2: Python for AI/ML

**Domain:** 1 - Platform Foundations  
**Duration:** Week 2 (8-10 hours)  
**Prerequisites:** Basic Python programming

---

## Overview

This module builds your Python proficiency for machine learning workflows. You'll master NumPy operations essential for neural networks, learn efficient data manipulation with Pandas, create publication-quality visualizations, and understand how to profile and optimize Python code.

---

## Learning Outcomes

By the end of this module, you will be able to:

- ✅ Write efficient NumPy code for tensor operations
- ✅ Manipulate datasets using Pandas for ML preprocessing
- ✅ Create publication-quality visualizations with Matplotlib/Seaborn
- ✅ Profile and optimize Python code for performance

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 1.2.1 | Implement vectorized operations using NumPy broadcasting | Apply |
| 1.2.2 | Transform and clean datasets using Pandas operations | Apply |
| 1.2.3 | Create multi-panel visualizations for model analysis | Create |
| 1.2.4 | Profile Python code and identify performance bottlenecks | Analyze |

---

## Topics

### 1.2.1 NumPy Essentials
- Array creation, indexing, slicing
- Broadcasting rules and vectorization
- Linear algebra operations (dot, matmul, einsum)
- Memory layout (C-contiguous vs F-contiguous)

### 1.2.2 Pandas for ML
- DataFrame operations and transformations
- Handling missing data
- Feature engineering patterns
- Efficient I/O (parquet, feather)

### 1.2.3 Visualization
- Matplotlib fundamentals
- Seaborn statistical plots
- Training curve visualization
- Attention heatmaps and activation maps

### 1.2.4 Performance Optimization
- Profiling with cProfile and line_profiler
- Numba JIT compilation
- Memory profiling

---

## Labs

### Lab 1.2.1: NumPy Broadcasting Lab
**Time:** 2 hours

Master broadcasting for efficient tensor operations.

**Instructions:**
1. Implement batch matrix multiplication using broadcasting
2. Create outer products without explicit loops
3. Compare loop vs vectorized implementations
4. Measure and document timing differences

**Deliverable:** Notebook with timing comparisons (expect 100x+ speedup)

---

### Lab 1.2.2: Dataset Preprocessing Pipeline
**Time:** 2 hours

Build a reusable preprocessing pipeline.

**Instructions:**
1. Download a real dataset (Titanic, Housing, or similar)
2. Handle missing values with multiple strategies
3. Encode categorical variables
4. Implement feature scaling (StandardScaler, MinMaxScaler)
5. Create a reusable `Preprocessor` class

**Deliverable:** `preprocessing_pipeline.py` with reusable class

---

### Lab 1.2.3: Visualization Dashboard
**Time:** 2 hours

Create a multi-panel figure for model analysis.

**Instructions:**
1. Create a 2x2 subplot figure
2. Panel 1: Training/validation loss curves
3. Panel 2: Confusion matrix heatmap
4. Panel 3: Feature importance bar chart
5. Panel 4: Prediction distribution histogram
6. Use consistent styling and color scheme

**Deliverable:** Publication-ready visualization notebook

---

### Lab 1.2.4: Einsum Mastery
**Time:** 2 hours

Master einsum notation for attention mechanisms.

**Instructions:**
1. Implement the following using einsum:
   - Matrix multiplication
   - Batch matrix multiplication
   - Outer product
   - Trace
   - Attention scores: `softmax(QK^T / sqrt(d))`
2. Compare with explicit implementations
3. Document einsum notation patterns

**Deliverable:** Einsum reference notebook with attention implementation

---

### Lab 1.2.5: Profiling Exercise
**Time:** 2 hours

Optimize slow code using profiling.

**Instructions:**
1. Start with provided slow function (nested loops)
2. Profile with `cProfile` and `line_profiler`
3. Identify bottlenecks
4. Optimize using vectorization and/or Numba
5. Achieve 10x+ speedup
6. Document before/after comparison

**Deliverable:** Optimization notebook with profiling results

---

## Guidance

### Einsum Notation Reference

```python
# Matrix multiply: (M, K) @ (K, N) -> (M, N)
np.einsum('mk,kn->mn', A, B)

# Batch matrix multiply: (B, M, K) @ (B, K, N) -> (B, M, N)
np.einsum('bmk,bkn->bmn', A, B)

# Attention scores: (B, H, S, D) @ (B, H, S, D).T -> (B, H, S, S)
np.einsum('bhsd,bhtd->bhst', Q, K)

# Trace
np.einsum('ii->', A)

# Outer product
np.einsum('i,j->ij', a, b)
```

### Memory Efficiency Tips

```python
# Use float32 instead of float64
x = np.array(data, dtype=np.float32)

# Avoid copies with in-place operations
x += 1  # In-place
x = x + 1  # Creates copy

# Check if array is contiguous
x.flags['C_CONTIGUOUS']
```

### Profiling Quick Start

```python
# cProfile (stdlib - always available)
import cProfile
cProfile.run('my_function()', sort='cumulative')

# line_profiler
# For x86_64 systems: pip install line_profiler
# For DGX Spark (ARM64): Use version from NGC container or conda
#   conda install -c conda-forge line_profiler
%load_ext line_profiler
%lprun -f my_function my_function()

# Memory profiler
# For basic tracking: tracemalloc (stdlib) works without install
# For detailed profiling: pip install memory_profiler psutil
%load_ext memory_profiler
%memit my_function()
```

### Comparing Implementations

Use the `compare_implementations` utility to benchmark different approaches:

```python
from scripts.profiling_utils import compare_implementations

# Example: Compare loop vs vectorized implementations
def slow_sum(arr):
    total = 0
    for x in arr:
        total += x
    return total

def fast_sum(arr):
    return np.sum(arr)

arr = np.random.rand(100000)

results = compare_implementations(
    functions=[slow_sum, fast_sum],
    names=['Loop', 'NumPy'],
    args=(arr,),
    n_runs=10
)
# Prints comparison table with speedup factors
```

### DGX Spark Compatibility Notes

On DGX Spark with ARM64 architecture:
- **cProfile and tracemalloc** work out of the box (stdlib)
- **line_profiler and memory_profiler** may need conda installation
- Use NGC containers which have pre-built ARM64 packages

### Memory Requirements

| Lab | Peak Memory (Est.) | Notes |
|-----|-------------------|-------|
| Lab 1.2.1: NumPy Broadcasting | ~500 MB | Large array operations for timing comparisons |
| Lab 1.2.2: Preprocessing | ~200 MB | Dataset loading + transformations |
| Lab 1.2.3: Visualization | ~300 MB | Multiple figures + datasets in memory |
| Lab 1.2.4: Einsum | ~400 MB | Batch attention matrix operations |
| Lab 1.2.5: Profiling | ~500 MB | Memory profiling may require additional overhead |

**Total recommended:** 2 GB RAM minimum for comfortable execution.

> **Note:** Memory estimates are approximate. Actual usage depends on dataset sizes and batch configurations. Run cleanup cells between labs to free memory.

---

## Milestone Checklist

- [ ] Broadcasting lab with 100x+ speedup demonstrated
- [ ] Reusable preprocessing pipeline class created
- [ ] Multi-panel visualization dashboard complete
- [ ] Einsum attention implementation working
- [ ] Achieved 10x+ speedup in profiling exercise
- [ ] All notebooks documented with markdown explanations

---

---

## 📖 Study Materials

| Document | Purpose |
|----------|---------|
| [QUICKSTART.md](./QUICKSTART.md) | See 100x+ speedup from vectorization in 5 minutes |
| [STUDY_GUIDE.md](./STUDY_GUIDE.md) | Learning roadmap and core concepts |
| [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) | NumPy, Pandas, einsum, profiling cheatsheet |
| [LAB_PREP.md](./LAB_PREP.md) | Environment setup and lab checklists |
| [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) | Common errors and fixes |
| [FAQ.md](./FAQ.md) | Frequently asked questions |

---

## Next Steps

After completing this module:
1. ✅ Verify all milestones are checked
2. 📁 Save preprocessing pipeline to `scripts/`
3. ➡️ Proceed to [Module 1.3: CUDA Python & GPU Programming](../module-1.3-cuda-python/)

---

## Resources

- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)
- [Einsum Tutorial](https://ajcr.net/Basic-guide-to-einsum/)
