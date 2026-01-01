# Module 1.2: Python for AI/ML - Study Guide

## 🎯 Learning Objectives

These objectives match the Learning Outcomes in [README.md](./README.md).

By the end of this module, you will be able to:
1. ✅ **Write** efficient NumPy code for tensor operations (vectorization, broadcasting)
2. ✅ **Manipulate** datasets using Pandas for ML preprocessing
3. ✅ **Create** publication-quality visualizations with Matplotlib/Seaborn
4. ✅ **Profile** and optimize Python code for performance

## 🗺️ Module Roadmap

| # | Lab | Focus | Time | Key Outcome |
|---|-----|-------|------|-------------|
| 1 | NumPy Broadcasting | Vectorization | ~2 hr | 100x+ speedup over loops |
| 2 | Dataset Preprocessing | Pandas pipelines | ~2 hr | Reusable Preprocessor class |
| 3 | Visualization Dashboard | Matplotlib/Seaborn | ~2 hr | Publication-ready figures |
| 4 | Einsum Mastery | Matrix notation | ~2 hr | Attention mechanism implementation |
| 5 | Profiling Exercise | Performance tuning | ~2 hr | 10x+ optimization |

**Total time**: ~10 hours

## 🔑 Core Concepts

### NumPy Broadcasting
**What**: Rules that allow NumPy to perform operations on arrays of different shapes without explicit loops.
**Why it matters**: Eliminates Python loops, achieving 10-100x speedups. Essential for efficient tensor operations.
**First appears in**: Lab 1 (NumPy Broadcasting)

### Vectorization
**What**: Replacing Python loops with array operations that execute in optimized C code.
**Why it matters**: Python loops are slow; NumPy operations are fast. This is the #1 optimization technique.
**First appears in**: Lab 1

### Einsum Notation
**What**: A compact notation for expressing complex tensor operations like batched matrix multiplication.
**Why it matters**: Used in attention mechanisms. `einsum('bhsd,bhtd->bhst', Q, K)` computes attention scores.
**First appears in**: Lab 4 (Einsum Mastery)

### Memory Layout
**What**: How arrays are stored in memory (row-major vs column-major).
**Why it matters**: Accessing memory in order is 10-100x faster than random access.
**First appears in**: Lab 1

## 🔗 How This Module Connects

```
    Module 1.1              Module 1.2                Module 1.3
    ───────────────────────────────────────────────────────────────
    DGX Spark Setup   ──►   Python for AI/ML    ──►   CUDA Python

    Container ready         NumPy fundamentals        GPU arrays
    GPU verified            Broadcasting              CuPy (NumPy on GPU)
                            Profiling                 Kernel optimization
```

**Builds on**:
- Module 1.1: NGC container setup
- Basic Python programming (prerequisite)

**Prepares for**:
- **Module 1.3**: CuPy extends NumPy to GPU (same syntax!)
- **Module 1.4**: NumPy for manual gradient computation
- **Module 1.5**: Building neural networks with NumPy
- **All modules**: Data preprocessing patterns used throughout

## 📖 Recommended Approach

**Standard path** (10 hours):
1. Lab 1: Master broadcasting—this is foundational
2. Lab 2: Build preprocessing pipeline you'll reuse
3. Lab 3: Learn visualization patterns
4. Lab 4: Einsum is crucial for understanding transformers
5. Lab 5: Profile and optimize

**Quick path** (if experienced with NumPy, 5-6 hours):
1. Skim Lab 1, focus on broadcasting exercises
2. Complete Lab 2 preprocessing pipeline
3. Skip Lab 3 if familiar with Matplotlib
4. Focus on Lab 4 einsum attention implementation
5. Complete Lab 5 profiling

## 📊 Key Patterns to Master

### Broadcasting Rules
```python
# Shapes are compatible when:
# 1. They are equal, OR
# 2. One of them is 1

(10, 3)  + (3,)     # ✅ (10, 3) - (3,) broadcasts to (1, 3)
(10, 3)  + (10, 1)  # ✅ (10, 3) - columns broadcast
(10, 3)  + (5, 3)   # ❌ 10 ≠ 5 and neither is 1
```

### Einsum Patterns
```python
# Matrix multiply: (M, K) @ (K, N) -> (M, N)
np.einsum('mk,kn->mn', A, B)

# Batch matrix multiply: (B, M, K) @ (B, K, N) -> (B, M, N)
np.einsum('bmk,bkn->bmn', A, B)

# Attention scores: Q @ K.T
np.einsum('bhsd,bhtd->bhst', Q, K)  # (batch, heads, seq, dim)
```

### Performance Rules of Thumb
| Operation | Relative Speed |
|-----------|----------------|
| Python loop | 1x (baseline) |
| NumPy vectorized | 100x |
| Numba JIT | 50-200x |
| CuPy (GPU) | 500-1000x |

## 📋 Before You Start

1. **Verify Prerequisites**: Complete the self-check in [PREREQUISITES.md](./PREREQUISITES.md)
2. **Try the Quickstart**: See [QUICKSTART.md](./QUICKSTART.md) for a 5-minute vectorization demo
3. **Have Reference Handy**: Keep [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) open for einsum patterns
4. **Start NGC Container**: Ensure the container is running (see [LAB_PREP.md](./LAB_PREP.md))
