# Module 1.2: Python for AI/ML - Troubleshooting Guide

This guide expands on the Common Issues from [README.md](./README.md) with detailed solutions.

## 🔍 Quick Diagnostic

**Before diving into specific errors:**
1. Check you're in NGC container: `python -c "import torch; print(torch.__version__)"`
2. Verify NumPy/Pandas: `python -c "import numpy as np; import pandas as pd; print('OK')"`
3. Check memory: `python -c "import psutil; print(f'{psutil.virtual_memory().percent}% used')"`

> **DGX Spark Note:** With 128GB unified memory, memory issues are rare. If you encounter them, ensure you're using `float32` instead of `float64`.

---

## 🚨 Environment Issues

### Error: ModuleNotFoundError: No module named 'numpy'

**Symptoms:**
```
ModuleNotFoundError: No module named 'numpy'
```

**Cause:** Running Python outside the NGC container, or using a bare Python installation.

**Solution:**
Always use the NGC container which has all required packages pre-installed:
```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

**Never** use `pip install numpy` or `pip install torch` on DGX Spark—the NGC container has ARM64-optimized versions.

---

## 🚨 NumPy Errors

### Error: Broadcasting shapes don't match

**Symptoms**:
```
ValueError: operands could not be broadcast together with shapes (10,3) (5,3)
```

**Cause**: Attempting to broadcast arrays with incompatible shapes.

**Solution**:
```python
# Check shapes
print(f"A shape: {A.shape}")
print(f"B shape: {B.shape}")

# Broadcasting rules:
# 1. Align shapes from the RIGHT
# 2. Dimensions must be equal OR one must be 1

# Fix: Add dimension where needed
A = A[:, None, :]  # (10, 3) → (10, 1, 3)
B = B[None, :, :]  # (5, 3) → (1, 5, 3)
# Now broadcasts to (10, 5, 3)
```

---

### Error: Memory allocation failed

**Symptoms**:
```
MemoryError: Unable to allocate X GiB for an array
```

**Solutions**:
```python
# Solution 1: Use smaller dtype
arr = np.zeros((10000, 10000), dtype=np.float32)  # Not float64

# Solution 2: Process in chunks
chunk_size = 1000
for i in range(0, len(data), chunk_size):
    chunk = data[i:i+chunk_size]
    process(chunk)

# Solution 3: Clear existing arrays
del old_array
import gc
gc.collect()
```

---

### Error: Einsum subscripts incorrect

**Symptoms**:
```
ValueError: einstein sum subscripts string contains too many subscripts
```

**Solution**:
```python
# Common mistakes:

# Wrong: repeating output index
np.einsum('ij,jk->ik', A, B)  # ✅ Correct
np.einsum('ij,jk->ii', A, B)  # ❌ 'i' repeated in output

# Wrong: missing index in output
np.einsum('ij,jk->i', A, B)   # ❌ 'k' missing (unless you want sum)
np.einsum('ij,jk->ik', A, B)  # ✅ Correct

# Check dimensions match subscripts
# If A is (3, 4) and B is (4, 5):
# A has 2 dims: i, j
# B has 2 dims: j, k
# j must match (both are 4)
```

---

## 📊 Pandas Errors

### Error: SettingWithCopyWarning

**Symptoms**:
```
SettingWithCopyWarning: A value is trying to be set on a copy of a slice
```

**Cause**: Modifying a view instead of the original DataFrame.

**Solutions**:
```python
# Bad - creates view
subset = df[df['x'] > 0]
subset['y'] = 1  # Warning!

# Good - explicit copy
subset = df[df['x'] > 0].copy()
subset['y'] = 1  # No warning

# Good - use .loc
df.loc[df['x'] > 0, 'y'] = 1  # Modify original directly
```

---

### Error: KeyError when accessing column

**Symptoms**:
```
KeyError: 'column_name'
```

**Solutions**:
```python
# Check column names (watch for spaces/case)
print(df.columns.tolist())

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Check for typos
# 'Age' vs 'age' vs ' Age'
```

---

### Error: Cannot convert to numeric

**Symptoms**:
```
ValueError: could not convert string to float: 'N/A'
```

**Solution**:
```python
# Replace non-numeric values first
df['col'] = pd.to_numeric(df['col'], errors='coerce')  # 'N/A' → NaN

# Then handle NaN
df['col'] = df['col'].fillna(0)  # or appropriate value
```

---

## 📈 Matplotlib Errors

### Issue: Plot not showing in Jupyter

**Solution**:
```python
# Add this at the top of notebook
%matplotlib inline

# Or for interactive plots
%matplotlib widget
```

---

### Issue: Overlapping labels/text

**Solutions**:
```python
# Adjust figure size
fig, ax = plt.subplots(figsize=(12, 8))

# Rotate labels
plt.xticks(rotation=45, ha='right')

# Use tight layout
plt.tight_layout()

# Adjust padding
plt.subplots_adjust(bottom=0.2)
```

---

### Error: Colorbar issues with subplots

**Solution**:
```python
# Wrong: colorbar takes space from all subplots
fig, axes = plt.subplots(1, 2)
im = axes[0].imshow(data)
plt.colorbar(im)  # Messes up layout

# Right: specify which axes
fig, axes = plt.subplots(1, 2)
im = axes[0].imshow(data)
plt.colorbar(im, ax=axes[0])  # Only affects axes[0]
```

---

## 🔧 Profiling Issues

### Issue: line_profiler not working

**Symptoms**:
```
ModuleNotFoundError: No module named 'line_profiler'
```

**Solutions**:
```bash
# Inside NGC container
pip install line_profiler

# Or use conda (more reliable on ARM64)
conda install -c conda-forge line_profiler
```

**Alternative**: Use cProfile (always available)
```python
import cProfile
cProfile.run('my_function()', sort='cumulative')
```

---

### Issue: memory_profiler fails on ARM64

**Solution**: Use tracemalloc (stdlib, always works)
```python
import tracemalloc

tracemalloc.start()
# ... your code ...
current, peak = tracemalloc.get_traced_memory()
print(f"Peak: {peak / 1e6:.1f} MB")
tracemalloc.stop()
```

---

### Issue: Profiler shows wrong function as slow

**Cause**: Profiling overhead or measuring wrong scope.

**Solution**:
```python
# Profile specific function, not whole script
%lprun -f target_function target_function(args)

# Not:
%lprun -f outer_function outer_function()  # Includes all calls
```

---

## 🔄 Performance Issues

### Issue: NumPy operation unexpectedly slow

**Possible causes and fixes**:

```python
# 1. Array not contiguous
if not arr.flags['C_CONTIGUOUS']:
    arr = np.ascontiguousarray(arr)

# 2. Using Python objects instead of native types
arr = np.array([1, 2, 3], dtype=np.float32)  # Not object dtype

# 3. Accidentally using loops
# Bad:
for i in range(len(arr)):
    arr[i] = arr[i] ** 2

# Good:
arr = arr ** 2  # Vectorized
```

---

### Issue: Pandas apply() is slow

**Solution**: Use vectorized operations instead
```python
# Slow - applies Python function row by row
df['new'] = df['col'].apply(lambda x: x ** 2 + 1)

# Fast - vectorized
df['new'] = df['col'] ** 2 + 1

# For complex logic, use np.where or np.select
df['category'] = np.where(df['value'] > 50, 'high', 'low')
```

---

## 🔄 Reset Procedures

### Clear All Variables
```python
# In Jupyter
%reset -f

# Or manually
for var in list(globals().keys()):
    if not var.startswith('_'):
        del globals()[var]
```

### Restart Kernel
```
Kernel → Restart Kernel
```

### Memory Cleanup
```python
import gc

# Delete specific variables
del large_array

# Force garbage collection
gc.collect()

# Check memory
import psutil
print(f"Memory: {psutil.virtual_memory().percent}%")
```

---

## ❓ Frequently Asked Questions

### NumPy & Vectorization

**Q: Why is vectorization so much faster than loops?**

Python loops have overhead for each iteration (type checking, function calls). NumPy operations are implemented in C and process entire arrays at once, avoiding this overhead. The difference is typically 50-200x.

```python
# Loop: Python overhead per element
for i in range(1000000):
    result[i] = data[i] ** 2  # Millions of Python operations

# Vectorized: One C function call
result = data ** 2  # One fast operation on entire array
```

---

**Q: When should I use einsum vs @ for matrix multiplication?**

Use `@` for simple cases, `einsum` for complex multi-dimensional operations.

| Use Case | Syntax |
|----------|--------|
| Simple matmul | `A @ B` |
| Batch matmul | `np.einsum('bij,bjk->bik', A, B)` |
| Attention scores | `np.einsum('bhsd,bhtd->bhst', Q, K)` |
| Custom contractions | `einsum` always |

---

**Q: What's the difference between `np.dot`, `@`, and `np.matmul`?**

- `@` and `np.matmul`: Same thing, handle batched operations correctly
- `np.dot`: Works differently for >2D arrays (rarely what you want)

**Recommendation**: Always use `@` for matrix multiplication.

---

**Q: How do I know if broadcasting will work?**

Align shapes from the right. Dimensions must be equal OR one must be 1.

```python
(10, 3)  + (3,)     # ✅ (3,) broadcasts to (1, 3)
(10, 3)  + (10, 1)  # ✅ (1,) broadcasts across columns
(10, 3)  + (5, 3)   # ❌ 10 ≠ 5 and neither is 1
```

---

### Pandas

**Q: When should I use Pandas vs NumPy?**

| Use Case | Tool |
|----------|------|
| Labeled data, mixed types | Pandas |
| Pure numerical computation | NumPy |
| Data cleaning, exploration | Pandas |
| Matrix operations | NumPy |
| Reading CSV/Excel | Pandas |
| Neural network inputs | NumPy (via `df.values`) |

---

**Q: Why is my Pandas operation slow?**

Common causes:
1. **Using `apply()` with Python functions**: Replace with vectorized operations
2. **Iterating with `iterrows()`**: Never do this for large data
3. **String operations without `.str`**: Use `df['col'].str.contains()` not `apply()`

---

**Q: What file format should I use for large datasets?**

Use Parquet or Feather, not CSV.

| Format | Read Speed | File Size | Preserves Types |
|--------|------------|-----------|-----------------|
| CSV | Slow | Large | No |
| Parquet | Fast | Small | Yes |
| Feather | Fastest | Small | Yes |

---

### Visualization

**Q: Why aren't my Jupyter plots showing?**

Add magic command at top of notebook:

```python
# For static plots
%matplotlib inline

# For interactive plots
%matplotlib widget
```

---

### Profiling

**Q: Which profiler should I use?**

| Profiler | Best For | Availability |
|----------|----------|--------------|
| `%timeit` | Quick timing | Always (Jupyter) |
| `cProfile` | Function-level | Always (stdlib) |
| `line_profiler` | Line-by-line | pip install |
| `tracemalloc` | Memory tracking | Always (stdlib) |

Start with `%timeit`, use `cProfile` for bottleneck hunting.

---

**Q: My optimization only gave 2x speedup. Is that good?**

It depends on what's achievable:
- Loop → vectorized: Expect 50-200x
- NumPy → CuPy (GPU): Expect 10-100x
- Minor refactoring: 1.5-3x is good
- Already optimized: <2x is normal

---

### Memory & Performance

**Q: How much memory does my array use?**

```python
arr = np.zeros((1000, 1000), dtype=np.float32)
print(f"Size: {arr.nbytes / 1e6:.1f} MB")  # 4.0 MB
```

Quick reference: float64 = 8 bytes, float32 = 4 bytes, int32 = 4 bytes, int8 = 1 byte.

---

**Q: Should I always use float32 instead of float64?**

For ML, yes. For scientific computing, it depends.

| Precision | Size | Use Case |
|-----------|------|----------|
| float64 | 8 bytes | Scientific computing, numerical stability |
| float32 | 4 bytes | Machine learning, GPU operations |
| float16 | 2 bytes | Inference only (some training) |
| bfloat16 | 2 bytes | Training (better range than float16) |

---

### DGX Spark Specific

**Q: Why does profiling show different results than on my laptop?**

DGX Spark uses ARM64/aarch64 architecture, which has different performance characteristics than x86. With 128GB unified memory, DGX Spark can handle much larger datasets than typical laptops.

---

**Q: Why must I use the NGC container instead of pip install?**

DGX Spark uses ARM64/aarch64 architecture. Many Python packages don't have pre-built ARM64 wheels on PyPI. **Never use `pip install torch` on DGX Spark**—it will fail or install an incompatible version.

---

**Q: Can I use Numba on DGX Spark?**

Yes, but use the version from the NGC container. Don't pip install numba—use the version pre-installed in the NGC container, which is compiled for ARM64.

---

## 📞 Still Stuck?

1. **Check the docstring**: `help(function_name)` or `function_name?` in Jupyter
2. **Check array shapes**: `print(arr.shape)` before operations
3. **Check dtypes**: `print(arr.dtype)` or `df.dtypes`
4. **Simplify**: Test with tiny arrays first
5. **Search error message**: Copy exact error text
