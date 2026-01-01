# Module 1.2: Python for AI/ML - Troubleshooting Guide

## 🔍 Quick Diagnostic

**Before diving into specific errors:**
1. Check you're in NGC container: `import torch; print(torch.__version__)`
2. Verify NumPy/Pandas: `import numpy as np; import pandas as pd`
3. Check memory: `import psutil; print(f"{psutil.virtual_memory().percent}% used")`

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

## 📞 Still Stuck?

1. **Check the docstring**: `help(function_name)` or `function_name?` in Jupyter
2. **Check array shapes**: `print(arr.shape)` before operations
3. **Check dtypes**: `print(arr.dtype)` or `df.dtypes`
4. **Simplify**: Test with tiny arrays first
5. **Search error message**: Copy exact error text
