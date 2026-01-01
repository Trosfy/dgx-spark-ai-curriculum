# Module 1.2: Python for AI/ML - FAQ

## NumPy & Vectorization

### Q: Why is vectorization so much faster than loops?
**A**: Python loops have overhead for each iteration (type checking, function calls). NumPy operations are implemented in C and process entire arrays at once, avoiding this overhead. The difference is typically 50-200x.

```python
# Loop: Python overhead per element
for i in range(1000000):
    result[i] = data[i] ** 2  # Millions of Python operations

# Vectorized: One C function call
result = data ** 2  # One fast operation on entire array
```

---

### Q: When should I use einsum vs @ for matrix multiplication?
**A**: Use `@` for simple cases, `einsum` for complex multi-dimensional operations.

| Use Case | Syntax |
|----------|--------|
| Simple matmul | `A @ B` |
| Batch matmul | `np.einsum('bij,bjk->bik', A, B)` |
| Attention scores | `np.einsum('bhsd,bhtd->bhst', Q, K)` |
| Custom contractions | `einsum` always |

---

### Q: What's the difference between `np.dot`, `@`, and `np.matmul`?
**A**:
- `@` and `np.matmul`: Same thing, handle batched operations correctly
- `np.dot`: Works differently for >2D arrays (rarely what you want)

**Recommendation**: Always use `@` for matrix multiplication.

---

### Q: How do I know if broadcasting will work?
**A**: Align shapes from the right. Dimensions must be equal OR one must be 1.

```python
(10, 3)  + (3,)     # ✅ (3,) broadcasts to (1, 3)
(10, 3)  + (10, 1)  # ✅ (1,) broadcasts across columns
(10, 3)  + (5, 3)   # ❌ 10 ≠ 5 and neither is 1
```

---

## Pandas

### Q: When should I use Pandas vs NumPy?
**A**:

| Use Case | Tool |
|----------|------|
| Labeled data, mixed types | Pandas |
| Pure numerical computation | NumPy |
| Data cleaning, exploration | Pandas |
| Matrix operations | NumPy |
| Reading CSV/Excel | Pandas |
| Neural network inputs | NumPy (via `df.values`) |

---

### Q: Why is my Pandas operation slow?
**A**: Common causes:

1. **Using `apply()` with Python functions**: Replace with vectorized operations
2. **Iterating with `iterrows()`**: Never do this for large data
3. **String operations without `.str`**: Use `df['col'].str.contains()` not `apply()`

```python
# Slow
df['new'] = df.apply(lambda row: row['a'] + row['b'], axis=1)

# Fast
df['new'] = df['a'] + df['b']
```

---

### Q: What file format should I use for large datasets?
**A**: Use Parquet or Feather, not CSV.

| Format | Read Speed | File Size | Preserves Types |
|--------|------------|-----------|-----------------|
| CSV | Slow | Large | No |
| Parquet | Fast | Small | Yes |
| Feather | Fastest | Small | Yes |

```python
# Save
df.to_parquet('data.parquet')

# Load
df = pd.read_parquet('data.parquet')
```

---

## Visualization

### Q: How do I make plots look publication-ready?
**A**: Key settings:

```python
import matplotlib.pyplot as plt

# Set style globally
plt.style.use('seaborn-v0_8-whitegrid')

# Figure settings
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

# Font sizes
ax.set_xlabel('X Label', fontsize=12)
ax.set_ylabel('Y Label', fontsize=12)
ax.set_title('Title', fontsize=14, fontweight='bold')

# Clean up
plt.tight_layout()
plt.savefig('figure.png', dpi=300, bbox_inches='tight')
```

---

### Q: Why aren't my Jupyter plots showing?
**A**: Add magic command at top of notebook:

```python
# For static plots
%matplotlib inline

# For interactive plots
%matplotlib widget
```

---

## Profiling

### Q: Which profiler should I use?
**A**:

| Profiler | Best For | Availability |
|----------|----------|--------------|
| `%timeit` | Quick timing | Always (Jupyter) |
| `cProfile` | Function-level | Always (stdlib) |
| `line_profiler` | Line-by-line | pip install |
| `tracemalloc` | Memory tracking | Always (stdlib) |

Start with `%timeit`, use `cProfile` for bottleneck hunting.

---

### Q: How do I interpret profiling output?
**A**: Focus on:

1. **cumtime**: Total time including subcalls (where to optimize)
2. **tottime**: Time in function only (actual bottleneck)
3. **ncalls**: Number of calls (reduce if high)

```
   ncalls  tottime  cumtime  filename:lineno(function)
     1000    5.234    5.234  slow_function:10(process)  ← Optimize this
        1    0.001   10.456  main:1(run)                ← Just a wrapper
```

---

### Q: My optimization only gave 2x speedup. Is that good?
**A**: It depends on what's achievable:

- Loop → vectorized: Expect 50-200x
- NumPy → CuPy (GPU): Expect 10-100x
- Minor refactoring: 1.5-3x is good
- Already optimized: <2x is normal

If you're only getting 2x on a loop→vectorized change, check that you actually eliminated the loop!

---

## Memory & Performance

### Q: How much memory does my array use?
**A**:

```python
arr = np.zeros((1000, 1000), dtype=np.float32)
print(f"Size: {arr.nbytes / 1e6:.1f} MB")  # 4.0 MB
```

Quick reference:
- float64: 8 bytes per element
- float32: 4 bytes per element
- int32: 4 bytes per element
- int8: 1 byte per element

---

### Q: Should I always use float32 instead of float64?
**A**: For ML, yes. For scientific computing, it depends.

| Precision | Size | Use Case |
|-----------|------|----------|
| float64 | 8 bytes | Scientific computing, numerical stability |
| float32 | 4 bytes | Machine learning, GPU operations |
| float16 | 2 bytes | Inference only (some training) |
| bfloat16 | 2 bytes | Training (better range than float16) |

---

## DGX Spark Specific

### Q: Why does profiling show different results than on my laptop?
**A**: DGX Spark uses ARM64/aarch64 architecture, which has different performance characteristics than x86:
- Some operations are faster (optimized SIMD operations)
- Some operations are slower (x86-optimized libraries may not be available)
- Memory bandwidth is different (273 GB/s on DGX Spark)

With 128GB unified memory, DGX Spark can handle much larger datasets than typical laptops. Always benchmark on target hardware for accurate performance comparisons.

---

### Q: Why must I use the NGC container instead of pip install?
**A**: DGX Spark uses ARM64/aarch64 architecture. Many Python packages (especially PyTorch and its dependencies) don't have pre-built ARM64 wheels on PyPI. The NGC containers include:
- Pre-built ARM64-compatible packages
- CUDA-optimized libraries for the Blackwell GPU
- Tested, validated configurations

**Never use `pip install torch` on DGX Spark**—it will fail or install an incompatible version.

---

### Q: Can I use Numba on DGX Spark?
**A**: Yes, but use the version from the NGC container:

```python
from numba import jit

@jit(nopython=True)
def fast_function(x):
    return x ** 2
```

Don't pip install numba—use the version pre-installed in the NGC container, which is compiled for ARM64.

---

## Still Have Questions?

- Check [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for error-specific help
- See [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) for syntax
- Review [STUDY_GUIDE.md](./STUDY_GUIDE.md) for concepts
