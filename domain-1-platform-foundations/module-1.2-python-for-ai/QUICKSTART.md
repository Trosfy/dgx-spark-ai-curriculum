# Module 1.2: Python for AI/ML - Quickstart

## ⏱️ Time: ~5 minutes

## 🎯 What You'll Build
Experience a 100x+ speedup by replacing loops with NumPy vectorization.

## ✅ Before You Start
- [ ] Verified Python prerequisites (see [PREREQUISITES.md](./PREREQUISITES.md))
- [ ] NGC PyTorch container running (see [LAB_PREP.md](./LAB_PREP.md))
- [ ] Python shell or Jupyter ready

## 🚀 Let's Go!

### Step 1: Create Test Data
```python
import numpy as np
import time

# 1 million random numbers
data = np.random.rand(1_000_000)
```

### Step 2: The Slow Way (Loops)
```python
def slow_sum_squares(arr):
    total = 0
    for x in arr:
        total += x ** 2
    return total

start = time.time()
result_slow = slow_sum_squares(data)
slow_time = time.time() - start
print(f"Loop: {slow_time:.4f}s")
```

### Step 3: The Fast Way (Vectorized)
```python
def fast_sum_squares(arr):
    return np.sum(arr ** 2)

start = time.time()
result_fast = fast_sum_squares(data)
fast_time = time.time() - start
print(f"NumPy: {fast_time:.4f}s")
```

### Step 4: See the Speedup
```python
speedup = slow_time / fast_time
print(f"\n🚀 Speedup: {speedup:.1f}x faster!")
print(f"Results match: {np.isclose(result_slow, result_fast)}")
```

**Expected output:**
```
Loop: 0.4521s
NumPy: 0.0024s

🚀 Speedup: 188.4x faster!
Results match: True
```

## 🎉 You Did It!

You just saw the power of vectorization:
- ✅ Same result, 100x+ faster
- ✅ Cleaner, more readable code
- ✅ Foundation for all AI/ML work

In the full module, you'll learn:
- Broadcasting for complex operations
- Einsum for attention mechanisms
- Pandas for data preprocessing
- Profiling to find bottlenecks

## ▶️ Next Steps
1. **Understand broadcasting**: Read [STUDY_GUIDE.md](./STUDY_GUIDE.md)
2. **See all patterns**: Check [QUICK_REFERENCE.md](./QUICK_REFERENCE.md)
3. **Start Lab 1**: Open `labs/lab-1.2.1-numpy-broadcasting-lab.ipynb`
