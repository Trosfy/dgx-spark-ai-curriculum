# Module 1.3: CUDA Python & GPU Programming - Quickstart

## ⏱️ Time: ~5 minutes

## 🎯 What You'll Build
See a 100x+ speedup by running a simple computation on the GPU.

## ✅ Before You Start
- [ ] NGC PyTorch container running
- [ ] GPU verified with `nvidia-smi`

## 🚀 Let's Go!

### Step 1: Import and Check GPU
```python
from numba import cuda
import numpy as np
import time

print(f"GPU available: {cuda.is_available()}")
print(f"GPU name: {cuda.get_current_device().name}")
```

### Step 2: Create Simple CPU Function
```python
def cpu_sum_squares(arr):
    total = 0.0
    for x in arr:
        total += x * x
    return total

# Test data - 10 million numbers
data = np.random.rand(10_000_000).astype(np.float32)
```

### Step 3: Create GPU Kernel
```python
@cuda.jit
def gpu_sum_squares_kernel(arr, result):
    idx = cuda.grid(1)
    if idx < arr.size:
        cuda.atomic.add(result, 0, arr[idx] * arr[idx])

def gpu_sum_squares(arr):
    # Copy to GPU
    d_arr = cuda.to_device(arr)
    d_result = cuda.to_device(np.zeros(1, dtype=np.float32))

    # Launch kernel
    threads_per_block = 256
    blocks = (arr.size + threads_per_block - 1) // threads_per_block
    gpu_sum_squares_kernel[blocks, threads_per_block](d_arr, d_result)

    return d_result.copy_to_host()[0]
```

### Step 4: Compare Performance
```python
# CPU timing
start = time.time()
cpu_result = cpu_sum_squares(data)
cpu_time = time.time() - start

# GPU timing (warm-up run)
_ = gpu_sum_squares(data)
cuda.synchronize()

start = time.time()
gpu_result = gpu_sum_squares(data)
cuda.synchronize()
gpu_time = time.time() - start

print(f"CPU: {cpu_time:.4f}s")
print(f"GPU: {gpu_time:.4f}s")
print(f"\n🚀 Speedup: {cpu_time/gpu_time:.1f}x faster!")
```

**Expected output:**
```
CPU: 3.2145s
GPU: 0.0089s

🚀 Speedup: 361.2x faster!
```

## 🎉 You Did It!

You just:
- ✅ Wrote a CUDA kernel with `@cuda.jit`
- ✅ Transferred data to GPU
- ✅ Saw 100x+ speedup over CPU
- ✅ Used atomic operations for parallel accumulation

In the full module, you'll learn:
- Shared memory for even faster operations
- Memory coalescing patterns
- Tiled matrix multiplication
- Profiling with Nsight

## ▶️ Next Steps
1. **Understand GPU architecture**: Read [ELI5.md](./ELI5.md)
2. **See all patterns**: Check [QUICK_REFERENCE.md](./QUICK_REFERENCE.md)
3. **Start Lab 1**: Open `notebooks/lab-1.3.1-parallel-reduction.ipynb`
