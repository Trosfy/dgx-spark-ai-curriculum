# Module 1.3: CUDA Python & GPU Programming - Quick Reference

## 🚀 Numba CUDA Basics

### Basic Kernel Pattern
```python
from numba import cuda
import numpy as np

@cuda.jit
def my_kernel(input_arr, output_arr):
    idx = cuda.grid(1)  # 1D grid
    if idx < input_arr.size:
        output_arr[idx] = input_arr[idx] * 2

# Launch
threads_per_block = 256
blocks = (arr.size + threads_per_block - 1) // threads_per_block
my_kernel[blocks, threads_per_block](d_input, d_output)
```

### 2D Kernel Pattern
```python
@cuda.jit
def kernel_2d(matrix, output):
    i, j = cuda.grid(2)  # 2D grid
    if i < matrix.shape[0] and j < matrix.shape[1]:
        output[i, j] = matrix[i, j] ** 2

# Launch 2D
threads = (16, 16)  # 256 threads per block
blocks = (
    (rows + threads[0] - 1) // threads[0],
    (cols + threads[1] - 1) // threads[1]
)
kernel_2d[blocks, threads](d_matrix, d_output)
```

### Thread Indexing
```python
# 1D grid
idx = cuda.grid(1)
# Equivalent to:
idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

# 2D grid
i, j = cuda.grid(2)
# Equivalent to:
i = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
j = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
```

## 📦 Memory Operations

### Host ↔ Device Transfer
```python
# Host to device
d_arr = cuda.to_device(np_array)

# Device to host
result = d_arr.copy_to_host()

# Pre-allocate on device
d_arr = cuda.device_array((1000, 1000), dtype=np.float32)
d_arr = cuda.device_array_like(np_array)
```

### Shared Memory
```python
@cuda.jit
def kernel_with_shared(input_arr, output_arr):
    # Allocate shared memory
    shared = cuda.shared.array(256, dtype=float32)

    tx = cuda.threadIdx.x
    idx = cuda.grid(1)

    # Load to shared
    if idx < input_arr.size:
        shared[tx] = input_arr[idx]

    # Wait for all threads
    cuda.syncthreads()

    # Use shared memory
    if idx < output_arr.size:
        output_arr[idx] = shared[tx] * 2
```

### Tiled Matrix Multiply Pattern
```python
TILE_SIZE = 16

@cuda.jit
def matmul_tiled(A, B, C):
    sA = cuda.shared.array((TILE_SIZE, TILE_SIZE), dtype=float32)
    sB = cuda.shared.array((TILE_SIZE, TILE_SIZE), dtype=float32)

    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    row = cuda.blockIdx.y * TILE_SIZE + ty
    col = cuda.blockIdx.x * TILE_SIZE + tx

    acc = 0.0
    for t in range((A.shape[1] + TILE_SIZE - 1) // TILE_SIZE):
        # Load tiles
        if row < A.shape[0] and t * TILE_SIZE + tx < A.shape[1]:
            sA[ty, tx] = A[row, t * TILE_SIZE + tx]
        else:
            sA[ty, tx] = 0

        if t * TILE_SIZE + ty < B.shape[0] and col < B.shape[1]:
            sB[ty, tx] = B[t * TILE_SIZE + ty, col]
        else:
            sB[ty, tx] = 0

        cuda.syncthreads()

        # Compute
        for k in range(TILE_SIZE):
            acc += sA[ty, k] * sB[k, tx]

        cuda.syncthreads()

    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = acc
```

## 🔄 Atomic Operations

```python
@cuda.jit
def kernel_with_atomic(data, result):
    idx = cuda.grid(1)
    if idx < data.size:
        # Atomic add to avoid race conditions
        cuda.atomic.add(result, 0, data[idx])

# Available atomic operations:
# cuda.atomic.add(array, index, value)
# cuda.atomic.max(array, index, value)
# cuda.atomic.min(array, index, value)
# cuda.atomic.compare_and_swap(array, index, old, val)
```

## 🐍 CuPy (NumPy on GPU)

### Basic Usage
```python
import cupy as cp

# Create on GPU
a = cp.array([1, 2, 3])
b = cp.zeros((1000, 1000))
c = cp.random.rand(1000, 1000)

# Operations (same as NumPy)
d = a + b
e = cp.dot(c, c.T)
f = cp.sum(c, axis=0)

# Transfer to CPU
result = cp.asnumpy(d)  # or d.get()
```

### NumPy ↔ CuPy
```python
# NumPy → CuPy
np_arr = np.random.rand(1000)
cp_arr = cp.asarray(np_arr)

# CuPy → NumPy
np_result = cp.asnumpy(cp_arr)
# or
np_result = cp_arr.get()
```

### CuPy ↔ PyTorch
```python
import torch
import cupy as cp
from cupy.cuda import Device

# CuPy → PyTorch (zero-copy)
cp_arr = cp.array([1, 2, 3])
tensor = torch.as_tensor(cp_arr, device='cuda')

# PyTorch → CuPy (zero-copy)
tensor = torch.randn(100, device='cuda')
cp_arr = cp.from_dlpack(tensor)
```

## 📊 Profiling Commands

### Basic Timing
```python
import time
from numba import cuda

# Warm-up
kernel[blocks, threads](args)
cuda.synchronize()

# Timed run
start = time.time()
kernel[blocks, threads](args)
cuda.synchronize()  # Wait for GPU to finish!
elapsed = time.time() - start
```

### CUDA Events (More Precise)
```python
start = cuda.event()
end = cuda.event()

start.record()
kernel[blocks, threads](args)
end.record()
end.synchronize()

elapsed_ms = cuda.event_elapsed_time(start, end)
```

### Nsight Systems (Command Line)
```bash
# Profile script
nsys profile python my_script.py

# Generate report
nsys stats report.nsys-rep

# View in GUI
nsys-ui report.nsys-rep
```

## ⚠️ Common Patterns

### Bounds Checking (Always Do This!)
```python
@cuda.jit
def safe_kernel(arr, output):
    idx = cuda.grid(1)
    if idx < arr.size:  # ← Always check bounds!
        output[idx] = arr[idx] * 2
```

### Coalesced Access (Good)
```python
# ✅ Good - adjacent threads access adjacent memory
@cuda.jit
def coalesced(data, output):
    idx = cuda.grid(1)
    if idx < data.size:
        output[idx] = data[idx] * 2  # Thread 0→data[0], Thread 1→data[1]
```

### Strided Access (Bad)
```python
# ❌ Bad - adjacent threads access strided memory
@cuda.jit
def strided(data, output, stride):
    idx = cuda.grid(1)
    if idx < data.size // stride:
        output[idx] = data[idx * stride]  # Thread 0→data[0], Thread 1→data[stride]
```

### Synchronization
```python
@cuda.jit
def with_sync(shared_data):
    tx = cuda.threadIdx.x

    # Load data
    shared_data[tx] = ...

    cuda.syncthreads()  # ← All threads must reach here!

    # Use data loaded by other threads
    result = shared_data[(tx + 1) % 32]
```

## 📊 Launch Configuration

### Threads Per Block
```python
# Common choices
threads = 128  # Good for simple kernels
threads = 256  # Most common, good occupancy
threads = 512  # For high register usage
threads = 1024  # Max allowed

# For 2D:
threads = (16, 16)  # 256 total
threads = (32, 32)  # 1024 total
```

### Calculate Blocks
```python
# 1D
blocks = (n + threads - 1) // threads

# 2D
blocks = (
    (cols + threads[0] - 1) // threads[0],
    (rows + threads[1] - 1) // threads[1]
)
```

## ⚠️ Common Mistakes

| Mistake | Fix |
|---------|-----|
| Missing bounds check | Add `if idx < arr.size` |
| Missing `cuda.syncthreads()` | Add after shared memory writes |
| Forgot `cuda.synchronize()` | Add before timing measurement |
| Using pip numba on ARM64 | Use NGC container (pre-configured for DGX Spark) |
| Not using float32 | Specify `dtype=np.float32` |
| Strided memory access | Rearrange for coalesced access |

## 🔗 Quick Links
- [Numba CUDA Docs](https://numba.readthedocs.io/en/stable/cuda/index.html)
- [CuPy Docs](https://docs.cupy.dev/en/stable/)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [CUDA C++ Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
