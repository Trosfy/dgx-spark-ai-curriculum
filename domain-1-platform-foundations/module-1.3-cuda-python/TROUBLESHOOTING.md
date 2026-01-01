# Module 1.3: CUDA Python & GPU Programming - Troubleshooting

## Quick Fixes

| Issue | Solution |
|-------|----------|
| `numba.cuda.cudadrv.error.CudaSupportError` | Use NGC container, not pip-installed Numba |
| Kernel runs but wrong results | Check bounds: `if idx < data.size` |
| Slow performance despite parallelization | Check for thread divergence or uncoalesced access |
| `cuda.syncthreads()` deadlock | Ensure all threads in block reach the barrier |
| Out of memory on kernel launch | Reduce block size or shared memory usage |

---

## Detailed Solutions

### CudaSupportError: CUDA not available

**Symptoms:**
```python
numba.cuda.cudadrv.error.CudaSupportError: Error at driver init:
CUDA driver library cannot be found
```

**Cause:** Numba installed via pip doesn't have ARM64-compatible CUDA bindings.

**Solution:** Use the NGC container which has Numba pre-configured for DGX Spark:

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

**Verification:**
```python
from numba import cuda
print(f"CUDA available: {cuda.is_available()}")
# Expected: CUDA available: True
```

---

### Kernel Produces Wrong Results

**Symptoms:** Kernel runs without errors but output values are incorrect or contain garbage.

**Common Causes:**

1. **Missing bounds check:**
```python
# ❌ WRONG: No bounds check
@cuda.jit
def kernel(data, output):
    idx = cuda.grid(1)
    output[idx] = data[idx] * 2  # Crashes or wrong values if idx >= size

# ✅ CORRECT: Always check bounds
@cuda.jit
def kernel(data, output):
    idx = cuda.grid(1)
    if idx < data.size:  # ← Safety check
        output[idx] = data[idx] * 2
```

2. **Missing synchronization after shared memory writes:**
```python
# ❌ WRONG: Reading before write completes
shared[tx] = input[idx]
result = shared[(tx + 1) % 32]  # Race condition!

# ✅ CORRECT: Wait for all threads
shared[tx] = input[idx]
cuda.syncthreads()  # ← All threads wait here
result = shared[(tx + 1) % 32]
```

3. **Integer overflow in index calculation:**
```python
# ❌ WRONG: Overflow for large arrays
idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # May overflow int32

# ✅ CORRECT: Use cuda.grid() helper
idx = cuda.grid(1)  # Handles index calculation correctly
```

---

### Slow Performance Despite Parallelization

**Symptoms:** GPU code is slower than expected or even slower than CPU.

**Diagnostic checklist:**

1. **Thread divergence:**
```python
# ❌ BAD: Different threads take different paths
@cuda.jit
def kernel(data, output):
    idx = cuda.grid(1)
    if data[idx] > 0:    # Half threads do this
        output[idx] = data[idx] * 2
    else:                 # Half threads wait, then do this
        output[idx] = data[idx] * 3
# Both branches execute sequentially for the warp!
```

2. **Uncoalesced memory access:**
```python
# ❌ BAD: Strided access (slow)
output[idx] = data[idx * stride]  # Adjacent threads access non-adjacent memory

# ✅ GOOD: Coalesced access (fast)
output[idx] = data[idx]  # Adjacent threads access adjacent memory
```

3. **Kernel launch overhead dominates:**
```python
# ❌ BAD: Launching kernel for tiny array
data = np.ones(100)  # Too small for GPU!
kernel[blocks, threads](data)

# ✅ GOOD: Use GPU for large arrays (10,000+ elements)
data = np.ones(1_000_000)
kernel[blocks, threads](data)
```

4. **Not warming up before benchmarking:**
```python
# ❌ WRONG: Timing includes compilation
start = time.time()
kernel[blocks, threads](data)  # First call includes JIT compilation!
elapsed = time.time() - start

# ✅ CORRECT: Warm up first
kernel[blocks, threads](data)  # Warm-up
cuda.synchronize()
start = time.time()
kernel[blocks, threads](data)  # Timed run
cuda.synchronize()
elapsed = time.time() - start
```

---

### syncthreads() Deadlock

**Symptoms:** Kernel hangs indefinitely.

**Cause:** Not all threads in a block reach `cuda.syncthreads()`.

```python
# ❌ WRONG: syncthreads inside conditional
@cuda.jit
def kernel(data):
    tx = cuda.threadIdx.x
    if tx < 128:  # Only half the threads!
        # ... do work ...
        cuda.syncthreads()  # ← DEADLOCK! Other threads never reach this

# ✅ CORRECT: All threads reach syncthreads
@cuda.jit
def kernel(data):
    tx = cuda.threadIdx.x
    if tx < 128:
        pass  # ... do work ...
    cuda.syncthreads()  # ← All threads reach this
```

**Rule:** `cuda.syncthreads()` must be reachable by **ALL** threads in the block.

---

### Out of Memory on Kernel Launch

**Symptoms:**
```python
numba.cuda.cudadrv.driver.CudaAPIError:
[701] Call to cuLaunchKernel results in CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
```

**Causes and solutions:**

1. **Too much shared memory:**
```python
# ❌ BAD: Shared memory too large
sdata = cuda.shared.array(shape=(1024, 1024), dtype=float32)  # 4 MB! Too big

# ✅ GOOD: Keep shared memory small
sdata = cuda.shared.array(shape=(256,), dtype=float32)  # 1 KB
```

2. **Too many registers per thread:**
```python
# Complex kernels with many local variables use more registers
# Reduce block size to compensate
threads = 128  # Instead of 256 or 512
```

3. **Too many threads per block:**
```python
# ❌ BAD: Exceeds maximum
threads = (32, 64)  # 2048 threads! Max is usually 1024

# ✅ GOOD: Stay within limits
threads = (32, 32)  # 1024 threads
```

---

### CuPy Not Available

**Symptoms:**
```python
ModuleNotFoundError: No module named 'cupy'
```

**Solution:** Do NOT use `pip install cupy-cuda12x` on DGX Spark (ARM64). Use the NGC container:

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

The NGC container has CuPy pre-installed and optimized for ARM64.

---

### Incorrect Timing Measurements

**Symptoms:** Benchmarks show unrealistically fast GPU times.

**Cause:** GPU operations are asynchronous. You're measuring launch time, not execution time.

```python
# ❌ WRONG: Not synchronizing
start = time.time()
kernel[blocks, threads](data)  # Async! Returns immediately
elapsed = time.time() - start  # Measures launch time only (~0.1ms)

# ✅ CORRECT: Synchronize before measuring
start = time.time()
kernel[blocks, threads](data)
cuda.synchronize()  # Wait for GPU to finish
elapsed = time.time() - start  # Measures actual execution time
```

---

### Memory Leaks in Jupyter

**Symptoms:** GPU memory usage grows over time; eventually out of memory.

**Solution:** Clean up after experiments:

```python
import gc

# Delete large arrays
del large_array
del d_array

# Python garbage collection
gc.collect()

# Clear CUDA context (Numba)
from numba import cuda
try:
    cuda.current_context().reset()
except Exception:
    pass

# Clear PyTorch cache
import torch
torch.cuda.empty_cache()

# Clear CuPy memory pool
import cupy as cp
cp.get_default_memory_pool().free_all_blocks()
```

---

### Can't Import torch.cuda

**Symptoms:**
```python
>>> torch.cuda.is_available()
False
```

**Cause:** PyTorch installed via pip doesn't support ARM64 CUDA.

**Solution:** Use NGC container (PyTorch pre-installed with CUDA support):

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

---

## Getting Help

If you encounter issues not covered here:

1. Check the [Numba CUDA Documentation](https://numba.readthedocs.io/en/stable/cuda/index.html)
2. Check the [CuPy Documentation](https://docs.cupy.dev/en/stable/)
3. Review your kernel for common patterns (bounds check, syncthreads, coalescing)
4. Use `print()` statements in host code to debug (kernels don't support print)

---

## Hardware Reference (DGX Spark)

| Spec | Value |
|------|-------|
| GPU | NVIDIA Blackwell GB10 Superchip |
| CUDA Cores | 6,144 |
| Tensor Cores | 192 (5th generation) |
| Memory | 128GB unified memory |
| Memory Bandwidth | 273 GB/s |
| Warp Size | 32 threads |
| Max Threads/Block | 1024 |
| Architecture | ARM64/aarch64 |
