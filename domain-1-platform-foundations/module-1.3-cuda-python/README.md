# Module 1.3: CUDA Python & GPU Programming

**Domain:** 1 - Platform Foundations
**Duration:** Week 3 (10-12 hours)
**Prerequisites:** Module 1.2 (Python for AI/ML), basic understanding of computer architecture
**Priority:** P0 Critical

---

## Overview

This module teaches you to harness the full power of your DGX Spark's 6,144 CUDA cores. You'll understand GPU architecture, write parallel algorithms, and optimize memory access patterns—skills essential for maximizing the 128GB unified memory advantage.

By the end of this module, you'll be able to write custom CUDA kernels, profile GPU code, and understand why certain operations are fast or slow on the GPU.

---

## Learning Outcomes

By the end of this module, you will be able to:

- ✅ Explain GPU architecture and parallel computing principles
- ✅ Write CUDA kernels using Numba and CuPy
- ✅ Optimize memory access patterns for GPU performance
- ✅ Profile and debug GPU code using NVIDIA tools

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 1.3.1 | Explain GPU memory hierarchy (global, shared, registers) | Understand |
| 1.3.2 | Write parallel algorithms using CUDA Python (Numba) | Apply |
| 1.3.3 | Optimize memory coalescing and reduce bank conflicts | Apply |
| 1.3.4 | Profile GPU code with Nsight Systems/Compute | Analyze |

---

## Topics

### 1.3.1 GPU Architecture Fundamentals

- **SIMT Execution Model**
  - Single Instruction, Multiple Threads
  - How 6,144 CUDA cores work together
  - Warp execution (32 threads)
  - Thread divergence and its performance impact

- **Streaming Multiprocessors (SMs)**
  - How work is distributed
  - Occupancy and resource limits
  - Blackwell-specific features

- **Tensor Cores**
  - 192 fifth-generation Tensor Cores
  - FP4/FP8/BF16 acceleration
  - When they're used automatically

### 1.3.2 Memory Hierarchy

- **Global Memory**
  - 128GB unified memory pool
  - Coalesced vs uncoalesced access
  - Memory bandwidth (273 GB/s)

- **Shared Memory**
  - Per-block fast memory
  - Bank conflicts and how to avoid them
  - Tile-based algorithms

- **Registers and Local Memory**
  - Per-thread storage
  - Register pressure and spilling

- **Unified Memory on DGX Spark**
  - No explicit CPU↔GPU transfers
  - How page migration works
  - When to use explicit management

### 1.3.3 CUDA Python with Numba

- **@cuda.jit decorator**
  - Writing your first kernel
  - Thread indexing: `cuda.threadIdx`, `cuda.blockIdx`
  - Grid configuration

- **Synchronization**
  - `cuda.syncthreads()`
  - Atomic operations
  - Memory fences

- **Shared Memory in Numba**
  - `cuda.shared.array()`
  - Tile-based matrix operations

### 1.3.4 CuPy for Array Operations

- **NumPy-Compatible GPU Arrays**
  - Drop-in replacement for NumPy
  - Automatic GPU acceleration
  - Memory management

- **Custom Kernels with RawKernel**
  - Writing raw CUDA C
  - When CuPy isn't enough

- **Interoperability**
  - CuPy ↔ PyTorch
  - Zero-copy transfers

### 1.3.5 Profiling and Optimization

- **Nsight Systems**
  - Timeline analysis
  - Finding bottlenecks
  - CPU↔GPU synchronization

- **Nsight Compute**
  - Kernel profiling
  - Memory throughput analysis
  - Occupancy optimization

- **Common Performance Pitfalls**
  - Thread divergence
  - Uncoalesced memory access
  - Insufficient parallelism

---

## Labs

### Lab 1.3.1: Parallel Reduction
**Time:** 3 hours

Implement parallel sum reduction with progressively optimized versions.

**Instructions:**
1. Open `notebooks/lab-1.3.1-parallel-reduction.ipynb`
2. Implement naive parallel reduction (one thread per element)
3. Add shared memory tiling
4. Implement warp shuffle reduction
5. Benchmark each version, document speedups

**Deliverable:** Notebook showing 100x+ speedup vs CPU, with analysis of each optimization step

---

### Lab 1.3.2: Matrix Multiplication
**Time:** 3 hours

Implement tiled matrix multiplication using shared memory.

**Instructions:**
1. Open `notebooks/lab-1.3.2-matrix-multiplication.ipynb`
2. Implement naive matrix multiply (one thread per output element)
3. Add shared memory tiles
4. Compare with cuBLAS
5. Analyze memory access patterns

**Deliverable:** Notebook with tiled implementation within 2x of cuBLAS performance

---

### Lab 1.3.3: Custom Embedding Lookup
**Time:** 2 hours

Write a CUDA kernel for batched embedding lookup (common in LLMs).

**Instructions:**
1. Open `notebooks/lab-1.3.3-embedding-lookup.ipynb`
2. Understand PyTorch's `nn.Embedding` internals
3. Write custom CUDA kernel for batched lookup
4. Optimize memory access patterns
5. Compare with PyTorch implementation

**Deliverable:** Custom embedding kernel with benchmark comparison

---

### Lab 1.3.4: CuPy Integration
**Time:** 2 hours

Port a NumPy preprocessing pipeline to CuPy.

**Instructions:**
1. Open `notebooks/lab-1.3.4-cupy-integration.ipynb`
2. Load a large dataset (1M+ rows)
3. Port preprocessing steps to CuPy
4. Measure speedup on each operation
5. Demonstrate CuPy ↔ PyTorch interop

**Deliverable:** Preprocessing pipeline with measured 10x+ speedup

---

### Lab 1.3.5: Profiling Workshop
**Time:** 2 hours

Profile a PyTorch training loop and identify bottlenecks.

**Instructions:**
1. Open `notebooks/lab-1.3.5-profiling-workshop.ipynb`
2. Run a simple training loop with Nsight Systems
3. Identify data loading bottlenecks
4. Find CPU↔GPU sync points
5. Implement optimizations
6. Generate before/after profile comparison

**Deliverable:** Nsight profiling report with optimization recommendations implemented

---

## Guidance

### DGX Spark Specifics

The unified memory architecture means CPU↔GPU transfers are faster than discrete GPUs, but understanding access patterns still matters for Tensor Core utilization.

```python
# DGX Spark: No explicit .to(device) needed for large tensors
# But still good practice for clarity
device = torch.device("cuda")
```

### Thread Configuration Pattern

```python
from numba import cuda
import numpy as np

@cuda.jit
def kernel(data):
    idx = cuda.grid(1)  # 1D grid
    if idx < data.size:
        data[idx] *= 2

# Launch configuration
data = cuda.to_device(np.arange(1000000, dtype=np.float32))
threads_per_block = 256
blocks = (data.size + threads_per_block - 1) // threads_per_block
kernel[blocks, threads_per_block](data)
```

### Memory Coalescing Example

```python
@cuda.jit
def coalesced_access(data, output):
    """Good: Adjacent threads access adjacent memory"""
    idx = cuda.grid(1)
    if idx < data.size:
        output[idx] = data[idx] * 2  # ✅ Coalesced

@cuda.jit
def strided_access(data, output, stride):
    """Bad: Adjacent threads access strided memory"""
    idx = cuda.grid(1)
    if idx < data.size // stride:
        output[idx] = data[idx * stride] * 2  # ❌ Uncoalesced
```

### Shared Memory Tile Pattern

```python
@cuda.jit
def tiled_operation(A, B, C):
    # Allocate shared memory
    sA = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE), dtype=float32)
    sB = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE), dtype=float32)

    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y

    # Load tiles into shared memory
    sA[ty, tx] = A[...]
    sB[ty, tx] = B[...]

    cuda.syncthreads()  # ⚠️ Critical: wait for all threads

    # Compute using fast shared memory
    # ...
```

---

## Milestone Checklist

Use this checklist to track your progress:

- [ ] Parallel reduction achieving >100x speedup vs CPU
- [ ] Tiled matrix multiplication within 2x of cuBLAS
- [ ] Custom embedding kernel working correctly
- [ ] CuPy pipeline with measured 10x+ speedup
- [ ] Nsight profiling report with optimizations applied
- [ ] Can explain memory coalescing concept
- [ ] Can explain warp execution and divergence
- [ ] Understand when to use shared memory

---

## Common Issues

| Issue | Solution |
|-------|----------|
| `numba.cuda.cudadrv.error.CudaSupportError` | Use NGC container, not pip-installed Numba |
| Kernel runs but wrong results | Check bounds: `if idx < data.size` |
| Slow performance despite parallelization | Check for thread divergence or uncoalesced access |
| `cuda.syncthreads()` deadlock | Ensure all threads in block reach the barrier |
| Out of memory on kernel launch | Reduce block size or shared memory usage |

---

## Next Steps

After completing this module:
1. ✅ Verify all milestones are checked
2. 📁 Save completed notebooks and scripts
3. ➡️ Proceed to [Module 1.4: Mathematics for Deep Learning](../module-1.4-math-foundations/)

---

## Resources

- [Numba CUDA Documentation](https://numba.readthedocs.io/en/stable/cuda/index.html)
- [CuPy Documentation](https://docs.cupy.dev/en/stable/)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [../../docs/NGC_CONTAINERS.md](../../docs/NGC_CONTAINERS.md)
