# Module 1.3: CUDA Python & GPU Programming - Study Guide

## 🎯 Learning Objectives
By the end of this module, you will be able to:
1. **Explain** GPU architecture and parallel computing principles (SIMT, warps, SMs)
2. **Write** CUDA kernels using Numba and CuPy
3. **Optimize** memory access patterns for GPU performance
4. **Profile** GPU code using NVIDIA Nsight tools

## 🗺️ Module Roadmap

| # | Lab | Focus | Time | Key Outcome |
|---|-----|-------|------|-------------|
| 1 | Parallel Reduction | Basic kernels | ~3 hr | 100x+ speedup over CPU |
| 2 | Matrix Multiplication | Shared memory | ~3 hr | Within 2x of cuBLAS |
| 3 | Embedding Lookup | Custom kernels | ~2 hr | LLM-relevant optimization |
| 4 | CuPy Integration | NumPy on GPU | ~2 hr | 10x+ preprocessing speedup |
| 5 | Profiling Workshop | Performance analysis | ~2 hr | Nsight profiling skills |

**Total time**: ~12 hours

## 🔑 Core Concepts

### SIMT Execution Model
**What**: Single Instruction, Multiple Threads—6,144 CUDA cores execute the same instruction on different data simultaneously.
**Why it matters**: Understanding SIMT helps you write code that runs efficiently on all cores.
**First appears in**: Lab 1 (Parallel Reduction)

### Warp Execution
**What**: 32 threads that execute in lockstep. All must execute the same instruction.
**Why it matters**: Thread divergence (if/else) causes serialization and kills performance.
**First appears in**: Lab 1

### Memory Hierarchy
**What**: Global (128GB, slow), Shared (per-block, fast), Registers (per-thread, fastest).
**Why it matters**: Moving data to fast memory before computation gives massive speedups.
**First appears in**: Lab 2 (Matrix Multiplication)

### Memory Coalescing
**What**: Adjacent threads accessing adjacent memory addresses.
**Why it matters**: Coalesced access is 10-100x faster than random access.
**First appears in**: Lab 1, Lab 2

## 🔗 How This Module Connects

```
    Module 1.2              Module 1.3                Module 1.4
    ───────────────────────────────────────────────────────────────
    Python for AI/ML   ──►   CUDA Python        ──►   Math for DL

    NumPy patterns          GPU acceleration          Manual gradients
    Vectorization           Custom kernels            Optimization
    Profiling               Tensor Cores              Loss landscapes
```

**Builds on**:
- Module 1.1: Container setup, GPU verification
- Module 1.2: NumPy patterns (CuPy mirrors NumPy)

**Prepares for**:
- **Module 1.4**: GPU-accelerated gradient computation
- **Module 1.5**: Understanding why PyTorch is fast
- **All DL modules**: Knowing what's happening under the hood

## 📊 Key Architecture Values

### DGX Spark GPU Specs
| Component | Specification |
|-----------|---------------|
| CUDA Cores | 6,144 |
| Tensor Cores | 192 (5th gen) |
| Memory | 128GB unified memory |
| Bandwidth | 273 GB/s |
| Warp Size | 32 threads |
| Max Threads/Block | 1024 |

### Performance Hierarchy
| Memory Type | Access Time | Size |
|-------------|-------------|------|
| Registers | 1 cycle | ~256 KB total |
| Shared Memory | ~5 cycles | 64-228 KB/SM |
| L2 Cache | ~100 cycles | 24 MB |
| Global Memory | ~500 cycles | 128GB |

### Precision Performance
| Format | TFLOPS | Best For |
|--------|--------|----------|
| NVFP4 | 1,000 (1 PFLOP) | Inference |
| FP8 | ~209 | Inference |
| BF16 | ~100 | Training |
| FP32 | ~31 | Debugging |

## 📖 Recommended Approach

**Standard path** (12 hours):
1. Lab 1: Learn thread indexing and atomic operations
2. Lab 2: Master shared memory with tiling
3. Lab 3: Apply to real LLM operation
4. Lab 4: Use CuPy for quick GPU code
5. Lab 5: Profile and optimize

**Quick path** (if experienced with CUDA, 6-7 hours):
1. Skim Lab 1, focus on Numba syntax
2. Focus on Lab 2 shared memory
3. Complete Lab 4 CuPy integration
4. Complete Lab 5 profiling

## 🚨 DGX Spark Performance Tip

> The Blackwell GB10 natively supports **bfloat16** at ~100 TFLOPS. While these labs use float32 for educational clarity, production code should use bfloat16 for 2x memory efficiency and optimal Tensor Core performance.

## 📋 Before You Start
→ See [QUICKSTART.md](./QUICKSTART.md) for 5-minute GPU speedup demo
→ See [ELI5.md](./ELI5.md) for jargon-free GPU concepts
→ See [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) for kernel patterns
→ Ensure NGC container is running
