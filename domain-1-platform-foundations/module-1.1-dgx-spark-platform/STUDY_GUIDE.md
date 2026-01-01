# Module 1.1: DGX Spark Platform Mastery - Study Guide

## 🎯 Learning Objectives
By the end of this module, you will be able to:
1. **Explain** the Grace Blackwell GB10 architecture (CPU cores, GPU specs, unified memory)
2. **Execute** system monitoring commands to verify GPU and memory status
3. **Configure** NGC containers for PyTorch and other frameworks
4. **Differentiate** between compatible and incompatible open-source tools

## 🗺️ Module Roadmap

| # | Lab | Focus | Time | Key Outcome |
|---|-----|-------|------|-------------|
| 1 | System Exploration | Hardware discovery | ~1 hr | Document your DGX Spark specs |
| 2 | Memory Architecture | Unified memory behavior | ~1.5 hr | Understand memory allocation patterns |
| 3 | NGC Container Setup | Docker configuration | ~1.5 hr | Working docker-compose.yml |
| 4 | Compatibility Matrix | Ecosystem research | ~2 hr | Know which tools work |
| 5 | Ollama Benchmarking | Performance testing | ~2 hr | Benchmark results for LLMs |

**Total time**: ~8 hours

## 🔑 Core Concepts

### Grace Blackwell GB10 Superchip
**What**: NVIDIA's integrated CPU+GPU superchip combining ARM cores with Blackwell GPU architecture.
**Why it matters**: Unlike discrete GPUs, CPU and GPU share 128GB of memory with no transfer overhead.
**First appears in**: Lab 1 (System Exploration)

### Unified Memory Architecture
**What**: Single 128GB LPDDR5X memory pool shared between CPU and GPU (273 GB/s bandwidth).
**Why it matters**: Load 70B+ parameter models that wouldn't fit on typical 24GB GPUs. No explicit memory copies needed.
**First appears in**: Lab 2 (Memory Architecture)

### NGC Containers
**What**: NVIDIA GPU Cloud containers with pre-built, optimized AI frameworks for ARM64.
**Why it matters**: Standard pip packages don't work on DGX Spark—NGC containers are required.
**First appears in**: Lab 3 (NGC Container Setup)

### Compute Capabilities
**What**: Hardware acceleration modes—NVFP4 (1 PFLOP), FP8 (~209 TFLOPS), BF16 (~100 TFLOPS).
**Why it matters**: Choose the right precision for inference vs training workloads.
**First appears in**: Lab 5 (Ollama Benchmarking)

## 🔗 How This Module Connects

```
                        ┌─────────────────────────────┐
                        │     Module 1.1              │
    No Prerequisites    │  DGX Spark Platform         │     All Future Modules
    ──────────────────► │  Mastery                    │ ──────────────────────►
                        │                             │
                        │  • Hardware understanding   │     Every module uses:
                        │  • NGC containers           │     • NGC containers
                        │  • Memory management        │     • GPU memory
                        │  • Ecosystem knowledge      │     • Ollama for testing
                        └─────────────────────────────┘
```

**Builds on**:
- Basic command line knowledge (prerequisite)

**Prepares for**:
- **Module 1.2** will use NumPy/Pandas inside NGC containers
- **Module 1.3** will use CUDA Python with the GPU you've verified
- **All modules** rely on the container setup from this module

## 📖 Recommended Approach

**Standard path** (8 hours):
1. Start with Lab 1 - understand your hardware
2. Complete Lab 2 - essential memory concepts
3. Work through Lab 3 - you'll use this docker-compose.yml constantly
4. Complete Lab 4 - know what works before you try things
5. Finish with Lab 5 - hands-on with Ollama

**Quick path** (if experienced with NVIDIA GPUs, 4-5 hours):
1. Skim Lab 1, run the verification commands
2. Focus on Lab 2's unified memory experiments - this is different!
3. Set up docker-compose.yml from Lab 3
4. Review Lab 4's compatibility table
5. Complete Lab 5 benchmarks

## 📊 Key Values to Remember

| Specification | Value | Notes |
|---------------|-------|-------|
| Total Memory | 128GB | LPDDR5X unified |
| Memory Bandwidth | 273 GB/s | CPU and GPU share |
| CUDA Cores | 6,144 | Blackwell architecture |
| Tensor Cores | 192 | 5th generation |
| ARM CPU Cores | 20 | 10 Cortex-X925 + 10 Cortex-A725 |
| NVFP4 Performance | 1 PFLOP | Blackwell exclusive |
| FP8 Performance | ~209 TFLOPS | Inference acceleration |
| BF16 Performance | ~100 TFLOPS | Training sweet spot |

## 📋 Before You Start
→ See [LAB_PREP.md](./LAB_PREP.md) for environment setup
→ See [QUICKSTART.md](./QUICKSTART.md) for 5-minute verification
→ See [ELI5.md](./ELI5.md) for jargon-free explanations
