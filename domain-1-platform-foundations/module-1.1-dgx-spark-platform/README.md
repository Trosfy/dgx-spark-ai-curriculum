# Module 1.1: DGX Spark Platform Mastery

**Domain:** 1 - Platform Foundations  
**Duration:** Week 1 (5-8 hours)  
**Prerequisites:** Basic command line knowledge

---

## Overview

This module introduces you to the NVIDIA DGX Spark hardware and software environment. You'll understand the unique Grace Blackwell architecture, configure your development environment, and learn which open-source tools work (and don't work) on this bleeding-edge platform.

By the end of this module, you'll have a fully configured DGX Spark ready for AI development and a deep understanding of its capabilities and limitations.

---

## Learning Outcomes

By the end of this module, you will be able to:

- ✅ Explain the DGX Spark hardware architecture and its advantages for AI workloads
- ✅ Navigate the DGX OS environment and utilize pre-installed AI tools
- ✅ Configure JupyterLab for optimal AI development workflows
- ✅ Identify which open-source projects are compatible with DGX Spark

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 1.1.1 | Describe the Grace Blackwell GB10 architecture including CPU cores, GPU specs, and unified memory | Understand |
| 1.1.2 | Execute system monitoring commands to verify GPU and memory status | Apply |
| 1.1.3 | Configure NGC containers for PyTorch and other frameworks | Apply |
| 1.1.4 | Differentiate between compatible and incompatible open-source tools | Analyze |

---

## Topics

### 1.1.1 Hardware Architecture Deep-Dive

- **Grace Blackwell GB10 Superchip**
  - 20 ARM v9.2 cores (10 Cortex-X925 + 10 Cortex-A725)
  - 6,144 CUDA cores
  - 192 fifth-generation Tensor Cores
  - 24MB L2 cache

- **Unified Memory Architecture**
  - 128GB LPDDR5X memory
  - 273 GB/s bandwidth
  - NVLink-C2C interconnect
  - Shared CPU/GPU memory pool

- **Compute Capabilities**
  - NVFP4: 1 PFLOP (Blackwell exclusive)
  - FP8: ~209 TFLOPS
  - BF16: ~100 TFLOPS
  - FP32: ~31 TFLOPS

### 1.1.2 Software Environment

- DGX OS (Ubuntu 24.04 LTS)
- CUDA 13.0.2 and cuDNN
- Pre-installed JupyterLab
- Docker with NVIDIA Container Runtime
- TensorRT and TensorRT-LLM

### 1.1.3 Ecosystem Compatibility

| Status | Tools |
|--------|-------|
| ✅ Full Support | Ollama, llama.cpp, NeMo |
| ⚠️ NGC Required | PyTorch, JAX, Hugging Face |
| ⚠️ Partial | vLLM, TensorRT-LLM, DeepSpeed |
| ❌ Not Compatible | Standard pip PyTorch |

---

## Labs

### Lab 1.1.1: System Exploration
**Time:** 1 hour

Explore your DGX Spark hardware and document its configuration.

**Instructions:**
1. Open the notebook `lab-1.1.1-system-exploration.ipynb`
2. Run system commands: `nvidia-smi`, `lscpu`, `free -h`, `df -h`
3. Document your findings in markdown cells
4. Take screenshots of key outputs

**Deliverable:** Completed notebook with system documentation

---

### Lab 1.1.2: Memory Architecture Lab
**Time:** 1.5 hours

Understand how unified memory works by allocating tensors of various sizes.

**Instructions:**
1. Open `lab-1.1.2-memory-architecture-lab.ipynb`
2. Allocate PyTorch tensors from 1GB to 100GB
3. Monitor memory with `torch.cuda.memory_summary()`
4. Observe when memory is shared vs dedicated
5. Document the memory behavior

**Deliverable:** Notebook with memory allocation experiments and analysis

---

### Lab 1.1.3: NGC Container Setup
**Time:** 1.5 hours

Configure NGC containers for development.

**Instructions:**
1. Open `lab-1.1.3-ngc-container-setup.ipynb`
2. Pull the PyTorch NGC container
3. Create a `docker-compose.yml` with proper volume mounts
4. Verify GPU access inside container
5. Test PyTorch CUDA availability

**Deliverable:** Working docker-compose.yml and verification notebook

---

### Lab 1.1.4: Compatibility Matrix
**Time:** 2 hours

Research and document the DGX Spark ecosystem compatibility.

**Instructions:**
1. Open `lab-1.1.4-compatibility-matrix.ipynb`
2. Research 20 popular AI/ML tools
3. Test or research their DGX Spark compatibility
4. Document workarounds for partial support
5. Create a comprehensive compatibility table

**Deliverable:** Markdown compatibility matrix with status and notes

---

### Lab 1.1.5: Ollama Benchmarking
**Time:** 2 hours

Benchmark Ollama models using direct API calls for accurate metrics. Results can be verified through the Ollama Web UI.

**Instructions:**
1. Open `lab-1.1.5-ollama-benchmarking.ipynb`
2. Verify Ollama service: `curl http://localhost:11434/api/tags`
3. Pull models: `ollama pull llama3.2:3b llama3.1:8b llama3.1:70b`
4. Use the benchmark utility from `utils/benchmark_utils.py`
5. Measure prefill tok/s and decode tok/s via API
6. Record memory usage per model
7. Compare with NVIDIA published specs and verify in Ollama Web UI

**Key:** Use direct API calls for precise timing metrics (Ollama Web UI adds rendering overhead)

```python
# Example: Direct API measurement
import requests
import time

response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "llama3.1:8b", "prompt": "Hello", "stream": False}
)
data = response.json()

# Ollama returns timing in nanoseconds
prefill_tps = data["prompt_eval_count"] / (data["prompt_eval_duration"] / 1e9)
decode_tps = data["eval_count"] / (data["eval_duration"] / 1e9)
```

**Deliverable:** Benchmark results table with tok/s metrics

**Expected Results on DGX Spark:**
| Model | Prefill (tok/s) | Decode (tok/s) | Memory (GB) |
|-------|-----------------|----------------|-------------|
| 3B Q4 | ~5,000 | ~80 | ~3 |
| 8B Q4 | ~3,000 | ~45 | ~6 |
| 70B Q4 | ~500 | ~15 | ~45 |

---

## Guidance

### Critical: NGC Containers Required

Standard PyTorch installations via pip do NOT work on DGX Spark. Always use NGC containers:

```bash
# Pull PyTorch container
docker pull nvcr.io/nvidia/pytorch:25.11-py3

# Run with GPU access
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    bash
```

### Buffer Cache Management

Before memory-intensive operations, clear the buffer cache:

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

### Unified Memory Behavior

The 128GB memory is shared between CPU and GPU. Key behaviors:
- No explicit memory transfers needed
- Large models (70B+) fit entirely in memory
- Linux buffer cache competes with GPU allocations
- Clear cache before loading large models

### JupyterLab Best Practices

Create a startup configuration:
```python
# ~/.jupyter/startup.py
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

## Milestone Checklist

Use this checklist to track your progress:

- [ ] Successfully ran `nvidia-smi` showing GB10 GPU
- [ ] Created system specification notebook
- [ ] NGC PyTorch container running with GPU access
- [ ] docker-compose.yml created and tested
- [ ] Compatibility matrix with 20+ tools documented
- [ ] Ollama serving models (3B, 8B, 70B tested)
- [ ] Benchmark results documented
- [ ] All notebooks complete with explanations

---

## Common Issues

| Issue | Solution |
|-------|----------|
| `torch.cuda.is_available()` returns False | Use NGC container, not pip PyTorch |
| Out of memory loading 70B model | Clear buffer cache first |
| Docker can't access GPU | Add `--gpus all` flag |
| Ollama not responding | Check `systemctl status ollama` |

---

## Next Steps

After completing this module:
1. ✅ Verify all milestones are checked
2. 📁 Save all notebooks to your repository
3. ➡️ Proceed to [Module 1.2: Python for AI/ML](../module-1.2-python-for-ai/)

---

## Resources

- [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/)
- [NGC Container Catalog](https://catalog.ngc.nvidia.com/)
- [DGX Spark Playbooks](https://build.nvidia.com/spark)
- [../../docs/NGC_CONTAINERS.md](../../docs/NGC_CONTAINERS.md)
