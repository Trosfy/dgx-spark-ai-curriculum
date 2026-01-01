# Module 3.3: Model Deployment & Inference Engines

**Domain:** 3 - LLM Systems
**Duration:** Weeks 21-22 (12-15 hours)
**Prerequisites:** Module 3.2 (Quantization)
**Priority:** P1 Expanded (SGLang, Speculative Decoding)

---

## Overview

Choose the right inference engine for your use case. This module covers all major options—from simple Ollama to high-performance TensorRT-LLM—with special focus on SGLang's RadixAttention and speculative decoding techniques that can achieve 2-3x speedups.

Deep dive into SGLang (29-45% faster than vLLM), Medusa heads, and EAGLE-3 speculative decoding for interactive speedups.

---

## Learning Outcomes

By the end of this module, you will be able to:

- ✅ Deploy models using multiple inference engines (Ollama, vLLM, SGLang, TensorRT-LLM)
- ✅ Implement speculative decoding for faster inference
- ✅ Optimize inference for latency and throughput
- ✅ Select the right inference engine for different requirements

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 3.3.1 | Compare inference engines and select optimal for use case | Analyze |
| 3.3.2 | Implement speculative decoding with Medusa/EAGLE | Apply |
| 3.3.3 | Configure continuous batching and PagedAttention | Apply |
| 3.3.4 | Deploy production-ready REST APIs | Apply |

---

## Topics

### 3.3.1 Inference Engine Overview

| Engine | Prefill | Decode | Best For | DGX Spark |
|--------|---------|--------|----------|-----------|
| **Ollama** | ⭐⭐⭐ | ⭐⭐⭐⭐ | Easy setup, model management | ✅ Native |
| **llama.cpp** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Fastest decode, GGUF | ✅ Excellent |
| **vLLM** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | High throughput, batching | ✅ --enforce-eager |
| **TensorRT-LLM** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Best prefill, NVIDIA optimized | ✅ Native |
| **SGLang** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | RadixAttention, spec decoding | ✅ 29-45% faster |

### 3.3.2 Speculative Decoding [P1 Expansion]

- **Theory: Draft-Verify Paradigm**
  - Small draft model generates candidates
  - Target model verifies in parallel
  - Accept/reject decisions

- **Medusa**
  - Multiple prediction heads
  - 2-3x speedup, no draft model needed
  - Works with any transformer

- **EAGLE (Efficient Acceleration)**
  - Feature-level draft tokens
  - Better than Medusa for long sequences

- **EAGLE-3**
  - Latest improvements
  - Optimized for production

- **When Speculative Decoding Helps**
  - Interactive chat (latency-sensitive)
  - Longer outputs
  - Simple/predictable text patterns

### 3.3.3 SGLang [P1 Expansion]

- **RadixAttention**
  - Cache and reuse prefix computations
  - 29-45% faster than vLLM on shared prefixes
  - Ideal for chat with system prompts

- **Key Features**
  - Structured generation (JSON, regex)
  - Efficient batching
  - Speculative decoding integration

### 3.3.4 Optimization Techniques

- **Continuous Batching**
  - Dynamic batch formation
  - Maximize GPU utilization

- **PagedAttention (vLLM)**
  - Efficient KV cache management
  - Memory fragmentation prevention

- **KV Cache Quantization**
  - INT8/FP8 KV cache
  - Reduce memory, minimal quality loss

### 3.3.5 Production Infrastructure

- REST API design patterns
- Streaming responses (SSE)
- Health checks and monitoring
- Load balancing strategies

---

## Labs

| # | Task | Time | Deliverable |
|---|------|------|-------------|
| 3.3.1 | Engine Benchmark | 3h | Comprehensive comparison report |
| 3.3.2 | SGLang Deployment | 2h | RadixAttention with shared prefixes |
| 3.3.3 | vLLM Continuous Batching | 2h | Throughput under load |
| 3.3.4 | Medusa Speculative Decoding | 2h | Configure Medusa, measure speedup |
| 3.3.5 | EAGLE-3 Implementation | 2h | Compare with Medusa |
| 3.3.6 | TensorRT-LLM Optimization | 2h | Build TRT engine, benchmark prefill |
| 3.3.7 | Production API | 2h | FastAPI with streaming, monitoring |

---

## DGX Spark Compatibility Notes

> **Important:** DGX Spark uses ARM64 (aarch64) architecture with NVIDIA Blackwell GPU. Always verify container ARM64 support before use.

- Use NGC PyTorch containers as the base for most workloads
- Always include `--gpus all` and `--ipc=host` in docker commands
- Use `bfloat16` (not float16) for native Blackwell support
- Clear buffer cache before loading large models (70B+):
  ```bash
  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
  ```

### Container Versions

This module was tested with the following NGC container versions. Check NGC for the latest ARM64-compatible releases:

| Container | Version Tested | NGC Catalog Link |
|-----------|----------------|------------------|
| PyTorch | `nvcr.io/nvidia/pytorch:25.11-py3` | [NGC PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) |
| Triton + TRT-LLM | `nvcr.io/nvidia/tritonserver:25.11-trtllm-python-py3` | [NGC Triton](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver) |

**Before using any container on DGX Spark:**
1. Visit the NGC catalog link
2. Check the "Tags" or "Supported Platforms" section
3. Verify `linux/arm64` or `aarch64` is listed
4. If not available, use PyTorch NGC container and install packages from source

---

## Guidance

### vLLM on DGX Spark

```bash
# Option 1: Use PyTorch NGC container with vLLM installed
docker run --gpus all -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN=$HF_TOKEN \
    --ipc=host \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    bash -c "pip install vllm && python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3-8B-Instruct \
        --enforce-eager \
        --max-model-len 4096 \
        --dtype bfloat16"

# Option 2: Use official vLLM container (verify ARM64 support)
# docker run --gpus all -p 8000:8000 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     --ipc=host \
#     vllm/vllm-openai:latest \
#     --model Qwen/Qwen3-8B-Instruct \
#     --enforce-eager
```

**Key flags for DGX Spark:**
- `--enforce-eager`: Required for ARM64 (disables CUDA graphs)
- `--dtype bfloat16`: Native Blackwell support
- `--ipc=host`: Required for DataLoader workers

### TensorRT-LLM Engine Build

```bash
# Build optimized engine (45-90 minutes)
python -m tensorrt_llm.commands.build \
    --model_dir ./model \
    --output_dir ./trt_engine \
    --dtype bfloat16 \
    --use_fused_mlp
```

### FastAPI Production Server

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    async def generate():
        async for chunk in model.stream(request.messages):
            yield f"data: {chunk}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
```

### SGLang Deployment

```python
# Start SGLang server with RadixAttention
# python -m sglang.launch_server \
#     --model-path Qwen/Qwen3-8B-Instruct \
#     --port 30000 \
#     --dtype bfloat16

import sglang as sgl

@sgl.function
def chat_with_context(s, system_prompt, user_message):
    s += sgl.system(system_prompt)
    s += sgl.user(user_message)
    s += sgl.assistant(sgl.gen("response", max_tokens=256))

# RadixAttention caches the system prompt!
# Subsequent calls with same system prompt are 29-45% faster
result = chat_with_context.run(
    system_prompt="You are a helpful assistant.",
    user_message="Hello!"
)
```

### Speculative Decoding with Medusa

```python
# Medusa adds prediction heads to the model
# No separate draft model needed!

from transformers import AutoModelForCausalLM
from medusa import MedusaConfig, add_medusa_heads
import torch

# Standard model loading pattern for DGX Spark
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B-Instruct",
    torch_dtype=torch.bfloat16,  # Native Blackwell support
    device_map="auto"
)

# Add Medusa heads (typically 3-5 heads)
config = MedusaConfig(num_heads=4)
model = add_medusa_heads(model, config)

# 2-3x speedup on interactive generation!
```

### Engine Selection Guide

| Use Case | Recommended Engine | Why |
|----------|-------------------|-----|
| Development/testing | Ollama | Easy setup, model management |
| Interactive chat | SGLang + Medusa | Lowest latency, prefix caching |
| High throughput API | vLLM | Continuous batching, PagedAttention |
| Batch processing | TensorRT-LLM | Best prefill performance |
| Edge/embedded | llama.cpp | Low resource usage |

### DGX Spark Performance Benchmarks

| Engine | Model | Prefill (tok/s) | Decode (tok/s) |
|--------|-------|-----------------|----------------|
| llama.cpp | Llama 3.1 8B | ~1,500 | ~59 |
| Ollama | Llama 3.1 8B | ~3,000 | ~45 |
| vLLM | Llama 3.1 8B | ~5,000 | ~40 |
| SGLang | Llama 3.1 8B | ~6,000 | ~42 |
| SGLang + Medusa | Llama 3.1 8B | ~6,000 | ~90-120 |
| TensorRT-LLM | Llama 3.1 8B | ~10,000+ | ~38 |

---

## Milestone Checklist

- [ ] Comprehensive engine benchmark report
- [ ] SGLang deployment with RadixAttention tested
- [ ] vLLM with continuous batching working
- [ ] Medusa speculative decoding speedup measured
- [ ] EAGLE-3 comparison completed
- [ ] TensorRT-LLM engine built and tested
- [ ] Production FastAPI server implemented

---

## Common Issues

| Issue | Solution |
|-------|----------|
| vLLM CUDA graphs fail | Use `--enforce-eager` on ARM64 |
| SGLang slow first request | RadixAttention needs warmup for prefix caching |
| Medusa low acceptance rate | Reduce number of heads or lower tree depth |
| TensorRT build slow | Expected: 45-90 minutes for large models |
| OOM during benchmark | Clear cache between engine tests |

---

## Next Steps

After completing this module:
1. ✅ Verify all milestones are checked
2. 📁 Save your deployment configurations
3. ➡️ Proceed to [Module 3.4: Test-Time Compute & Reasoning](../module-3.4-test-time-compute/)

---

## Module Navigation

| Previous | Current | Next |
|----------|---------|------|
| [Module 3.2: Quantization](../module-3.2-quantization/) | **Module 3.3: Deployment** | [Module 3.4: Test-Time Compute](../module-3.4-test-time-compute/) |

---

## 📖 Study Materials

| Document | Purpose |
|----------|---------|
| [QUICKSTART.md](./QUICKSTART.md) | Deploy with vLLM in 5 minutes |
| [ELI5.md](./ELI5.md) | Inference engines, batching, speculative decoding explained |
| [PREREQUISITES.md](./PREREQUISITES.md) | Self-check before starting |
| [STUDY_GUIDE.md](./STUDY_GUIDE.md) | Learning objectives and lab roadmap |
| [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) | Engine commands and patterns |
| [LAB_PREP.md](./LAB_PREP.md) | Environment setup and port mapping |
| [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) | vLLM, SGLang, TensorRT errors and FAQ |

---

## Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [SGLang Paper](https://arxiv.org/abs/2312.07104) - RadixAttention
- [Medusa Paper](https://arxiv.org/abs/2401.10774) - Multi-head speculation
- [EAGLE Paper](https://arxiv.org/abs/2401.15077) - Feature-level draft
- [FastAPI](https://fastapi.tiangolo.com/)
