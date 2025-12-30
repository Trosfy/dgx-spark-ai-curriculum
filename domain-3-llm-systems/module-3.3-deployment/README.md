# Module 3.3: Model Deployment & Inference Engines

**Domain:** 3 - LLM Systems  
**Duration:** Weeks 20-21 (10-12 hours)  
**Prerequisites:** Module 11 (Quantization)

---

## Overview

Choose the right inference engine for your use case. This module covers all major options—from simple Ollama to high-performance TensorRT-LLM—and helps you understand when to use each.

---

## Learning Outcomes

By the end of this module, you will be able to:

- ✅ Deploy models using various inference engines
- ✅ Optimize inference for latency and throughput
- ✅ Implement serving APIs for production use
- ✅ Select the right inference engine for different requirements

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 3.3.1 | Compare inference engines (Ollama, llama.cpp, vLLM, TensorRT-LLM) | Analyze |
| 3.3.2 | Deploy models as REST APIs | Apply |
| 3.3.3 | Implement continuous batching for throughput | Apply |
| 3.3.4 | Configure and use speculative decoding | Apply |

---

## Topics

### Inference Engine Comparison

| Engine | Prefill | Decode | Best For |
|--------|---------|--------|----------|
| **Ollama** | ⭐⭐⭐ | ⭐⭐⭐⭐ | Easy setup, model management |
| **llama.cpp** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Fastest decode, GGUF |
| **vLLM** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | High throughput, batching |
| **TensorRT-LLM** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Best prefill, NVIDIA optimized |
| **SGLang** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Speculative decoding |

### Key Concepts
- Continuous batching
- KV cache optimization
- Speculative decoding
- Tensor parallelism

---

## Labs

| # | Task | Time | Deliverable |
|---|------|------|-------------|
| 3.3.1 | Engine Benchmark | 3h | Comprehensive comparison report |
| 3.3.2 | vLLM Deployment | 2h | Continuous batching under load |
| 3.3.3 | TensorRT-LLM Optimization | 3h | Build TRT engine, benchmark prefill |
| 3.3.4 | Speculative Decoding | 2h | SGLang with EAGLE-3, measure speedup |
| 3.3.5 | Production API | 2h | FastAPI with streaming, monitoring |
| 3.3.6 | Ollama Web UI Integration | 2h | Enhance your existing UI |

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
        --model meta-llama/Llama-3.1-8B-Instruct \
        --enforce-eager \
        --max-model-len 4096 \
        --dtype bfloat16"

# Option 2: Use official vLLM container (verify ARM64 support)
# docker run --gpus all -p 8000:8000 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     --ipc=host \
#     vllm/vllm-openai:latest \
#     --model meta-llama/Llama-3.1-8B-Instruct \
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

---

## Milestone Checklist

- [ ] Comprehensive engine benchmark report
- [ ] vLLM with continuous batching
- [ ] TensorRT-LLM engine built and tested
- [ ] Speculative decoding speedup measured
- [ ] Production FastAPI server implemented
- [ ] Ollama Web UI enhanced

---

## Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [SGLang](https://github.com/sgl-project/sglang)
- [FastAPI](https://fastapi.tiangolo.com/)
