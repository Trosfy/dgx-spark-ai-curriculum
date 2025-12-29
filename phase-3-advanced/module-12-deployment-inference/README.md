# Module 12: Model Deployment & Inference Engines

**Phase:** 3 - Advanced  
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
| 12.1 | Compare inference engines (Ollama, llama.cpp, vLLM, TensorRT-LLM) | Analyze |
| 12.2 | Deploy models as REST APIs | Apply |
| 12.3 | Implement continuous batching for throughput | Apply |
| 12.4 | Configure and use speculative decoding | Apply |

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

## Tasks

| # | Task | Time | Deliverable |
|---|------|------|-------------|
| 12.1 | Engine Benchmark | 3h | Comprehensive comparison report |
| 12.2 | vLLM Deployment | 2h | Continuous batching under load |
| 12.3 | TensorRT-LLM Optimization | 3h | Build TRT engine, benchmark prefill |
| 12.4 | Speculative Decoding | 2h | SGLang with EAGLE-3, measure speedup |
| 12.5 | Production API | 2h | FastAPI with streaming, monitoring |
| 12.6 | Ollama Web UI Integration | 2h | Enhance your existing UI |

---

## Guidance

### vLLM on DGX Spark

```bash
# Use NVIDIA container
docker pull nvcr.io/nvidia/vllm:spark

# Run with --enforce-eager flag
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --enforce-eager \
    --max-model-len 4096
```

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
