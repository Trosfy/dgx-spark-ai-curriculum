# Module 3.3: Model Deployment & Inference Engines - Quick Reference

## 🚀 Essential Commands

### NGC Container with Port Mapping
```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8000:8000 \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

### Memory Clearing
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

## 📊 Engine Comparison

| Engine | Prefill | Decode | Best For | DGX Spark Notes |
|--------|---------|--------|----------|-----------------|
| **Ollama** | ⭐⭐⭐ | ⭐⭐⭐⭐ | Easy setup | Native ARM64 |
| **llama.cpp** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Fastest decode | GGUF required |
| **vLLM** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | High throughput | Use `--enforce-eager` |
| **SGLang** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Prefix caching | 29-45% faster for chat |
| **TensorRT-LLM** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Best prefill | Native NVIDIA optimization |

### DGX Spark Performance Benchmarks
| Engine | Model | Prefill (tok/s) | Decode (tok/s) |
|--------|-------|-----------------|----------------|
| llama.cpp | Llama 3.1 8B | ~1,500 | ~59 |
| Ollama | Llama 3.1 8B | ~3,000 | ~45 |
| vLLM | Llama 3.1 8B | ~5,000 | ~40 |
| SGLang | Llama 3.1 8B | ~6,000 | ~42 |
| SGLang + Medusa | Llama 3.1 8B | ~6,000 | ~90-120 |
| TensorRT-LLM | Llama 3.1 8B | ~10,000+ | ~38 |

## 🔧 Common Patterns

### Pattern: vLLM Server (DGX Spark)
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --enforce-eager \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --port 8000
```

### Pattern: vLLM Client
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Pattern: SGLang Server
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 30000 \
    --dtype bfloat16
```

### Pattern: SGLang with RadixAttention
```python
import sglang as sgl

@sgl.function
def chat_with_context(s, system_prompt, user_message):
    s += sgl.system(system_prompt)
    s += sgl.user(user_message)
    s += sgl.assistant(sgl.gen("response", max_tokens=256))

# RadixAttention caches the system prompt!
result = chat_with_context.run(
    system_prompt="You are a helpful assistant.",
    user_message="Hello!"
)
# Second call with same system prompt is 29-45% faster!
```

### Pattern: Medusa Speculative Decoding
```python
from transformers import AutoModelForCausalLM
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Add Medusa heads (if using Medusa library)
from medusa import MedusaConfig, add_medusa_heads

config = MedusaConfig(num_heads=4)
model = add_medusa_heads(model, config)

# 2-3x speedup on interactive generation!
```

### Pattern: TensorRT-LLM Engine Build
```bash
# Build optimized engine (takes 45-90 minutes)
python -m tensorrt_llm.commands.build \
    --model_dir ./model \
    --output_dir ./trt_engine \
    --dtype bfloat16 \
    --use_fused_mlp \
    --max_batch_size 8 \
    --max_input_len 2048 \
    --max_seq_len 4096
```

### Pattern: FastAPI Production Server
```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    async def generate():
        async for chunk in model.stream(request.messages):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

@app.get("/health")
async def health():
    return {"status": "healthy", "gpu_memory": get_gpu_memory()}
```

## 📝 Engine Selection Guide

```
Development/testing?      ──► Ollama (easiest)
Interactive chat?         ──► SGLang + Medusa (fastest latency)
High throughput API?      ──► vLLM (continuous batching)
Batch processing?         ──► TensorRT-LLM (best prefill)
Edge/CPU deployment?      ──► llama.cpp (lightweight)
Not sure?                 ──► Start with vLLM
```

## ⚠️ Common Mistakes

| Mistake | Fix |
|---------|-----|
| vLLM CUDA graphs fail | Use `--enforce-eager` on ARM64 (DGX Spark) |
| SGLang slow first request | Warmup needed for RadixAttention caching |
| Medusa low acceptance rate | Reduce heads (3-4) or lower tree depth |
| TensorRT build slow | Expected: 45-90 minutes for large models |
| OOM during benchmark | Clear cache between engine tests |
| Port already in use | Kill previous server or use different port |
| Missing HF token | Set `HF_TOKEN` env var for gated models |

## 🔗 Quick Links
- [vLLM Documentation](https://docs.vllm.ai/)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [Medusa Paper](https://arxiv.org/abs/2401.10774)
- [EAGLE Paper](https://arxiv.org/abs/2401.15077)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
