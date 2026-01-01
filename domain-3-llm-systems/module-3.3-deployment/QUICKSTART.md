# Module 3.3: Model Deployment & Inference Engines - Quickstart

## ⏱️ Time: ~5 minutes

## 🎯 What You'll Build
Deploy a model with vLLM and see continuous batching in action.

## ✅ Before You Start
- [ ] DGX Spark NGC container running
- [ ] Model downloaded (or will use Ollama)

## 🚀 Let's Go!

### Step 1: Start Container with Port Mapping
```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8000:8000 \
    nvcr.io/nvidia/pytorch:25.11-py3
```

### Step 2: Install vLLM
```bash
pip install vllm
```

### Step 3: Start vLLM Server
```bash
python -m vllm.entrypoints.openai.api_server \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --enforce-eager \
    --dtype bfloat16 \
    --port 8000 &
```

Wait 30 seconds for the server to start, then:

### Step 4: Test the API
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 50
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

**Expected output**:
```
Hello! How can I help you today?
```

### Step 5: Check Server Metrics
```bash
curl http://localhost:8000/metrics | grep vllm
```

**Expected output**:
```
vllm:num_requests_running 0
vllm:num_requests_waiting 0
vllm:gpu_cache_usage_perc 0.05
```

## 🎉 You Did It!

You just deployed a model with vLLM's continuous batching! This same setup scales to:
- **70B models** with quantization
- **Multiple concurrent users** with automatic batching
- **Streaming responses** with Server-Sent Events

In the full module, you'll learn:
- **SGLang**: 29-45% faster than vLLM with RadixAttention
- **Speculative decoding**: 2-3x speedup with Medusa/EAGLE
- **TensorRT-LLM**: Maximum prefill performance
- **Production APIs**: FastAPI with monitoring

## ▶️ Next Steps
1. **Benchmark engines**: See [Lab 3.3.1](./labs/lab-3.3.1-engine-benchmark.ipynb)
2. **Try SGLang**: See [Lab 3.3.2](./labs/lab-3.3.2-sglang-deployment.ipynb)
3. **Full setup**: Start with [LAB_PREP.md](./LAB_PREP.md)
