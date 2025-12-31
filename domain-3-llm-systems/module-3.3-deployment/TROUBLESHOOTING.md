# Module 3.3: Model Deployment & Inference Engines - Troubleshooting Guide

## 🔍 Quick Diagnostic

**Before diving into specific errors, try these:**
1. Check GPU memory: `nvidia-smi`
2. Check port availability: `lsof -i :8000`
3. Clear cache: `torch.cuda.empty_cache()`
4. Verify HF token: `huggingface-cli whoami`
5. Check container network: `curl http://localhost:8000/health`

---

## 🚨 Error Categories

### vLLM Errors

#### Error: `CUDA graphs not supported` or vLLM crashes
**Symptoms**:
```
RuntimeError: CUDA error: an illegal instruction was encountered
```

**Cause**: vLLM's CUDA graphs don't work on ARM64 (DGX Spark).

**Solution**:
```bash
# Always use --enforce-eager on DGX Spark
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --enforce-eager \  # Required for ARM64!
    --dtype bfloat16
```

---

#### Error: `Address already in use`
**Symptoms**:
```
OSError: [Errno 98] Address already in use
```

**Solution**:
```bash
# Find what's using the port
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port
python -m vllm.entrypoints.openai.api_server --port 8001
```

---

#### Error: `Model not found` or `401 Unauthorized`
**Cause**: HuggingFace token not set for gated models.

**Solution**:
```bash
# Set HF token
export HF_TOKEN="hf_..."

# Or use in Python
from huggingface_hub import login
login(token="hf_...")

# For vLLM server
HF_TOKEN=$HF_TOKEN python -m vllm.entrypoints.openai.api_server ...
```

---

### SGLang Errors

#### Error: `SGLang slow first request`
**Cause**: RadixAttention needs warmup to build prefix cache.

**Solution**:
```python
# Send warmup requests before benchmarking
for _ in range(3):
    result = chat_with_context.run(
        system_prompt="You are a helpful assistant.",
        user_message="Hello!"
    )

# Now subsequent requests with same system prompt are cached
```

---

#### Error: `Connection refused` to SGLang server
**Solution**:
```bash
# Check if server is running
curl http://localhost:30000/health

# If not running, start it
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 30000 \
    --dtype bfloat16

# Wait for "Server started" message before testing
```

---

### Speculative Decoding Errors

#### Error: `Medusa low acceptance rate (<30%)`
**Cause**: Too many heads or text not predictable enough.

**Solution**:
```python
# Reduce number of heads
from medusa import MedusaConfig

config = MedusaConfig(
    num_heads=3,  # Reduce from 5 to 3
    top_k=5       # Reduce candidates per head
)

# Or use for more predictable text (code, structured output)
# Speculative decoding works better on predictable patterns
```

---

#### Error: `Draft model doesn't match target model`
**Cause**: Using incompatible draft model for speculative decoding.

**Solution**:
```python
# Draft model should be from same family
# Good: Llama-8B with Llama-1B draft
# Bad: Llama-8B with Mistral draft

# For Medusa, heads are trained specifically for the target model
# Check Medusa repo for compatible head checkpoints
```

---

### TensorRT-LLM Errors

#### Error: `TensorRT build fails`
**Symptoms**:
```
[TensorRT] ERROR: ...
```

**Solutions**:
```bash
# 1. Use matching container version
# Check NGC for latest TensorRT-LLM container

# 2. Reduce complexity for testing
python -m tensorrt_llm.commands.build \
    --model_dir ./model \
    --output_dir ./trt_engine \
    --dtype bfloat16 \
    --max_batch_size 1  # Start simple

# 3. Check disk space (need 2x model size)
df -h /workspace
```

---

#### Error: `TensorRT engine too slow`
**Cause**: Engine not optimized for batch size or sequence length.

**Solution**:
```bash
# Rebuild with correct max values
python -m tensorrt_llm.commands.build \
    --model_dir ./model \
    --output_dir ./trt_engine \
    --max_batch_size 8       # Match your use case
    --max_input_len 2048     # Match expected inputs
    --max_seq_len 4096       # Match total context
    --use_fused_mlp          # Enable optimizations
```

---

### Production API Errors

#### Error: `Streaming not working`
**Cause**: Incorrect media type or missing async.

**Solution**:
```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    async def generate():
        async for chunk in model.stream(request.messages):
            # Use SSE format
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",  # Must be SSE
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )
```

---

#### Error: `Request timeout` under load
**Cause**: Server can't keep up with requests.

**Solutions**:
```python
# 1. Enable continuous batching (vLLM default)
# Already enabled by default

# 2. Limit max concurrent requests
from fastapi import FastAPI
from fastapi.middleware import Middleware
import asyncio

semaphore = asyncio.Semaphore(10)  # Max 10 concurrent

@app.post("/v1/chat/completions")
async def chat(request):
    async with semaphore:
        return await process_request(request)

# 3. Add request timeout
from fastapi import HTTPException
import asyncio

async def chat_with_timeout(request, timeout=60):
    try:
        return await asyncio.wait_for(
            process_request(request),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        raise HTTPException(504, "Request timeout")
```

---

### Memory Errors

#### Error: `OOM during benchmarking`
**Cause**: Not clearing memory between engine tests.

**Solution**:
```python
import torch
import gc

def clear_memory():
    """Full memory clear between benchmark runs."""
    # Delete models
    for name in list(globals()):
        if isinstance(globals()[name], torch.nn.Module):
            del globals()[name]

    # Clear caches
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Verify
    print(f"Memory after clear: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Use between testing different engines
clear_memory()
```

---

#### Error: `KV cache OOM with long sequences`
**Cause**: KV cache grows with sequence length.

**Solution**:
```bash
# Limit max sequence length
python -m vllm.entrypoints.openai.api_server \
    --model ... \
    --max-model-len 4096  # Limit context window

# Or use KV cache quantization (if supported)
--kv-cache-dtype fp8
```

---

## 🔄 Reset Procedures

### Full Engine Reset
```bash
# 1. Kill all servers
pkill -f "vllm"
pkill -f "sglang"
pkill -f "uvicorn"

# 2. Clear memory
python -c "import torch; torch.cuda.empty_cache()"

# 3. Clear buffer cache
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# 4. Restart container if needed
exit
# Re-run docker command
```

### Quick Memory Reset
```python
import torch
import gc

# Kill any running inference
torch.cuda.empty_cache()
gc.collect()

# Force CUDA synchronization
torch.cuda.synchronize()

# Report
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

---

## 📞 Still Stuck?

1. **Check server logs** - Most engines print helpful errors
2. **Verify network** - Can you `curl` the health endpoint?
3. **Check resource usage** - `nvidia-smi` for GPU, `htop` for CPU
4. **Try simpler config** - Reduce batch size, sequence length
5. **Update packages** - Inference libraries update frequently
