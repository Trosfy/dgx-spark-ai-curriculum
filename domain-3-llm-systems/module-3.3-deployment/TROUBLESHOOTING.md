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

## ❓ Frequently Asked Questions

### Setup & Environment

**Q: Which inference engine should I use?**

**A**: It depends on your use case:

| Use Case | Recommended | Why |
|----------|-------------|-----|
| Development/learning | Ollama | Easiest setup, model management |
| Interactive chat | SGLang + Medusa | Lowest latency, prefix caching |
| High-throughput API | vLLM | Continuous batching, PagedAttention |
| Batch processing | TensorRT-LLM | Best prefill performance |
| CPU/Edge deployment | llama.cpp | Lightweight, runs anywhere |

**Default recommendation**: Start with vLLM—it's the most popular and well-documented.

---

**Q: Why do I need `--enforce-eager` on DGX Spark?**

**A**: DGX Spark uses ARM64 (aarch64) architecture. vLLM's CUDA graphs optimization currently has limited ARM64 support and can cause crashes. `--enforce-eager` disables CUDA graphs and uses standard "eager" execution, which works reliably on ARM64.

```bash
# Always include for DGX Spark
python -m vllm.entrypoints.openai.api_server \
    --model ... \
    --enforce-eager \
    --dtype bfloat16
```

---

**Q: How do I choose between vLLM and SGLang?**

**A**:

| Factor | vLLM | SGLang |
|--------|------|--------|
| **Setup complexity** | Simple | Slightly more complex |
| **Documentation** | Excellent | Good |
| **Prefix caching** | Limited | RadixAttention (29-45% faster) |
| **Speculative decoding** | Built-in | Built-in |
| **OpenAI compatibility** | Full | Partial |
| **Community size** | Larger | Growing |

**Choose SGLang if**: You have chat with system prompts (prefix caching helps a lot).
**Choose vLLM if**: You want maximum compatibility and documentation.

---

### Concepts

**Q: What's the difference between prefill and decode?**

**A**:

**Prefill (Prompt Processing)**:
- Processes all input tokens at once (parallel)
- Compute-bound (faster GPU = faster prefill)
- Speed in "prefill tok/s"
- Example: 10,000 tok/s prefill

**Decode (Token Generation)**:
- Generates one token at a time (sequential)
- Memory-bandwidth-bound
- Speed in "decode tok/s"
- Example: 40 tok/s decode

That's why TensorRT-LLM has 10,000 prefill but only 38 decode tok/s—different bottlenecks!

---

**Q: How does speculative decoding achieve 2-3x speedup?**

**A**: Instead of generating one token per forward pass:

1. **Draft model** (small, fast) guesses 5-10 tokens ahead
2. **Target model** (large, accurate) verifies all guesses in ONE pass
3. If 5 guesses are correct, you skip 4 forward passes!

The target model does the same amount of work per token, but you batch multiple token verifications together. On predictable text (code, structured output), acceptance rates are high.

---

**Q: What is PagedAttention and why does it matter?**

**A**: The KV cache (storing attention keys/values) can waste memory:

**Without PagedAttention**:
- Allocate fixed memory for max sequence length
- Short conversations waste most of it
- Memory fragmentation limits concurrent users

**With PagedAttention** (vLLM):
- Allocate memory in "pages" as needed
- Short conversations use less memory
- No fragmentation, more concurrent users

This is why vLLM can handle 100 users while naive serving handles 1-2.

---

**Q: Why is TensorRT-LLM so fast for prefill but not decode?**

**A**: TensorRT-LLM optimizes GPU computation (matrix multiplications, kernels). Prefill is compute-heavy, so optimizations help a lot.

Decode is memory-bandwidth-heavy (reading KV cache). TensorRT's compute optimizations don't help as much there. llama.cpp's memory-efficient design actually decodes faster!

---

### Common Issues

**Q: My vLLM server won't start. What should I check?**

**A**: In order:
1. **Port available?** `lsof -i :8000`
2. **Memory available?** `nvidia-smi`
3. **Model accessible?** HuggingFace token set?
4. **Using `--enforce-eager`?** Required for DGX Spark
5. **Correct dtype?** Use `bfloat16` for Blackwell

---

**Q: My benchmarks are inconsistent. Why?**

**A**: Common causes:
1. **No warmup** - First few requests are slower (JIT compilation, cache warmup)
2. **Memory fragmentation** - Clear cache between tests
3. **Thermal throttling** - GPU might throttle under sustained load
4. **Background processes** - Check nothing else is using GPU

```python
# Standard benchmark procedure
# 1. Clear memory
torch.cuda.empty_cache()

# 2. Warmup (3-5 requests)
for _ in range(5):
    model.generate(...)

# 3. Benchmark
results = []
for prompt in test_prompts:
    start = time.time()
    model.generate(prompt)
    results.append(time.time() - start)
```

---

**Q: Speculative decoding isn't faster. What's wrong?**

**A**: Check these:
1. **Acceptance rate** - If <30%, text is too unpredictable
2. **Draft model too slow** - Draft should be much smaller than target
3. **Short outputs** - Speculation helps more on longer outputs
4. **High temperature** - Random sampling reduces acceptance

Speculative decoding works best on:
- Code generation
- Structured output (JSON)
- Predictable patterns
- Longer outputs (100+ tokens)

---

### Beyond the Basics

**Q: Can I serve multiple models on one GPU?**

**A**: Yes, with care:
- **Memory budget**: Sum of all model sizes must fit
- **vLLM**: Use model registry for switching
- **Ollama**: Natural multi-model support
- **Trade-off**: Switching has latency cost

```python
# vLLM multi-model (load on demand)
from vllm import LLM

llm_small = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
# When needed, unload small and load large
del llm_small
torch.cuda.empty_cache()
llm_large = LLM(model="meta-llama/Llama-3.1-70B-Instruct")
```

---

**Q: How do I monitor my production API?**

**A**: Key metrics to track:
- **Latency**: Time to first token, total time
- **Throughput**: Requests/second, tokens/second
- **GPU utilization**: Should be >80% under load
- **Memory**: Track for leaks
- **Error rate**: Track failures

```python
# Simple monitoring endpoint
@app.get("/metrics")
async def metrics():
    return {
        "gpu_memory_used_gb": torch.cuda.memory_allocated()/1e9,
        "gpu_memory_reserved_gb": torch.cuda.memory_reserved()/1e9,
        "requests_total": request_counter.value,
        "requests_in_flight": current_requests.value,
        "average_latency_ms": latency_tracker.average()
    }
```

---

**Q: How do I scale beyond one GPU?**

**A**: DGX Spark has one GPU, but for reference:
- **Tensor Parallelism**: Split model across GPUs (vLLM, TRT-LLM)
- **Pipeline Parallelism**: Split layers across GPUs
- **Load Balancing**: Multiple servers, route requests

For DGX Spark's single GPU, focus on:
- Quantization (fit larger models)
- Speculative decoding (faster generation)
- Efficient batching (more concurrent users)

---

## 📞 Still Stuck?

1. **Check server logs** - Most engines print helpful errors
2. **Verify network** - Can you `curl` the health endpoint?
3. **Check resource usage** - `nvidia-smi` for GPU, `htop` for CPU
4. **Try simpler config** - Reduce batch size, sequence length
5. **Update packages** - Inference libraries update frequently
