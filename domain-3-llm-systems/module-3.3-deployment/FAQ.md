# Module 3.3: Model Deployment & Inference Engines - Frequently Asked Questions

## Setup & Environment

### Q: Which inference engine should I use?
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

### Q: Why do I need `--enforce-eager` on DGX Spark?
**A**: DGX Spark uses ARM64 (aarch64) architecture. vLLM's CUDA graphs optimization currently has limited ARM64 support and can cause crashes. `--enforce-eager` disables CUDA graphs and uses standard "eager" execution, which works reliably on ARM64.

```bash
# Always include for DGX Spark
python -m vllm.entrypoints.openai.api_server \
    --model ... \
    --enforce-eager \
    --dtype bfloat16
```

---

### Q: How do I choose between vLLM and SGLang?
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

## Concepts

### Q: What's the difference between prefill and decode?
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

### Q: How does speculative decoding achieve 2-3x speedup?
**A**: Instead of generating one token per forward pass:

1. **Draft model** (small, fast) guesses 5-10 tokens ahead
2. **Target model** (large, accurate) verifies all guesses in ONE pass
3. If 5 guesses are correct, you skip 4 forward passes!

The target model does the same amount of work per token, but you batch multiple token verifications together. On predictable text (code, structured output), acceptance rates are high.

---

### Q: What is PagedAttention and why does it matter?
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

### Q: Why is TensorRT-LLM so fast for prefill but not decode?
**A**: TensorRT-LLM optimizes GPU computation (matrix multiplications, kernels). Prefill is compute-heavy, so optimizations help a lot.

Decode is memory-bandwidth-heavy (reading KV cache). TensorRT's compute optimizations don't help as much there. llama.cpp's memory-efficient design actually decodes faster!

---

## Troubleshooting

### Q: My vLLM server won't start. What should I check?
**A**: In order:
1. **Port available?** `lsof -i :8000`
2. **Memory available?** `nvidia-smi`
3. **Model accessible?** HuggingFace token set?
4. **Using `--enforce-eager`?** Required for DGX Spark
5. **Correct dtype?** Use `bfloat16` for Blackwell

---

### Q: My benchmarks are inconsistent. Why?
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

### Q: Speculative decoding isn't faster. What's wrong?
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

## Beyond the Basics

### Q: Can I serve multiple models on one GPU?
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

### Q: How do I monitor my production API?
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

### Q: How do I scale beyond one GPU?
**A**: DGX Spark has one GPU, but for reference:
- **Tensor Parallelism**: Split model across GPUs (vLLM, TRT-LLM)
- **Pipeline Parallelism**: Split layers across GPUs
- **Load Balancing**: Multiple servers, route requests

For DGX Spark's single GPU, focus on:
- Quantization (fit larger models)
- Speculative decoding (faster generation)
- Efficient batching (more concurrent users)

---

## Still Have Questions?

- Check [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for error-specific help
- Review [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) for correct patterns
- See [ELI5.md](./ELI5.md) for concept explanations
- Consult engine docs: [vLLM](https://docs.vllm.ai/), [SGLang](https://github.com/sgl-project/sglang)
