# Module 3.3: Model Deployment & Inference Engines - Prerequisites Check

## 🎯 Purpose
This module builds on quantization knowledge to deploy models efficiently. Use this self-check to ensure you're ready.

## ⏱️ Estimated Time
- **If all prerequisites met**: Jump straight to [QUICKSTART.md](./QUICKSTART.md)
- **If 1-2 gaps**: ~2-3 hours of review
- **If multiple gaps**: Complete Modules 3.1-3.2 first

---

## Required Skills

### 1. Model Loading: Quantized Models

**Can you do this?**
```python
# Without looking anything up:
# 1. Load a model with 4-bit quantization
# 2. Run inference on it
# 3. Check memory usage
```

<details>
<summary>✅ Check your answer</summary>

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Run inference
inputs = tokenizer("Hello!", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0]))

# Check memory
print(f"Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```

**Key points**:
- Use `bfloat16` compute dtype for DGX Spark
- `device_map="auto"` handles placement
- Quantized models reduce memory 3-4x

</details>

**Not ready?** Review: [Module 3.2: Quantization](../module-3.2-quantization/)

---

### 2. REST APIs: Basic HTTP Concepts

**Do you know these terms?**
| Term | Your Definition |
|------|-----------------|
| REST API | [Write yours here] |
| POST request | [Write yours here] |
| JSON payload | [Write yours here] |
| Streaming (SSE) | [Write yours here] |

<details>
<summary>✅ Check definitions</summary>

| Term | Definition |
|------|------------|
| REST API | Interface for communicating with a service over HTTP using standard methods (GET, POST, etc.) |
| POST request | HTTP method for sending data to a server (used for LLM inference requests) |
| JSON payload | Structured data format used to send/receive information (model, messages, parameters) |
| Streaming (SSE) | Server-Sent Events: receiving data incrementally as it's generated (tokens as they're produced) |

</details>

**Not ready?** This is quick to learn—the labs include examples.

---

### 3. Docker: Container Networking

**Can you answer this?**
> How do you expose a port from inside a Docker container to the host machine?

<details>
<summary>✅ Check your answer</summary>

Use the `-p` flag to map ports:
```bash
docker run -p 8000:8000 ...  # Host port 8000 → Container port 8000
```

For multiple ports:
```bash
docker run -p 8000:8000 -p 8888:8888 ...
```

For accessing host services from container, use `--network=host`:
```bash
docker run --network=host ...  # Container shares host network
```

**For this module**:
- vLLM typically uses port 8000
- Jupyter uses port 8888
- Ollama uses port 11434

</details>

**Not ready?** Review: [Module 1.1: DGX Spark Platform](../../domain-1-platform-foundations/module-1.1-dgx-spark-platform/)

---

### 4. Benchmarking: Measuring Performance

**Can you answer this?**
> What's the difference between latency and throughput for LLM inference?

<details>
<summary>✅ Check your answer</summary>

**Latency**: Time for one request (end-to-end)
- "How long until the user sees the full response?"
- Measured in seconds or milliseconds
- Lower is better

**Throughput**: Requests per unit time
- "How many users can we serve simultaneously?"
- Measured in requests/second or tokens/second
- Higher is better

**Tradeoff**: Batching increases throughput but may increase latency

**For LLMs, we also track**:
- **Prefill tok/s**: Speed of processing input
- **Decode tok/s**: Speed of generating output
- **Time to First Token (TTFT)**: Latency for first output token

</details>

**Not ready?** This module will teach you—it's a core focus!

---

### 5. GPU Memory: KV Cache Understanding

**Can you answer this?**
> What is the KV cache and why does it grow during generation?

<details>
<summary>✅ Check your answer</summary>

**KV Cache (Key-Value Cache)**:
- Stores attention key and value vectors for all previous tokens
- Avoids recomputing attention for tokens already processed

**Why it grows**:
- Each new token adds its K and V vectors to the cache
- Cache size = (num_layers × 2 × hidden_dim × num_tokens) × bytes_per_value
- For 70B model, 2048 tokens: ~10-20GB just for KV cache!

**Memory formula for inference**:
```
Total Memory = Model Weights + KV Cache + Activations

Model Weights: Fixed (based on quantization)
KV Cache: Grows with sequence length and batch size
Activations: Temporary, usually small
```

**PagedAttention** (vLLM) solves KV cache memory fragmentation.

</details>

**Not ready?** Review: [Module 3.2: Quantization](../module-3.2-quantization/) and attention mechanism concepts

---

### 6. DGX Spark: ARM64 Considerations

**Can you answer this?**
> Why do we need `--enforce-eager` when running vLLM on DGX Spark?

<details>
<summary>✅ Check your answer</summary>

**DGX Spark uses ARM64 (aarch64) architecture**, not x86.

**CUDA Graphs** (vLLM's default optimization):
- Pre-compile GPU operations for faster execution
- Currently have limited ARM64 support
- Can cause errors or hangs on DGX Spark

**`--enforce-eager` flag**:
- Disables CUDA graphs
- Uses "eager mode" where operations run immediately
- Works reliably on ARM64

**Other DGX Spark considerations**:
- Use `dtype bfloat16` (Blackwell native)
- Check container ARM64 compatibility
- Some packages may need source builds

</details>

**Not ready?** Review: [Module 1.1: DGX Spark Platform](../../domain-1-platform-foundations/module-1.1-dgx-spark-platform/)

---

## Optional But Helpful

These aren't required but will accelerate your learning:

### FastAPI Basics
**Why it helps**: Lab 3.3.7 builds a production API.
**Quick primer**: FastAPI is a modern Python web framework. You define endpoints with decorators and async functions.

### Async/Await in Python
**Why it helps**: Streaming responses and concurrent handling.
**Quick primer**: `async def` and `await` allow non-blocking I/O, essential for handling multiple requests.

---

## Ready?

- [ ] I can load and run quantized models
- [ ] I understand REST APIs and HTTP methods
- [ ] I can map Docker container ports
- [ ] I know the difference between latency and throughput
- [ ] I understand KV cache and why it matters
- [ ] I know about ARM64 considerations for DGX Spark

**All boxes checked?** → Start with [QUICKSTART.md](./QUICKSTART.md)!

**Some gaps?** → Review Module 3.2 first—deployment builds on quantization.
