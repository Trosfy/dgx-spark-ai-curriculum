# Module 2.4: Efficient Architectures - Quick Reference

## Loading Models

### Mamba

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Requires transformers >= 4.46.0
model = AutoModelForCausalLM.from_pretrained(
    "state-spaces/mamba-2.8b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
```

### MoE (Mixtral)

```python
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-v0.1",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
# For DGX Spark: Fits in 128GB in BF16 (~90GB)
```

### MoE with Quantization

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True  # ~45GB for Mixtral
)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto"
)
```

### Jamba (Hybrid)

```python
model = AutoModelForCausalLM.from_pretrained(
    "ai21labs/Jamba-v0.1",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

---

## Architecture Comparison

| Model | Total Params | Active Params | Max Context | Memory (BF16) |
|-------|-------------|---------------|-------------|---------------|
| Mamba-2.8B | 2.8B | 2.8B | 100K+ | ~6 GB |
| Llama-3-8B | 8B | 8B | 8K (128K ext) | ~16 GB |
| Mixtral-8x7B | 45B | ~12B | 32K | ~90 GB |
| DeepSeekMoE-16B | 16B | 2.5B | 4K | ~32 GB |
| Jamba-v0.1 | 52B | ~12B | 256K | ~104 GB |

---

## Key Patterns

### Pattern: Benchmark Inference Speed

```python
import time
import torch

def benchmark_generation(model, tokenizer, prompt, max_new_tokens=100, num_runs=5):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10)

    # Benchmark
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()
        times.append(time.time() - start)

    tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
    avg_time = sum(times) / len(times)
    tok_per_sec = tokens_generated / avg_time

    return {
        'avg_time': avg_time,
        'tok_per_sec': tok_per_sec,
        'tokens_generated': tokens_generated
    }
```

### Pattern: Memory Comparison

```python
def measure_memory(model, tokenizer, seq_lengths=[1024, 4096, 16384]):
    results = []

    for seq_len in seq_lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Create input
        prompt = "Hello " * (seq_len // 2)
        inputs = tokenizer(prompt, return_tensors="pt", max_length=seq_len,
                          truncation=True).to("cuda")

        # Forward pass
        with torch.no_grad():
            _ = model(**inputs)

        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        results.append({
            'seq_len': seq_len,
            'peak_memory_gb': peak_memory
        })

    return results
```

### Pattern: Access MoE Router Weights

```python
# For Mixtral-style models
def get_router_weights(model, layer_idx):
    """Extract router weights from MoE layer"""
    layer = model.model.layers[layer_idx]

    # Mixtral structure
    if hasattr(layer, 'block_sparse_moe'):
        router = layer.block_sparse_moe.gate
        return router.weight  # (num_experts, hidden_size)

    # DeepSeek structure
    if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
        return layer.mlp.gate.weight

    return None
```

### Pattern: Analyze Expert Selection

```python
def analyze_expert_selection(model, tokenizer, texts):
    """Track which experts are selected for different inputs"""
    expert_counts = {}

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to("cuda")

        # Hook to capture router decisions
        router_outputs = []
        def hook(module, input, output):
            router_outputs.append(output)

        # Register hooks on all router layers
        hooks = []
        for layer in model.model.layers:
            if hasattr(layer, 'block_sparse_moe'):
                hooks.append(layer.block_sparse_moe.gate.register_forward_hook(hook))

        # Forward pass
        with torch.no_grad():
            model(**inputs)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Analyze selections
        for output in router_outputs:
            selected = output.topk(2, dim=-1).indices  # top-2 routing
            for expert_idx in selected.flatten().tolist():
                expert_counts[expert_idx] = expert_counts.get(expert_idx, 0) + 1

    return expert_counts
```

### Pattern: Simplified Selective Scan (Educational)

```python
def simplified_selective_scan(x, A, B, C, delta):
    """
    Simplified selective scan for understanding (not optimized)

    x: input sequence (batch, seq_len, dim)
    A, B, C: state space parameters (learned)
    delta: step size (learned, input-dependent)

    Returns: output sequence
    """
    batch, seq_len, dim = x.shape
    state_dim = A.shape[0]

    h = torch.zeros(batch, state_dim, device=x.device)  # Initial state
    outputs = []

    for t in range(seq_len):
        # Discretize A using delta (this is the "selective" part)
        A_discrete = torch.exp(delta[:, t] * A)

        # State update: h = A_discrete * h + B * x_t
        h = A_discrete * h + B @ x[:, t].unsqueeze(-1)

        # Output: y_t = C * h
        y = (C @ h).squeeze(-1)
        outputs.append(y)

    return torch.stack(outputs, dim=1)
```

---

## Key Values

| Architecture | Complexity | KV Cache | Best For |
|--------------|-----------|----------|----------|
| Transformer | O(n²) | Yes, grows with context | Most tasks, highest quality |
| Mamba | O(n) | No, constant state | Long context, streaming |
| MoE | O(n×k/N) | Yes | Large-scale, multi-domain |
| Jamba | Mixed | Reduced | Long context with precision |

Where: n = sequence length, k = active experts, N = total experts

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| `ImportError: Mamba not found` | `pip install transformers>=4.46.0` |
| MoE out of memory | Use `load_in_8bit=True` |
| Mamba slow first token | Expected - initialization overhead |
| Can't access router | Check model structure with `print(model)` |
| Inconsistent benchmarks | Warmup runs before timing |

---

## Quick Comparison Commands

```python
# Compare generation speed
mamba_speed = benchmark_generation(mamba_model, mamba_tokenizer, prompt)
transformer_speed = benchmark_generation(transformer_model, transformer_tokenizer, prompt)

print(f"Mamba: {mamba_speed['tok_per_sec']:.1f} tok/s")
print(f"Transformer: {transformer_speed['tok_per_sec']:.1f} tok/s")

# Compare memory at different context lengths
mamba_mem = measure_memory(mamba_model, mamba_tokenizer, [4096, 16384, 32768])
transformer_mem = measure_memory(transformer_model, transformer_tokenizer, [4096, 16384, 32768])

for m, t in zip(mamba_mem, transformer_mem):
    print(f"Context {m['seq_len']}: Mamba {m['peak_memory_gb']:.1f}GB, Transformer {t['peak_memory_gb']:.1f}GB")
```

---

## Quick Links

- [Mamba Paper](https://arxiv.org/abs/2312.00752)
- [Mamba-2 Paper](https://arxiv.org/abs/2405.21060)
- [Mixtral Paper](https://arxiv.org/abs/2401.04088)
- [DeepSeekMoE Paper](https://arxiv.org/abs/2401.06066)
- [Jamba Paper](https://arxiv.org/abs/2403.19887)
- [HuggingFace Mamba Guide](https://huggingface.co/docs/transformers/model_doc/mamba)
