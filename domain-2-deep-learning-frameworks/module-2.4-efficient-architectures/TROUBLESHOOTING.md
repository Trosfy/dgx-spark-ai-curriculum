# Module 2.4: Efficient Architectures - Troubleshooting Guide

## Quick Diagnostic

**Before diving into specific errors, try these:**

1. Check transformers version: `pip show transformers` (need >= 4.46.0)
2. Check GPU memory: `nvidia-smi`
3. Clear cache: `torch.cuda.empty_cache()`
4. Verify model exists on HuggingFace Hub

---

## Mamba Errors

### Error: `ImportError: Mamba not found in transformers`

**Cause**: transformers version too old.

**Solution**:
```bash
pip install transformers>=4.46.0 --upgrade
```

Verify:
```python
from transformers import MambaConfig  # Should work
```

---

### Error: Mamba first token is slow

**Symptoms**: First token takes 2-3 seconds, then generation is fast.

**Cause**: This is expected behavior - Mamba has initialization overhead.

**Explanation**:
```
Mamba timeline:
├── First token: 2-3s (initializing state, compiling kernels)
└── Subsequent tokens: Very fast (0.01s each)

This is different from Transformers where first token is usually fastest.
```

**Workaround**:
```python
# Warmup before benchmarking
with torch.no_grad():
    model.generate(**inputs, max_new_tokens=10)  # Warmup

# Now benchmark
start = time.time()
outputs = model.generate(**inputs, max_new_tokens=100)
```

---

### Error: Mamba output quality seems lower

**Cause**: Mamba-2.8B is smaller than popular transformers (Llama-8B, etc.)

**Solution**: Compare with similar-sized transformers:
```python
# Fair comparison (similar parameter count):
# Mamba-2.8B vs Phi-2 (2.7B) or Gemma-2B
# NOT Mamba-2.8B vs Llama-3-8B
```

---

## MoE Errors

### Error: `CUDA out of memory` loading Mixtral

**Cause**: Mixtral-8x7B is ~90GB in BF16.

**Solutions**:

```python
# Solution 1: 8-bit quantization
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto"
)

# Solution 2: Use smaller MoE
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-moe-16b-base",  # ~32GB
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Solution 3: 4-bit quantization (smallest)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

---

### Error: Can't access router weights

**Symptoms**: `AttributeError: 'layer' has no attribute 'gate'`

**Cause**: Model structure varies between MoE implementations.

**Solution**: Explore model structure first:
```python
# Print model structure
print(model)

# Or explore programmatically
def find_routers(model):
    routers = []
    for name, module in model.named_modules():
        if 'gate' in name.lower() or 'router' in name.lower():
            routers.append((name, module))
    return routers

print(find_routers(model))
```

Common structures:
```python
# Mixtral
router = model.model.layers[0].block_sparse_moe.gate

# DeepSeek
router = model.model.layers[0].mlp.gate

# Some models use different naming
# Always check with print(model) first
```

---

### Error: MoE expert activation not matching expectations

**Symptoms**: Same experts selected for all tokens.

**Causes**:
1. Model not fully trained (collapsed routing)
2. All inputs too similar
3. Wrong layer being analyzed

**Solutions**:
```python
# Solution 1: Check multiple layers
for i in [0, 5, 10, 15]:
    router = model.model.layers[i].block_sparse_moe.gate
    # Analyze each layer separately

# Solution 2: Use diverse inputs
test_inputs = [
    "Write a Python function that",  # Code
    "The weather in Paris today is",  # Factual
    "Once upon a time in a land",     # Creative
    "According to the research paper", # Academic
]

# Solution 3: Verify top-k selection
router_output = router(hidden_states)
top_k = router_output.topk(2, dim=-1)
print(f"Selected experts: {top_k.indices}")
print(f"Weights: {top_k.values}")
```

---

## Architecture Comparison Errors

### Error: Inconsistent benchmark results

**Symptoms**: Same model gives different speeds on different runs.

**Solutions**:
```python
# Solution 1: Warmup runs
for _ in range(3):
    model.generate(**inputs, max_new_tokens=10)

# Solution 2: Synchronize CUDA
torch.cuda.synchronize()
start = time.time()
outputs = model.generate(**inputs, max_new_tokens=100)
torch.cuda.synchronize()
end = time.time()

# Solution 3: Multiple runs and average
times = []
for _ in range(5):
    torch.cuda.synchronize()
    start = time.time()
    model.generate(**inputs, max_new_tokens=100)
    torch.cuda.synchronize()
    times.append(time.time() - start)

avg_time = sum(times) / len(times)
std_time = (sum((t - avg_time)**2 for t in times) / len(times)) ** 0.5
print(f"Time: {avg_time:.2f} ± {std_time:.2f}s")
```

---

### Error: Memory comparison seems wrong

**Symptoms**: Mamba uses more memory than expected.

**Cause**: Might be measuring at wrong point.

**Solution**:
```python
# Reset memory stats before measurement
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Run forward pass
with torch.no_grad():
    outputs = model(**inputs)

# Get peak memory (not current)
peak_memory = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak memory: {peak_memory:.2f} GB")
```

---

## Fine-tuning Errors

### Error: LoRA not working with Mamba

**Symptoms**: Training fails or no improvement.

**Cause**: Wrong target modules.

**Solution**:
```python
from peft import LoraConfig, get_peft_model

# Mamba-specific target modules
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "in_proj",   # Mamba-specific
        "out_proj",  # Mamba-specific
        "x_proj",    # Mamba-specific
        "dt_proj"    # Mamba-specific
    ],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
```

---

### Error: Gradient explosion during Mamba training

**Symptoms**: NaN loss or very high loss values.

**Solutions**:
```python
# Solution 1: Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Solution 2: Lower learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Solution 3: Gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(**batch).loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
```

---

## Jamba Errors

### Error: Jamba OOM

**Cause**: Jamba-v0.1 is ~104GB in BF16.

**Solution**:
```python
# Use 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    "ai21labs/Jamba-v0.1",
    load_in_8bit=True,
    device_map="auto"
)
# ~52GB with 8-bit
```

---

## Reset Procedures

### Memory Reset

```python
import gc
import torch

# Delete models
del model
gc.collect()
torch.cuda.empty_cache()

# Verify
print(f"Memory after cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```

### Full Environment Reset

```bash
# Exit and restart container
exit

docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3

# Reinstall dependencies
pip install transformers>=4.46.0 peft accelerate
```

---

## Still Stuck?

1. **Check HuggingFace model page** - Often has usage examples
2. **Print model structure** - `print(model)` reveals architecture
3. **Use smaller models first** - Debug with Mamba-130M before 2.8B
4. **Check solution notebooks** - in `solutions/` folder

**Debug template for MoE:**
```python
def debug_moe_layer(model, layer_idx, hidden_states):
    layer = model.model.layers[layer_idx]

    if hasattr(layer, 'block_sparse_moe'):
        moe = layer.block_sparse_moe

        # Router output
        router_logits = moe.gate(hidden_states)
        print(f"Router logits shape: {router_logits.shape}")
        print(f"Router logits range: [{router_logits.min():.2f}, {router_logits.max():.2f}]")

        # Top-k selection
        top_k = torch.topk(router_logits, 2, dim=-1)
        print(f"Selected experts: {top_k.indices[0, :5]}")  # First 5 tokens
        print(f"Expert weights: {top_k.values[0, :5]}")
```
