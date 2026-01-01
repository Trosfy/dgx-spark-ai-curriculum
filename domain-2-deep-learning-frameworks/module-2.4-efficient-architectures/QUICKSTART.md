# Module 2.4: Efficient Architectures - Quickstart

## Time: ~5 minutes

## What You'll Do
Compare a Mamba model with a Transformer - see why Mamba's O(n) complexity matters.

## Before You Start
- [ ] DGX Spark container running
- [ ] `transformers >= 4.46.0` installed

## Let's Go!

### Step 1: Load Mamba Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

# Load Mamba (requires transformers >= 4.46.0)
print("Loading Mamba-2.8B...")
mamba_model = AutoModelForCausalLM.from_pretrained(
    "state-spaces/mamba-2.8b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
mamba_tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
print(f"Mamba loaded: {sum(p.numel() for p in mamba_model.parameters())/1e9:.1f}B params")
```

### Step 2: Generate with Mamba
```python
prompt = "The key insight of Mamba compared to Transformers is"
inputs = mamba_tokenizer(prompt, return_tensors="pt").to("cuda")

start = time.time()
outputs = mamba_model.generate(**inputs, max_new_tokens=50)
mamba_time = time.time() - start

print(f"\nMamba generation ({mamba_time:.2f}s):")
print(mamba_tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Step 3: Compare with Transformer (Optional)
```python
# Load a similar-sized transformer
print("\nLoading Phi-2 (2.7B transformer)...")
phi_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

inputs = phi_tokenizer(prompt, return_tensors="pt").to("cuda")
start = time.time()
outputs = phi_model.generate(**inputs, max_new_tokens=50)
phi_time = time.time() - start

print(f"\nPhi-2 generation ({phi_time:.2f}s):")
print(phi_tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Step 4: See the Memory Difference
```python
import gc

# Check memory after both models
print(f"\nGPU Memory Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Key insight: Try with longer sequences!
# Mamba's constant memory shines with 16K+ token contexts
```

## You Did It!

You just:
- Loaded a Mamba (State Space Model) on DGX Spark
- Generated text with Mamba's O(n) complexity
- Saw that inference works just like Transformers (same API!)

In the full module, you'll learn:
- Why Mamba uses O(n) instead of O(n²) memory
- The selective state space mechanism (what makes Mamba work)
- Mixture of Experts (MoE) and sparse activation
- When to use Mamba vs Transformer vs MoE
- Fine-tuning Mamba with LoRA

## Next Steps
1. **Understand the mechanism**: Start with [Lab 2.4.2](./labs/lab-2.4.2-mamba-architecture.ipynb)
2. **Benchmark long sequences**: See where Mamba really shines (32K+ tokens)
3. **Explore MoE**: Try [Lab 2.4.3](./labs/lab-2.4.3-moe-exploration.ipynb)
