# Module 2.4: Efficient Architectures - Prerequisites

## Required Prior Knowledge

Before starting this module, verify you can confidently complete these skill checks.

---

## From Module 2.3: NLP & Transformers

### Skill 1: Transformer Architecture Fundamentals
**Can you explain what the attention mechanism does?**

<details>
<summary>Self-Check Answer</summary>

The attention mechanism allows each token in a sequence to "attend to" (compute relevance scores for) every other token. Key concepts:
- **Query, Key, Value (Q, K, V)**: Projections of the input that compute attention scores
- **Attention Score**: `softmax(Q @ K.T / sqrt(d_k)) @ V`
- **O(n^2) Complexity**: Every token attends to every other token, creating quadratic scaling

This quadratic complexity is exactly what Mamba (State Space Models) and MoE aim to address!
</details>

### Skill 2: Loading and Running Transformer Models
**Can you load a pretrained model with HuggingFace and generate text?**

<details>
<summary>Self-Check Code</summary>

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

# Generate text
inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0]))
```
</details>

### Skill 3: Understanding Model Memory Requirements
**Can you estimate how much GPU memory a model needs?**

<details>
<summary>Self-Check Answer</summary>

Basic formula for model weights:
- **FP32**: Parameters × 4 bytes
- **FP16/BF16**: Parameters × 2 bytes
- **INT8**: Parameters × 1 byte
- **INT4**: Parameters × 0.5 bytes

Example: A 7B parameter model in BF16 = 7 × 10^9 × 2 bytes = 14 GB

For inference, add:
- KV cache (grows with context length for transformers)
- Activation memory
- Framework overhead (~10-20%)

On DGX Spark with 128GB unified memory, you can run models up to ~50B in FP16 inference.
</details>

---

## From Module 2.1: Deep Learning with PyTorch

### Skill 4: Basic PyTorch Operations
**Can you work with tensors and understand device placement?**

<details>
<summary>Self-Check Code</summary>

```python
import torch

# Create tensors
x = torch.randn(2, 3)

# Move to GPU
x_gpu = x.to("cuda")
x_gpu = x.cuda()  # Alternative

# Check device
print(x_gpu.device)  # cuda:0

# Memory management
torch.cuda.empty_cache()
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```
</details>

### Skill 5: Model Forward Pass and Generation
**Do you understand how autoregressive generation works?**

<details>
<summary>Self-Check Answer</summary>

Autoregressive generation:
1. Input prompt is tokenized: "Hello" → [15496]
2. Model produces next token probability: P(token | [15496])
3. Sample or select next token: "world" → [995]
4. Append and repeat: P(token | [15496, 995])
5. Continue until stop token or max length

For this module, you'll compare HOW different architectures compute these next-token probabilities:
- **Transformers**: Full attention over all previous tokens
- **Mamba**: Compressed state representation
- **MoE**: Sparse expert selection
</details>

---

## From Module 1.1: DGX Spark Platform

### Skill 6: NGC Container Usage
**Can you start an NGC container with proper GPU access?**

<details>
<summary>Self-Check Command</summary>

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

Key flags:
- `--gpus all`: Enable GPU access
- `--ipc=host`: Required for PyTorch DataLoader workers
- `-v` mounts: Persist workspace and HuggingFace cache
- Container: `nvcr.io/nvidia/pytorch:25.11-py3`
</details>

### Skill 7: DGX Spark Hardware Awareness
**Do you know DGX Spark's key specifications?**

<details>
<summary>Self-Check Answer</summary>

| Spec | Value |
|------|-------|
| GPU | NVIDIA Blackwell GB10 Superchip |
| Memory | 128GB unified (LPDDR5X) |
| CUDA Cores | 6,144 |
| Tensor Cores | 192 (5th generation) |
| NVFP4 Performance | 1 PFLOP |
| Architecture | ARM64/aarch64 |

For this module, the 128GB unified memory is critical - it allows:
- Mamba: 100K+ token contexts (constant memory!)
- MoE: Full Mixtral 8x7B without quantization (~90GB)
- Side-by-side architecture comparison
</details>

---

## Mathematical Prerequisites

### Linear Algebra Basics
**Required**: Matrix multiplication, eigenvalues (conceptual)

For understanding Mamba's state space equations:
```
h[t] = A·h[t-1] + B·x[t]    (State update)
y[t] = C·h[t]                (Output)
```

### Probability and Statistics
**Required**: Softmax, probability distributions

For understanding MoE routing:
```
router_weights = softmax(router_logits)
top_k_experts = topk(router_weights, k=2)
```

---

## Ready to Start?

### Quick Readiness Check

Run this code to verify your environment:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Memory: {mem:.1f} GB")

    if mem > 100:
        print("\n✅ DGX Spark detected! Ready for all labs.")
    else:
        print("\n⚠️ Limited GPU memory. Some labs may need adjustment.")

# Check transformers version for Mamba support
min_version = tuple(map(int, "4.46.0".split('.')))
current = tuple(map(int, transformers.__version__.split('.')[:3]))
if current >= min_version:
    print(f"✅ Transformers version OK for Mamba support")
else:
    print(f"⚠️ Upgrade transformers: pip install transformers>=4.46.0")
```

### If You're Not Ready

| Missing Skill | Review Module |
|---------------|---------------|
| Transformer attention | Module 2.3: NLP & Transformers |
| PyTorch basics | Module 2.1: Deep Learning with PyTorch |
| Model loading with HuggingFace | Module 2.3 or Module 2.5: Hugging Face Ecosystem |
| DGX Spark container usage | Module 1.1: DGX Spark Platform Mastery |
| Memory estimation | Module 3.2: Quantization & Optimization |

---

## What You'll Learn in This Module

After completing the prerequisites, you're ready to explore:

1. **Mamba (State Space Models)**: O(n) complexity alternative to transformers
2. **Mixture of Experts (MoE)**: Sparse activation for efficient large models
3. **Hybrid architectures**: Jamba and similar approaches
4. **Architecture comparison**: Benchmarking and selection criteria

These concepts build directly on your transformer understanding to show what comes NEXT in efficient AI architectures.
