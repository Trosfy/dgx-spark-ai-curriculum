# LoRA: Low-Rank Adaptation for Large Language Models

## Introduction

Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique that enables adapting large language models with minimal computational resources. Instead of updating all model parameters, LoRA freezes the pretrained weights and injects trainable low-rank decomposition matrices into each layer.

## The Problem LoRA Solves

### Traditional Fine-Tuning Challenges
Full fine-tuning of large language models presents significant challenges:

- **Memory Requirements**: A 70B model requires ~280GB just for weights in FP16
- **Storage Costs**: Each fine-tuned version creates a complete model copy
- **Training Time**: Updating billions of parameters is slow
- **Catastrophic Forgetting**: Risk of losing pretrained knowledge

### LoRA's Solution
LoRA addresses these challenges by:
- Training only 0.1-1% of the parameters
- Reducing memory requirements by 90%+
- Creating small adapter files (~50-100MB)
- Preserving the original model weights

## How LoRA Works

### Low-Rank Decomposition
The key insight of LoRA is that the weight updates during fine-tuning have low "intrinsic rank." Instead of updating a weight matrix W directly, LoRA learns a low-rank decomposition:

```
W' = W + BA
```

Where:
- W is the frozen pretrained weight (d × k)
- B is a trainable matrix (d × r)
- A is a trainable matrix (r × k)
- r is the rank (typically 8, 16, 32, or 64)

### Rank Selection
The rank r is a key hyperparameter:
- **r = 8**: Minimal overhead, good for simple adaptations
- **r = 16**: Common default, balanced performance
- **r = 32-64**: Better quality, more parameters
- **r = 128+**: Approaching full fine-tuning quality

### Alpha Scaling
LoRA uses a scaling factor alpha to control the magnitude of updates:

```
W' = W + (alpha/r) * BA
```

Common settings:
- alpha = r: No scaling (1x)
- alpha = 2*r: 2x scaling
- alpha = 32, r = 16: 2x scaling

## Target Modules

### Standard Targets
LoRA can be applied to different weight matrices. Common choices:

**Query and Value Projections (Q, V)**
- Most commonly targeted
- Good balance of quality and efficiency
- Default for many implementations

**All Attention Projections (Q, K, V, O)**
- Higher quality adaptations
- More parameters to train
- Recommended for complex tasks

**Feed-Forward Layers (up_proj, down_proj, gate_proj)**
- Often overlooked but impactful
- Especially useful for knowledge injection

### Module Naming by Model
Different model architectures use different naming conventions:

| Component | LLaMA/Mistral | GPT-2 | BERT |
|-----------|---------------|-------|------|
| Query | q_proj | c_attn | query |
| Key | k_proj | c_attn | key |
| Value | v_proj | c_attn | value |
| Output | o_proj | c_proj | dense |
| FF Up | up_proj | c_fc | intermediate |
| FF Down | down_proj | c_proj | output |

## QLoRA: Quantized LoRA

QLoRA combines LoRA with quantization for even greater efficiency:

### How QLoRA Works
1. Load base model in 4-bit (NF4) quantization
2. Keep base model frozen and quantized
3. Train LoRA adapters in FP16/BF16
4. Compute gradients through quantized weights

### Memory Savings
For a 70B model:
- Full FP16: ~140GB
- 4-bit Quantized: ~35GB
- 4-bit + LoRA: ~40GB (with adapters)

This enables fine-tuning 70B models on a single DGX Spark!

### QLoRA Configuration
```python
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

## Training Tips

### Learning Rate
LoRA typically requires higher learning rates than full fine-tuning:
- Full fine-tuning: 1e-5 to 5e-5
- LoRA: 1e-4 to 3e-4

### Batch Size
Larger batch sizes often help with LoRA:
- Accumulate gradients if memory is limited
- Target effective batch size of 32-128

### Warmup
Use learning rate warmup:
- 3-10% of total steps
- Linear warmup works well

### Dropout
LoRA supports its own dropout:
- lora_dropout=0.05 to 0.1
- Helps prevent overfitting

## Advanced Techniques

### DoRA: Weight-Decomposed LoRA
DoRA decomposes weights into magnitude and direction:
- Improves learning dynamics
- Often +3-4% improvement over LoRA

### NEFTune: Noisy Embeddings
Adding noise to embeddings during training:
- Significantly improves instruction following
- 29.8% -> 64.7% on AlpacaEval in one study

### AdaLoRA: Adaptive Rank
Dynamically adjusts rank during training:
- Allocates more rank to important layers
- Can improve efficiency and quality

## Merging LoRA Adapters

### Why Merge?
- Eliminate adapter overhead during inference
- Combine multiple adapters
- Simplify deployment

### Merging Process
```python
# Merge adapter into base model
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("merged_model/")
```

### Multiple Adapters
PEFT supports loading multiple adapters:
```python
model.load_adapter("adapter1", adapter_name="task1")
model.load_adapter("adapter2", adapter_name="task2")
model.set_adapter("task1")  # Switch between them
```

## Comparison with Other Methods

### LoRA vs Full Fine-Tuning
| Aspect | LoRA | Full Fine-Tuning |
|--------|------|------------------|
| Parameters | 0.1-1% | 100% |
| Memory | 10-20% | 100% |
| Quality | 95-99% | 100% |
| Speed | 2-10x faster | Baseline |

### LoRA vs Prefix Tuning
- LoRA typically achieves better quality
- LoRA adds no inference latency when merged
- Prefix tuning works better for some tasks

### LoRA vs Adapters
- LoRA can be merged into weights
- Adapters add permanent inference overhead
- Both achieve similar quality

## Conclusion

LoRA has become the de facto standard for efficient fine-tuning of large language models. Its combination of low resource requirements, high quality, and ease of use makes it an essential technique for any practitioner working with LLMs.
