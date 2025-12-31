# Quantization Methods for Large Language Models

## Introduction

Quantization is the process of reducing the precision of model weights and activations from higher bit-widths (like FP32 or FP16) to lower bit-widths (like INT8, INT4, or FP4). This technique is essential for deploying large language models on resource-constrained hardware.

## Why Quantization Matters

### Memory Reduction
A 70B parameter model requires:
- FP32: 280 GB
- FP16: 140 GB
- INT8: 70 GB
- INT4: 35 GB

### Speed Improvements
Lower precision enables:
- Faster memory transfers
- More efficient compute on modern hardware
- Higher throughput

### Accuracy Trade-offs
The challenge is maintaining model quality:
- Well-designed 4-bit quantization can preserve 95%+ of quality
- Poor quantization can severely degrade performance
- Different tasks have different sensitivity

## Quantization Methods Overview

### Post-Training Quantization (PTQ)
Quantizes an already-trained model without additional training:
- Fast and simple to apply
- May result in accuracy loss for very low bit-widths
- Examples: GPTQ, AWQ, GGUF

### Quantization-Aware Training (QAT)
Simulates quantization during training:
- Better accuracy preservation
- Requires retraining
- More computationally expensive

## GPTQ (GPT Quantization)

### How GPTQ Works
GPTQ is a one-shot weight quantization method based on approximate second-order information:

1. Process one layer at a time
2. For each column of weights, find optimal quantized values
3. Update remaining columns to compensate for quantization error
4. Use a calibration dataset to guide the process

### GPTQ Configuration
```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True,  # Activation order
    damp_percent=0.1,
)

model = AutoGPTQForCausalLM.from_pretrained(
    model_path,
    quantize_config,
    device_map="auto"
)
```

### GPTQ Characteristics
- Very accurate for 4-bit quantization
- Relatively slow quantization process
- Fast inference
- Widely supported (HuggingFace, vLLM, etc.)

## AWQ (Activation-aware Weight Quantization)

### How AWQ Works
AWQ protects "salient" weights that are most important for performance:

1. Analyze activation patterns on calibration data
2. Identify weights that correspond to high activations
3. Scale these weights before quantization
4. Apply inverse scaling to preserve original computation

### AWQ Configuration
```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

model.quantize(tokenizer, quant_config=quant_config)
```

### AWQ Characteristics
- Often better quality than GPTQ at same bit-width
- Faster quantization than GPTQ
- Excellent inference performance
- Good hardware acceleration support

## GGUF (GGML Universal Format)

### What is GGUF?
GGUF is a file format for storing quantized models, designed for use with llama.cpp and similar inference engines:

- Successor to the older GGML format
- Self-contained with metadata
- Supports multiple quantization types

### GGUF Quantization Types
| Type | Bits | Quality | Size | Description |
|------|------|---------|------|-------------|
| F16 | 16 | Excellent | 1.0x | Half precision |
| Q8_0 | 8 | Excellent | 0.5x | Simple 8-bit |
| Q6_K | 6 | Very Good | 0.38x | K-quant 6-bit |
| Q5_K_M | 5.5 | Good | 0.35x | K-quant 5.5-bit |
| Q4_K_M | 4.8 | Good | 0.30x | K-quant medium |
| Q4_K_S | 4.5 | Acceptable | 0.28x | K-quant small |
| Q3_K_M | 3.5 | Lower | 0.25x | K-quant 3.5-bit |
| Q2_K | 2.8 | Lowest | 0.20x | 2-bit (experimental) |

### Converting to GGUF
```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp

# Convert HuggingFace model to GGUF
python convert_hf_to_gguf.py model_path --outfile model.gguf

# Quantize to specific format
./llama-quantize model.gguf model-q4_k_m.gguf Q4_K_M
```

## NVFP4: Native Blackwell Quantization

### What is NVFP4?
NVFP4 is NVIDIA's native 4-bit floating-point format exclusive to Blackwell architecture:

- Uses 4-bit floating point with micro-block scaling
- 3.5x memory reduction vs FP16
- Native hardware support in Blackwell tensor cores
- Excellent quality preservation

### NVFP4 on DGX Spark
The DGX Spark's Blackwell GPU can run ~200B parameter models at NVFP4:
- Llama 3.1 405B fits on dual Spark (NVLink connected)
- Superior to integer quantization for many tasks

### Using NVFP4
```python
import tensorrt_llm
from tensorrt_llm.quantization import QuantMode

# Configure FP4 quantization
quant_mode = QuantMode.from_description(
    quantize_weights=True,
    quantize_activations=True,
    per_group=True,
    use_fp4_weights=True,
    use_fp4_activations=True
)
```

## FP8 Quantization

### Understanding FP8
FP8 provides an excellent balance of speed and quality:

| Format | Exponent | Mantissa | Use Case |
|--------|----------|----------|----------|
| E4M3 | 4 bits | 3 bits | Inference |
| E5M2 | 5 bits | 2 bits | Training |

### FP8 on DGX Spark
- Native Blackwell support
- ~90-100B models fit in 128GB
- Minimal quality degradation
- 2x speedup vs FP16

## BitsAndBytes Quantization

### 4-bit Quantization with bitsandbytes
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # or "fp4"
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # Nested quantization
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto"
)
```

### NF4 vs FP4
- **NF4**: Optimal for normally-distributed weights (most LLMs)
- **FP4**: Standard floating-point representation

### Double Quantization
Quantizes the quantization constants themselves:
- Additional ~0.4 bits/parameter savings
- Minimal quality impact

## Quality Comparison

### Benchmark Results (Llama 2 70B on WikiText-2)
| Method | Perplexity | Memory | Speed |
|--------|------------|--------|-------|
| FP16 | 3.12 | 140GB | 1.0x |
| GPTQ-4bit | 3.21 | 35GB | 2.5x |
| AWQ-4bit | 3.18 | 35GB | 2.8x |
| GGUF-Q4_K_M | 3.25 | 35GB | 2.2x |
| FP8 | 3.13 | 70GB | 1.8x |

## Best Practices

### Choosing a Quantization Method
1. **For deployment**: AWQ or GPTQ with vLLM
2. **For local/edge**: GGUF with llama.cpp
3. **For DGX Spark**: NVFP4 for maximum model size
4. **For training**: bitsandbytes with QLoRA

### Calibration Data
- Use data representative of your use case
- More calibration samples generally helps (up to a point)
- 128-512 samples is often sufficient

### Validation
Always validate quantized models:
- Test on your specific tasks
- Compare perplexity scores
- Check for obvious degradation

## Conclusion

Quantization is essential for practical LLM deployment. Modern methods like AWQ, GPTQ, and NVFP4 enable running large models with minimal quality loss. The DGX Spark's Blackwell architecture provides native support for the latest quantization formats.
