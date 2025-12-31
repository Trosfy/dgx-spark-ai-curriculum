# Module 3.2: Quantization & Optimization - Frequently Asked Questions

## Setup & Environment

### Q: Do I need a Blackwell GPU for this module?
**A**: Not entirely. Most quantization methods (GPTQ, AWQ, GGUF, bitsandbytes) work on any modern GPU. However, NVFP4 is **Blackwell-exclusive** (compute capability 10.0+). DGX Spark's unique advantage is native NVFP4 and FP8 tensor core support.

| Method | Ampere (8.x) | Hopper (9.x) | Blackwell (10.x) |
|--------|--------------|--------------|------------------|
| bitsandbytes 4-bit | ✅ | ✅ | ✅ |
| GPTQ | ✅ | ✅ | ✅ |
| AWQ | ✅ | ✅ | ✅ |
| GGUF | ✅ | ✅ | ✅ |
| FP8 | ⚠️ Emulated | ✅ Native | ✅ Native |
| NVFP4 | ❌ | ❌ | ✅ Native |

---

### Q: Which quantization method should I use?
**A**:

| If you need... | Use... | Why |
|----------------|--------|-----|
| Quick 4-bit for fine-tuning | bitsandbytes | Integrated with HuggingFace |
| Best inference quality | AWQ | Activation-aware, minimal quality loss |
| Fastest quantization | GPTQ | Well-optimized process |
| Ollama/llama.cpp compatibility | GGUF | Universal format |
| Maximum DGX Spark performance | NVFP4 | Hardware-accelerated |
| Training in reduced precision | FP8 | Native Blackwell support |

**Default recommendation**: Start with bitsandbytes for development, AWQ for production inference, GGUF for Ollama deployment.

---

### Q: How much quality do I lose with 4-bit quantization?
**A**: Less than you'd expect! With proper calibration:

| Quantization | Typical Perplexity Increase | MMLU Accuracy Loss |
|--------------|----------------------------|-------------------|
| FP16 → INT8 | ~0.05 | <0.5% |
| FP16 → INT4 (GPTQ) | ~0.2-0.3 | ~1-2% |
| FP16 → INT4 (AWQ) | ~0.1-0.2 | ~0.5-1% |
| FP16 → NVFP4 | ~0.05 | <0.1% |

NVFP4's dual-level scaling achieves near-FP16 quality at 4-bit size.

---

## Concepts

### Q: What's the difference between NF4 and INT4?
**A**:

**INT4** (regular 4-bit integer):
- 16 evenly spaced values: -8, -7, -6, ... 0, ... 6, 7
- Works poorly for neural networks (weights aren't uniform)

**NF4** (NormalFloat4):
- 16 values optimized for normally-distributed data
- More values near 0 (where most weights are)
- Much better quality for the same 4 bits

```
INT4:  |-8    -4    0    4    8|  (uniform spacing)
NF4:   ||||||  |    |  |||||||   (clustered near zero)
```

This is why bitsandbytes uses NF4 by default.

---

### Q: What is calibration data and why does it matter?
**A**: Calibration data is a small sample (128-512 examples) used during quantization to:

1. Determine activation ranges for each layer
2. Set optimal scale factors
3. Identify which weights are "important"

**Bad calibration** = poor quantization:
- If you calibrate on English but run on code → wrong scale factors
- If you use too few samples → high variance, unstable quality

**Best practice**: Use data similar to your actual use case.

---

### Q: What's the difference between PTQ and QAT?
**A**:

**PTQ (Post-Training Quantization)**:
- Quantize after training is complete
- Fast: minutes to hours
- Slight quality loss
- Methods: GPTQ, AWQ, NVFP4

**QAT (Quantization-Aware Training)**:
- Train with quantization simulation
- Slow: full training run
- Better quality
- Methods: FP8 training, some advanced approaches

For most LLM use cases, **PTQ is sufficient** and much faster.

---

### Q: Why does GGUF have so many quantization levels (Q2, Q3, Q4, Q5, Q6, Q8)?
**A**: GGUF is designed for flexibility across different hardware:

| Level | Use Case |
|-------|----------|
| Q2_K | Extreme memory constraints (not recommended for quality) |
| Q3_K | Very tight memory, acceptable for simple tasks |
| Q4_K_M | **Best balance** for most users |
| Q5_K_M | Quality priority, more memory |
| Q6_K | Near-original quality |
| Q8_0 | Maximum quality in GGUF format |

The "_K" variants use "k-quant" (smarter quantization) and "_M" is medium granularity.

---

## Troubleshooting

### Q: Why is my GPTQ quantization taking hours?
**A**: Common causes:
1. **Using CPU instead of GPU** - Ensure CUDA is available
2. **Too many calibration samples** - 128 is usually enough
3. **Large model without streaming** - Use gradient checkpointing

```python
# Check GPU is being used
import torch
print(torch.cuda.is_available())  # Should be True

# Reduce samples for faster testing
gptq_config = GPTQConfig(bits=4, dataset="c4", num_samples=64)
```

---

### Q: My quantized model gives wrong/gibberish outputs. Why?
**A**: Check these in order:
1. **Tokenizer mismatch** - Load tokenizer from same path as model
2. **Incomplete quantization** - Process may have failed partway
3. **Too aggressive quantization** - Try Q5 instead of Q4, or AWQ instead of GPTQ
4. **Chat template issues** - Quantized model may need same template as original

---

### Q: How do I check if quantization worked correctly?
**A**:
```python
# 1. Check model size
import os
original_size = os.path.getsize("original_model/model.safetensors")
quantized_size = os.path.getsize("quantized_model/model.safetensors")
print(f"Compression: {original_size/quantized_size:.1f}x")

# 2. Check memory usage
print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# 3. Run quick quality test
from lm_eval import evaluator
results = evaluator.simple_evaluate(model, tasks=["wikitext"])
print(f"Perplexity: {results['results']['wikitext']['word_perplexity']}")

# 4. Sanity check generation
output = model.generate(tokenizer("Hello, world!", return_tensors="pt").to("cuda"))
print(tokenizer.decode(output[0]))
```

---

## Beyond the Basics

### Q: Can I quantize any model?
**A**: Most transformer-based models yes, but:
- **Well-supported**: Llama, Mistral, Qwen, Falcon, GPT-NeoX
- **Partial support**: Some architectures need custom handling
- **Check first**: Look for existing quantized versions on HuggingFace

---

### Q: Should I quantize for fine-tuning or just inference?
**A**:
- **For fine-tuning**: Use bitsandbytes (QLoRA). The base model is quantized, adapters are FP16.
- **For inference**: Use GPTQ, AWQ, or NVFP4 for best performance.
- **Don't mix**: A QLoRA model and a GPTQ model are different. Choose based on your use case.

---

### Q: What's the best quantization for production deployment?
**A**: Depends on your deployment target:

| Target | Recommended | Why |
|--------|-------------|-----|
| DGX Spark (TensorRT) | NVFP4 | Hardware-accelerated, best perf |
| vLLM/SGLang | AWQ or GPTQ | Good integration, fast serving |
| Ollama | GGUF Q4_K_M | Universal, easy deployment |
| Cloud API | AWQ | Good balance of quality and speed |

---

### Q: How do I combine quantization with speculative decoding?
**A**: Yes, and it's powerful! In Module 3.3 you'll learn:
- Draft model: Small, FP16 (fast)
- Target model: Large, quantized (memory-efficient)
- Result: 2-3x speedup on already-quantized models

---

## Still Have Questions?

- Check [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for error-specific help
- Review [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) for correct patterns
- See [ELI5.md](./ELI5.md) for concept explanations
- Consult the specific library docs (GPTQ, AWQ, llama.cpp)
