# Module 3.1: LLM Fine-Tuning - Frequently Asked Questions

## Setup & Environment

### Q: Why can't I just pip install PyTorch? Why use the NGC container?
**A**: The NGC container includes ARM64-optimized builds of PyTorch, CUDA libraries, and bitsandbytes that work correctly on DGX Spark's Blackwell architecture. Installing from pip may get you x86 builds that won't work or won't have full GPU support.

**See also**: [LAB_PREP.md](./LAB_PREP.md) for container setup

---

### Q: How do I get access to Llama models?
**A**:
1. Create a HuggingFace account
2. Go to https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
3. Click "Agree and access repository"
4. Run `huggingface-cli login` and paste your token
5. Wait ~15 minutes for access to propagate

**Alternative**: Use `mistralai/Mistral-7B-Instruct-v0.2` which requires no approval.

---

### Q: How much disk space do I need?
**A**:
- TinyLlama (testing): ~2GB
- Llama 3.1 8B: ~16GB
- Llama 3.1 70B: ~140GB (downloads full, uses ~35GB when quantized)
- Total recommended: **200GB free** for comfortable work

---

## Concepts

### Q: What's the difference between LoRA and QLoRA?
**A**:
| Aspect | LoRA | QLoRA |
|--------|------|-------|
| Base model | Full precision (FP16/BF16) | 4-bit quantized |
| Memory for 70B | ~140GB (won't fit) | ~45GB (fits on Spark) |
| Training speed | Faster | Slightly slower |
| Quality | Baseline | Very close to LoRA |

Use QLoRA when model won't fit in memory. For 8B models on DGX Spark, regular LoRA is fine.

---

### Q: What rank (r) should I use for LoRA?
**A**:
- **r=8**: Quick experiments, simple tasks
- **r=16**: Default for most tasks (start here)
- **r=32**: Complex tasks, domain-specific knowledge
- **r=64**: Maximum quality, approaching full fine-tuning

Higher rank = more trainable parameters = better adaptation but more memory.

---

### Q: Why does NEFTune work? Adding noise sounds counterproductive.
**A**: NEFTune adds small random noise to embeddings during training only. This forces the model to be robust to input variations, similar to data augmentation in vision. The model learns to focus on the essential meaning rather than overfitting to exact input patterns. At inference time, no noise is added.

**See also**: [ELI5.md](./ELI5.md) for the coffee shop analogy

---

### Q: When should I use DPO vs SimPO vs ORPO vs KTO?
**A**:

| Method | Best For |
|--------|----------|
| **DPO** | Proven baseline, well-understood behavior |
| **SimPO** | Best quality (+6.4 pts), simpler (no reference model) |
| **ORPO** | Memory-constrained situations (50% less memory) |
| **KTO** | When you only have thumbs up/down, not preference pairs |

**Default recommendation**: Start with SimPO unless you have constraints.

---

### Q: Do I need preference pairs for preference optimization?
**A**: For DPO, SimPO, and ORPO: **Yes**, you need pairs of (prompt, chosen_response, rejected_response).

For KTO: **No**, you only need (prompt, response, is_good) where is_good is True/False.

---

## Troubleshooting

### Q: I'm getting OOM on 70B. What's wrong?
**A**: Most likely the buffer cache isn't cleared. Run this **before** starting Jupyter:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

Also ensure you're using QLoRA with 4-bit quantization:
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
```

**See also**: [TROUBLESHOOTING.md](./TROUBLESHOOTING.md#error-outofmemoryerror-when-loading-70b)

---

### Q: My training loss is NaN. What happened?
**A**: Common causes:
1. **Learning rate too high** - Try 1e-5 or lower
2. **Empty sequences in data** - Check for truncation issues
3. **Gradient explosion** - Enable `max_grad_norm=1.0`

```python
# Safe starting configuration
training_args = TrainingArguments(
    learning_rate=1e-5,
    warmup_ratio=0.1,
    max_grad_norm=1.0,
)
```

---

### Q: How long should training take?
**A**: Rough estimates on DGX Spark:

| Model | Dataset Size | Time |
|-------|--------------|------|
| 8B (LoRA) | 1K examples | ~10 min |
| 8B (LoRA) | 10K examples | ~1.5 hr |
| 70B (QLoRA) | 1K examples | ~30 min |
| 70B (QLoRA) | 10K examples | ~4 hr |

Actual time depends on sequence length, batch size, and number of epochs.

---

### Q: My fine-tuned model gives weird/repetitive outputs.
**A**: Possible issues:
1. **Overfitting** - Use fewer epochs or more diverse data
2. **Dataset format wrong** - Check chat template matches model
3. **Temperature too low** - Try temperature=0.7 for generation

---

## Beyond the Basics

### Q: Can I fine-tune for my production use case?
**A**: Yes, but consider:
1. **Data quality** > quantity - Clean, diverse data matters most
2. **Evaluation** - Create a test set before training to measure improvement
3. **Deployment** - Plan for Ollama/vLLM deployment (Module 3.3)
4. **Monitoring** - Track quality over time

---

### Q: Should I use full fine-tuning instead of LoRA?
**A**: Generally no. LoRA achieves similar quality with:
- 10-100x fewer trainable parameters
- Much lower memory requirements
- Ability to store multiple task adapters
- Less risk of catastrophic forgetting

Only consider full fine-tuning for fundamental behavior changes with massive datasets.

---

### Q: How do I deploy my fine-tuned model?
**A**:
1. **Merge LoRA weights** with base model
2. **Convert to GGUF** for Ollama
3. **Import to Ollama** and test

See Lab 3.1.10 for the complete workflow.

```python
# Quick merge
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(model, adapter_path)
merged = model.merge_and_unload()
merged.save_pretrained("./merged_model")
```

---

### Q: What about training on multiple GPUs?
**A**: DGX Spark has a single Blackwell GPU with 128GB unified memory—enough for most tasks. Multi-GPU training patterns (FSDP, DeepSpeed) aren't needed here. The focus is on efficient single-GPU techniques like QLoRA.

---

## Still Have Questions?

- Check [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for error-specific help
- Review [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) for correct code patterns
- See [ELI5.md](./ELI5.md) for concept explanations
- Consult the [TRL Documentation](https://huggingface.co/docs/trl) for library-specific questions
