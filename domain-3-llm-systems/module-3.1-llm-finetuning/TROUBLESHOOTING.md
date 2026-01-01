# Module 3.1: LLM Fine-Tuning - Troubleshooting Guide

## 🔍 Quick Diagnostic

**Before diving into specific errors, try these:**
1. Check GPU memory: `nvidia-smi` or `torch.cuda.memory_summary()`
2. Clear cache: `torch.cuda.empty_cache(); gc.collect()`
3. Restart kernel/container
4. Check you're in correct directory
5. Verify HuggingFace login: `huggingface-cli whoami`

---

## 🚨 Error Categories

### Memory Errors

#### Error: `CUDA out of memory`
**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate X GiB
```

**Causes**:
1. Model too large for available memory
2. Previous model still loaded
3. Batch size too high
4. Buffer cache not cleared (especially for 70B)

**Solutions**:
```python
# Solution 1: Clear memory thoroughly
import torch, gc

# Delete existing model references
if 'model' in dir():
    del model
if 'tokenizer' in dir():
    del tokenizer

torch.cuda.empty_cache()
gc.collect()

# Verify memory freed
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

```bash
# Solution 2: Clear buffer cache (for 70B models)
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

```python
# Solution 3: Reduce batch size + gradient accumulation
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # Minimum
    gradient_accumulation_steps=16,  # Simulate larger batch
    gradient_checkpointing=True,     # Trade compute for memory
)
```

```python
# Solution 4: Use more aggressive quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True  # Extra memory savings
)
```

**Prevention**: Always clear memory before loading new models.

---

#### Error: `OutOfMemoryError` when loading 70B
**Symptoms**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 35.00 GiB
```

**Cause**: DGX Spark's unified memory needs buffer cache cleared first.

**Solution**:
```bash
# BEFORE starting Python/Jupyter:
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# Then in Python:
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    max_memory={0: "120GiB"}  # Leave headroom
)
```

---

### Import/Dependency Errors

#### Error: `ModuleNotFoundError: No module named 'peft'`
**Solution**:
```bash
pip install peft>=0.10.0
```

#### Error: `ModuleNotFoundError: No module named 'bitsandbytes'`
**Solution**:
```bash
# Inside NGC container (ARM64)
pip install bitsandbytes>=0.41.0
```

#### Error: `ImportError: cannot import name 'SimPOTrainer' from 'trl'`
**Cause**: Old TRL version doesn't have SimPO.

**Solution**:
```bash
pip install trl>=0.8.0
```

#### Error: `bitsandbytes CUDA error`
**Symptoms**:
```
CUDA Setup failed despite GPU being available
```

**Solution**:
```bash
# Check CUDA version matches
python -c "import torch; print(torch.version.cuda)"

# Reinstall bitsandbytes for your CUDA version
pip uninstall bitsandbytes
pip install bitsandbytes --no-cache-dir
```

---

### Model Loading Errors

#### Error: `OSError: meta-llama/Llama-3.1-8B-Instruct is not a local folder`
**Cause**: Need to accept license on HuggingFace.

**Solution**:
1. Go to https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. Click "Agree and access repository"
3. Login: `huggingface-cli login`
4. Retry the download

---

#### Error: `ValueError: Tokenizer class LlamaTokenizer does not exist`
**Cause**: Using specific tokenizer class instead of Auto class.

**Solution**:
```python
# Wrong:
# from transformers import LlamaTokenizer
# tokenizer = LlamaTokenizer.from_pretrained(...)

# Correct:
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
```

---

#### Error: `KeyError: 'model.embed_tokens.weight'`
**Cause**: Trying to load LoRA adapter without base model.

**Solution**:
```python
from peft import PeftModel, PeftConfig

# First load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Then load adapter
model = PeftModel.from_pretrained(base_model, "./my_lora_adapter")
```

---

### Training Errors

#### Error: `ValueError: Unable to create tensor, you should probably activate truncation`
**Cause**: Input sequences longer than model's max length.

**Solution**:
```python
tokenizer.model_max_length = 2048  # Or your desired length

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,        # Enable truncation
        max_length=2048,        # Set maximum
        padding="max_length"
    )
```

---

#### Error: `RuntimeError: element 0 of tensors does not require grad`
**Cause**: Trying to compute gradients on frozen parameters.

**Solution**:
```python
# Make sure LoRA is applied correctly
from peft import get_peft_model, LoraConfig

model = get_peft_model(model, lora_config)

# Verify trainable params exist
model.print_trainable_parameters()
# Should show: trainable params: X || all params: Y || trainable%: Z
```

---

#### Error: `AssertionError: No pad token` or similar
**Cause**: Model tokenizer doesn't have a pad token defined.

**Solution**:
```python
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# For training, also update model config
model.config.pad_token_id = tokenizer.pad_token_id
```

---

#### Error: Loss is `nan` or training diverges
**Causes**:
1. Learning rate too high
2. Gradient explosion
3. Bad data (empty sequences, wrong format)

**Solutions**:
```python
# Solution 1: Lower learning rate
training_args = TrainingArguments(
    learning_rate=1e-5,  # Start low, increase if stable
    warmup_ratio=0.1,    # Warm up learning rate
)

# Solution 2: Enable gradient clipping
training_args = TrainingArguments(
    max_grad_norm=1.0,  # Clip gradients
)

# Solution 3: Check data
for i, example in enumerate(dataset[:5]):
    tokens = tokenizer(example["text"])
    print(f"Example {i}: {len(tokens['input_ids'])} tokens")
    if len(tokens['input_ids']) == 0:
        print("WARNING: Empty sequence!")
```

---

### GGUF Conversion Errors

#### Error: `llama.cpp convert script not found`
**Solution**:
```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build
make -j$(nproc)

# Convert (after merging LoRA)
python convert_hf_to_gguf.py ../merged_model --outfile model.gguf
```

---

#### Error: Ollama won't load converted model
**Causes**:
1. Incomplete conversion
2. Wrong quantization format
3. Missing metadata

**Solution**:
```bash
# Use Ollama's built-in conversion when possible
ollama create my-model -f Modelfile

# Modelfile contents:
FROM ./model.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.9
```

---

## 🔄 Reset Procedures

### Full Environment Reset
```bash
# 1. Exit container
exit

# 2. Clear HuggingFace cache (if needed)
rm -rf ~/.cache/huggingface/hub/models--meta-llama--*

# 3. Clear buffer cache
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# 4. Restart container
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    nvcr.io/nvidia/pytorch:25.11-py3
```

### Memory-Only Reset
```python
import torch
import gc

def full_reset():
    # Clear all variables that might hold tensors
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            obj.data = torch.empty(0)

    # Force garbage collection
    gc.collect()

    # Clear CUDA cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Report
    print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")

full_reset()
```

### Restart Jupyter Kernel
In Jupyter:
- Kernel → Restart Kernel
- Or: Press `0` twice in command mode

---

## ❓ Frequently Asked Questions

### Setup & Environment

**Q: Why can't I just pip install PyTorch? Why use the NGC container?**

**A**: The NGC container includes ARM64-optimized builds of PyTorch, CUDA libraries, and bitsandbytes that work correctly on DGX Spark's Blackwell architecture. Installing from pip may get you x86 builds that won't work or won't have full GPU support.

**See also**: [LAB_PREP.md](./LAB_PREP.md) for container setup

---

**Q: How do I get access to Llama models?**

**A**:
1. Create a HuggingFace account
2. Go to https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
3. Click "Agree and access repository"
4. Run `huggingface-cli login` and paste your token
5. Wait ~15 minutes for access to propagate

**Alternative**: Use `mistralai/Mistral-7B-Instruct-v0.2` which requires no approval.

---

**Q: How much disk space do I need?**

**A**:
- TinyLlama (testing): ~2GB
- Llama 3.1 8B: ~16GB
- Llama 3.1 70B: ~140GB (downloads full, uses ~35GB when quantized)
- Total recommended: **200GB free** for comfortable work

---

### Concepts

**Q: What's the difference between LoRA and QLoRA?**

**A**:
| Aspect | LoRA | QLoRA |
|--------|------|-------|
| Base model | Full precision (FP16/BF16) | 4-bit quantized |
| Memory for 70B | ~140GB (won't fit) | ~45GB (fits on Spark) |
| Training speed | Faster | Slightly slower |
| Quality | Baseline | Very close to LoRA |

Use QLoRA when model won't fit in memory. For 8B models on DGX Spark, regular LoRA is fine.

---

**Q: What rank (r) should I use for LoRA?**

**A**:
- **r=8**: Quick experiments, simple tasks
- **r=16**: Default for most tasks (start here)
- **r=32**: Complex tasks, domain-specific knowledge
- **r=64**: Maximum quality, approaching full fine-tuning

Higher rank = more trainable parameters = better adaptation but more memory.

---

**Q: Why does NEFTune work? Adding noise sounds counterproductive.**

**A**: NEFTune adds small random noise to embeddings during training only. This forces the model to be robust to input variations, similar to data augmentation in vision. The model learns to focus on the essential meaning rather than overfitting to exact input patterns. At inference time, no noise is added.

**See also**: [ELI5.md](./ELI5.md) for the coffee shop analogy

---

**Q: When should I use DPO vs SimPO vs ORPO vs KTO?**

**A**:

| Method | Best For |
|--------|----------|
| **DPO** | Proven baseline, well-understood behavior |
| **SimPO** | Best quality (+6.4 pts), simpler (no reference model) |
| **ORPO** | Memory-constrained situations (50% less memory) |
| **KTO** | When you only have thumbs up/down, not preference pairs |

**Default recommendation**: Start with SimPO unless you have constraints.

---

**Q: Do I need preference pairs for preference optimization?**

**A**: For DPO, SimPO, and ORPO: **Yes**, you need pairs of (prompt, chosen_response, rejected_response).

For KTO: **No**, you only need (prompt, response, is_good) where is_good is True/False.

---

### Common Issues

**Q: I'm getting OOM on 70B. What's wrong?**

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

---

**Q: My training loss is NaN. What happened?**

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

**Q: How long should training take?**

**A**: Rough estimates on DGX Spark:

| Model | Dataset Size | Time |
|-------|--------------|------|
| 8B (LoRA) | 1K examples | ~10 min |
| 8B (LoRA) | 10K examples | ~1.5 hr |
| 70B (QLoRA) | 1K examples | ~30 min |
| 70B (QLoRA) | 10K examples | ~4 hr |

Actual time depends on sequence length, batch size, and number of epochs.

---

**Q: My fine-tuned model gives weird/repetitive outputs.**

**A**: Possible issues:
1. **Overfitting** - Use fewer epochs or more diverse data
2. **Dataset format wrong** - Check chat template matches model
3. **Temperature too low** - Try temperature=0.7 for generation

---

### Beyond the Basics

**Q: Can I fine-tune for my production use case?**

**A**: Yes, but consider:
1. **Data quality** > quantity - Clean, diverse data matters most
2. **Evaluation** - Create a test set before training to measure improvement
3. **Deployment** - Plan for Ollama/vLLM deployment (Module 3.3)
4. **Monitoring** - Track quality over time

---

**Q: Should I use full fine-tuning instead of LoRA?**

**A**: Generally no. LoRA achieves similar quality with:
- 10-100x fewer trainable parameters
- Much lower memory requirements
- Ability to store multiple task adapters
- Less risk of catastrophic forgetting

Only consider full fine-tuning for fundamental behavior changes with massive datasets.

---

**Q: How do I deploy my fine-tuned model?**

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

**Q: What about training on multiple GPUs?**

**A**: DGX Spark has a single Blackwell GPU with 128GB unified memory—enough for most tasks. Multi-GPU training patterns (FSDP, DeepSpeed) aren't needed here. The focus is on efficient single-GPU techniques like QLoRA.

---

## 📞 Still Stuck?

1. **Check the lab notebook comments** - Often contain hints
2. **Review prerequisites** - See [PREREQUISITES.md](./PREREQUISITES.md)
3. **Check QUICK_REFERENCE.md** - Correct code patterns
4. **Search the error message** - Include library name (e.g., "peft ValueError tensor")
5. **Ask with context** - Include:
   - Full error traceback
   - Code that caused the error
   - What you've already tried
   - GPU memory status (`nvidia-smi`)
