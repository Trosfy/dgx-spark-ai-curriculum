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
