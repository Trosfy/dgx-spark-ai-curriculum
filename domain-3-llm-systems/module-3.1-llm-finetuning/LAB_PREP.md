# Module 3.1: LLM Fine-Tuning - Lab Preparation Guide

## ⏱️ Time Estimates
| Lab | Preparation | Execution | Total |
|-----|-------------|-----------|-------|
| Lab 3.1.1: LoRA Theory | 10 min | 2 hr | 2.2 hr |
| Lab 3.1.2: DoRA Comparison | 5 min | 2 hr | 2.1 hr |
| Lab 3.1.3: NEFTune Magic | 5 min | 1 hr | 1.1 hr |
| Lab 3.1.4: 8B LoRA Fine-tuning | 15 min | 3 hr | 3.3 hr |
| Lab 3.1.5: 70B QLoRA ⭐ | 30 min | 4 hr | 4.5 hr |
| Lab 3.1.6: Dataset Preparation | 10 min | 2 hr | 2.2 hr |
| Lab 3.1.7: DPO Training | 10 min | 2 hr | 2.2 hr |
| Lab 3.1.8: SimPO vs ORPO | 10 min | 2 hr | 2.2 hr |
| Lab 3.1.9: KTO Binary Feedback | 10 min | 2 hr | 2.2 hr |
| Lab 3.1.10: Ollama Integration | 15 min | 2 hr | 2.3 hr |

## 📦 Required Downloads

### Models (Download Before Labs)

```bash
# For quick testing (Lab 3.1.1-3.1.3)
# TinyLlama - 1.1B parameters, ~2GB
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0

# For 8B training (Lab 3.1.4)
# Llama 3.1 8B - ~16GB (requires approval)
# First: Request access at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct

# For 70B training (Lab 3.1.5) ⭐
# Llama 3.1 70B - ~140GB (will be quantized to ~35GB)
huggingface-cli download meta-llama/Llama-3.1-70B-Instruct

# Alternative (no approval required):
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2
```

**Total download size**: ~160GB (for all models)
**Estimated download time**: 30-60 minutes on fast connection

### Datasets
```bash
# Datasets are automatically downloaded during labs
# Pre-download for offline work:
python -c "from datasets import load_dataset; load_dataset('tatsu-lab/alpaca')"
python -c "from datasets import load_dataset; load_dataset('HuggingFaceH4/ultrafeedback_binarized')"
```

### Additional Packages
```bash
# Beyond base NGC container
pip install peft transformers accelerate bitsandbytes
pip install trl datasets
pip install unsloth  # Optional: 2x faster training
pip install psutil   # Memory monitoring
```

## 🔧 Environment Setup

### 1. Start Container
```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

### 2. Verify GPU Access
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print(f"Compute Capability: {torch.cuda.get_device_capability()}")
```

**Expected output**:
```
CUDA available: True
Device: NVIDIA Blackwell GB10 Superchip  # DGX Spark GPU
Memory: 128.0 GB
Compute Capability: (10, 0)  # Blackwell architecture
```

### 3. Clear Memory (Fresh Start)
```python
import torch, gc
torch.cuda.empty_cache()
gc.collect()
```

### 4. Verify HuggingFace Access
```python
from huggingface_hub import HfApi
api = HfApi()
try:
    api.whoami()
    print("✅ HuggingFace login successful")
except:
    print("❌ Run: huggingface-cli login")
```

## ✅ Pre-Lab Checklists

### Lab 3.1.1-3.1.3: LoRA Basics
- [ ] TinyLlama model downloaded
- [ ] peft, transformers installed
- [ ] At least 8GB GPU memory free
- [ ] Reviewed: LoRA concept in [ELI5.md](./ELI5.md)

### Lab 3.1.4: 8B LoRA Fine-tuning
- [ ] Llama 3.1 8B downloaded (or Mistral-7B)
- [ ] Completed Labs 3.1.1-3.1.3
- [ ] At least 20GB GPU memory free
- [ ] trl, datasets installed
- [ ] Reviewed: QLoRA config in [QUICK_REFERENCE.md](./QUICK_REFERENCE.md)

### Lab 3.1.5: 70B QLoRA ⭐
- [ ] Llama 3.1 70B downloaded
- [ ] Completed Lab 3.1.4 successfully
- [ ] Buffer cache cleared (see command below)
- [ ] At least 60GB GPU memory free
- [ ] bitsandbytes installed and working
- [ ] Reviewed: Memory management tips

**Critical for 70B:**
```bash
# Run BEFORE starting Jupyter:
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

### Lab 3.1.6: Dataset Preparation
- [ ] Completed at least one training lab
- [ ] Sample data files in `data/` directory
- [ ] datasets library installed

### Lab 3.1.7-3.1.9: Preference Optimization
- [ ] Completed Lab 3.1.6 (dataset prep)
- [ ] trl >= 0.8.0 installed (for SimPO)
- [ ] Preference dataset prepared
- [ ] At least 30GB GPU memory free

### Lab 3.1.10: Ollama Integration
- [ ] Completed at least one training lab with saved adapter
- [ ] Ollama installed on host system
- [ ] llama.cpp dependencies available
- [ ] Docker with `--network=host` or proper port mapping

## 🚫 Common Setup Mistakes

| Mistake | Consequence | Prevention |
|---------|-------------|------------|
| Not clearing buffer cache | OOM when loading 70B | Run `sync; echo 3 > /proc/sys/vm/drop_caches` |
| Old bitsandbytes version | Quantization errors | `pip install bitsandbytes>=0.41.0` |
| Missing HF login | Model download fails | `huggingface-cli login` before starting |
| No Meta access request | Llama models 403 error | Request at huggingface.co/meta-llama first |
| Wrong container flags | GPU not visible | Include `--gpus all --ipc=host` |
| Insufficient disk space | Download fails | Need ~200GB free for all models |

## 📁 Expected File Structure
After preparation, your workspace should look like:
```
/workspace/
├── module-3.1-llm-finetuning/
│   ├── labs/
│   │   ├── lab-3.1.1-lora-theory.ipynb
│   │   ├── lab-3.1.2-dora-comparison.ipynb
│   │   └── ... (10 notebooks)
│   ├── data/
│   │   ├── sample_instruction_data.json
│   │   ├── sample_preference_data.json
│   │   └── sample_conversation_data.json
│   ├── scripts/
│   │   ├── dataset_utils.py
│   │   └── lora_utils.py
│   ├── configs/
│   │   └── (your saved configs)
│   └── outputs/
│       └── (created during labs)
└── .cache/
    └── huggingface/
        └── hub/
            └── models--meta-llama--Llama-3.1-...
```

## ⚡ Quick Start Commands
```bash
# Copy-paste this block to set up everything:
cd /workspace

# Install dependencies
pip install peft transformers accelerate bitsandbytes trl datasets

# Login to HuggingFace
huggingface-cli login

# Download small model for testing
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Verify setup
python -c "
import torch
from peft import LoraConfig
print('✅ PyTorch:', torch.__version__)
print('✅ CUDA:', torch.cuda.is_available())
print('✅ PEFT: imported successfully')
print('✅ Memory:', torch.cuda.get_device_properties(0).total_memory/1e9, 'GB')
"
```

## 🔄 Memory Recovery Script
Save this for when you run out of memory:

```python
# memory_recovery.py
import torch
import gc
import subprocess

def full_memory_clear():
    """Aggressive memory clearing for DGX Spark."""
    # Clear Python garbage
    gc.collect()

    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Force garbage collection again
    gc.collect()

    # Report status
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

    return True

if __name__ == "__main__":
    full_memory_clear()
    print("Memory cleared. For 70B models, also run:")
    print("sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'")
```
