# Module 2.4: Efficient Architectures - Lab Preparation Guide

## Time Estimates

| Lab | Preparation | Execution | Total |
|-----|-------------|-----------|-------|
| 2.4.1 Mamba Inference | 15 min | 2 hr | ~2.5 hr |
| 2.4.2 Mamba Architecture | 5 min | 2 hr | ~2 hr |
| 2.4.3 MoE Exploration | 20 min | 2 hr | ~2.5 hr |
| 2.4.4 MoE Router | 5 min | 2 hr | ~2 hr |
| 2.4.5 Architecture Comparison | 10 min | 2 hr | ~2 hr |
| 2.4.6 Mamba Fine-tuning | 10 min | 2 hr | ~2 hr |

**Total**: ~10-12 hours

---

## Required Downloads

### Models (Auto-download)

```python
# Mamba-2.8B (~5.5 GB)
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-2.8b-hf")

# Mixtral-8x7B (~90 GB in BF16) - optional, use DeepSeekMoE for lighter option
# Or smaller MoE:
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-moe-16b-base")

# For comparison transformer:
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
```

### Packages

```bash
# Ensure latest transformers (Mamba support)
pip install transformers>=4.46.0 datasets accelerate
```

---

## Environment Setup

### 1. Start NGC Container

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

### 2. Upgrade Transformers

```bash
# Inside container
pip install transformers>=4.46.0 --upgrade
```

### 3. Verify Mamba Support

```python
from transformers import AutoModelForCausalLM
import torch

# Quick test with smaller model
print("Testing Mamba loading...")
model = AutoModelForCausalLM.from_pretrained(
    "state-spaces/mamba-130m-hf",  # Smaller version for testing
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print(f"Mamba loaded: {sum(p.numel() for p in model.parameters()/1e6):.0f}M params")
print("Mamba support verified!")
```

---

## Pre-Lab Checklist

### Lab 2.4.1: Mamba Inference
- [ ] transformers >= 4.46.0 installed
- [ ] Mamba-2.8B accessible (or use 130M for testing)
- [ ] Comparison transformer model ready

### Lab 2.4.2: Mamba Architecture
- [ ] Completed Lab 2.4.1
- [ ] Understand RNN/recurrent concepts
- [ ] matplotlib installed for visualization

### Lab 2.4.3: MoE Exploration
- [ ] DeepSeekMoE or Mixtral accessible
- [ ] Understand expert concept from ELI5
- [ ] 50+ GB free memory (or use 8-bit quantization)

### Lab 2.4.4: MoE Router
- [ ] Completed Lab 2.4.3
- [ ] Understand softmax and top-k selection
- [ ] Can access model internals with hooks

### Lab 2.4.5: Architecture Comparison
- [ ] Multiple models downloaded
- [ ] benchmarking utilities ready
- [ ] Understand metrics (tok/s, memory, perplexity)

### Lab 2.4.6: Mamba Fine-tuning
- [ ] PEFT library installed (`pip install peft`)
- [ ] Training dataset ready
- [ ] Understand LoRA from Module 2.5 concepts

---

## Model Memory Requirements

| Model | BF16 Memory | INT8 Memory | Notes |
|-------|------------|-------------|-------|
| Mamba-130M | ~0.3 GB | N/A | Testing only |
| Mamba-2.8B | ~5.5 GB | ~3 GB | Main Mamba model |
| Phi-2 (2.7B) | ~5.5 GB | ~3 GB | Comparison transformer |
| DeepSeekMoE-16B | ~32 GB | ~16 GB | Lighter MoE option |
| Mixtral-8x7B | ~90 GB | ~45 GB | Full MoE experience |
| Jamba-v0.1 | ~104 GB | ~52 GB | Hybrid architecture |

DGX Spark's 128GB easily handles all models in BF16.

---

## Resource Requirements by Lab

| Lab | GPU Memory | Models Needed | Notes |
|-----|------------|---------------|-------|
| 2.4.1 | ~12 GB | Mamba + Phi-2 | Side-by-side comparison |
| 2.4.2 | ~6 GB | Mamba only | Architecture study |
| 2.4.3 | ~35 GB | DeepSeekMoE | Or Mixtral with 8-bit |
| 2.4.4 | ~35 GB | Same as 2.4.3 | Router analysis |
| 2.4.5 | ~50 GB | Multiple models | Comprehensive comparison |
| 2.4.6 | ~12 GB | Mamba + LoRA | Fine-tuning overhead |

---

## Quick Start Commands

```bash
# Inside NGC container
cd /workspace/domain-2-deep-learning-frameworks/module-2.4-efficient-architectures

# Upgrade transformers and install dependencies
pip install transformers>=4.46.0 peft datasets accelerate --upgrade

# Pre-download models (optional but recommended)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print('Downloading Mamba-2.8B...')
AutoModelForCausalLM.from_pretrained(
    'state-spaces/mamba-2.8b-hf',
    torch_dtype=torch.bfloat16
)
AutoTokenizer.from_pretrained('state-spaces/mamba-2.8b-hf')

print('Downloading Phi-2...')
AutoModelForCausalLM.from_pretrained(
    'microsoft/phi-2',
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
AutoTokenizer.from_pretrained('microsoft/phi-2', trust_remote_code=True)

print('All downloads complete!')
"
```

---

## Expected File Structure

```
/workspace/
├── domain-2-deep-learning-frameworks/
│   └── module-2.4-efficient-architectures/
│       ├── README.md
│       ├── QUICKSTART.md
│       ├── STUDY_GUIDE.md
│       ├── QUICK_REFERENCE.md
│       ├── ELI5.md
│       ├── LAB_PREP.md
│       ├── TROUBLESHOOTING.md
│       ├── labs/
│       │   ├── lab-2.4.1-mamba-inference.ipynb
│       │   ├── lab-2.4.2-mamba-architecture.ipynb
│       │   ├── lab-2.4.3-moe-exploration.ipynb
│       │   ├── lab-2.4.4-moe-router.ipynb
│       │   ├── lab-2.4.5-architecture-comparison.ipynb
│       │   └── lab-2.4.6-mamba-finetuning.ipynb
│       ├── scripts/
│       └── solutions/
```

---

## Common Setup Issues

| Issue | Solution |
|-------|----------|
| `Mamba not in transformers` | `pip install transformers>=4.46.0` |
| Mixtral OOM | Use `load_in_8bit=True` or use DeepSeekMoE |
| Slow model download | Use HF cache: `-v $HOME/.cache/huggingface:/root/.cache/huggingface` |
| CUDA OOM during comparison | Run models sequentially, not simultaneously |

---

## DGX Spark Advantages for This Module

| Feature | Benefit |
|---------|---------|
| 128GB Memory | Load full Mixtral without quantization |
| 128GB Memory | Run Jamba hybrid with 256K context |
| Fast Memory | Handle Mamba's sequential state updates efficiently |
| BF16 Native | Train/fine-tune at full precision |
