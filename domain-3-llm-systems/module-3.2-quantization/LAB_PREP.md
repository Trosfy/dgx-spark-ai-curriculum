# Module 3.2: Quantization & Optimization - Lab Preparation Guide

## ⏱️ Time Estimates
| Lab | Preparation | Execution | Total |
|-----|-------------|-----------|-------|
| Lab 3.2.1: Data Type Exploration | 10 min | 1.5 hr | 1.7 hr |
| Lab 3.2.2: NVFP4 Quantization ⭐ | 30 min | 3 hr | 3.5 hr |
| Lab 3.2.3: FP8 Training & Inference | 15 min | 2 hr | 2.3 hr |
| Lab 3.2.4: GPTQ Quantization | 15 min | 2 hr | 2.3 hr |
| Lab 3.2.5: AWQ Quantization | 10 min | 1.5 hr | 1.7 hr |
| Lab 3.2.6: GGUF Conversion | 20 min | 2 hr | 2.3 hr |
| Lab 3.2.7: Quality Benchmark Suite | 15 min | 2 hr | 2.3 hr |
| Lab 3.2.8: TensorRT-LLM Engine | 30 min | 2 hr | 2.5 hr |

## 📦 Required Downloads

### Models (Download Before Labs)

```bash
# For quick testing (Labs 3.2.1, 3.2.3-3.2.7)
# Small model for fast iteration
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0

# For 8B quantization (most labs)
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct

# For 70B NVFP4 showcase (Lab 3.2.2) ⭐
huggingface-cli download meta-llama/Llama-3.1-70B-Instruct
```

**Total download size**: ~160GB
**Estimated download time**: 30-60 minutes on fast connection

### Calibration Datasets
```bash
# Automatically downloaded during labs, but can pre-cache:
python -c "from datasets import load_dataset; load_dataset('wikitext', 'wikitext-2-raw-v1')"
python -c "from datasets import load_dataset; load_dataset('c4', 'en', split='train', streaming=True)"
```

### Additional Packages
```bash
# Core quantization libraries
pip install auto-gptq autoawq bitsandbytes

# TensorRT Model Optimizer (for NVFP4)
pip install nvidia-modelopt

# Evaluation
pip install lm-eval

# GGUF conversion (install llama.cpp separately)
pip install gguf sentencepiece
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

### 2. Verify Blackwell Hardware
```python
import torch

def verify_hardware():
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    cc = torch.cuda.get_device_capability()
    print(f"Compute Capability: {cc[0]}.{cc[1]}")

    if cc[0] >= 10:
        print("✅ Blackwell GPU - NVFP4 native support!")
    elif cc[0] >= 8:
        print("⚠️ Ampere/Hopper GPU - FP8 supported, NVFP4 emulated")
    else:
        print("⚠️ Older GPU - limited quantization support")

    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Memory: {mem_gb:.1f} GB")

verify_hardware()
```

**Expected output for DGX Spark**:
```
PyTorch: 2.x.x
CUDA: 12.x
GPU: NVIDIA GH200 480GB
Compute Capability: 10.0
✅ Blackwell GPU - NVFP4 native support!
Memory: 128.0 GB
```

### 3. Clear Memory (For 70B Models)
```bash
# Run BEFORE starting Jupyter for 70B work:
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

### 4. Install llama.cpp (For Lab 3.2.6)
```bash
# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j$(nproc) GGML_CUDA=1  # With CUDA support

# Verify
./llama-quantize --help
```

## ✅ Pre-Lab Checklists

### Lab 3.2.1: Data Type Exploration
- [ ] TinyLlama or 8B model downloaded
- [ ] transformers, torch installed
- [ ] At least 8GB GPU memory free
- [ ] Reviewed: Data type concepts in [ELI5.md](./ELI5.md)

### Lab 3.2.2: NVFP4 Quantization ⭐
- [ ] Llama 3.1 70B downloaded
- [ ] Blackwell GPU verified (CC ≥ 10.0)
- [ ] Buffer cache cleared
- [ ] nvidia-modelopt installed
- [ ] At least 60GB GPU memory free
- [ ] Calibration dataset ready

### Lab 3.2.3: FP8 Training & Inference
- [ ] 8B model downloaded
- [ ] transformer-engine installed (in NGC container)
- [ ] Reviewed: E4M3 vs E5M2 differences

### Lab 3.2.4-3.2.5: GPTQ & AWQ
- [ ] 8B model downloaded
- [ ] auto-gptq installed
- [ ] autoawq installed
- [ ] At least 24GB GPU memory free

### Lab 3.2.6: GGUF Conversion
- [ ] Model to convert ready (fine-tuned or base)
- [ ] llama.cpp built with CUDA
- [ ] gguf Python package installed
- [ ] Ollama installed on host (for testing)

### Lab 3.2.7: Quality Benchmark Suite
- [ ] Multiple quantized models ready
- [ ] lm-eval installed
- [ ] Evaluation datasets cached
- [ ] Baseline FP16 results for comparison

### Lab 3.2.8: TensorRT-LLM Engine
- [ ] TensorRT-LLM container or package installed
- [ ] 8B or 70B model ready
- [ ] At least 40GB GPU memory free
- [ ] 45-90 minutes for engine build time

## 🚫 Common Setup Mistakes

| Mistake | Consequence | Prevention |
|---------|-------------|------------|
| Not checking compute capability | NVFP4 errors | Verify CC ≥ 10.0 for Blackwell |
| Mixing quantization methods | Conflicting configs | Use one method per model |
| Small calibration dataset | Poor quantization quality | Use 128-512 samples minimum |
| Wrong llama.cpp version | GGUF format incompatible | Match version to model architecture |
| Insufficient disk space | Quantization fails mid-process | Need 2x model size free |
| Skipping buffer cache clear | OOM on 70B | Always clear before large models |

## 📁 Expected File Structure
After preparation, your workspace should look like:
```
/workspace/
├── module-3.2-quantization/
│   ├── labs/
│   │   ├── lab-3.2.1-data-type-exploration.ipynb
│   │   ├── lab-3.2.2-nvfp4-quantization.ipynb
│   │   └── ... (8 notebooks)
│   ├── scripts/
│   │   ├── quantization_utils.py
│   │   ├── memory_utils.py
│   │   └── perplexity.py
│   └── outputs/
│       ├── gptq_models/
│       ├── awq_models/
│       └── gguf_models/
├── llama.cpp/  (for GGUF conversion)
│   ├── convert_hf_to_gguf.py
│   └── llama-quantize
└── .cache/
    └── huggingface/
        └── hub/
            └── models--meta-llama--...
```

## ⚡ Quick Start Commands
```bash
# Copy-paste this block to set up everything:
cd /workspace

# Install all quantization packages
pip install auto-gptq autoawq bitsandbytes nvidia-modelopt lm-eval gguf

# Download test model
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Build llama.cpp for GGUF
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j$(nproc) GGML_CUDA=1 && cd ..

# Verify setup
python -c "
import torch
from auto_gptq import AutoGPTQForCausalLM
from awq import AutoAWQForCausalLM
print('✅ PyTorch:', torch.__version__)
print('✅ CUDA:', torch.cuda.is_available())
print('✅ GPTQ: imported')
print('✅ AWQ: imported')
print('✅ Memory:', torch.cuda.get_device_properties(0).total_memory/1e9, 'GB')
cc = torch.cuda.get_device_capability()
print(f'✅ Compute Capability: {cc[0]}.{cc[1]}')
"
```

## 🔄 Quantization Time Estimates

| Model Size | GPTQ | AWQ | GGUF (convert + quantize) |
|------------|------|-----|---------------------------|
| 1B | ~5 min | ~5 min | ~3 min |
| 8B | ~30 min | ~45 min | ~15 min |
| 70B | ~4 hr | ~6 hr | ~2 hr |

**Note**: TensorRT-LLM engine build adds 45-90 minutes depending on model size and optimization level.
