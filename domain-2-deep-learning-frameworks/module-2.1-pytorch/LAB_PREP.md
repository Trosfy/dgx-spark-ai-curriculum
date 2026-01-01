# Module 2.1: Deep Learning with PyTorch - Lab Preparation Guide

## Time Estimates

| Lab | Preparation | Execution | Total |
|-----|-------------|-----------|-------|
| 2.1.1 Custom Module | 10 min | 2 hr | ~2 hr |
| 2.1.2 Dataset Pipeline | 15 min | 2 hr | ~2 hr |
| 2.1.3 Autograd Deep Dive | 5 min | 2 hr | ~2 hr |
| 2.1.4 Mixed Precision | 5 min | 2 hr | ~2 hr |
| 2.1.5 Profiling Workshop | 10 min | 2 hr | ~2 hr |
| 2.1.6 Checkpointing | 5 min | 2 hr | ~2 hr |

**Total**: ~12-15 hours

---

## Required Downloads

### Datasets (Auto-download in Lab)

CIFAR-10 and MNIST will download automatically during labs:

```python
# These download automatically when first accessed
from torchvision.datasets import CIFAR10, MNIST

# First run will download ~170MB (CIFAR-10) + ~50MB (MNIST)
train_dataset = CIFAR10('./data', train=True, download=True)
```

**Total download size**: ~220 MB
**Note**: Downloads happen inside the lab notebooks - no pre-download needed

### Pre-trained Weights (Optional)

For comparison in Lab 2.1.1:
```python
# These download automatically when first used
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
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

**Critical flags**:
- `--gpus all`: Required for GPU access
- `--ipc=host`: **Required** for DataLoader with `num_workers > 0`
- `-p 8888:8888`: Expose Jupyter port

### 2. Verify GPU Access

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
```

**Expected output**:
```
PyTorch version: 2.x.x
CUDA available: True
Device: NVIDIA Graphics Device  # DGX Spark: Blackwell GB10 Superchip
Memory: 128.0 GB
```

### 3. Verify BF16 Support

```python
# Should work without errors on DGX Spark
x = torch.randn(100, 100, dtype=torch.bfloat16, device='cuda')
y = torch.matmul(x, x.T)
print(f"BF16 matmul works: {y.dtype}")
```

**Expected**: `BF16 matmul works: torch.bfloat16`

### 4. Clear Memory (Fresh Start)

```python
import torch
import gc

torch.cuda.empty_cache()
gc.collect()

print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

---

## Pre-Lab Checklist

### Lab 2.1.1: Custom Module Lab
- [ ] Container running with GPU access
- [ ] Understand `nn.Module` basics (from PREREQUISITES.md)
- [ ] Review ResNet architecture (skip connections)

### Lab 2.1.2: Dataset Pipeline
- [ ] Container running with `--ipc=host` flag
- [ ] CIFAR-10 downloaded (or will download in lab)
- [ ] Understand Python generators and `__getitem__`

### Lab 2.1.3: Autograd Deep Dive
- [ ] Completed MicroGrad capstone (Module 1.7)
- [ ] Understand chain rule for derivatives
- [ ] Know what `requires_grad=True` means

### Lab 2.1.4: Mixed Precision Training
- [ ] Understand why lower precision saves memory
- [ ] Know difference between FP32, FP16, BF16
- [ ] Completed Lab 2.1.1 (have a model to train)

### Lab 2.1.5: Profiling Workshop
- [ ] Have trained a model (Labs 2.1.1-2.1.4)
- [ ] Know how to read flame graphs (helpful)
- [ ] Understand CPU vs GPU execution

### Lab 2.1.6: Checkpointing System
- [ ] Have a training loop from previous labs
- [ ] Understand Python file I/O basics
- [ ] Know what `state_dict` contains

---

## Common Setup Mistakes

| Mistake | Consequence | Prevention |
|---------|-------------|------------|
| Missing `--ipc=host` | DataLoader crashes with shared memory error | Always include this flag |
| Wrong port mapping | Can't access Jupyter | Use `-p 8888:8888` |
| Not mounting workspace | Work lost when container exits | Use `-v $HOME/workspace:/workspace` |
| Forgetting `--gpus all` | No GPU access | Always include for DGX Spark |

---

## Expected File Structure

After setup, your workspace should look like:

```
/workspace/
├── domain-2-deep-learning-frameworks/
│   └── module-2.1-pytorch/
│       ├── README.md
│       ├── QUICKSTART.md
│       ├── PREREQUISITES.md
│       ├── STUDY_GUIDE.md
│       ├── QUICK_REFERENCE.md
│       ├── LAB_PREP.md           # This file
│       ├── TROUBLESHOOTING.md
│       ├── labs/
│       │   ├── lab-2.1.1-custom-module-lab.ipynb
│       │   ├── lab-2.1.2-dataset-pipeline.ipynb
│       │   ├── lab-2.1.3-autograd-deep-dive.ipynb
│       │   ├── lab-2.1.4-mixed-precision-training.ipynb
│       │   ├── lab-2.1.5-profiling-workshop.ipynb
│       │   └── lab-2.1.6-checkpointing-system.ipynb
│       ├── scripts/
│       │   ├── __init__.py
│       │   ├── resnet_blocks.py
│       │   └── ...
│       ├── solutions/
│       └── data/                  # Created when datasets download
│           ├── cifar-10-batches-py/
│           └── MNIST/
```

---

## Quick Start Commands

Copy-paste this block to verify your setup:

```bash
# Inside the NGC container
cd /workspace

# Clone curriculum if not already present
# git clone <curriculum-repo> domain-2-deep-learning-frameworks

# Navigate to module
cd domain-2-deep-learning-frameworks/module-2.1-pytorch

# Verify Python environment
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
print(f'BF16: {torch.cuda.is_bf16_supported()}')
print('Setup complete!')
"

# Start Jupyter if not already running
# jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

---

## Resource Requirements by Lab

| Lab | GPU Memory | CPU | Disk |
|-----|------------|-----|------|
| 2.1.1 | ~4 GB | Low | ~200 MB |
| 2.1.2 | ~2 GB | High (data loading) | ~200 MB |
| 2.1.3 | ~1 GB | Low | Minimal |
| 2.1.4 | ~8 GB (comparing FP32 vs BF16) | Low | Minimal |
| 2.1.5 | ~4 GB | Medium | ~50 MB (traces) |
| 2.1.6 | ~4 GB | Low | ~100 MB (checkpoints) |

All labs fit comfortably within DGX Spark's 128GB unified memory.
