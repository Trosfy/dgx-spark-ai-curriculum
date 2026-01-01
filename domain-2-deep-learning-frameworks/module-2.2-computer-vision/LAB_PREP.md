# Module 2.2: Computer Vision - Lab Preparation Guide

## Time Estimates

| Lab | Preparation | Execution | Total |
|-----|-------------|-----------|-------|
| 2.2.1 CNN Architecture Study | 15 min | 3 hr | ~3 hr |
| 2.2.2 Transfer Learning | 15 min | 3 hr | ~3 hr |
| 2.2.3 Object Detection | 20 min | 2 hr | ~2.5 hr |
| 2.2.4 Segmentation | 15 min | 3 hr | ~3 hr |
| 2.2.5 Vision Transformer | 10 min | 3 hr | ~3 hr |
| 2.2.6 SAM Integration | 20 min | 2 hr | ~2.5 hr |

**Total**: ~14-16 hours

---

## Required Downloads

### Datasets (Auto-download in Labs)

```python
# CIFAR-10 (Labs 2.2.1, 2.2.5) - ~170 MB
from torchvision.datasets import CIFAR10
dataset = CIFAR10('./data', download=True)

# CIFAR-100 (Lab 2.2.2) - ~170 MB
from torchvision.datasets import CIFAR100
dataset = CIFAR100('./data', download=True)

# VOC Segmentation (Lab 2.2.4) - ~2 GB
from torchvision.datasets import VOCSegmentation
dataset = VOCSegmentation('./data', download=True)
```

### Pre-trained Models

```bash
# YOLOv8 models (Lab 2.2.3) - auto-download
pip install ultralytics
# First use downloads ~140MB for yolov8x.pt

# SAM model (Lab 2.2.6) - ~2.5 GB
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### Additional Packages

```bash
# Install in NGC container
pip install timm ultralytics scikit-learn segment-anything
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

### 2. Install Additional Packages

```bash
# Run once inside container
pip install timm ultralytics scikit-learn

# For SAM (Lab 2.2.6)
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 3. Verify Setup

```python
import torch
import torchvision
import timm

print(f"PyTorch: {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print(f"timm: {timm.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test loading a model
model = timm.create_model('resnet50', pretrained=True)
print(f"ResNet50 loaded: {sum(p.numel() for p in model.parameters()):,} params")
```

### 4. Verify YOLO

```python
from ultralytics import YOLO

# This will auto-download the model
model = YOLO('yolov8n.pt')  # nano version for quick test
print("YOLOv8 ready!")
```

---

## Pre-Lab Checklist

### Lab 2.2.1: CNN Architecture Study
- [ ] Container running with GPU
- [ ] Understand convolution operation conceptually
- [ ] Module 2.1 PyTorch skills (nn.Module, DataLoader)

### Lab 2.2.2: Transfer Learning
- [ ] Completed Lab 2.2.1
- [ ] timm library installed
- [ ] Understand freezing vs fine-tuning

### Lab 2.2.3: Object Detection
- [ ] ultralytics package installed
- [ ] Sample images available (or use provided)
- [ ] Understand bounding boxes (x1, y1, x2, y2)

### Lab 2.2.4: Segmentation
- [ ] VOC dataset downloaded (~2 GB)
- [ ] Understand pixel-wise classification
- [ ] Understand encoder-decoder architecture

### Lab 2.2.5: Vision Transformer
- [ ] Understand attention mechanism (from README examples)
- [ ] Completed at least Labs 2.2.1-2.2.2
- [ ] Know what patches and embeddings are

### Lab 2.2.6: SAM Integration
- [ ] SAM checkpoint downloaded (2.5 GB)
- [ ] segment-anything package installed
- [ ] Have sample images to test with

---

## Common Setup Mistakes

| Mistake | Consequence | Prevention |
|---------|-------------|------------|
| Not installing timm | Can't load advanced models | `pip install timm` |
| Wrong ultralytics version | YOLO API changes | `pip install ultralytics --upgrade` |
| SAM checkpoint in wrong location | FileNotFoundError | Put in `/workspace/models/` |
| VOC download interrupted | Corrupted dataset | Delete and re-download |

---

## Expected File Structure

```
/workspace/
├── domain-2-deep-learning-frameworks/
│   └── module-2.2-computer-vision/
│       ├── README.md
│       ├── QUICKSTART.md
│       ├── PREREQUISITES.md
│       ├── STUDY_GUIDE.md
│       ├── QUICK_REFERENCE.md
│       ├── ELI5.md
│       ├── LAB_PREP.md           # This file
│       ├── TROUBLESHOOTING.md
│       ├── labs/
│       ├── scripts/
│       ├── solutions/
│       ├── data/                  # Datasets download here
│       │   ├── cifar-10-batches-py/
│       │   ├── cifar-100-python/
│       │   └── VOCdevkit/
│       └── models/                # Downloaded models
│           └── sam_vit_h_4b8939.pth
```

---

## Resource Requirements by Lab

| Lab | GPU Memory | Disk Space | Notes |
|-----|------------|------------|-------|
| 2.2.1 | ~4 GB | ~200 MB | CIFAR-10 |
| 2.2.2 | ~8 GB | ~200 MB | EfficientNet fine-tuning |
| 2.2.3 | ~4 GB | ~200 MB | YOLO inference |
| 2.2.4 | ~8 GB | ~2 GB | VOC dataset |
| 2.2.5 | ~8 GB | ~200 MB | ViT training |
| 2.2.6 | ~12 GB | ~2.5 GB | SAM ViT-H |

All labs fit easily within DGX Spark's 128GB unified memory.

---

## Quick Start Commands

```bash
# Inside NGC container
cd /workspace/domain-2-deep-learning-frameworks/module-2.2-computer-vision

# Install required packages
pip install timm ultralytics scikit-learn

# Download SAM model (for Lab 2.2.6)
mkdir -p models
wget -P models https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Verify setup
python -c "
import torch
import torchvision
import timm
from ultralytics import YOLO

print('PyTorch:', torch.__version__)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
print('timm:', timm.__version__)
print('All packages ready!')
"
```

---

## Downloading SAM Model (Lab 2.2.6)

The SAM ViT-H model is 2.5 GB. Download before the lab:

```bash
# Option 1: wget
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Option 2: curl
curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Option 3: Python
import urllib.request
urllib.request.urlretrieve(
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "sam_vit_h_4b8939.pth"
)
```

Smaller variants available:
- `sam_vit_l_0b3195.pth` - ViT-Large (~1.2 GB)
- `sam_vit_b_01ec64.pth` - ViT-Base (~375 MB)
