# Module 2.2: Computer Vision

**Domain:** 2 - Deep Learning Frameworks
**Duration:** Weeks 9-10 (12-15 hours)
**Prerequisites:** Module 6 (PyTorch)

---

## 🚀 Quick Start with DGX Spark

```bash
# Launch NGC PyTorch container with GPU support
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser

# ⚠️ IMPORTANT FLAGS:
# --gpus all    : Required for GPU access
# --ipc=host    : Required for DataLoader with num_workers > 0
# -v .cache/huggingface : Preserves downloaded models between sessions
```

**Pre-installed in NGC container:** PyTorch, torchvision, numpy, matplotlib, tqdm

**Install additional packages (run once):**
```bash
pip install timm ultralytics scikit-learn
```

---

## Directory Structure

```
module-2.2-computer-vision/
├── README.md                    # This file
├── labs/
│   ├── 01-cnn-architecture-study.ipynb
│   ├── 02-transfer-learning-project.ipynb
│   ├── 03-object-detection-demo.ipynb
│   ├── 04-segmentation-lab.ipynb
│   ├── 05-vision-transformer.ipynb
│   └── 06-sam-integration.ipynb
├── scripts/
│   ├── cnn_architectures.py     # CNN implementations (LeNet, AlexNet, VGG, ResNet, U-Net)
│   ├── training_utils.py        # Training helpers and Trainer class
│   ├── visualization_utils.py   # Visualization helpers
│   └── metrics.py               # Evaluation metrics (IoU, mAP, etc.)
├── solutions/
│   └── exercise-solutions.ipynb # Solutions for all exercises
└── data/
    └── README.md                # Dataset documentation
```

---

## Using the Scripts Directory

The `scripts/` directory contains production-ready implementations that you can import and reuse:

```python
# Import CNN architectures
from scripts.cnn_architectures import LeNet5, AlexNet, VGG11, ResNet18, UNet, get_model

# Import training utilities
from scripts.training_utils import Trainer, get_optimizer, get_scheduler, EarlyStopping

# Import visualization helpers
from scripts.visualization_utils import (
    plot_training_history,
    visualize_predictions,
    plot_confusion_matrix,
    visualize_segmentation
)

# Import metrics
from scripts.metrics import ClassificationMetrics, SegmentationMetrics, compute_iou, compute_dice
```

**Note:** The notebooks implement architectures from scratch for educational purposes. For production use or quick experimentation, use the scripts instead.

---

## Overview

This module covers the fundamentals of computer vision with deep learning. You'll implement CNN architectures, apply transfer learning, explore object detection, and work with modern vision transformers.

---

## Learning Outcomes

By the end of this module, you will be able to:

- ✅ Implement and train CNN architectures for image classification
- ✅ Apply transfer learning for custom image tasks
- ✅ Perform object detection using pre-trained models
- ✅ Understand and implement image segmentation

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 2.2.1 | Explain the evolution from LeNet to modern architectures | Understand |
| 2.2.2 | Implement data augmentation pipelines for image data | Apply |
| 2.2.3 | Fine-tune pre-trained models on custom datasets | Apply |
| 2.2.4 | Evaluate model performance using appropriate CV metrics | Analyze |

---

## Topics

### 2.2.1 CNN Architectures
- Convolution and pooling operations
- LeNet, AlexNet, VGG
- ResNet and skip connections
- Modern: EfficientNet, ConvNeXt

### 2.2.2 Transfer Learning
- Pre-trained model selection
- Feature extraction vs fine-tuning
- Learning rate strategies

### 2.2.3 Object Detection
- R-CNN family
- YOLO, SSD
- Using pre-trained detectors

### 2.2.4 Image Segmentation
- Semantic vs instance segmentation
- U-Net architecture
- Segment Anything Model (SAM)

### 2.2.5 Vision Transformers
- ViT architecture
- Patch embeddings
- DeiT training tricks

---

## Labs

| # | Task | Time | Deliverable |
|---|------|------|-------------|
| 2.2.1 | CNN Architecture Study | 3h | LeNet, AlexNet, ResNet comparison on CIFAR-10 |
| 2.2.2 | Transfer Learning Project | 3h | Fine-tuned EfficientNet with >90% accuracy |
| 2.2.3 | Object Detection Demo | 2h | YOLOv8 inference with DGX Spark benchmarks |
| 2.2.4 | Segmentation Lab | 3h | U-Net implementation for semantic segmentation |
| 2.2.5 | Vision Transformer | 3h | ViT implementation trained on CIFAR-10 |
| 2.2.6 | SAM Integration | 2h | Segment Anything demo notebook |

---

## Guidance

### Data Augmentation Best Practices

```python
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

### Transfer Learning Strategy

```python
# Freeze backbone, train head
for param in model.backbone.parameters():
    param.requires_grad = False

# Use smaller LR for backbone
optimizer = torch.optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-5},
    {'params': model.head.parameters(), 'lr': 1e-3}
])
```

---

## Milestone Checklist

- [ ] Three CNN architectures implemented and compared
- [ ] Transfer learning achieving >90% accuracy
- [ ] Object detection demo working
- [ ] U-Net segmentation trained
- [ ] ViT implemented from scratch
- [ ] SAM demo complete

---

## DGX Spark Advantages

This module is optimized for DGX Spark's unique capabilities:

| Feature | Benefit |
|---------|---------|
| 128GB Unified Memory | Load SAM ViT-H (2.5GB) with room to spare |
| Blackwell GPU | Fast inference for real-time detection |
| Tensor Cores | Accelerated training with mixed precision |

### Memory Usage Guide

| Task | Typical VRAM | DGX Spark |
|------|-------------|-----------|
| Training ResNet-18 on CIFAR-10 | ~2 GB | ✓ Easy |
| Fine-tuning EfficientNet-B3 | ~8 GB | ✓ Easy |
| YOLOv8-X inference | ~4 GB | ✓ Easy |
| SAM ViT-H + multiple images | ~12 GB | ✓ Easy |
| Training ViT-Large | ~16 GB | ✓ Easy |

---

## Resources

- [CS231n](http://cs231n.stanford.edu/)
- [timm library](https://github.com/huggingface/pytorch-image-models)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Segment Anything](https://segment-anything.com/)
- [PyTorch Vision](https://pytorch.org/vision/stable/index.html)
- [Hugging Face Transformers (Vision)](https://huggingface.co/docs/transformers/model_doc/vit)
