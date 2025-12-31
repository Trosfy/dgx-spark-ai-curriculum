# Module 2.2: Computer Vision

**Domain:** 2 - Deep Learning Frameworks
**Duration:** Weeks 10-11 (14-16 hours)
**Prerequisites:** Module 2.1 (PyTorch)
**Priority:** P2 Expanded (ViT, YOLO, Object Detection)

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
│   ├── lab-2.2.1-cnn-architecture-study.ipynb
│   ├── lab-2.2.2-transfer-learning-project.ipynb
│   ├── lab-2.2.3-object-detection-demo.ipynb
│   ├── lab-2.2.4-segmentation-lab.ipynb
│   ├── lab-2.2.5-vision-transformer.ipynb
│   └── lab-2.2.6-sam-integration.ipynb
├── scripts/
│   ├── __init__.py              # Package init
│   ├── cnn_architectures.py     # CNN implementations (LeNet, AlexNet, VGG, ResNet, U-Net)
│   ├── training_utils.py        # Training helpers and Trainer class
│   ├── visualization_utils.py   # Visualization helpers
│   └── metrics.py               # Evaluation metrics (IoU, mAP, etc.)
├── solutions/
│   ├── lab-2.2.1-cnn-architecture-study-solution.ipynb
│   ├── lab-2.2.2-transfer-learning-project-solution.ipynb
│   ├── lab-2.2.3-object-detection-demo-solution.ipynb
│   ├── lab-2.2.4-segmentation-lab-solution.ipynb
│   ├── lab-2.2.5-vision-transformer-solution.ipynb
│   └── lab-2.2.6-sam-integration-solution.ipynb
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
- ✅ Perform object detection using YOLO and Faster R-CNN
- ✅ Understand and implement Vision Transformers (ViT) from scratch

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 2.2.1 | Explain the evolution from LeNet to modern architectures | Understand |
| 2.2.2 | Train and deploy YOLO for object detection | Apply |
| 2.2.3 | Implement Vision Transformer (ViT) from scratch | Apply |
| 2.2.4 | Fine-tune pre-trained models on custom datasets | Apply |

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

### 2.2.3 Object Detection [P2 Expansion]
- Region-based methods (R-CNN family, Faster R-CNN)
- Single-shot detectors (YOLO family, SSD)
- Anchor-free detectors (FCOS, CenterNet)
- Using YOLOv8/YOLOv11 on DGX Spark

### 2.2.4 Image Segmentation
- Semantic vs instance segmentation
- U-Net architecture
- Segment Anything Model (SAM)

### 2.2.5 Vision Transformers [P2 Expansion]
- ViT architecture and patch embeddings
- Positional embeddings for images
- DeiT training tricks (distillation)
- Swin Transformer (hierarchical)
- Comparison with CNNs: when to use each

---

## Labs

| # | Task | Time | Deliverable |
|---|------|------|-------------|
| 2.2.1 | CNN Architecture Study | 2h | LeNet, AlexNet, ResNet comparison on CIFAR-10 |
| 2.2.2 | Transfer Learning Project | 2h | Fine-tuned EfficientNet with >90% accuracy |
| 2.2.3 | YOLO Object Detection | 3h | YOLOv8 inference, custom training demo |
| 2.2.4 | Semantic Segmentation | 2h | U-Net implementation, VOC dataset training |
| 2.2.5 | Vision Transformer from Scratch | 3h | ViT implementation, compare with CNN |
| 2.2.6 | SAM Integration | 2h | Segment Anything demo with interactive prompts |

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

- [ ] Three CNN architectures implemented and compared (LeNet, AlexNet, ResNet)
- [ ] Transfer learning achieving >90% accuracy on CIFAR-100
- [ ] YOLOv8 object detection working with inference and custom training
- [ ] U-Net semantic segmentation trained on VOC dataset
- [ ] Vision Transformer implemented from scratch with attention visualization
- [ ] SAM interactive segmentation demo complete

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

## Next Steps

After completing this module:
1. ✅ Verify all milestones are checked
2. 📁 Save reusable implementations to `scripts/`
3. ➡️ Proceed to [Module 2.3: NLP & Transformers](../module-2.3-nlp-transformers/)

---

## Module Navigation

| Previous | Current | Next |
|----------|---------|------|
| [Module 2.1: PyTorch](../module-2.1-pytorch/) | **Module 2.2: Computer Vision** | [Module 2.3: NLP & Transformers](../module-2.3-nlp-transformers/) |

---

## Resources

- [CS231n](http://cs231n.stanford.edu/)
- [timm library](https://github.com/huggingface/pytorch-image-models)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Segment Anything](https://segment-anything.com/)
- [PyTorch Vision](https://pytorch.org/vision/stable/index.html)
- [Hugging Face Transformers (Vision)](https://huggingface.co/docs/transformers/model_doc/vit)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [Swin Transformer Paper](https://arxiv.org/abs/2103.14030)
