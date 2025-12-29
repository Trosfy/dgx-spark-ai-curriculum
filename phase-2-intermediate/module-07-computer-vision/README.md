# Module 7: Computer Vision

**Phase:** 2 - Intermediate  
**Duration:** Weeks 9-10 (12-15 hours)  
**Prerequisites:** Module 6 (PyTorch)

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
| 7.1 | Explain the evolution from LeNet to modern architectures | Understand |
| 7.2 | Implement data augmentation pipelines for image data | Apply |
| 7.3 | Fine-tune pre-trained models on custom datasets | Apply |
| 7.4 | Evaluate model performance using appropriate CV metrics | Analyze |

---

## Topics

### 7.1 CNN Architectures
- Convolution and pooling operations
- LeNet, AlexNet, VGG
- ResNet and skip connections
- Modern: EfficientNet, ConvNeXt

### 7.2 Transfer Learning
- Pre-trained model selection
- Feature extraction vs fine-tuning
- Learning rate strategies

### 7.3 Object Detection
- R-CNN family
- YOLO, SSD
- Using pre-trained detectors

### 7.4 Image Segmentation
- Semantic vs instance segmentation
- U-Net architecture
- Segment Anything Model (SAM)

### 7.5 Vision Transformers
- ViT architecture
- Patch embeddings
- DeiT training tricks

---

## Tasks

| # | Task | Time | Deliverable |
|---|------|------|-------------|
| 7.1 | CNN Architecture Study | 3h | LeNet, AlexNet, ResNet comparison on CIFAR-10 |
| 7.2 | Transfer Learning Project | 3h | Fine-tuned EfficientNet with >90% accuracy |
| 7.3 | Object Detection Demo | 2h | YOLOv8 inference with DGX Spark benchmarks |
| 7.4 | Segmentation Lab | 3h | U-Net implementation for semantic segmentation |
| 7.5 | Vision Transformer | 3h | ViT implementation trained on CIFAR-10 |
| 7.6 | SAM Integration | 2h | Segment Anything demo notebook |

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

## Resources

- [CS231n](http://cs231n.stanford.edu/)
- [timm library](https://github.com/huggingface/pytorch-image-models)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Segment Anything](https://segment-anything.com/)
