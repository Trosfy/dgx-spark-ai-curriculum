# Module 2.2: Computer Vision - Study Guide

## Learning Objectives

By the end of this module, you will be able to:

1. **Explain CNN evolution** from LeNet to modern architectures (ResNet, EfficientNet)
2. **Implement CNN architectures** from scratch with skip connections
3. **Apply transfer learning** to achieve >90% accuracy on custom datasets
4. **Use object detection** with YOLOv8 for real-world applications
5. **Build Vision Transformer (ViT)** from scratch and compare with CNNs
6. **Apply SAM** (Segment Anything) for interactive segmentation

---

## Module Roadmap

| # | Lab | Focus | Time | Key Outcome |
|---|-----|-------|------|-------------|
| 1 | lab-2.2.1-cnn-architecture-study.ipynb | CNN fundamentals | ~3 hr | LeNet, AlexNet, ResNet comparison |
| 2 | lab-2.2.2-transfer-learning-project.ipynb | Transfer learning | ~3 hr | Fine-tuned model >90% accuracy |
| 3 | lab-2.2.3-object-detection-demo.ipynb | Object detection | ~2 hr | YOLOv8 inference and training |
| 4 | lab-2.2.4-segmentation-lab.ipynb | Segmentation | ~3 hr | U-Net implementation |
| 5 | lab-2.2.5-vision-transformer.ipynb | ViT architecture | ~3 hr | ViT from scratch |
| 6 | lab-2.2.6-sam-integration.ipynb | Modern segmentation | ~2 hr | SAM interactive demo |

**Total time**: ~14-16 hours

---

## Core Concepts

### Convolution
**What**: Operation that slides a filter across an image, detecting local patterns
**Why it matters**: Foundation of all CNNs; learns hierarchical features (edges → textures → objects)
**First appears in**: Lab 2.2.1

### Skip Connection (Residual)
**What**: Direct path that bypasses layers, adding input to output
**Why it matters**: Enables training very deep networks (100+ layers); solves vanishing gradients
**First appears in**: Lab 2.2.1 (ResNet implementation)

### Transfer Learning
**What**: Using pre-trained weights as starting point for new task
**Why it matters**: Achieves high accuracy with limited data; saves training time
**First appears in**: Lab 2.2.2

### Object Detection
**What**: Locating and classifying multiple objects in an image (bounding boxes + classes)
**Why it matters**: Real-world applications (autonomous vehicles, medical imaging, security)
**First appears in**: Lab 2.2.3

### Vision Transformer (ViT)
**What**: Transformer architecture adapted for images (patches as tokens)
**Why it matters**: State-of-the-art on many benchmarks; bridges vision and language
**First appears in**: Lab 2.2.5

---

## How This Module Connects

```
Previous                    This Module                 Next
─────────────────────────────────────────────────────────────
Module 2.1              ►   Module 2.2              ►  Module 2.3
PyTorch                     Computer Vision            NLP/Transformers
(building blocks)           (visual applications)      (text + attention)
```

**Builds on**:
- **nn.Module and DataLoader** from Module 2.1 (now build complex architectures)
- **Mixed precision training** from Module 2.1 (use BF16 for faster training)
- **Profiling skills** from Module 2.1 (optimize CV pipelines)

**Prepares for**:
- **Module 2.3** will apply similar attention mechanisms to text
- **Module 2.4** will explore efficient alternatives to transformers
- **Module 4.1** (Multimodal) will combine vision and language

---

## Recommended Approach

### Standard Path (14-16 hours)
1. **Start with Lab 2.2.1** - Build intuition for CNN architectures
2. **Work through 2.2.2** - Transfer learning is the practical skill you'll use most
3. **Complete 2.2.3** - Object detection for real-world applications
4. **Understand 2.2.4** - Segmentation extends classification to pixels
5. **Master 2.2.5** - ViT connects to transformers (crucial for Domain 3)
6. **Explore 2.2.6** - SAM is cutting-edge; great for interactive applications

### Quick Path (8-10 hours, if experienced with CV)
1. Skim Lab 2.2.1 - Focus on ResNet skip connections
2. Focus on Lab 2.2.2 - Transfer learning techniques
3. Complete Lab 2.2.5 - ViT is the key modern architecture
4. Try Lab 2.2.6 - SAM for interactive segmentation

### Deep-Dive Path (20+ hours)
1. Implement additional architectures in Lab 2.2.1 (VGG, DenseNet)
2. Train on your own dataset in Lab 2.2.2
3. Train YOLO on custom objects in Lab 2.2.3
4. Extend U-Net with attention in Lab 2.2.4
5. Implement DeiT training tricks in Lab 2.2.5

---

## Lab-by-Lab Summary

### Lab 2.2.1: CNN Architecture Study
**Goal**: Understand evolution from LeNet to ResNet
**Key skills**:
- Implementing convolution, pooling, fully-connected layers
- Skip connections and residual blocks
- Batch normalization placement
- Architecture comparison on CIFAR-10

### Lab 2.2.2: Transfer Learning Project
**Goal**: Fine-tune EfficientNet to >90% accuracy
**Key skills**:
- Loading pre-trained weights
- Freezing backbone, training head
- Learning rate strategies (discriminative LR)
- Data augmentation for limited data

### Lab 2.2.3: Object Detection Demo
**Goal**: Run and train YOLOv8
**Key skills**:
- YOLO inference on images/video
- Understanding anchor boxes and NMS
- Custom training with Ultralytics
- Evaluating with mAP

### Lab 2.2.4: Semantic Segmentation
**Goal**: Implement U-Net for pixel-wise classification
**Key skills**:
- Encoder-decoder architecture
- Skip connections in U-Net (different from ResNet!)
- Training on VOC dataset
- IoU metric for segmentation

### Lab 2.2.5: Vision Transformer (ViT)
**Goal**: Build ViT from scratch
**Key skills**:
- Patch embedding (image → tokens)
- Positional encoding for 2D
- Self-attention for images
- CLS token for classification

### Lab 2.2.6: SAM Integration
**Goal**: Use Segment Anything with interactive prompts
**Key skills**:
- Loading SAM ViT-H on DGX Spark
- Point and box prompts
- Mask generation and refinement
- Integration into applications

---

## DGX Spark Advantages

| Task | Consumer GPU | DGX Spark (128GB) |
|------|--------------|-------------------|
| Training ResNet-18 on CIFAR-10 | ~2 GB | Easy |
| Fine-tuning EfficientNet-B3 | ~8 GB | Easy |
| YOLOv8-X inference | ~4 GB | Easy |
| SAM ViT-H + multiple images | ~12 GB | Easy |
| Training ViT-Large | ~16 GB | Easy |
| SDXL generation (future) | ~14 GB | Easy |

**Key advantage**: You can load SAM ViT-H (2.5GB model) with plenty of room for batch processing and experimentation.

---

## Before You Start

- See [PREREQUISITES.md](./PREREQUISITES.md) for skill self-check
- See [LAB_PREP.md](./LAB_PREP.md) for environment setup
- See [QUICKSTART.md](./QUICKSTART.md) for 5-minute first success

---

## Common Challenges

| Challenge | Solution |
|-----------|----------|
| "CNN not learning" | Check data normalization (ImageNet stats), learning rate |
| "Transfer learning accuracy low" | Unfreeze more layers, use discriminative LR |
| "YOLO detections wrong" | Check image preprocessing, confidence threshold |
| "ViT takes forever to train" | Use smaller patches, pre-trained weights |
| "SAM masks not accurate" | Provide better prompts (points + boxes) |

---

## Success Metrics

You've mastered this module when you can:

- [ ] Explain why ResNet can be 100+ layers but VGG can't
- [ ] Fine-tune any pre-trained model on a new dataset
- [ ] Run object detection on custom images/video
- [ ] Implement a segmentation model (U-Net)
- [ ] Explain how ViT adapts transformers for images
- [ ] Use SAM for interactive segmentation tasks
