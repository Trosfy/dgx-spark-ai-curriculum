# Frequently Asked Questions: Module 2.2 Computer Vision

**Module:** 2.2 - Computer Vision
**Last Updated:** January 2025

---

## General Questions

### Q: Why do we use NGC containers instead of pip install for PyTorch?

**A:** DGX Spark uses ARM64/aarch64 architecture. PyTorch pip packages are primarily built for x86_64. The NGC container (`nvcr.io/nvidia/pytorch:25.11-py3`) comes pre-built and optimized for ARM64 with full CUDA support, including native bfloat16 support for the Blackwell GPU.

---

### Q: Why does my DataLoader hang when using num_workers > 0?

**A:** When running in Docker, you must use the `--ipc=host` flag. PyTorch DataLoader uses Linux shared memory (IPC) for multiprocessing, which requires this flag:

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for more details.

---

### Q: What batch sizes can DGX Spark handle for computer vision models?

**A:** With 128GB unified memory, DGX Spark can handle significantly larger batch sizes than typical GPUs:

| Model | Typical GPU (8-16GB) | DGX Spark (128GB) |
|-------|---------------------|-------------------|
| ResNet-50 | 32-64 | 256-512 |
| ViT-Base | 16-32 | 128-256 |
| YOLOv8 | 8-16 | 64-128 |
| SAM (ViT-H) | Doesn't fit | Easily loads |

Larger batch sizes generally lead to faster training and smoother gradients.

---

## CNN Questions

### Q: Why do we use 3×3 convolutions instead of larger kernels?

**A:** VGG (2014) demonstrated that stacking two 3×3 convolutions has the same receptive field as one 5×5 convolution, but with:
- **Fewer parameters**: 2×(3×3) = 18 vs 5×5 = 25
- **More non-linearity**: Two ReLU activations instead of one
- **Better feature learning**: More layers of abstraction

This insight is used in most modern architectures.

---

### Q: What's the difference between AvgPool and MaxPool?

**A:**
- **MaxPool** keeps the strongest activation (good for detecting if a feature exists)
- **AvgPool** averages activations (good for summarizing overall feature presence)

For classification, MaxPool is typically preferred in early layers (detect edges strongly), while global AvgPool is used before the final classifier (summarize all features).

---

### Q: How do skip connections in ResNet prevent vanishing gradients?

**A:** Skip connections create a direct path for gradients to flow backward:

```
Without skip:     gradient × layer × layer × layer → vanishes
With skip:        gradient + gradient × layer × layer × layer → preserved!
```

The identity shortcut ensures gradients can always flow back unchanged, even if the learned transformations have small gradients.

---

## Transfer Learning Questions

### Q: Should I freeze all layers when fine-tuning a pre-trained model?

**A:** It depends on your dataset size and similarity to ImageNet:

| Dataset | Strategy |
|---------|----------|
| Small + Similar to ImageNet | Freeze all, train classifier only |
| Small + Different domain | Freeze early layers, fine-tune later layers |
| Large + Any domain | Fine-tune entire model with lower LR for pre-trained layers |

See Lab 2.2.2 for detailed examples.

---

### Q: Why do I need to normalize images the same way as the pre-trained model?

**A:** Pre-trained models learned features expecting specific input distributions. ImageNet normalization is:

```python
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

Using different normalization means the pre-trained features won't work correctly, and you'll need to retrain more of the network.

---

## Object Detection Questions

### Q: What's the difference between YOLOv8n, YOLOv8s, YOLOv8m, and YOLOv8x?

**A:** They're different size variants trading speed vs accuracy:

| Model | Parameters | mAP@0.5 | Speed | Use Case |
|-------|-----------|---------|-------|----------|
| YOLOv8n | 3.2M | 37.3% | Fastest | Real-time video, edge devices |
| YOLOv8s | 11.2M | 44.9% | Fast | Good balance |
| YOLOv8m | 25.9M | 50.2% | Medium | Higher accuracy needs |
| YOLOv8x | 68.2M | 53.9% | Slowest | Maximum accuracy |

For DGX Spark with 128GB unified memory, you can easily run YOLOv8x.

---

### Q: What is mAP (mean Average Precision)?

**A:** mAP measures detection quality by considering both:
- **Precision**: Of all detections, how many are correct?
- **Recall**: Of all ground truth objects, how many did we find?

mAP@0.5 means we count a detection as correct if its IoU (Intersection over Union) with ground truth is ≥ 50%.

---

## Segmentation Questions

### Q: What's the difference between semantic, instance, and panoptic segmentation?

**A:**
- **Semantic**: Label each pixel with a class (all cats are "cat")
- **Instance**: Separate individual objects (cat1, cat2, cat3)
- **Panoptic**: Both! Each pixel has a class AND instance ID

SAM (Segment Anything Model) performs instance-level segmentation for any object you prompt.

---

### Q: Why do we need Dice Loss for segmentation?

**A:** Cross-entropy loss can be dominated by the background class (often 90%+ of pixels). Dice Loss directly measures overlap between prediction and ground truth:

```
Dice = 2 × |Prediction ∩ Truth| / (|Prediction| + |Truth|)
```

This naturally handles class imbalance since it focuses on the object itself.

---

## Vision Transformer (ViT) Questions

### Q: Why do Vision Transformers need so much data to train?

**A:** CNNs have **inductive biases** built in:
- Locality: Nearby pixels are related
- Translation equivariance: A cat is a cat regardless of position

ViTs learn these properties from data instead of having them built in, so they need more examples. Solutions:
- Use pre-trained models (trained on massive datasets)
- Strong data augmentation (RandAugment, Mixup)
- Distillation from CNN teachers (DeiT approach)

---

### Q: What's the purpose of the [CLS] token in ViT?

**A:** The [CLS] token is a learnable embedding that:
1. Attends to all image patches through self-attention
2. Aggregates global information about the image
3. Serves as the representation for classification

It's borrowed from BERT in NLP where it serves a similar purpose.

---

## SAM (Segment Anything) Questions

### Q: Why is SAM called a "foundation model"?

**A:** SAM is called a foundation model because:
1. **Zero-shot generalization**: Works on objects never seen during training
2. **Promptable**: Can be guided by points, boxes, or text
3. **Massive training**: 11 million images, 1.1 billion masks
4. **Transfer**: Same model works for medical imaging, satellite imagery, photos, etc.

---

### Q: Why does SAM encode the image once but can segment many times?

**A:** SAM's architecture separates:
1. **Image encoder** (heavy, ~1 second): Computes rich image features once
2. **Mask decoder** (light, ~50ms): Takes prompts and produces masks

This design enables interactive applications where you click multiple times on the same image.

---

## DGX Spark Specific Questions

### Q: How do I clear GPU memory between experiments?

**A:** Use the standard cleanup pattern:

```python
import gc

del model  # Delete model reference
torch.cuda.empty_cache()  # Release cached memory
gc.collect()  # Python garbage collection

# Verify
print(f"Free: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")
```

---

### Q: What precision should I use for training on DGX Spark?

**A:** Use bfloat16 mixed precision (native Blackwell support):

```python
from torch.amp import autocast

with autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)
```

This is faster and uses less memory than FP32 while maintaining training stability.

---

### Q: Can I train a ViT-Large from scratch on DGX Spark?

**A:** Yes! With 128GB unified memory, DGX Spark can handle:
- ViT-Large (304M params) with batch size 128-256
- ViT-Huge (632M params) with batch size 64-128

However, training from scratch requires massive datasets. For small datasets, use pre-trained weights and fine-tune.

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [QUICKSTART.md](./QUICKSTART.md) | Get started in 5 minutes |
| [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) | Solutions to common errors |
| [PREREQUISITES.md](./PREREQUISITES.md) | Required knowledge |
| [STUDY_GUIDE.md](./STUDY_GUIDE.md) | Learning objectives |
| [LAB_PREP.md](./LAB_PREP.md) | Lab setup instructions |
| [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) | Commands cheat sheet |
| [ELI5.md](./ELI5.md) | Simple explanations |

---

*Module 2.2 - Computer Vision | DGX Spark AI Curriculum v2.0*
