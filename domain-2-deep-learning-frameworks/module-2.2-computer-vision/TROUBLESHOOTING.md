# Module 2.2: Computer Vision - Troubleshooting Guide

## Quick Diagnostic

**Before diving into specific errors, try these:**

1. Check GPU access: `torch.cuda.is_available()`
2. Check memory: `nvidia-smi`
3. Verify packages: `pip list | grep -E "timm|ultralytics|torchvision"`
4. Restart kernel if memory issues

---

## Import/Dependency Errors

### Error: `ModuleNotFoundError: No module named 'timm'`

**Solution**:
```bash
pip install timm
```

---

### Error: `ModuleNotFoundError: No module named 'ultralytics'`

**Solution**:
```bash
pip install ultralytics
```

---

### Error: `No module named 'segment_anything'`

**Solution**:
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

---

## Model Loading Errors

### Error: `FileNotFoundError: sam_vit_h_4b8939.pth`

**Cause**: SAM checkpoint not downloaded.

**Solution**:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Or use smaller variant:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

---

### Error: `RuntimeError: Error(s) in loading state_dict for ResNet`

**Cause**: Mismatch between model architecture and checkpoint.

**Solution**:
```python
# Make sure you're loading the right model
from torchvision import models

# This loads model + weights together
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# NOT this (architecture only, no weights)
model = models.resnet50()  # weights=None by default
```

---

### Error: `HTTPError 403: Forbidden` when downloading models

**Cause**: Model requires authentication or is rate-limited.

**Solution**:
```python
# For Hugging Face models, login first
from huggingface_hub import login
login()  # Enter your token

# Or set token in environment
import os
os.environ["HF_TOKEN"] = "your_token_here"
```

---

## Training Errors

### Error: CNN accuracy stuck at random (10% for CIFAR-10)

**Causes**:
1. Learning rate too high or too low
2. Wrong normalization
3. Labels not matching

**Solutions**:
```python
# Check 1: Verify data normalization matches pre-trained model
# ImageNet normalization:
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Check 2: Verify learning rate
# Try: 1e-3 (fine-tuning head), 1e-4 (full model), 1e-5 (pre-trained backbone)

# Check 3: Verify one batch
for images, labels in dataloader:
    print(f"Images shape: {images.shape}")
    print(f"Labels: {labels[:10]}")
    print(f"Image range: [{images.min():.2f}, {images.max():.2f}]")
    break
```

---

### Error: Transfer learning not improving

**Symptoms**: Fine-tuned model accuracy barely better than random.

**Solutions**:
```python
# Solution 1: Unfreeze more layers
for name, param in model.named_parameters():
    if 'layer4' in name or 'fc' in name:  # Unfreeze last layer + head
        param.requires_grad = True
    else:
        param.requires_grad = False

# Solution 2: Use discriminative learning rates
optimizer = torch.optim.Adam([
    {'params': model.layer3.parameters(), 'lr': 1e-5},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])

# Solution 3: More epochs and warmup
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, epochs=20, steps_per_epoch=len(dataloader)
)
```

---

### Error: `RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.cuda.HalfTensor) should be the same`

**Cause**: Model and input have different dtypes.

**Solution**:
```python
# Ensure consistent dtype
model = model.to(torch.bfloat16)
input = input.to(torch.bfloat16)

# Or use autocast
from torch.amp import autocast
with autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input)
```

---

## YOLO Errors

### Error: YOLO inference returns empty detections

**Causes**:
1. Confidence threshold too high
2. Image not preprocessed correctly
3. Wrong model for task

**Solutions**:
```python
from ultralytics import YOLO

model = YOLO('yolov8x.pt')

# Lower confidence threshold
results = model('image.jpg', conf=0.1)  # Default is 0.25

# Check image format
print(f"Results: {len(results[0].boxes)} detections")
for r in results:
    print(f"Boxes: {r.boxes.xyxy}")
    print(f"Confidences: {r.boxes.conf}")
```

---

### Error: `ModuleNotFoundError: No module named 'lap'`

**Cause**: Missing dependency for YOLO tracking.

**Solution**:
```bash
pip install lap
```

---

## Segmentation Errors

### Error: U-Net output shape doesn't match input

**Cause**: Padding or architecture mismatch.

**Solution**:
```python
# Ensure symmetric encoding/decoding
# For input 256x256:
# Encoder: 256 → 128 → 64 → 32 → 16 (4 downsamples)
# Decoder: 16 → 32 → 64 → 128 → 256 (4 upsamples)

# Use same padding in all convolutions
nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)  # Maintains size

# Match skip connection dimensions
self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
# Concatenate: [batch, out_ch + skip_ch, H, W]
```

---

### Error: IoU score is 0 or very low

**Causes**:
1. Predictions all one class
2. Wrong class indexing
3. Threshold too high/low

**Solutions**:
```python
# Check prediction distribution
pred = model(input).argmax(dim=1)
print(f"Unique predictions: {pred.unique()}")
print(f"Unique targets: {target.unique()}")

# Visualize predictions
import matplotlib.pyplot as plt
plt.imshow(pred[0].cpu())
plt.colorbar()
plt.show()
```

---

## SAM Errors

### Error: SAM predictions are empty masks

**Cause**: Prompts not formatted correctly.

**Solution**:
```python
from segment_anything import SamPredictor

# Points must be numpy arrays with correct shape
point_coords = np.array([[500, 375]])  # Shape: (N, 2)
point_labels = np.array([1])           # Shape: (N,)

# Boxes must be [x1, y1, x2, y2]
box = np.array([100, 100, 400, 400])

masks, scores, logits = predictor.predict(
    point_coords=point_coords,
    point_labels=point_labels,
    multimask_output=True
)

# Check output
print(f"Masks shape: {masks.shape}")  # (3, H, W) for multimask
print(f"Scores: {scores}")
```

---

### Error: SAM out of memory

**Cause**: Processing high-resolution image.

**Solution**:
```python
# Resize image before processing
from PIL import Image

image = Image.open('large_image.jpg')
image = image.resize((1024, 1024))
image_array = np.array(image)

predictor.set_image(image_array)
```

---

## Vision Transformer Errors

### Error: ViT training is extremely slow

**Cause**: Not using pre-trained weights.

**Solution**:
```python
# Use pre-trained ViT for faster convergence
import timm

# Pre-trained on ImageNet-21k, fine-tuned on ImageNet-1k
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=10)
```

---

### Error: ViT accuracy much lower than CNN

**Cause**: ViTs need more data or careful hyperparameters.

**Solutions**:
```python
# Solution 1: Use stronger augmentation
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(),  # Strong augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Solution 2: Use DeiT training tricks
# Lower LR, longer warmup, weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

# Solution 3: Use pre-trained weights
model = timm.create_model('deit_small_patch16_224', pretrained=True)
```

---

## Reset Procedures

### Memory Reset

```python
import torch
import gc

# Clear GPU memory
torch.cuda.empty_cache()
gc.collect()

# Verify
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```

### Full Environment Reset

```bash
# Exit and restart container
exit
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

---

## ❓ Frequently Asked Questions

### General Questions

**Q: Why do we use NGC containers instead of pip install for PyTorch?**

DGX Spark uses ARM64/aarch64 architecture. PyTorch pip packages are primarily built for x86_64. The NGC container (`nvcr.io/nvidia/pytorch:25.11-py3`) comes pre-built and optimized for ARM64 with full CUDA support, including native bfloat16 support for the Blackwell GPU.

---

**Q: Why does my DataLoader hang when using num_workers > 0?**

When running in Docker, you must use the `--ipc=host` flag. PyTorch DataLoader uses Linux shared memory (IPC) for multiprocessing, which requires this flag:

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

---

**Q: What batch sizes can DGX Spark handle for computer vision models?**

With 128GB unified memory, DGX Spark can handle significantly larger batch sizes than typical GPUs:

| Model | Typical GPU (8-16GB) | DGX Spark (128GB) |
|-------|---------------------|-------------------|
| ResNet-50 | 32-64 | 256-512 |
| ViT-Base | 16-32 | 128-256 |
| YOLOv8 | 8-16 | 64-128 |
| SAM (ViT-H) | Doesn't fit | Easily loads |

Larger batch sizes generally lead to faster training and smoother gradients.

---

### CNN Questions

**Q: Why do we use 3×3 convolutions instead of larger kernels?**

VGG (2014) demonstrated that stacking two 3×3 convolutions has the same receptive field as one 5×5 convolution, but with:
- **Fewer parameters**: 2×(3×3) = 18 vs 5×5 = 25
- **More non-linearity**: Two ReLU activations instead of one
- **Better feature learning**: More layers of abstraction

This insight is used in most modern architectures.

---

**Q: What's the difference between AvgPool and MaxPool?**

- **MaxPool** keeps the strongest activation (good for detecting if a feature exists)
- **AvgPool** averages activations (good for summarizing overall feature presence)

For classification, MaxPool is typically preferred in early layers (detect edges strongly), while global AvgPool is used before the final classifier (summarize all features).

---

**Q: How do skip connections in ResNet prevent vanishing gradients?**

Skip connections create a direct path for gradients to flow backward:

```
Without skip:     gradient × layer × layer × layer → vanishes
With skip:        gradient + gradient × layer × layer × layer → preserved!
```

The identity shortcut ensures gradients can always flow back unchanged, even if the learned transformations have small gradients.

---

### Transfer Learning Questions

**Q: Should I freeze all layers when fine-tuning a pre-trained model?**

It depends on your dataset size and similarity to ImageNet:

| Dataset | Strategy |
|---------|----------|
| Small + Similar to ImageNet | Freeze all, train classifier only |
| Small + Different domain | Freeze early layers, fine-tune later layers |
| Large + Any domain | Fine-tune entire model with lower LR for pre-trained layers |

See Lab 2.2.2 for detailed examples.

---

**Q: Why do I need to normalize images the same way as the pre-trained model?**

Pre-trained models learned features expecting specific input distributions. ImageNet normalization is:

```python
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

Using different normalization means the pre-trained features won't work correctly, and you'll need to retrain more of the network.

---

### Object Detection Questions

**Q: What's the difference between YOLOv8n, YOLOv8s, YOLOv8m, and YOLOv8x?**

They're different size variants trading speed vs accuracy:

| Model | Parameters | mAP@0.5 | Speed | Use Case |
|-------|-----------|---------|-------|----------|
| YOLOv8n | 3.2M | 37.3% | Fastest | Real-time video, edge devices |
| YOLOv8s | 11.2M | 44.9% | Fast | Good balance |
| YOLOv8m | 25.9M | 50.2% | Medium | Higher accuracy needs |
| YOLOv8x | 68.2M | 53.9% | Slowest | Maximum accuracy |

For DGX Spark with 128GB unified memory, you can easily run YOLOv8x.

---

**Q: What is mAP (mean Average Precision)?**

mAP measures detection quality by considering both:
- **Precision**: Of all detections, how many are correct?
- **Recall**: Of all ground truth objects, how many did we find?

mAP@0.5 means we count a detection as correct if its IoU (Intersection over Union) with ground truth is ≥ 50%.

---

### Segmentation Questions

**Q: What's the difference between semantic, instance, and panoptic segmentation?**

- **Semantic**: Label each pixel with a class (all cats are "cat")
- **Instance**: Separate individual objects (cat1, cat2, cat3)
- **Panoptic**: Both! Each pixel has a class AND instance ID

SAM (Segment Anything Model) performs instance-level segmentation for any object you prompt.

---

**Q: Why do we need Dice Loss for segmentation?**

Cross-entropy loss can be dominated by the background class (often 90%+ of pixels). Dice Loss directly measures overlap between prediction and ground truth:

```
Dice = 2 × |Prediction ∩ Truth| / (|Prediction| + |Truth|)
```

This naturally handles class imbalance since it focuses on the object itself.

---

### Vision Transformer (ViT) Questions

**Q: Why do Vision Transformers need so much data to train?**

CNNs have **inductive biases** built in:
- Locality: Nearby pixels are related
- Translation equivariance: A cat is a cat regardless of position

ViTs learn these properties from data instead of having them built in, so they need more examples. Solutions:
- Use pre-trained models (trained on massive datasets)
- Strong data augmentation (RandAugment, Mixup)
- Distillation from CNN teachers (DeiT approach)

---

**Q: What's the purpose of the [CLS] token in ViT?**

The [CLS] token is a learnable embedding that:
1. Attends to all image patches through self-attention
2. Aggregates global information about the image
3. Serves as the representation for classification

It's borrowed from BERT in NLP where it serves a similar purpose.

---

### SAM (Segment Anything) Questions

**Q: Why is SAM called a "foundation model"?**

SAM is called a foundation model because:
1. **Zero-shot generalization**: Works on objects never seen during training
2. **Promptable**: Can be guided by points, boxes, or text
3. **Massive training**: 11 million images, 1.1 billion masks
4. **Transfer**: Same model works for medical imaging, satellite imagery, photos, etc.

---

**Q: Why does SAM encode the image once but can segment many times?**

SAM's architecture separates:
1. **Image encoder** (heavy, ~1 second): Computes rich image features once
2. **Mask decoder** (light, ~50ms): Takes prompts and produces masks

This design enables interactive applications where you click multiple times on the same image.

---

### DGX Spark Specific Questions

**Q: How do I clear GPU memory between experiments?**

Use the standard cleanup pattern:

```python
import gc

del model  # Delete model reference
torch.cuda.empty_cache()  # Release cached memory
gc.collect()  # Python garbage collection

# Verify
print(f"Free: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")
```

---

**Q: What precision should I use for training on DGX Spark?**

Use bfloat16 mixed precision (native Blackwell support):

```python
from torch.amp import autocast

with autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)
```

This is faster and uses less memory than FP32 while maintaining training stability.

---

**Q: Can I train a ViT-Large from scratch on DGX Spark?**

Yes! With 128GB unified memory, DGX Spark can handle:
- ViT-Large (304M params) with batch size 128-256
- ViT-Huge (632M params) with batch size 64-128

However, training from scratch requires massive datasets. For small datasets, use pre-trained weights and fine-tune.

---

## Still Stuck?

1. **Check solution notebooks** in `solutions/` folder
2. **Print tensor shapes** at each step - most bugs are shape mismatches
3. **Verify data preprocessing** - wrong normalization is a common issue
4. **Check model.train() vs model.eval()** - critical for BatchNorm and Dropout

**Debug template**:
```python
def debug_forward(model, x):
    print(f"Input: {x.shape}, dtype={x.dtype}, device={x.device}")
    for name, layer in model.named_children():
        x = layer(x)
        print(f"{name}: {x.shape}")
    return x
```
