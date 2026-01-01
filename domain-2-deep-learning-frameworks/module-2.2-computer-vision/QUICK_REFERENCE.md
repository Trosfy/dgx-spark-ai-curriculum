# Module 2.2: Computer Vision - Quick Reference

## Essential Commands

### Loading Pre-trained Models

```python
import torch
from torchvision import models

# ResNet-50 (ImageNet pre-trained)
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# EfficientNet-B3
model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)

# ViT-B/16
model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

# Move to GPU with BF16
model = model.cuda().to(torch.bfloat16)
```

### Standard ImageNet Preprocessing

```python
from torchvision import transforms

# Standard ImageNet transforms
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

---

## Key Patterns

### Pattern: CNN Basic Block

```python
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
```

### Pattern: ResNet Skip Connection

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection (identity or projection)
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)  # Add skip connection
```

### Pattern: Transfer Learning

```python
# Load pre-trained model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace classifier head
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Unfreeze last layer
for param in model.fc.parameters():
    param.requires_grad = True

# Discriminative learning rates
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-5},  # Fine-tune
    {'params': model.fc.parameters(), 'lr': 1e-3}       # New head
])
```

### Pattern: YOLOv8 Inference

```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolov8x.pt')  # x = extra large

# Inference on image
results = model('image.jpg')

# Access detections
for result in results:
    boxes = result.boxes.xyxy  # Bounding boxes
    confs = result.boxes.conf  # Confidence scores
    classes = result.boxes.cls  # Class indices

    # Draw results
    result.show()
    result.save('output.jpg')
```

### Pattern: YOLOv8 Custom Training

```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolov8n.pt')  # n = nano for speed

# Train on custom dataset
model.train(
    data='dataset.yaml',  # YAML with paths and classes
    epochs=100,
    imgsz=640,
    batch=16,
    device=0  # GPU
)
```

### Pattern: U-Net Architecture

```python
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=21):
        super().__init__()
        # Encoder (downsampling)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder (upsampling)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = self.conv_block(1024, 512)  # 512 + 512 from skip
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = self.conv_block(512, 256)
        # ... continue pattern

        self.final = nn.Conv2d(64, num_classes, 1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        # ... continue
        return self.final(d1)
```

### Pattern: ViT Patch Embedding

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2

        # Conv2d with kernel_size=patch_size extracts patches
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.proj(x)
        # Flatten spatial dims: (B, embed_dim, N) -> (B, N, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x
```

### Pattern: SAM Usage

```python
from segment_anything import sam_model_registry, SamPredictor

# Load SAM (fits easily in DGX Spark 128GB)
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
sam = sam.cuda()
predictor = SamPredictor(sam)

# Set image
predictor.set_image(image)  # RGB numpy array

# Predict with point prompts
masks, scores, logits = predictor.predict(
    point_coords=np.array([[500, 375]]),  # (N, 2) points
    point_labels=np.array([1]),           # 1 = foreground, 0 = background
    multimask_output=True
)

# Predict with box prompt
masks, scores, logits = predictor.predict(
    box=np.array([100, 100, 400, 400])  # [x1, y1, x2, y2]
)
```

---

## Key Values to Remember

| What | Value | Notes |
|------|-------|-------|
| ImageNet mean | [0.485, 0.456, 0.406] | RGB order |
| ImageNet std | [0.229, 0.224, 0.225] | RGB order |
| Standard input size | 224×224 | Most models |
| ViT patch size | 16×16 | B/16 variant |
| SAM ViT-H memory | ~2.5 GB | Model weights |
| YOLO confidence threshold | 0.25 | Default |

---

## Output Size Formulas

```
# Convolution
output = (input + 2*padding - kernel_size) / stride + 1

# Same padding (maintain size)
padding = (kernel_size - 1) / 2  # for stride=1

# MaxPool2d(2, 2)
output = input / 2

# ConvTranspose2d(kernel=2, stride=2)
output = input * 2
```

---

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Wrong normalization | Use ImageNet stats for pre-trained models |
| Forgetting `model.eval()` | Always set for inference (disables dropout, BN training) |
| Mismatched input size | Check model's expected input size |
| Not unfreezing layers | Unfreeze some backbone layers for better accuracy |
| Wrong channel order | PIL is RGB, OpenCV is BGR |
| Forgetting to move to GPU | `model.cuda()` and `input.cuda()` |

---

## Quick Links

- [torchvision models](https://pytorch.org/vision/stable/models.html)
- [timm library](https://github.com/huggingface/pytorch-image-models)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Segment Anything](https://segment-anything.com/)
- [CS231n Notes](http://cs231n.stanford.edu/)
