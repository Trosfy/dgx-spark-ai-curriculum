# Module 2.2: Computer Vision - Quickstart

## Time: ~5 minutes

## What You'll Build
A pre-trained ResNet classifying images with a few lines of code.

## Before You Start
- [ ] DGX Spark container running
- [ ] Module 2.1 concepts understood (nn.Module, DataLoader)

## Let's Go!

### Step 1: Load a Pre-trained Model
```python
import torch
from torchvision import models, transforms
from PIL import Image
import urllib.request

# Download ImageNet class labels
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
urllib.request.urlretrieve(url, "imagenet_classes.txt")
with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# Load pre-trained ResNet
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model = model.cuda().eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
```

### Step 2: Prepare an Image
```python
# Download a sample image
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
urllib.request.urlretrieve(url, "cat.jpg")

# Standard ImageNet preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open("cat.jpg")
input_tensor = transform(image).unsqueeze(0).cuda()
print(f"Input shape: {input_tensor.shape}")
```

### Step 3: Run Inference
```python
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.softmax(output[0], dim=0)

# Top 5 predictions
top5_prob, top5_idx = probabilities.topk(5)
for i in range(5):
    print(f"{labels[top5_idx[i]]}: {top5_prob[i].item()*100:.1f}%")
```

### Step 4: See the Result
**Expected output** (something like):
```
tabby: 45.2%
tiger cat: 23.1%
Egyptian cat: 18.7%
Persian cat: 5.3%
lynx: 2.1%
```

## You Did It!

You just:
- Loaded a pre-trained ImageNet model
- Applied standard image preprocessing
- Ran inference on GPU
- Decoded predictions to class labels

In the full module, you'll learn:
- Building CNNs from scratch (LeNet, AlexNet, ResNet)
- Transfer learning for custom datasets
- Object detection with YOLOv8
- Vision Transformers (ViT)
- Segment Anything Model (SAM)

## Next Steps
1. **Build from scratch**: Start with [Lab 2.2.1](./labs/lab-2.2.1-cnn-architecture-study.ipynb)
2. **Try your own image**: Replace `cat.jpg` with any image
3. **Full setup**: See [LAB_PREP.md](./LAB_PREP.md) for complete environment
