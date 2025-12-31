# Module 2.2: Computer Vision - Prerequisites Check

## Purpose
This module builds directly on Module 2.1 (PyTorch). Use this self-check to ensure you're ready.

## Estimated Time
- **If all prerequisites met**: Start with [QUICKSTART.md](./QUICKSTART.md)
- **If 1-2 gaps**: ~1-2 hours of review
- **If multiple gaps**: Complete Module 2.1 first

---

## Required Skills

### 1. PyTorch: nn.Module

**Can you write this without looking anything up?**
```python
# Create a simple two-layer neural network using nn.Module
class TwoLayerNet(nn.Module):
    # Your implementation here
    pass
```

<details>
<summary>Check your answer</summary>

```python
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.layer1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden, out_features)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
```

**Key points**:
- `super().__init__()` is required
- Layers defined in `__init__`, used in `forward`
- Return the output from `forward`

</details>

**Not ready?** Review: Module 2.1, Lab 2.1.1

---

### 2. PyTorch: DataLoader and Transforms

**Can you do this?**
```python
# Load CIFAR-10 with transforms and create a DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Create transform, dataset, and dataloader
```

<details>
<summary>Check your answer</summary>

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

**Key points**:
- `transforms.Compose` chains multiple transforms
- `ToTensor()` converts PIL Image to tensor (H,W,C → C,H,W)
- `Normalize` standardizes values

</details>

**Not ready?** Review: Module 2.1, Lab 2.1.2

---

### 3. Convolutions: Basic Understanding

**Can you answer this?**
> What does a 3x3 convolution do? What is the output shape if input is 32x32 with padding=1?

<details>
<summary>Check your answer</summary>

**What a 3x3 convolution does**:
- Slides a 3x3 filter across the image
- At each position, element-wise multiply and sum
- Produces one output value per position
- Detects local patterns (edges, textures)

**Output shape with padding=1**:
- Input: 32x32
- Padding=1 adds 1 pixel border → 34x34
- 3x3 kernel with stride=1: output = 34 - 3 + 1 = 32
- Output: 32x32 (same as input!)

**Formula**: `output_size = (input_size + 2*padding - kernel_size) / stride + 1`

</details>

**Not ready?** Quick read: [CS231n Convolution Section](http://cs231n.github.io/convolutional-networks/)

---

### 4. Training Loop: Complete Flow

**Can you write a basic training loop?**
```python
# Write a training loop for one epoch
model.train()
for batch in dataloader:
    # Your implementation
    pass
```

<details>
<summary>Check your answer</summary>

```python
model.train()
for images, labels in dataloader:
    images, labels = images.cuda(), labels.cuda()

    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Key points**:
- `model.train()` enables training mode
- Move data to GPU
- `optimizer.zero_grad()` before backward
- `loss.backward()` computes gradients
- `optimizer.step()` updates parameters

</details>

**Not ready?** Review: Module 2.1, Lab 2.1.4

---

### 5. Mixed Precision: BF16 Usage

**Do you know how to use BF16 on DGX Spark?**

<details>
<summary>Check your answer</summary>

```python
from torch.amp import autocast

# BF16 training (native on DGX Spark Blackwell GPU)
with autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)

loss.backward()  # Outside autocast
optimizer.step()
```

**Key points**:
- `dtype=torch.bfloat16` for DGX Spark
- No GradScaler needed for BF16
- Backward pass can be outside autocast

</details>

**Not ready?** Review: Module 2.1, Lab 2.1.4

---

## Terminology Check

Do you know these terms?

| Term | Your Definition |
|------|-----------------|
| Convolution | |
| Feature map | |
| Pooling | |
| Stride | |
| Padding | |
| Receptive field | |

<details>
<summary>Check definitions</summary>

| Term | Definition |
|------|------------|
| **Convolution** | Operation that slides a filter across an image, computing element-wise products and sums |
| **Feature map** | Output of a convolutional layer; represents detected features at each location |
| **Pooling** | Downsampling operation (e.g., max pooling takes max value in each region) |
| **Stride** | Step size of the convolution; stride=2 halves the spatial dimensions |
| **Padding** | Adding pixels around the border (usually zeros) to control output size |
| **Receptive field** | Region of input that influences a particular output value |

</details>

---

## Optional But Helpful

### Linear Algebra: Matrix Operations
**Why it helps**: Convolutions are matrix operations; understanding helps with debugging
**Key concepts**: Matrix multiplication, element-wise operations

### Image Processing Basics
**Why it helps**: Understanding RGB, channels, resolution
**Quick primer**: Images are 3D tensors (C, H, W) - channels, height, width

---

## Ready Checklist

- [ ] I can write a custom nn.Module
- [ ] I understand DataLoader and transforms
- [ ] I know what convolutions do
- [ ] I can write a training loop
- [ ] I know how to use BF16 on DGX Spark
- [ ] My environment is set up (see [LAB_PREP.md](./LAB_PREP.md))

**All boxes checked?** Start with [QUICKSTART.md](./QUICKSTART.md)!

**Some gaps?** Review Module 2.1 first - it's the foundation for everything here.
