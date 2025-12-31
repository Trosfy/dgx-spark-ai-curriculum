# Module 2.1: Deep Learning with PyTorch - Quickstart

## Time: ~5 minutes

## What You'll Build
A simple neural network trained on MNIST with GPU acceleration and mixed precision.

## Before You Start
- [ ] DGX Spark container running (`docker run --gpus all ...`)
- [ ] GPU accessible (`nvidia-smi` shows GPU)

## Let's Go!

### Step 1: Verify PyTorch GPU Access
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

### Step 2: Create a Simple Model
```python
import torch.nn as nn

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
).cuda()

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Step 3: Load MNIST Data
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
```

### Step 4: Train with Mixed Precision (BF16)
```python
from torch.amp import autocast

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# One epoch with BF16 (native Blackwell support)
model.train()
for images, labels in loader:
    images, labels = images.cuda(), labels.cuda()

    with autocast(device_type='cuda', dtype=torch.bfloat16):
        outputs = model(images)
        loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Training complete! Final loss: {loss.item():.4f}")
```

### Step 5: Check Accuracy
```python
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in loader:
        outputs = model(images.cuda())
        _, predicted = outputs.max(1)
        correct += (predicted == labels.cuda()).sum().item()
        total += labels.size(0)

print(f"Accuracy: {100*correct/total:.1f}%")
```

## You Did It!

You just:
- Loaded data with DataLoader (multi-worker)
- Built a neural network with nn.Sequential
- Trained with BFloat16 mixed precision (DGX Spark optimized)
- Evaluated accuracy

In the full module, you'll learn:
- Custom nn.Module classes (ResNet blocks)
- Custom Dataset implementations
- Custom autograd functions
- Profiling with torch.profiler
- Robust checkpointing systems

## Next Steps
1. **Understand the components**: Start with [Lab 2.1.1](./labs/lab-2.1.1-custom-module-lab.ipynb)
2. **Try variations**: Change hidden size to 512, add another layer
3. **Full setup**: See [LAB_PREP.md](./LAB_PREP.md) for complete environment
