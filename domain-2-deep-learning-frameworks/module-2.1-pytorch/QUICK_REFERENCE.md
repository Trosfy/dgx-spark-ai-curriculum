# Module 2.1: Deep Learning with PyTorch - Quick Reference

## Essential Commands

### GPU and Memory

```python
import torch
import gc

# Check GPU
torch.cuda.is_available()              # True if GPU accessible
torch.cuda.get_device_name(0)          # Device name
torch.cuda.device_count()              # Number of GPUs

# Memory monitoring
torch.cuda.memory_allocated() / 1e9    # GB allocated
torch.cuda.memory_reserved() / 1e9     # GB reserved (cached)
torch.cuda.max_memory_allocated() / 1e9  # Peak usage

# Clear memory
torch.cuda.empty_cache()
gc.collect()

# Memory summary
print(torch.cuda.memory_summary())
```

### Model Operations

```python
# Create model
model = MyModel().cuda()                # Move to GPU

# Count parameters
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Save/Load
torch.save(model.state_dict(), 'model.pt')
model.load_state_dict(torch.load('model.pt'))

# Training/Eval mode
model.train()   # Enable dropout, batch norm training mode
model.eval()    # Disable dropout, batch norm eval mode
```

---

## Key Patterns

### Pattern: Custom nn.Module

```python
import torch.nn as nn

class MyBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Initialize weights
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out')

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out
```

### Pattern: ResNet Skip Connection

```python
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection with projection if dimensions change
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Skip connection
        return F.relu(out)
```

### Pattern: Custom Dataset

```python
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
```

### Pattern: Optimized DataLoader

```python
from torch.utils.data import DataLoader

# DGX Spark optimized settings
loader = DataLoader(
    dataset,
    batch_size=64,          # Start here, increase if memory allows
    shuffle=True,           # For training
    num_workers=8,          # 8-16 optimal for DGX Spark
    pin_memory=True,        # Faster GPU transfer
    prefetch_factor=2,      # Prefetch batches
    persistent_workers=True # Keep workers alive between epochs
)
```

### Pattern: Mixed Precision Training (BF16)

```python
from torch.amp import autocast, GradScaler

# BF16 on DGX Spark (no scaler needed)
model = model.cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, targets = batch[0].cuda(), batch[1].cuda()

        optimizer.zero_grad()

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
```

### Pattern: Custom Autograd Function

```python
class CustomActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Swish: x * sigmoid(x)
        sigmoid_x = torch.sigmoid(x)
        ctx.save_for_backward(x, sigmoid_x)
        return x * sigmoid_x

    @staticmethod
    def backward(ctx, grad_output):
        x, sigmoid_x = ctx.saved_tensors
        # d/dx (x * sigmoid(x)) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        grad = sigmoid_x + x * sigmoid_x * (1 - sigmoid_x)
        return grad_output * grad

# Usage
swish = CustomActivation.apply
output = swish(input_tensor)
```

### Pattern: Profiling

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    # Run training step(s)
    for _ in range(5):
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Print summary
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export for Chrome trace viewer
prof.export_chrome_trace("trace.json")
```

### Pattern: Complete Checkpointing

```python
def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(path, model, optimizer, scheduler):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']
```

---

## Key Values to Remember

| What | Value | Notes |
|------|-------|-------|
| DGX Spark Memory | 128 GB | Unified LPDDR5X |
| Typical batch_size start | 64 | Increase until slowdown |
| DataLoader num_workers | 8-16 | Test for your workload |
| BF16 memory savings | ~50% | vs FP32 |
| Tensor Cores generation | 5th | Native BF16 support |
| CUDA cores | 6,144 | Blackwell GB10 |

---

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Forgetting `.cuda()` | Model and data must both be on GPU |
| Not using `model.eval()` for inference | Disables dropout, BN training mode |
| `num_workers=0` bottleneck | Use `num_workers=8+` with `--ipc=host` |
| Forgetting `optimizer.zero_grad()` | Gradients accumulate by default |
| Using FP16 scaler with BF16 | BF16 doesn't need scaling |
| Not saving optimizer state | Can't properly resume training |
| Missing `torch.no_grad()` for inference | Wastes memory on gradient computation |

---

## Quick Links

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
- [AMP Documentation](https://pytorch.org/docs/stable/amp.html)
- [Profiler Guide](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
