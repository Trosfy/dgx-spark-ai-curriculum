# Module 2.1: Deep Learning with PyTorch

**Domain:** 2 - Deep Learning Frameworks
**Duration:** Weeks 8-9 (12-15 hours)
**Prerequisites:** Domain 1 complete (Module 1.7 Capstone)

---

## Overview

Now that you understand neural networks from scratch, it's time to use professional tools. This module covers PyTorch in depth—from custom modules to advanced training techniques. You'll learn to build, debug, and profile models effectively.

---

## Learning Outcomes

By the end of this module, you will be able to:

- ✅ Build complex neural networks using PyTorch's nn.Module
- ✅ Implement custom datasets and data loaders
- ✅ Utilize PyTorch's autograd for custom operations
- ✅ Debug and profile PyTorch models effectively

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 2.1.1 | Create custom nn.Module classes with proper initialization | Apply |
| 2.1.2 | Implement Dataset and DataLoader for custom data | Apply |
| 2.1.3 | Implement custom autograd functions and use hooks for introspection | Apply |
| 2.1.4 | Use mixed precision training with AMP for memory and speed gains | Apply |
| 2.1.5 | Profile models with PyTorch Profiler to identify bottlenecks | Analyze |
| 2.1.6 | Implement robust checkpointing with early stopping | Apply |

---

## Topics

### 2.1.1 PyTorch Fundamentals
- Tensors and operations
- Autograd mechanics
- GPU memory management
- Mixed precision with AMP

### 2.1.2 Building Models
- nn.Module architecture
- Sequential vs functional API
- Parameter registration
- State dict and checkpointing

### 2.1.3 Data Pipeline
- Dataset class implementation
- DataLoader with workers
- Transforms and augmentation
- Efficient data loading on DGX Spark

### 2.1.4 Training Infrastructure
- Training loops
- Validation and metrics
- Learning rate scheduling
- Gradient clipping and accumulation

---

## Labs

### Lab 2.1.1: Custom Module Lab
**Time:** 2 hours

Implement ResNet building blocks.

**Instructions:**
1. Implement `BasicBlock` (two 3x3 convs with skip connection)
2. Implement `Bottleneck` (1x1 → 3x3 → 1x1 with skip)
3. Stack blocks to create ResNet-18
4. Test on CIFAR-10

**Deliverable:** Working ResNet-18 implementation

---

### Lab 2.1.2: Dataset Pipeline
**Time:** 2 hours

Create efficient data loading.

**Instructions:**
1. Create custom `Dataset` for a local image folder
2. Implement data augmentation transforms
3. Create `DataLoader` with multiple workers
4. Benchmark loading speed
5. Optimize for DGX Spark (find optimal num_workers, batch_size)

**Deliverable:** Optimized data pipeline with benchmarks

---

### Lab 2.1.3: Autograd Deep Dive
**Time:** 2 hours

Create a custom autograd function.

**Instructions:**
1. Implement a novel activation (e.g., Swish, Mish)
2. Use `torch.autograd.Function` with custom forward/backward
3. Verify gradients with `torch.autograd.gradcheck`
4. Benchmark against built-in version

**Deliverable:** Custom autograd function with verified gradients

---

### Lab 2.1.4: Mixed Precision Training
**Time:** 2 hours

Use Automatic Mixed Precision (AMP).

**Instructions:**
1. Train baseline model in FP32
2. Convert to mixed precision with `torch.cuda.amp`
3. Compare memory usage
4. Compare training speed
5. Verify accuracy is maintained

**Deliverable:** AMP training notebook with comparisons

---

### Lab 2.1.5: Profiling Workshop
**Time:** 2 hours

Profile and optimize training.

**Instructions:**
1. Profile a training loop with `torch.profiler`
2. Generate Chrome trace
3. Identify bottlenecks (CPU, GPU, data loading)
4. Apply optimizations
5. Measure improvement

**Deliverable:** Profiling report with optimizations

---

### Lab 2.1.6: Checkpointing System
**Time:** 2 hours

Implement robust checkpointing.

**Instructions:**
1. Save/load complete training state
2. Implement best model tracking
3. Implement early stopping
4. Test interrupt and resume
5. Handle optimizer state correctly

**Deliverable:** Reusable checkpointing module

---

## Guidance

### DGX Spark Memory Tips

```python
# With 128GB unified memory, you can use larger batches
# Start with batch_size=64, increase until slowdown

# Clear cache between experiments
import gc
torch.cuda.empty_cache()
gc.collect()

# Monitor memory
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
```

### NGC Container for PyTorch

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

> **Important flags:**
> - `--gpus all`: Required to access GPU
> - `--ipc=host`: **Required** when using `num_workers > 0` in DataLoader. Without it, you'll get "unable to open shared memory" errors because PyTorch workers use shared memory for inter-process communication.
> - `-p 8888:8888`: Maps container port 8888 to host port 8888 for Jupyter access

> **Note:** The container tag (`25.11-py3`) may need updating. Check [NGC PyTorch Containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) for the latest version compatible with DGX Spark.

### Custom nn.Module Template

```python
class MyModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()
        
        # Initialize weights
        nn.init.kaiming_normal_(self.linear.weight)
    
    def forward(self, x):
        return self.activation(self.linear(x))
```

### Mixed Precision Training

> **DGX Spark Recommendation:** Use **BFloat16** (`torch.bfloat16`) instead of Float16 on DGX Spark with Blackwell GPU. BF16 has the same dynamic range as FP32, avoiding overflow/underflow issues and eliminating the need for gradient scaling.

```python
# PyTorch 2.0+ API - BFloat16 (Recommended for DGX Spark)
from torch.amp import autocast, GradScaler

# BF16 doesn't need gradient scaling
scaler = GradScaler('cuda', enabled=False)

for batch in dataloader:
    optimizer.zero_grad()

    # Use bfloat16 for Blackwell GPU native support
    with autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model(batch)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

For Float16 (legacy or comparison), use `dtype=torch.float16` with `GradScaler('cuda', enabled=True)` to prevent gradient underflow.

### Profiling

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for batch in dataloader:
        output = model(batch)
        loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("trace.json")
```

---

## Milestone Checklist

- [ ] ResNet-18 implemented with custom blocks
- [ ] Custom dataset pipeline optimized for DGX Spark
- [ ] Custom autograd function with verified gradients
- [ ] AMP training with memory/speed comparison
- [ ] Profiling report with identified bottlenecks
- [ ] Checkpointing system with resume capability

---

## Next Steps

After completing this module:
1. ✅ Verify all milestones are checked
2. 📁 Save reusable modules to `scripts/`
3. ➡️ Proceed to [Module 2.2: Computer Vision](../module-2.2-computer-vision/)

---

## Module Navigation

| Previous | Current | Next |
|----------|---------|------|
| [Module 1.7: Capstone — MicroGrad+](../../domain-1-platform-foundations/module-1.7-capstone-micrograd/) | **Module 2.1: PyTorch** | [Module 2.2: Computer Vision](../module-2.2-computer-vision/) |

---

## Resources

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
