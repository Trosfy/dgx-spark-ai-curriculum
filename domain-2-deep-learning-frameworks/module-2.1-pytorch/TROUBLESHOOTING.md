# Module 2.1: Deep Learning with PyTorch - Troubleshooting Guide

## Quick Diagnostic

**Before diving into specific errors, try these:**

1. Check GPU access: `torch.cuda.is_available()`
2. Check GPU memory: `nvidia-smi` or `torch.cuda.memory_summary()`
3. Clear cache: `torch.cuda.empty_cache(); gc.collect()`
4. Restart kernel/container
5. Verify you're in correct directory

---

## Memory Errors

### Error: `CUDA out of memory`

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate X GiB
(Y GiB allocated; Z GiB free; W GiB reserved)
```

**Causes**:
1. Model too large for available memory
2. Previous model/tensors still loaded
3. Batch size too high
4. Gradient accumulation without clearing

**Solutions**:

```python
# Solution 1: Clear memory
import torch
import gc
torch.cuda.empty_cache()
gc.collect()

# Solution 2: Reduce batch size
batch_size = 16  # Start small, increase gradually

# Solution 3: Use mixed precision (saves ~50% memory)
from torch.amp import autocast
with autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)

# Solution 4: Delete unused tensors
del old_tensor
torch.cuda.empty_cache()

# Solution 5: Use gradient checkpointing (trades compute for memory)
from torch.utils.checkpoint import checkpoint
out = checkpoint(self.layer, x, use_reentrant=False)
```

**Prevention**: Always clear memory before loading new models.

---

### Error: `RuntimeError: expected scalar type BFloat16 but found Float`

**Symptoms**:
```
RuntimeError: expected scalar type BFloat16 but found Float
```

**Cause**: Mixing precision types (some tensors BF16, some FP32).

**Solution**:
```python
# Ensure all tensors are same dtype within autocast
with autocast(device_type='cuda', dtype=torch.bfloat16):
    # Everything here should be BF16
    output = model(input.to(torch.bfloat16))  # If needed
```

Or convert model explicitly:
```python
model = model.to(torch.bfloat16)
```

---

## DataLoader Errors

### Error: `unable to open shared memory` / `Broken pipe`

**Symptoms**:
```
RuntimeError: unable to open shared memory object
```
or
```
BrokenPipeError: [Errno 32] Broken pipe
```

**Cause**: Container missing `--ipc=host` flag.

**Solution**:
```bash
# Restart container with --ipc=host
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    nvcr.io/nvidia/pytorch:25.11-py3
```

**Workaround** (if can't restart):
```python
# Use num_workers=0 (slower but works)
loader = DataLoader(dataset, batch_size=32, num_workers=0)
```

---

### Error: `DataLoader worker (pid X) exited unexpectedly`

**Symptoms**:
```
RuntimeError: DataLoader worker (pid 12345) exited unexpectedly
```

**Causes**:
1. Out of memory in worker process
2. Error in Dataset `__getitem__`
3. Missing `--ipc=host`

**Solutions**:
```python
# Solution 1: Reduce workers
loader = DataLoader(dataset, num_workers=4)  # Try fewer

# Solution 2: Add timeout
loader = DataLoader(dataset, num_workers=8, timeout=60)

# Solution 3: Debug with single worker
loader = DataLoader(dataset, num_workers=0)  # See actual error
```

---

## Model Loading Errors

### Error: `RuntimeError: Error(s) in loading state_dict`

**Symptoms**:
```
RuntimeError: Error(s) in loading state_dict for MyModel:
    Missing key(s) in state_dict: "layer.weight"
    Unexpected key(s) in state_dict: "module.layer.weight"
```

**Cause**: Model was saved with `DataParallel` wrapper or different architecture.

**Solutions**:
```python
# Solution 1: Remove "module." prefix
state_dict = torch.load('model.pt')
new_state_dict = {}
for k, v in state_dict.items():
    name = k.replace('module.', '')  # Remove "module." prefix
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)

# Solution 2: Load with strict=False (skip missing/extra keys)
model.load_state_dict(torch.load('model.pt'), strict=False)

# Solution 3: If saved as full model
model = torch.load('model_full.pt')  # Not recommended, but works
```

---

### Error: `ModuleNotFoundError: No module named 'scripts'`

**Symptoms**:
```
ModuleNotFoundError: No module named 'scripts'
```

**Cause**: Python can't find the scripts directory.

**Solution**:
```python
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path.cwd().parent))

# Now import works
from scripts import resnet_blocks
```

---

## Training Errors

### Error: `RuntimeError: one of the variables needed for gradient computation has been modified`

**Symptoms**:
```
RuntimeError: one of the variables needed for gradient computation
has been modified by an inplace operation
```

**Cause**: Inplace operation modified a tensor needed for backprop.

**Solutions**:
```python
# Bad: inplace operation
x += 1
x.add_(1)

# Good: create new tensor
x = x + 1
x = x.add(1)

# For ReLU, use inplace=False during debugging
nn.ReLU(inplace=False)  # Instead of inplace=True
```

---

### Error: `NaN loss` or `NaN gradients`

**Symptoms**:
```
Loss: nan
```

**Causes**:
1. Learning rate too high
2. Numerical overflow
3. Division by zero
4. Log of zero or negative

**Solutions**:
```python
# Solution 1: Lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Solution 2: Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Solution 3: Check for NaN and debug
if torch.isnan(loss):
    print("NaN detected!")
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"NaN gradient in {name}")

# Solution 4: Add epsilon for stability
loss = F.cross_entropy(output, target)  # More stable than manual
# or
log_probs = torch.log(probs + 1e-8)  # Avoid log(0)
```

---

### Error: Loss not decreasing

**Symptoms**: Loss stays constant or fluctuates randomly.

**Causes**:
1. Learning rate too low or too high
2. Forgot `optimizer.zero_grad()`
3. Model in eval mode during training
4. Labels not matching output format

**Solutions**:
```python
# Check 1: Are gradients flowing?
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad mean = {param.grad.mean():.6f}")

# Check 2: Proper training loop
model.train()  # Make sure training mode!
for batch in dataloader:
    optimizer.zero_grad()  # Don't forget!
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

# Check 3: Learning rate
# Try: 1e-3 (too high?), 1e-4 (common), 1e-5 (too low?)
```

---

## Profiling Errors

### Error: Chrome trace file not opening

**Symptoms**: `trace.json` won't open in Chrome.

**Solution**:
1. Open Chrome
2. Go to `chrome://tracing`
3. Click "Load" and select the trace file
4. If still failing, check file isn't empty:
```python
import os
print(f"Trace file size: {os.path.getsize('trace.json')} bytes")
```

---

### Error: Profiler shows no CUDA activity

**Symptoms**: Only CPU operations in profiler output.

**Causes**:
1. Model not on GPU
2. Data not on GPU
3. CUDA activity not recorded

**Solution**:
```python
# Ensure CUDA activities are recorded
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,  # Make sure this is included
    ]
) as prof:
    output = model(input.cuda())  # Ensure GPU execution

# Wait for CUDA ops to complete
torch.cuda.synchronize()
```

---

## Reset Procedures

### Full Environment Reset

```bash
# 1. Exit container
exit

# 2. Remove any cached data (if needed)
rm -rf ~/.cache/torch/

# 3. Restart container
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

### Memory-Only Reset

```python
import torch
import gc

# Clear all CUDA memory
torch.cuda.empty_cache()
gc.collect()

# Reset peak memory stats
torch.cuda.reset_peak_memory_stats()

# Verify
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

### Jupyter Kernel Reset

In Jupyter:
1. Kernel → Restart Kernel
2. Or: Kernel → Restart & Clear Output

---

## Still Stuck?

1. **Check the solution notebooks** - in `solutions/` folder
2. **Read the error carefully** - PyTorch errors often tell you exactly what's wrong
3. **Add print statements** - Print shapes, dtypes, devices
4. **Simplify** - Reduce to minimal reproducing example
5. **Search the error** - PyTorch forums, Stack Overflow often have solutions

**Useful debug snippet**:
```python
def debug_tensor(name, t):
    print(f"{name}: shape={t.shape}, dtype={t.dtype}, device={t.device}")
    if t.numel() > 0:
        print(f"  min={t.min():.4f}, max={t.max():.4f}, mean={t.float().mean():.4f}")
        print(f"  has_nan={torch.isnan(t).any()}, has_inf={torch.isinf(t).any()}")
```
