# Module 1.1: DGX Spark Platform Mastery - Quickstart

## ⏱️ Time: ~5 minutes

## 🎯 What You'll Do
Verify your DGX Spark is working and see GPU stats in action.

## ✅ Before You Start
- [ ] DGX Spark powered on and accessible
- [ ] Terminal or SSH access to the system

## 🚀 Let's Go!

### Step 1: Check GPU Status
```bash
nvidia-smi
```

**You should see:**
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.xxx       Driver Version: 560.xxx       CUDA Version: 13.0              |
|-----------------------------------------------------------------------------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+======================+======================|
|   0  NVIDIA Graphics Device         On | 00000009:01:00.0 Off |                    0 |
| N/A   45C    P0              25W / 100W |       0MiB / 131072MiB |      0%      Default |
+-----------------------------------------------------------------------------------------+
```

### Step 2: Start PyTorch Container
```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    nvcr.io/nvidia/pytorch:25.11-py3
```

### Step 3: Verify CUDA in Python
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

**Expected output:**
```
CUDA available: True
Device: NVIDIA Graphics Device
Memory: 128.0 GB
```

### Step 4: Quick Memory Test
```python
# Allocate a 10GB tensor - only possible with 128GB unified memory!
x = torch.zeros(10 * 1024**3 // 4, dtype=torch.float32, device='cuda')
print(f"Allocated: {x.numel() * 4 / 1e9:.1f} GB")
del x
torch.cuda.empty_cache()
```

**Expected output:**
```
Allocated: 10.0 GB
```

## 🎉 You Did It!

You just verified:
- ✅ GPU is accessible (nvidia-smi works)
- ✅ NGC container runs with GPU support
- ✅ PyTorch sees 128GB unified memory
- ✅ You can allocate large tensors

In the full module, you'll learn:
- Grace Blackwell GB10 architecture details
- How unified memory differs from discrete GPUs
- Which AI tools are compatible with DGX Spark
- How to benchmark models with Ollama

## ▶️ Next Steps
1. **Understand the hardware**: Read [STUDY_GUIDE.md](./STUDY_GUIDE.md)
2. **Set up your environment**: Follow [LAB_PREP.md](./LAB_PREP.md)
3. **Start Lab 1**: Open `labs/lab-1.1.1-system-exploration.ipynb`
