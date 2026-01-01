# Module 1.1: DGX Spark Platform Mastery - Troubleshooting Guide

## 🔍 Quick Diagnostic

**Before diving into specific errors, try these:**
1. Check GPU status: `nvidia-smi`
2. Check Docker GPU access: `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi`
3. Check Ollama: `curl http://localhost:11434/api/tags`
4. Clear memory: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`

---

## 🚨 Error Categories

### PyTorch / CUDA Errors

#### Error: `torch.cuda.is_available()` returns `False`

**Symptoms**:
```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**Causes**:
1. Using pip-installed PyTorch instead of NGC container
2. Docker container started without `--gpus all`
3. NVIDIA driver issue

**Solutions**:

```bash
# Solution 1: Use NGC container (MOST COMMON FIX)
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.11-py3

# Solution 2: Verify docker run command includes GPU flag
# ❌ Wrong:
docker run -it nvcr.io/nvidia/pytorch:25.11-py3

# ✅ Correct:
docker run --gpus all -it nvcr.io/nvidia/pytorch:25.11-py3

# Solution 3: Check NVIDIA runtime
docker info | grep -i runtime
# Should show: Runtimes: nvidia runc
```

**Prevention**: Never use `pip install torch` on DGX Spark. Always use NGC containers.

---

#### Error: `CUDA out of memory`

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate X GiB
```

**Causes**:
1. Linux buffer cache consuming memory
2. Previous model still loaded
3. Attempting to load model too large for available memory

**Solutions**:

```bash
# Solution 1: Clear Linux buffer cache
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

```python
# Solution 2: Clear PyTorch cache
import torch
import gc

# Delete any large tensors/models
del model  # or whatever variable holds your model

# Clear caches
torch.cuda.empty_cache()
gc.collect()

# Verify memory freed
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

```python
# Solution 3: Load model with lower precision
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Half the memory of float32
    device_map="auto"
)
```

**Prevention**: Always clear buffer cache before loading large models.

---

#### Error: `No CUDA GPUs are available`

**Symptoms**:
```
RuntimeError: No CUDA GPUs are available
```

**Solutions**:
```bash
# Check if GPU is visible
nvidia-smi

# If nvidia-smi fails, check driver
cat /proc/driver/nvidia/version

# Restart NVIDIA services if needed
sudo systemctl restart nvidia-persistenced
```

---

### Docker Errors

#### Error: Docker can't access GPU

**Symptoms**:
```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

**Solutions**:
```bash
# Install NVIDIA Container Toolkit (usually pre-installed on DGX Spark)
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify installation
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

---

#### Error: Container exits immediately

**Symptoms**:
Container starts and immediately stops.

**Solutions**:
```bash
# Add interactive flags
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.11-py3

# Check container logs
docker logs <container_id>

# Ensure you're using bash or a command
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.11-py3 bash
```

---

#### Error: Permission denied on mounted volumes

**Symptoms**:
```
PermissionError: [Errno 13] Permission denied: '/workspace/...'
```

**Solutions**:
```bash
# Check ownership of host directory
ls -la $HOME/workspace

# Fix permissions
sudo chown -R $(id -u):$(id -g) $HOME/workspace

# Or run container with your user ID
docker run --gpus all -it --rm \
    --user $(id -u):$(id -g) \
    -v $HOME/workspace:/workspace \
    nvcr.io/nvidia/pytorch:25.11-py3
```

---

### Ollama Errors

#### Error: Ollama not responding

**Symptoms**:
```
curl: (7) Failed to connect to localhost port 11434: Connection refused
```

**Solutions**:
```bash
# Check service status
systemctl status ollama

# Start if not running
sudo systemctl start ollama

# Enable auto-start
sudo systemctl enable ollama

# Check if port is in use
sudo lsof -i :11434
```

---

#### Error: Model download fails

**Symptoms**:
```
Error: pull model manifest: Get "https://...": dial tcp: lookup registry.ollama.ai: no such host
```

**Solutions**:
```bash
# Check internet connectivity
ping -c 3 google.com

# Check DNS
nslookup registry.ollama.ai

# Try with explicit DNS
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf

# Retry pull
ollama pull llama3.1:8b
```

---

#### Error: Ollama OOM (Out of Memory)

**Symptoms**:
Model fails to load or generates very slowly.

**Solutions**:
```bash
# Clear memory first
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# Unload any currently loaded model
curl http://localhost:11434/api/generate -d '{"model": "llama3.1:8b", "keep_alive": 0}'

# Use quantized model variant
ollama pull llama3.1:8b-q4_0  # Smaller quantization
```

---

### Import / Dependency Errors

#### Error: `ModuleNotFoundError: No module named 'xxx'`

**Symptoms**:
```
ModuleNotFoundError: No module named 'transformers'
```

**Solutions**:
```bash
# Inside NGC container, install package
pip install transformers

# For packages needing specific versions
pip install transformers>=4.40.0

# For persistent installs, add to requirements.txt
echo "transformers>=4.40.0" >> /workspace/requirements.txt
pip install -r /workspace/requirements.txt
```

---

#### Error: `pip install` fails on ARM64

**Symptoms**:
```
ERROR: Could not find a version that satisfies the requirement xxx
ERROR: No matching distribution found for xxx
```

**Causes**:
Package doesn't have ARM64 (aarch64) wheels.

**Solutions**:
```bash
# Option 1: Use conda inside container
conda install -c conda-forge package_name

# Option 2: Build from source
pip install package_name --no-binary :all:

# Option 3: Use NGC container that includes the package
docker pull nvcr.io/nvidia/pytorch:25.11-py3  # Includes most common packages
```

---

### Memory Issues

#### Issue: System feels slow / unresponsive

**Causes**:
Buffer cache consuming too much memory.

**Solutions**:
```bash
# Check memory usage
free -h

# Clear buffer cache
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# Check again
free -h
```

---

#### Issue: Memory usage doesn't decrease after deleting model

**Causes**:
PyTorch caches GPU memory for reuse.

**Solutions**:
```python
import torch
import gc

# Delete the variable
del model

# Force garbage collection
gc.collect()

# Clear PyTorch cache
torch.cuda.empty_cache()

# Verify
print(f"Now allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

---

## 🔄 Reset Procedures

### Full Environment Reset
```bash
# 1. Exit all containers
docker stop $(docker ps -aq) 2>/dev/null

# 2. Clear HuggingFace cache (if corrupted)
rm -rf ~/.cache/huggingface/hub/models--*

# 3. Clear buffer cache
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# 4. Restart Ollama
sudo systemctl restart ollama

# 5. Start fresh container
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    nvcr.io/nvidia/pytorch:25.11-py3
```

### GPU Reset (If GPU Appears Hung)
```bash
# Check for stuck processes
nvidia-smi

# Kill stuck GPU processes (use with caution)
sudo nvidia-smi --gpu-reset

# If that fails, reboot
sudo reboot
```

### Memory-Only Reset
```python
import torch
import gc

# Clear all CUDA memory
torch.cuda.empty_cache()
gc.collect()

# Verify
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

---

## 📊 Diagnostic Commands

### Quick System Health Check
```bash
#!/bin/bash
echo "=== DGX Spark Health Check ==="

echo -e "\n--- GPU Status ---"
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv

echo -e "\n--- Memory Status ---"
free -h

echo -e "\n--- Docker Status ---"
docker info 2>/dev/null | grep -E "Server Version|Runtimes"

echo -e "\n--- Ollama Status ---"
systemctl is-active ollama

echo -e "\n--- Disk Space ---"
df -h / $HOME

echo "=== Check Complete ==="
```

---

## ❓ Frequently Asked Questions

### Setup & Environment

**Q: Why can't I just `pip install torch` on DGX Spark?**

DGX Spark uses ARM64 (aarch64) architecture, not x86_64. Standard PyPI PyTorch wheels are only built for x86_64. NGC containers include PyTorch pre-built for ARM64 with GPU support.

---

**Q: Do I need to pull a new container for every module?**

No. The PyTorch NGC container (`nvcr.io/nvidia/pytorch:25.11-py3`) works for most modules. You only need different containers for:
- RAPIDS/cuML (Module 1.6): Use `nvcr.io/nvidia/rapidsai/base:25.11-py3`
- Some specialized tools

---

**Q: How do I keep my installed packages between container sessions?**

Use a requirements file or bind mount a pip cache:

```bash
# Option 1: Requirements file
echo "transformers>=4.40.0" > $HOME/workspace/requirements.txt
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    bash -c "pip install -r /workspace/requirements.txt && bash"

# Option 2: Persistent pip cache
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $HOME/.cache/pip:/root/.cache/pip \
    --ipc=host \
    nvcr.io/nvidia/pytorch:25.11-py3
```

---

**Q: Why is `--ipc=host` recommended in docker run commands?**

The `--ipc=host` flag shares the host's IPC namespace with the container. This is needed for:
- PyTorch DataLoader with `num_workers > 0` (uses shared memory)
- Multi-process training
- Large batch training

Without it, you may see "DataLoader worker exited unexpectedly" errors.

---

### Hardware Concepts

**Q: What's the difference between unified memory and regular GPU memory?**

| Aspect | Unified (DGX Spark) | Discrete (RTX/A100) |
|--------|---------------------|---------------------|
| Memory pool | Shared 128GB | Separate (e.g., 24GB GPU + 64GB RAM) |
| Transfers | Automatic, no copies | Explicit `.to(device)` copies |
| Max model size | ~128GB | Limited to GPU VRAM |
| Bandwidth | 273 GB/s shared | GPU: 900+ GB/s, PCIe: ~32 GB/s |

---

**Q: What do NVFP4, FP8, and BF16 mean?**

These are numerical precision formats:

| Format | Bits | Best For | DGX Spark Speed |
|--------|------|----------|-----------------|
| NVFP4 | 4 | Inference only | 1 PFLOP |
| FP8 | 8 | Inference | ~209 TFLOPS |
| BF16 | 16 | Training + Inference | ~100 TFLOPS |
| FP32 | 32 | Debugging, legacy | ~31 TFLOPS |

Use BF16 for training, FP8/NVFP4 for inference when accuracy permits.

---

**Q: How big of a model can I run on DGX Spark?**

Depends on precision and whether you're training or inferring:

| Task | Precision | Approximate Max |
|------|-----------|-----------------|
| Inference | FP16 | 50-55B parameters |
| Inference | FP8 | 90-100B parameters |
| Inference | NVFP4 | ~200B parameters |
| Fine-tuning (full) | FP16 | 12-16B parameters |
| Fine-tuning (QLoRA) | 4-bit | 100-120B parameters |

---

### Ollama & Models

**Q: Why use Ollama API calls instead of the Web UI for benchmarks?**

The Web UI adds rendering overhead that skews timing measurements. Direct API calls give you:
- Exact token counts (`prompt_eval_count`, `eval_count`)
- Precise timing in nanoseconds (`prompt_eval_duration`, `eval_duration`)
- No UI latency

---

**Q: How do I unload a model from Ollama to free memory?**

Send a generate request with `keep_alive: 0`:

```bash
curl http://localhost:11434/api/generate -d '{"model": "llama3.1:8b", "keep_alive": 0}'
```

Or wait for the default timeout (5 minutes).

---

**Q: Which Ollama model variants should I download?**

For this module, download at minimum:

| Model | Size | Required? |
|-------|------|-----------|
| `llama3.2:3b` | ~2 GB | Yes - quick tests |
| `llama3.1:8b` | ~5 GB | Yes - main benchmark |
| `llama3.1:70b` | ~45 GB | Recommended - shows unified memory advantage |

---

### Memory Management

**Q: Why do I need to "clear buffer cache" before loading models?**

Linux aggressively caches file data in RAM for faster access. On DGX Spark, this cache competes with GPU allocations. Clearing it frees that memory for your model. This is especially important before loading 70B+ models.

---

**Q: Why does `nvidia-smi` show less memory than 128GB?**

You may see ~126-127 GB usable because:
1. Small amount reserved for GPU firmware/ECC
2. Display buffers (if using GUI)
3. System overhead

This is normal. You still have access to unified memory.

---

**Q: My Python session shows memory but I can't allocate more tensors. Why?**

PyTorch caches GPU memory for reuse. Run:

```python
import torch, gc
torch.cuda.empty_cache()
gc.collect()
```

If still stuck, also clear buffer cache from terminal.

---

### Compatibility

**Q: Can I use vLLM on DGX Spark?**

Partial support. vLLM has ARM64 support but requires the `--enforce-eager` flag. Current status:
- Basic inference: Works with `--enforce-eager` flag
- Full features: Some may not work
- Workaround: `vllm serve model --enforce-eager`
- Recommendation: Use Ollama, SGLang, or TensorRT-LLM for production (SGLang is 29-45% faster)

---

**Q: Why is my favorite tool marked as "not compatible"?**

Common reasons:
1. **No ARM64 support**: Tool only has x86 binaries
2. **CUDA version mismatch**: Requires different CUDA than DGX Spark
3. **Architecture assumptions**: Hardcoded x86 assembly or instructions

Check if there's:
- An NGC container version
- A conda-forge ARM64 package
- Source code you can build

---

**Q: Can I run Windows or x86 Linux containers on DGX Spark?**

No. DGX Spark runs ARM64 Linux. You can only run ARM64 Linux containers. This is why NGC containers are essential—they're built for ARM64.

---

### Performance & Best Practices

**Q: How does DGX Spark compare to cloud GPU instances?**

| Aspect | DGX Spark | Cloud (A100 80GB) |
|--------|-----------|-------------------|
| Memory | 128GB unified | 80GB GPU + separate RAM |
| Cost | One-time purchase | Hourly (~$2-4/hr) |
| Max model (inference) | ~100B FP8 | ~40B FP16 |
| Best for | Local development, privacy | Burst capacity, training |

---

**Q: Can I use DGX Spark for training, not just inference?**

Yes, but with limits:
- **Full fine-tuning**: Models up to ~12-16B parameters
- **LoRA/QLoRA**: Models up to ~100B parameters
- **From-scratch training**: Good for experimentation, not production scale

For training large models from scratch, cloud or data center GPUs are more appropriate.

---

**Q: How do I maximize performance on DGX Spark?**

1. Use BF16 for training (not FP32)
2. Use FP8/NVFP4 quantization for inference
3. Use NGC containers (optimized for ARM64)
4. Clear buffer cache before large operations
5. Use batch sizes that fit in memory (larger = faster per-sample)

---

## 📞 Still Stuck?

1. **Check the lab notebook comments** - Often contain hints for specific issues
2. **Review prerequisites** - Missing foundation knowledge?
3. **Search error message** - Others may have encountered it
4. **Check NGC release notes** - Container-specific issues documented
5. **Ask with context** - Include: error message, command run, what you tried

### Information to Gather for Help
```bash
# Run this and share output when asking for help
echo "=== Debug Info ==="
nvidia-smi -L
docker --version
cat /etc/os-release | grep PRETTY_NAME
uname -m
python3 --version
echo "CUDA version:" && nvcc --version 2>/dev/null | tail -1 || echo "nvcc not in PATH"
```
