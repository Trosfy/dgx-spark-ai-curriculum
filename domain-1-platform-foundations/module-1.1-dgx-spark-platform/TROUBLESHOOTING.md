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
