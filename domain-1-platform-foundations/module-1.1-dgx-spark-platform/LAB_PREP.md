# Module 1.1: DGX Spark Platform Mastery - Lab Preparation Guide

## ⏱️ Time Estimates

| Lab | Preparation | Execution | Total |
|-----|-------------|-----------|-------|
| Lab 1.1.1: System Exploration | 10 min | 50 min | 1 hr |
| Lab 1.1.2: Memory Architecture | 15 min | 75 min | 1.5 hr |
| Lab 1.1.3: NGC Container Setup | 20 min | 70 min | 1.5 hr |
| Lab 1.1.4: Compatibility Matrix | 10 min | 110 min | 2 hr |
| Lab 1.1.5: Ollama Benchmarking | 30 min | 90 min | 2 hr |

**Total module time**: ~8 hours

## 📦 Required Downloads

### NGC Container (Download Before Starting)
```bash
# PyTorch container (~15 GB, takes 10-20 minutes)
docker pull nvcr.io/nvidia/pytorch:25.11-py3
```

### Ollama Models - 2025 Tier 1 (Download Before Lab 5)
```bash
# Primary teaching models
ollama pull qwen3:8b              # ~5 GB - Fast, hybrid thinking
ollama pull qwen3:32b             # ~20 GB - Best quality, primary teaching

# Reasoning model
ollama pull qwq:32b               # ~20 GB - SOTA reasoning (79.5% AIME)

# Reasoning model with vision + tools (Tier 1 alternative)
ollama pull magistral-small       # ~15 GB - 86% AIME, multimodal reasoning

# Legacy/comparison (optional)
ollama pull llama3.1:8b           # ~5 GB - For performance comparison
```

**Total download size**: ~70 GB
**Estimated download time**: 30-60 minutes on fast connection

## 🔧 Environment Setup

### 1. Verify DGX Spark Access
```bash
# Check you can run nvidia-smi
nvidia-smi

# Expected: Shows NVIDIA Graphics Device with 128 GB memory
```

### 2. Check Docker Installation
```bash
# Docker should be pre-installed
docker --version
# Expected: Docker version 24.x or higher

# Verify GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### 3. Verify Ollama Service
```bash
# Check if Ollama is running
systemctl status ollama

# If not running, start it
sudo systemctl start ollama

# Test API
curl http://localhost:11434/api/tags
```

### 4. Create Workspace Directory
```bash
# Create workspace if it doesn't exist
mkdir -p $HOME/workspace/module-1.1
mkdir -p $HOME/.cache/huggingface
```

## ✅ Pre-Lab Checklists

### Lab 1.1.1: System Exploration
- [ ] Can run `nvidia-smi` successfully
- [ ] Can run `lscpu` and `free -h`
- [ ] Have terminal access (direct or SSH)
- [ ] Notebook file accessible: `labs/lab-1.1.1-system-exploration.ipynb`

### Lab 1.1.2: Memory Architecture
- [ ] NGC PyTorch container downloaded
- [ ] At least 100 GB GPU memory free
- [ ] Understand basic Python/PyTorch
- [ ] Ready to observe memory allocation patterns

### Lab 1.1.3: NGC Container Setup
- [ ] Docker installed and working with GPU
- [ ] Familiar with basic Docker commands
- [ ] Have write access to $HOME/workspace
- [ ] Notebook file accessible: `labs/lab-1.1.3-ngc-container-setup.ipynb`

### Lab 1.1.4: Compatibility Matrix
- [ ] Internet access for research
- [ ] List of 20 AI/ML tools to research
- [ ] Understanding of ARM64 vs x86_64 differences
- [ ] Ready to document findings

### Lab 1.1.5: Ollama Benchmarking
- [ ] Ollama service running
- [ ] At least qwen3:8b downloaded (or llama3.1:8b for comparison)
- [ ] Optional: llama3.1:70b for full benchmarks
- [ ] Python environment with `requests` library

## 🚫 Common Setup Mistakes

| Mistake | Consequence | Prevention |
|---------|-------------|------------|
| Using pip to install PyTorch | Import errors, no GPU access | Always use NGC container |
| Forgetting `--gpus all` | Container has no GPU access | Use the full docker run command |
| Not clearing buffer cache | OOM loading large models | Run cache clear command first |
| Ollama service not started | API connection refused | Check with systemctl |
| Wrong container version | Missing libraries/features | Use exact version: `25.11-py3` |

## 📁 Expected File Structure

After preparation, your workspace should look like:
```
$HOME/
├── workspace/
│   └── module-1.1/
│       ├── notebooks/     # Will contain your work
│       └── outputs/       # Will store benchmark results
│
└── .cache/
    └── huggingface/       # Model cache (shared with containers)
```

## ⚡ Quick Start Commands

Copy-paste this block to set up everything:
```bash
# Create directories
mkdir -p $HOME/workspace/module-1.1/{notebooks,outputs}

# Pull NGC container (if not already done)
docker pull nvcr.io/nvidia/pytorch:25.11-py3

# Verify Ollama
systemctl status ollama || sudo systemctl start ollama

# Pull minimum required models (2025 Tier 1)
ollama pull qwen3:8b              # Primary fast model
ollama pull qwen3:32b             # Primary teaching model

# Clear buffer cache for clean start
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# Verify setup
echo "=== Verification ==="
nvidia-smi --query-gpu=name,memory.total --format=csv
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.11-py3 python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
curl -s http://localhost:11434/api/tags | head -5

echo "✅ Setup complete!"
```

## 🔄 Between Labs

After completing each lab, clean up memory:
```python
# In Python/Jupyter
import torch
import gc

# Delete large variables
# del large_tensor

# Clear caches
torch.cuda.empty_cache()
gc.collect()

print("Memory cleared")
```

```bash
# In terminal (before next lab)
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

## 📞 Pre-Flight Check

Run this before starting any lab:
```bash
#!/bin/bash
echo "=== DGX Spark Pre-Flight Check ==="

# GPU
echo -n "GPU: "
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "FAILED"

# Memory
echo -n "Memory: "
nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo "FAILED"

# Docker
echo -n "Docker GPU: "
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi -L 2>/dev/null | head -1 || echo "FAILED"

# Ollama
echo -n "Ollama: "
curl -s http://localhost:11434/api/tags >/dev/null && echo "OK" || echo "NOT RUNNING"

# Container
echo -n "NGC Container: "
docker images | grep -q "pytorch:25.11-py3" && echo "PULLED" || echo "NOT PULLED"

echo "=== Check Complete ==="
```
