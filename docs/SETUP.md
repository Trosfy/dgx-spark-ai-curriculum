# DGX Spark Environment Setup Guide

This guide walks you through setting up your NVIDIA DGX Spark for the AI curriculum.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [1. System Verification](#1-system-verification)
- [2. Docker and NGC Setup](#2-docker-and-ngc-setup)
- [3. JupyterLab Configuration](#3-jupyterlab-configuration)
- [4. Ollama Setup](#4-ollama-setup)
- [5. Python Environment](#5-python-environment)
- [6. Storage Configuration](#6-storage-configuration)
- [7. Networking](#7-networking)
- [8. Verification Checklist](#8-verification-checklist)

---

## Prerequisites

Before starting, ensure you have:

- NVIDIA DGX Spark with DGX OS installed
- Admin/sudo access to the system
- Internet connection for downloading containers and models
- At least 100GB free storage for models and datasets

---

## 1. System Verification

### Check Hardware

```bash
# GPU Information
nvidia-smi

# Expected output should show:
# - Driver Version: 580.x or higher
# - CUDA Version: 13.0 or higher
# - GPU: NVIDIA GB10 Superchip
```

```bash
# CPU Information
lscpu | grep -E "Model name|CPU\(s\)|Architecture"

# Expected:
# Architecture: aarch64
# CPU(s): 20
# Model name: ARMv9 (Cortex-X925/A725)
```

```bash
# Memory
free -h

# Expected: ~128GB total
```

```bash
# Storage
df -h /home

# Check available space
```

### Verify CUDA Installation

```bash
# Check CUDA version
nvcc --version

# Check CUDA libraries
ls /usr/local/cuda/lib64/ | head -10
```

### System Information Script

Create a script to capture full system info:

```bash
#!/bin/bash
# Save as: ~/system_info.sh

echo "=== DGX Spark System Information ==="
echo ""
echo "--- GPU ---"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo ""
echo "--- CPU ---"
lscpu | grep -E "Model name|CPU\(s\)|Architecture|Thread|Core"
echo ""
echo "--- Memory ---"
free -h
echo ""
echo "--- Storage ---"
df -h / /home
echo ""
echo "--- OS ---"
cat /etc/os-release | grep -E "NAME|VERSION"
echo ""
echo "--- CUDA ---"
nvcc --version 2>/dev/null || echo "nvcc not in PATH"
echo ""
echo "--- Docker ---"
docker --version
echo ""
echo "--- Python ---"
python3 --version
```

---

## 2. Docker and NGC Setup

### Verify Docker Installation

```bash
# Check Docker
docker --version
docker info | grep -E "Server Version|Storage Driver"

# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```

### NGC Authentication (Optional but Recommended)

```bash
# Create NGC account at https://ngc.nvidia.com
# Generate API key from NGC dashboard

# Login to NGC
docker login nvcr.io
# Username: $oauthtoken
# Password: <your-ngc-api-key>
```

### Pull Essential Containers

```bash
# PyTorch (Primary development container)
docker pull nvcr.io/nvidia/pytorch:25.11-py3

# vLLM for inference
docker pull nvcr.io/nvidia/vllm:spark

# TensorRT-LLM (optional, for advanced optimization)
docker pull nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3
```

### Create Docker Compose File

Create `~/docker-compose.yml`:

```yaml
version: '3.8'

services:
  pytorch-lab:
    image: nvcr.io/nvidia/pytorch:25.11-py3
    container_name: dgx-spark-lab
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ${HOME}:/home/user
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
      - ${HOME}/workspace:/workspace
    working_dir: /workspace
    ports:
      - "8888:8888"   # JupyterLab
      - "6006:6006"   # TensorBoard
      - "5000:5000"   # MLflow
    ipc: host
    network_mode: host
    command: >
      jupyter lab 
      --ip=0.0.0.0 
      --port=8888 
      --allow-root 
      --no-browser 
      --NotebookApp.token=''
      --NotebookApp.password=''

  mlflow:
    image: nvcr.io/nvidia/pytorch:25.11-py3
    container_name: dgx-spark-mlflow
    volumes:
      - ${HOME}/mlflow:/mlflow
    ports:
      - "5001:5000"
    command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri /mlflow
```

### Run Development Environment

```bash
# Start the environment
docker-compose up -d pytorch-lab

# View logs
docker-compose logs -f pytorch-lab

# Stop
docker-compose down
```

---

## 3. JupyterLab Configuration

### Native JupyterLab (Pre-installed)

DGX Spark comes with JupyterLab pre-installed. Access via:

```bash
# Check if JupyterLab is running
systemctl status jupyter

# Start if not running
jupyter lab --ip=0.0.0.0 --no-browser
```

### JupyterLab Extensions

Inside your container or native environment:

```bash
# Install useful extensions
pip install jupyterlab-git
pip install jupyterlab-lsp python-lsp-server
pip install jupyter-resource-usage

# Rebuild JupyterLab
jupyter lab build
```

### Custom JupyterLab Settings

Create `~/.jupyter/jupyter_lab_config.py`:

```python
c = get_config()

# Server settings
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True

# Security (for local use only)
c.ServerApp.token = ''
c.ServerApp.password = ''

# Performance
c.ServerApp.iopub_data_rate_limit = 10000000
c.ServerApp.rate_limit_window = 3

# Notebook settings
c.NotebookApp.notebook_dir = '/workspace'
```

### Notebook Startup Script

Create `~/.jupyter/startup.py`:

```python
"""Auto-run on notebook startup"""
import warnings
warnings.filterwarnings('ignore')

# Common imports available in all notebooks
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyTorch (if available)
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except ImportError:
    pass

print("Startup complete ✓")
```

---

## 4. Ollama Setup

### Verify Ollama Installation

```bash
# Ollama is pre-installed on DGX Spark
ollama --version

# Check if service is running
systemctl status ollama
```

### Configure Ollama

```bash
# Set environment variables (add to ~/.bashrc)
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_MODELS=/home/$USER/.ollama/models
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=2
```

### Pull Recommended Models (2025 Tier 1)

```bash
# General purpose (primary development)
ollama pull qwen3:8b              # Fast testing (~5GB, hybrid thinking)
ollama pull qwen3:32b             # Best quality (~20GB, 131K context)

# Reasoning model (extended CoT)
ollama pull qwq:32b               # SOTA reasoning (~20GB, matches R1 on math)

# Coding model
ollama pull qwen3-coder:30b       # 69.6% SWE-Bench (~19GB, 256K context)

# Embedding model
ollama pull qwen3-embedding:8b    # #1 MTEB multilingual (~8GB, 32K context)

# Vision model
ollama pull qwen3-vl:8b           # Design-to-code, 32-lang OCR (~8GB)

# Legacy comparison models (optional)
ollama pull llama3.1:8b           # For comparison with newer models
ollama pull deepseek-r1:8b        # SOTA 8B reasoning distillation

# List installed models
ollama list
```

> **2025 Model Notes:**
> - Qwen3 models support hybrid thinking mode (toggle with /think or /no_think)
> - QwQ-32B provides extended chain-of-thought reasoning by default
> - All Tier 1 models have Apache 2.0 licensing (except Llama)

### Ollama API Test

```python
# test_ollama.py
import requests
import json

def test_ollama():
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': 'qwen3:8b',
            'prompt': 'Hello! Respond with just "OK" if you can hear me.',
            'stream': False
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Ollama working: {result['response'][:50]}...")
    else:
        print(f"✗ Error: {response.status_code}")

if __name__ == "__main__":
    test_ollama()
```

---

## 5. Python Environment

### System Python Packages

For curriculum tasks, install these packages:

```bash
# Inside NGC container or with --break-system-packages flag
pip install --upgrade pip

# Core ML
pip install torch torchvision torchaudio  # Use NGC container version
pip install numpy pandas scipy scikit-learn

# Deep Learning
pip install transformers datasets accelerate
pip install peft bitsandbytes
pip install sentencepiece tokenizers

# Visualization
pip install matplotlib seaborn plotly

# Experiment tracking
pip install mlflow wandb tensorboard

# AI Agents
pip install langchain langchain-community langgraph
pip install llama-index chromadb faiss-cpu

# Utilities
pip install tqdm rich python-dotenv
pip install jupyter ipywidgets
```

### Requirements File

Create `requirements.txt` for the curriculum:

```text
# Core
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0

# Deep Learning
torch>=2.5.0
transformers>=4.46.0
datasets>=2.18.0
accelerate>=0.28.0
peft>=0.10.0
bitsandbytes>=0.43.0
sentencepiece>=0.2.0
tokenizers>=0.19.0

# Visualization
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.18.0

# Experiment Tracking
mlflow>=2.11.0
tensorboard>=2.16.0

# AI Agents
langchain>=0.1.0
langchain-community>=0.0.20
llama-index>=0.10.0
chromadb>=0.4.22

# Evaluation
lm-eval>=0.4.0

# Utilities
tqdm>=4.66.0
rich>=13.7.0
python-dotenv>=1.0.0
einops>=0.7.0
```

---

## 6. Storage Configuration

### Recommended Directory Structure

```bash
# Create workspace directories
mkdir -p ~/workspace/{projects,datasets,models,checkpoints,outputs}
mkdir -p ~/.cache/huggingface/{hub,datasets}

# Set permissions
chmod -R 755 ~/workspace
```

### Hugging Face Cache

```bash
# Set HF cache location (add to ~/.bashrc)
export HF_HOME=/home/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets

# Login to Hugging Face (for gated models)
huggingface-cli login
```

### Model Storage Best Practices

```bash
# Large models should go to dedicated storage
# If you have external NVMe or NAS:
ln -s /mnt/models ~/.cache/huggingface/hub

# Check cache size
du -sh ~/.cache/huggingface/hub
```

---

## 7. Networking

### Port Configuration

| Port | Service | Description |
|------|---------|-------------|
| 8888 | JupyterLab | Primary development |
| 11434 | Ollama | LLM inference API |
| 6006 | TensorBoard | Training visualization |
| 5000 | MLflow | Experiment tracking |
| 8000 | FastAPI | Custom API servers |
| 7860 | Gradio | ML demos |

### Firewall Configuration

```bash
# Allow required ports (if firewall is enabled)
sudo ufw allow 8888/tcp
sudo ufw allow 11434/tcp
sudo ufw allow 6006/tcp
sudo ufw allow 5000/tcp
```

### Remote Access

For accessing DGX Spark from another machine:

```bash
# SSH tunnel for JupyterLab
ssh -L 8888:localhost:8888 user@dgx-spark-ip

# Then access http://localhost:8888 on your local machine
```

---

## 8. Verification Checklist

Run this script to verify your setup:

```bash
#!/bin/bash
# Save as: ~/verify_setup.sh

echo "=== DGX Spark Curriculum Setup Verification ==="
echo ""

# Function to check command
check() {
    if $1 &>/dev/null; then
        echo "✓ $2"
        return 0
    else
        echo "✗ $2"
        return 1
    fi
}

# System
echo "--- System ---"
check "nvidia-smi" "NVIDIA GPU accessible"
check "docker --version" "Docker installed"
check "docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi" "Docker GPU access"

# Python
echo ""
echo "--- Python ---"
check "python3 --version" "Python 3 installed"
check "python3 -c 'import torch; assert torch.cuda.is_available()'" "PyTorch CUDA"
check "python3 -c 'import transformers'" "Transformers library"

# Ollama
echo ""
echo "--- Ollama ---"
check "ollama --version" "Ollama installed"
check "curl -s http://localhost:11434/api/tags" "Ollama API responding"

# JupyterLab
echo ""
echo "--- JupyterLab ---"
check "jupyter --version" "Jupyter installed"

# Storage
echo ""
echo "--- Storage ---"
echo "Workspace: $(du -sh ~/workspace 2>/dev/null || echo 'Not created')"
echo "HF Cache: $(du -sh ~/.cache/huggingface 2>/dev/null || echo 'Not created')"
echo "Free space: $(df -h /home | tail -1 | awk '{print $4}')"

echo ""
echo "=== Verification Complete ==="
```

### Quick Test Notebook

Create this notebook to test everything:

```python
# Cell 1: System Info
!nvidia-smi
!python --version

# Cell 2: PyTorch
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Cell 3: Memory Test
x = torch.randn(10000, 10000, device='cuda')
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
del x
torch.cuda.empty_cache()

# Cell 4: Hugging Face
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print("✓ Hugging Face working")

# Cell 5: Ollama
import requests
r = requests.get("http://localhost:11434/api/tags")
print(f"Ollama models: {[m['name'] for m in r.json().get('models', [])]}")
```

---

## Next Steps

Once setup is complete:

1. Clone the curriculum repository
2. Navigate to `domain-1-platform-foundations/module-1.1-dgx-spark-platform/`
3. Open the first notebook and begin!

For issues, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
