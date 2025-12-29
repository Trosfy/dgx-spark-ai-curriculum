# NGC Containers Guide for DGX Spark

NVIDIA NGC (NVIDIA GPU Cloud) containers are **essential** for DGX Spark development. Standard PyTorch pip installations do not work on ARM64 + CUDA architecture.

---

## Table of Contents

- [Why NGC Containers?](#why-ngc-containers)
- [Essential Containers](#essential-containers)
- [Container Usage Patterns](#container-usage-patterns)
- [Persistent Development Setup](#persistent-development-setup)
- [Container Customization](#container-customization)
- [Troubleshooting](#troubleshooting)

---

## Why NGC Containers?

### The ARM64 + CUDA Challenge

DGX Spark uses an **ARM64 CPU (Grace)** paired with a **Blackwell GPU**. This combination means:

| Installation Method | Works? | Notes |
|---------------------|--------|-------|
| `pip install torch` | ❌ No | PyPI wheels are x86_64 only |
| `conda install pytorch` | ❌ No | No ARM64 + CUDA builds |
| NGC Container | ✅ Yes | Pre-built for ARM64 + CUDA |
| Build from source | ⚠️ Maybe | Complex, time-consuming |

### NGC Advantages

- **Pre-optimized** for NVIDIA hardware
- **Tested** on DGX systems
- **Updated** with latest CUDA and cuDNN
- **Includes** common ML libraries pre-installed
- **Saves hours** of build/debug time

---

## Essential Containers

### 1. PyTorch (Primary Development)

```bash
# Latest PyTorch for DGX Spark
docker pull nvcr.io/nvidia/pytorch:25.11-py3

# Includes:
# - PyTorch 2.x
# - CUDA 13.0
# - cuDNN 9.x
# - NCCL
# - Apex
# - Transformer Engine
```

**Best for:** Most curriculum tasks, training, fine-tuning

### 2. vLLM (Inference)

```bash
# Optimized vLLM for DGX Spark
docker pull nvcr.io/nvidia/vllm:spark

# Includes:
# - vLLM with PagedAttention
# - Continuous batching
# - DGX Spark optimizations
```

**Best for:** High-throughput LLM inference, API serving

### 3. TensorRT-LLM (Maximum Performance)

```bash
# TensorRT-LLM with Triton
docker pull nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3

# Includes:
# - TensorRT-LLM
# - Triton Inference Server
# - FP4/FP8 quantization
```

**Best for:** Production inference, FP4 quantization, maximum throughput

### 4. NeMo (NVIDIA's ML Framework)

```bash
# NeMo for training and fine-tuning
docker pull nvcr.io/nvidia/nemo:25.01

# Includes:
# - NeMo Framework
# - Megatron-LM
# - Pre-built training pipelines
```

**Best for:** Large-scale training, NVIDIA-optimized workflows

### 5. RAPIDS (Data Science)

```bash
# GPU-accelerated data science
docker pull nvcr.io/nvidia/rapidsai/base:25.02-cuda12.0-py3.11

# Includes:
# - cuDF (GPU DataFrames)
# - cuML (GPU ML algorithms)
# - cuGraph
```

**Best for:** Large dataset processing, feature engineering

---

## Container Usage Patterns

### Basic Interactive Session

```bash
docker run --gpus all -it --rm \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    bash
```

### JupyterLab Development

```bash
docker run --gpus all -it --rm \
    -p 8888:8888 \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

### Full Development Environment

```bash
docker run --gpus all -it --rm \
    --name dgx-dev \
    -p 8888:8888 \
    -p 6006:6006 \
    -p 5000:5000 \
    -v $HOME:/home/user \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -w /workspace \
    --ipc=host \
    --net=host \
    -e HF_HOME=/root/.cache/huggingface \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token=''
```

### Running a Script

```bash
docker run --gpus all --rm \
    -v $PWD:/workspace \
    -w /workspace \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    python train.py --epochs 10
```

---

## Persistent Development Setup

### Docker Compose Configuration

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  # Primary development environment
  dev:
    image: nvcr.io/nvidia/pytorch:25.11-py3
    container_name: dgx-spark-dev
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - HF_HOME=/root/.cache/huggingface
      - TRANSFORMERS_CACHE=/root/.cache/huggingface/hub
    volumes:
      # Your home directory
      - ${HOME}:/home/user
      # Workspace for projects
      - ${HOME}/workspace:/workspace
      # Hugging Face cache (models, datasets)
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
      # Ollama models (if using Ollama inside container)
      - ${HOME}/.ollama:/root/.ollama
    working_dir: /workspace
    ports:
      - "8888:8888"   # JupyterLab
      - "6006:6006"   # TensorBoard
      - "5000:5000"   # MLflow
      - "7860:7860"   # Gradio
      - "8000:8000"   # FastAPI
    ipc: host
    network_mode: host
    stdin_open: true
    tty: true
    command: >
      bash -c "
        pip install -q jupyterlab-git jupyter-resource-usage &&
        jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser --NotebookApp.token=''
      "

  # vLLM inference server
  vllm:
    image: nvcr.io/nvidia/vllm:spark
    container_name: dgx-spark-vllm
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
    ports:
      - "8001:8000"
    ipc: host
    command: >
      --model meta-llama/Llama-3.1-8B-Instruct
      --enforce-eager
      --max-model-len 4096
    profiles:
      - inference

  # MLflow tracking server
  mlflow:
    image: nvcr.io/nvidia/pytorch:25.11-py3
    container_name: dgx-spark-mlflow
    volumes:
      - ${HOME}/mlflow-data:/mlflow
    ports:
      - "5001:5000"
    command: >
      mlflow server 
      --host 0.0.0.0 
      --port 5000 
      --backend-store-uri sqlite:///mlflow/mlflow.db
      --default-artifact-root /mlflow/artifacts
    profiles:
      - tracking
```

### Usage

```bash
# Start development environment
docker-compose up -d dev

# Start with MLflow
docker-compose --profile tracking up -d

# Start vLLM server
docker-compose --profile inference up -d vllm

# View logs
docker-compose logs -f dev

# Stop all
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Shell Aliases

Add to `~/.bashrc`:

```bash
# NGC container shortcuts
alias dgx-dev='docker-compose up -d dev && docker exec -it dgx-spark-dev bash'
alias dgx-jupyter='docker-compose up -d dev'
alias dgx-stop='docker-compose down'

# Quick PyTorch container
alias pytorch='docker run --gpus all -it --rm -v $PWD:/workspace -w /workspace nvcr.io/nvidia/pytorch:25.11-py3'

# Run Python script in container
dgx-run() {
    docker run --gpus all --rm \
        -v $HOME/.cache/huggingface:/root/.cache/huggingface \
        -v $PWD:/workspace \
        -w /workspace \
        nvcr.io/nvidia/pytorch:25.11-py3 \
        python "$@"
}
```

---

## Container Customization

### Building Custom Image

Create `Dockerfile`:

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Set working directory
WORKDIR /workspace

# Install additional packages
RUN pip install --no-cache-dir \
    transformers \
    datasets \
    accelerate \
    peft \
    bitsandbytes \
    langchain \
    langchain-community \
    llama-index \
    chromadb \
    mlflow \
    wandb \
    gradio \
    fastapi \
    uvicorn \
    lm-eval

# Install JupyterLab extensions
RUN pip install --no-cache-dir \
    jupyterlab-git \
    jupyter-resource-usage

# Set environment variables
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/hub

# Expose ports
EXPOSE 8888 6006 5000 7860 8000

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
```

Build and use:

```bash
# Build
docker build -t dgx-spark-curriculum:latest .

# Run
docker run --gpus all -it --rm \
    -p 8888:8888 \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    dgx-spark-curriculum:latest
```

### Installing Packages in Running Container

```bash
# Enter running container
docker exec -it dgx-spark-dev bash

# Install packages (they persist until container is removed)
pip install some-package

# To persist permanently, add to Dockerfile or requirements.txt
```

---

## Container-Specific Tips

### PyTorch Container

```python
# Verify GPU access
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Transformer Engine (for FP8)
import transformer_engine.pytorch as te
print("Transformer Engine available")
```

### vLLM Container

```bash
# Start server
docker run --gpus all -it --rm \
    -p 8000:8000 \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    nvcr.io/nvidia/vllm:spark \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --enforce-eager \
    --max-model-len 4096

# Test API
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "meta-llama/Llama-3.1-8B-Instruct", "prompt": "Hello", "max_tokens": 50}'
```

### TensorRT-LLM Container

```bash
# Enter container
docker run --gpus all -it --rm \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $PWD:/workspace \
    -w /workspace \
    nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3 \
    bash

# Inside container: convert model to TensorRT
python -m tensorrt_llm.commands.build \
    --model_dir /root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct \
    --output_dir ./trt_engines/llama-8b \
    --dtype bfloat16
```

---

## Troubleshooting

### "No CUDA GPUs are available"

```bash
# Check NVIDIA runtime
docker info | grep -i runtime

# Ensure --gpus all flag is used
docker run --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```

### "OOM (Out of Memory)" in Container

```bash
# Clear buffer cache before starting container
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# Use --ipc=host for shared memory
docker run --gpus all --ipc=host ...
```

### "Permission denied" for mounted volumes

```bash
# Run as root inside container (default for NGC)
docker run --gpus all -u root ...

# Or fix permissions on host
chmod -R 755 ~/workspace
```

### Container can't access Ollama

```bash
# Use host networking
docker run --gpus all --net=host ...

# Or access via host IP
# Inside container: curl http://host.docker.internal:11434/api/tags
```

### Slow model downloads

```bash
# Pre-download models on host
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct

# Mount cache in container
docker run -v $HOME/.cache/huggingface:/root/.cache/huggingface ...
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Start JupyterLab | `docker-compose up -d dev` |
| Enter container | `docker exec -it dgx-spark-dev bash` |
| Run script | `docker run --gpus all -v $PWD:/workspace -w /workspace nvcr.io/nvidia/pytorch:25.11-py3 python script.py` |
| Check GPU | `docker run --gpus all --rm nvcr.io/nvidia/pytorch:25.11-py3 nvidia-smi` |
| Pull latest | `docker pull nvcr.io/nvidia/pytorch:25.11-py3` |
| Clean unused | `docker system prune -a` |

---

## Further Resources

- [NGC Catalog](https://catalog.ngc.nvidia.com/)
- [DGX Spark Container Guide](https://docs.nvidia.com/dgx/dgx-spark/)
- [PyTorch NGC Container Release Notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/)
