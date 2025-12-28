# DGX Spark Environment Setup

## Prerequisites
- NVIDIA DGX Spark with DGX OS
- Docker with NVIDIA Container Runtime
- Git

## Initial Setup

### 1. Verify System
```bash
nvidia-smi
lscpu
free -h
```

### 2. Pull NGC Container
```bash
docker pull nvcr.io/nvidia/pytorch:25.11-py3
```

### 3. Run Container with Proper Mounts
```bash
docker run --gpus all --ipc=host --net=host \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $PWD:/workspace \
  -w /workspace \
  -it nvcr.io/nvidia/pytorch:25.11-py3 \
  jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

### 4. Clear Buffer Cache (Before Heavy Workloads)
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

## Ollama Setup
```bash
# Ollama is pre-installed on DGX Spark
ollama list
ollama pull llama3.1:8b
ollama pull llama3.1:70b
```

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues.
