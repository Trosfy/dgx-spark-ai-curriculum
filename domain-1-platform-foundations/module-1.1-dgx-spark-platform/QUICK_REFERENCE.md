# Module 1.1: DGX Spark Platform Mastery - Quick Reference

## 🚀 Essential Commands

### System Information
```bash
# GPU status and memory usage
nvidia-smi

# Watch GPU continuously (updates every 1 second)
watch -n 1 nvidia-smi

# CPU information
lscpu

# Memory (RAM) usage
free -h

# Disk space
df -h

# Check Ollama service
systemctl status ollama
```

### NGC Container Commands
```bash
# Pull PyTorch container (do this once)
docker pull nvcr.io/nvidia/pytorch:25.11-py3

# Run container with GPU access
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    nvcr.io/nvidia/pytorch:25.11-py3

# Run with JupyterLab
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

### Memory Management
```bash
# Clear Linux buffer cache before loading large models
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

```python
# Clear PyTorch GPU cache
import torch
import gc
torch.cuda.empty_cache()
gc.collect()

# Check memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### Ollama Commands
```bash
# List available models
ollama list

# Pull a model
ollama pull llama3.2:3b
ollama pull llama3.1:8b
ollama pull llama3.1:70b

# Run model interactively
ollama run llama3.1:8b

# Check API availability
curl http://localhost:11434/api/tags
```

## 📊 Key Specifications

### Hardware Specs
| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA Blackwell GB10 Superchip |
| Memory | 128 GB LPDDR5X unified |
| Bandwidth | 273 GB/s |
| CUDA Cores | 6,144 |
| Tensor Cores | 192 (5th gen) |
| CPU | 20 ARM v9.2 cores |
| Architecture | ARM64/aarch64 |

### Compute Performance
| Precision | Performance |
|-----------|-------------|
| NVFP4 | 1 PFLOP |
| FP8 | ~209 TFLOPS |
| BF16 | ~100 TFLOPS |
| FP32 | ~31 TFLOPS |

### Model Capacity (Approximate)
| Scenario | Max Model Size | Memory Used |
|----------|----------------|-------------|
| Full Fine-Tuning (FP16) | 12-16B | ~100-128 GB |
| QLoRA Fine-Tuning | 100-120B | ~50-70 GB |
| FP16 Inference | 50-55B | ~110-120 GB |
| FP8 Inference | 90-100B | ~90-100 GB |
| NVFP4 Inference | ~200B | ~100 GB |

### Ecosystem Compatibility
| Status | Tools |
|--------|-------|
| ✅ Full Support | Ollama, llama.cpp, NeMo |
| ⚠️ NGC Required | PyTorch, JAX, Hugging Face |
| ⚠️ Partial | vLLM, TensorRT-LLM, DeepSpeed |
| ❌ Not Compatible | Standard pip PyTorch |

## 🔧 Common Patterns

### Pattern: Verify GPU Access
```python
import torch

# Quick check
assert torch.cuda.is_available(), "CUDA not available!"

# Detailed info
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Compute: {torch.cuda.get_device_capability(0)}")
```

### Pattern: Ollama API Benchmark
```python
import requests
import time

response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "llama3.1:8b", "prompt": "Hello", "stream": False}
)
data = response.json()

# Extract timing (Ollama returns nanoseconds)
prefill_tps = data["prompt_eval_count"] / (data["prompt_eval_duration"] / 1e9)
decode_tps = data["eval_count"] / (data["eval_duration"] / 1e9)

print(f"Prefill: {prefill_tps:.0f} tok/s")
print(f"Decode: {decode_tps:.0f} tok/s")
```

### Pattern: docker-compose.yml
```yaml
version: '3.8'
services:
  pytorch:
    image: nvcr.io/nvidia/pytorch:25.11-py3
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - $HOME/workspace:/workspace
      - $HOME/.cache/huggingface:/root/.cache/huggingface
    ipc: host
    stdin_open: true
    tty: true
```

## ⚠️ Common Mistakes

| Mistake | Fix |
|---------|-----|
| `pip install torch` | Use NGC container instead |
| Forgot `--gpus all` | Add flag to docker run |
| Model won't load (OOM) | Clear buffer cache first |
| Ollama not responding | `systemctl start ollama` |
| Wrong Python version | NGC container uses correct version |

## 🔗 Quick Links
- [NVIDIA DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/)
- [NGC Container Catalog](https://catalog.ngc.nvidia.com/)
- [DGX Spark Playbooks](https://build.nvidia.com/spark)
- [Ollama Documentation](https://github.com/ollama/ollama)
