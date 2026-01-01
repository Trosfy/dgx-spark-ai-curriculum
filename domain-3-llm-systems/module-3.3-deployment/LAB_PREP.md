# Module 3.3: Model Deployment & Inference Engines - Lab Preparation Guide

## ⏱️ Time Estimates
| Lab | Preparation | Execution | Total |
|-----|-------------|-----------|-------|
| Lab 3.3.1: Engine Benchmark | 30 min | 3 hr | 3.5 hr |
| Lab 3.3.2: SGLang Deployment | 15 min | 2 hr | 2.3 hr |
| Lab 3.3.3: vLLM Continuous Batching | 15 min | 2 hr | 2.3 hr |
| Lab 3.3.4: Medusa Speculative Decoding | 20 min | 2 hr | 2.3 hr |
| Lab 3.3.5: EAGLE Implementation | 15 min | 2 hr | 2.3 hr |
| Lab 3.3.6: TensorRT-LLM Optimization | 45 min | 2 hr | 2.8 hr |
| Lab 3.3.7: Production API | 15 min | 2 hr | 2.3 hr |

## 📦 Required Downloads

### Models (Download Before Labs - 2025)

```bash
# Primary test models via Ollama (recommended)
ollama pull qwen3:8b              # Fast testing (~5GB)
ollama pull qwen3:32b             # Production quality (~20GB)

# For HuggingFace/vLLM/TensorRT-LLM benchmarks
huggingface-cli download Qwen/Qwen3-8B-Instruct
huggingface-cli download Qwen/Qwen3-32B-Instruct

# Legacy models for comparison (optional)
ollama pull llama3.1:8b           # ~5GB
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct

# For Medusa (if using pre-trained heads)
# Check Medusa repo for compatible heads
```

### Additional Packages
```bash
# Core inference engines
pip install vllm
pip install sglang

# For production API
pip install fastapi uvicorn

# For benchmarking
pip install locust httpx

# For Medusa (if available for your model)
pip install medusa-llm  # Check compatibility
```

## 🔧 Environment Setup

### 1. Start Container with All Ports
```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $HOME/.ollama:/root/.ollama \
    --ipc=host \
    --network=host \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

**Note**: Using `--network=host` for easy access to all services.

### 2. Verify GPU Access
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print(f"Compute Capability: {torch.cuda.get_device_capability()}")
```

### 3. Start Ollama (for comparisons)
```bash
# In a separate terminal
ollama serve
```

### 4. Verify All Engines
```python
def verify_engines():
    checks = []

    # vLLM
    try:
        import vllm
        checks.append(f"✅ vLLM: {vllm.__version__}")
    except:
        checks.append("❌ vLLM: pip install vllm")

    # SGLang
    try:
        import sglang
        checks.append("✅ SGLang: installed")
    except:
        checks.append("❌ SGLang: pip install sglang")

    # FastAPI
    try:
        import fastapi
        checks.append(f"✅ FastAPI: {fastapi.__version__}")
    except:
        checks.append("❌ FastAPI: pip install fastapi uvicorn")

    # Ollama
    import requests
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code == 200:
            checks.append("✅ Ollama: running")
        else:
            checks.append("⚠️ Ollama: not responding")
    except:
        checks.append("⚠️ Ollama: start with 'ollama serve'")

    print("\n".join(checks))

verify_engines()
```

## ✅ Pre-Lab Checklists

### Lab 3.3.1: Engine Benchmark
- [ ] All engines installed (vLLM, SGLang, Ollama)
- [ ] 8B model downloaded
- [ ] At least 20GB GPU memory free
- [ ] Port 8000 available
- [ ] Benchmark prompts ready (in `data/`)

### Lab 3.3.2: SGLang Deployment
- [ ] SGLang installed: `pip install sglang`
- [ ] 8B model downloaded
- [ ] Port 30000 available
- [ ] Understand prefix caching concept

### Lab 3.3.3: vLLM Continuous Batching
- [ ] vLLM installed: `pip install vllm`
- [ ] 8B model downloaded
- [ ] Port 8000 available
- [ ] httpx installed for load testing

### Lab 3.3.4-3.3.5: Speculative Decoding
- [ ] Base model loaded
- [ ] Medusa library installed (or using vLLM's built-in)
- [ ] At least 25GB GPU memory free
- [ ] Understand draft-verify paradigm

### Lab 3.3.6: TensorRT-LLM Optimization
- [ ] TensorRT-LLM installed or use Triton container
- [ ] 8B or 70B model ready
- [ ] At least 40GB GPU memory free
- [ ] 45-90 minutes for engine build
- [ ] Clear buffer cache before starting

### Lab 3.3.7: Production API
- [ ] FastAPI, uvicorn installed
- [ ] One inference engine working (vLLM recommended)
- [ ] Understanding of async Python
- [ ] Port 8000 available

## 🚫 Common Setup Mistakes

| Mistake | Consequence | Prevention |
|---------|-------------|------------|
| Not using `--enforce-eager` | vLLM crashes on DGX Spark | Always add for ARM64 |
| Port conflict | Server won't start | Check `lsof -i :8000` |
| Wrong dtype | Suboptimal performance | Use `bfloat16` for Blackwell |
| Not clearing cache between tests | Inconsistent benchmarks | `torch.cuda.empty_cache()` |
| HF token not set | Model download fails | `export HF_TOKEN=...` |
| TensorRT version mismatch | Build fails | Use matching NGC container |

## 📁 Expected File Structure
After preparation, your workspace should look like:
```
/workspace/
├── module-3.3-deployment/
│   ├── labs/
│   │   ├── lab-3.3.1-engine-benchmark.ipynb
│   │   └── ... (7 notebooks)
│   ├── api/
│   │   └── api_server.py
│   ├── data/
│   │   └── benchmark_prompts.json
│   ├── scripts/
│   │   ├── benchmark_utils.py
│   │   ├── inference_client.py
│   │   └── monitoring.py
│   └── outputs/
│       ├── benchmarks/
│       └── trt_engines/
└── .cache/
    └── huggingface/
        └── hub/
            └── models--meta-llama--...
```

## ⚡ Quick Start Commands
```bash
# Copy-paste this block to set up everything:
cd /workspace

# Install all inference libraries
pip install vllm sglang fastapi uvicorn httpx locust

# Download models (2025 Tier 1)
ollama pull qwen3:8b
# or for vLLM/TensorRT:
huggingface-cli download Qwen/Qwen3-8B-Instruct

# Test vLLM quickly
python -c "
import torch
print('Memory available:', torch.cuda.get_device_properties(0).total_memory/1e9, 'GB')
print('Checking vLLM...')
from vllm import LLM
print('✅ vLLM ready')
"

# Start Ollama in background (optional)
nohup ollama serve &> /tmp/ollama.log &
```

## 🔄 Port Reference

| Service | Default Port | Usage |
|---------|--------------|-------|
| vLLM | 8000 | OpenAI-compatible API |
| SGLang | 30000 | SGLang server |
| Ollama | 11434 | Ollama API |
| Jupyter | 8888 | Notebook interface |
| TensorRT Triton | 8001 | Triton inference |
| FastAPI (custom) | 8000 | Production API |

## 🔄 Benchmark Template

```python
import time
import torch

def benchmark_engine(engine_fn, prompts, warmup=3):
    """Standard benchmark for comparing engines."""
    # Warmup
    for _ in range(warmup):
        engine_fn(prompts[0])

    torch.cuda.synchronize()

    # Benchmark
    results = []
    for prompt in prompts:
        start = time.perf_counter()
        output = engine_fn(prompt)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        results.append({
            "input_tokens": len(prompt.split()),  # Approximate
            "output_tokens": len(output.split()),
            "time_seconds": elapsed
        })

    return results
```
