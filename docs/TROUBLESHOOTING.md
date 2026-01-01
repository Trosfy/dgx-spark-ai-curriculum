# Troubleshooting Guide

Common issues and solutions when working with DGX Spark for AI development.

---

## Table of Contents

- [GPU and CUDA Issues](#gpu-and-cuda-issues)
- [Memory Issues](#memory-issues)
- [PyTorch Issues](#pytorch-issues)
- [Container Issues](#container-issues)
- [Ollama Issues](#ollama-issues)
- [Hugging Face Issues](#hugging-face-issues)
- [Training Issues](#training-issues)
- [Inference Issues](#inference-issues)
- [Networking Issues](#networking-issues)
- [Performance Issues](#performance-issues)

---

## GPU and CUDA Issues

### `nvidia-smi` shows no GPU

**Symptoms:**
```
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

**Solutions:**
```bash
# 1. Check if driver is loaded
lsmod | grep nvidia

# 2. Reload driver
sudo modprobe nvidia

# 3. Restart NVIDIA services
sudo systemctl restart nvidia-persistenced

# 4. If still failing, reboot
sudo reboot
```

### CUDA version mismatch

**Symptoms:**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Solutions:**
```bash
# Check installed CUDA version
nvcc --version
cat /usr/local/cuda/version.txt

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Solution: Use NGC containers which have matching versions
docker pull nvcr.io/nvidia/pytorch:25.11-py3
```

### `torch.cuda.is_available()` returns False

**Symptoms:**
```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**Solutions:**
```bash
# 1. If using pip-installed PyTorch, it won't work on ARM64
# Solution: Use NGC container
docker run --gpus all -it nvcr.io/nvidia/pytorch:25.11-py3 python -c "import torch; print(torch.cuda.is_available())"

# 2. Check if GPU is visible
nvidia-smi

# 3. In Docker, ensure --gpus all flag
docker run --gpus all ...
```

---

## Memory Issues

### CUDA Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GiB
```

**Solutions:**

```bash
# 1. Clear buffer cache (CRITICAL for DGX Spark unified memory)
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

```python
# 2. Clear PyTorch cache
import torch
import gc
torch.cuda.empty_cache()
gc.collect()

# 3. Reduce batch size
batch_size = 4  # Start small, increase gradually

# 4. Use gradient checkpointing
model.gradient_checkpointing_enable()

# 5. Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    output = model(input)

# 6. For fine-tuning, use QLoRA instead of LoRA
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

### Memory not released after model deletion

**Symptoms:**
- Memory still shows as allocated after `del model`

**Solutions:**
```python
# Complete cleanup procedure
import torch
import gc

# 1. Delete model
del model

# 2. Clear references
gc.collect()

# 3. Clear CUDA cache
torch.cuda.empty_cache()

# 4. Verify
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### "Cannot allocate memory" during model loading

**Solutions:**
```python
# 1. Load model with device_map
model = AutoModelForCausalLM.from_pretrained(
    "model-name",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 2. Use quantization for large models
model = AutoModelForCausalLM.from_pretrained(
    "model-name",
    load_in_4bit=True,
    device_map="auto"
)

# 3. Clear cache before loading
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

---

## PyTorch Issues

### PyTorch not finding CUDA (in container)

**Symptoms:**
```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**Solutions:**
```bash
# Ensure --gpus all flag
docker run --gpus all -it nvcr.io/nvidia/pytorch:25.11-py3 bash

# Check inside container
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### `RuntimeError: cuDNN error`

**Solutions:**
```python
# 1. Disable cuDNN benchmarking
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 2. Use different cuDNN algorithm
torch.backends.cudnn.enabled = False  # Last resort
```

### Mixed precision training NaN loss

**Solutions:**
```python
# 1. Use GradScaler properly
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(batch)
        loss = criterion(output, target)
    
    # Check for NaN
    if torch.isnan(loss):
        print("NaN detected, skipping batch")
        continue
    
    scaler.scale(loss).backward()
    
    # Gradient clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    scaler.step(optimizer)
    scaler.update()
```

---

## Container Issues

### Container can't access GPU

**Symptoms:**
```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

**Solutions:**
```bash
# 1. Install NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 2. Verify installation
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```

### Permission denied for mounted volumes

**Solutions:**
```bash
# 1. Check ownership
ls -la ~/workspace

# 2. Fix permissions
chmod -R 755 ~/workspace

# 3. Or run container as root (default for NGC)
docker run --gpus all -u root ...

# 4. Or match user ID
docker run --gpus all -u $(id -u):$(id -g) ...
```

### Container runs out of shared memory

**Symptoms:**
```
RuntimeError: DataLoader worker exited unexpectedly
```

**Solutions:**
```bash
# Use --ipc=host
docker run --gpus all --ipc=host ...

# Or increase shm-size
docker run --gpus all --shm-size=16g ...
```

---

## Ollama Issues

### Ollama not responding

**Symptoms:**
```
Error: could not connect to ollama app
```

**Solutions:**
```bash
# 1. Check if service is running
systemctl status ollama

# 2. Start service
sudo systemctl start ollama

# 3. Check logs
journalctl -u ollama -f

# 4. Restart
sudo systemctl restart ollama

# 5. Check port
curl http://localhost:11434/api/tags
```

### Model loading fails in Ollama

**Symptoms:**
```
Error: model requires more memory than is available
```

**Solutions:**
```bash
# 1. Clear system cache
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# 2. Stop other GPU processes
nvidia-smi  # Check what's using GPU
kill <pid>  # Stop unnecessary processes

# 3. Use smaller quantization
ollama run qwen3:32b-instruct-q4_0  # Instead of q8
```

### Ollama slow response

**Solutions:**
```bash
# 1. Check if model is loaded
ollama ps

# 2. Increase context size limit
export OLLAMA_NUM_CTX=4096

# 3. Reduce parallel requests
export OLLAMA_NUM_PARALLEL=1

# 4. Use appropriate quantization for speed
ollama run qwen3:8b-instruct-q4_0
```

---

## Hugging Face Issues

### Gated model access denied

**Symptoms:**
```
OSError: You are trying to access a gated repo.
```

**Solutions:**
```bash
# 1. Login to Hugging Face
huggingface-cli login

# 2. Accept model license on HF website
# Visit: https://huggingface.co/Qwen/Qwen3-8B-Instruct
# Click "Agree and access repository"

# 3. Set token in code
from huggingface_hub import login
login(token="hf_xxxxx")
```

### Download interrupted / corrupted cache

**Solutions:**
```bash
# 1. Clear specific model cache
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Instruct

# 2. Force re-download
from transformers import AutoModel
model = AutoModel.from_pretrained("model-name", force_download=True)

# 3. Use CLI for reliable download
huggingface-cli download Qwen/Qwen3-8B-Instruct --resume-download
```

### Tokenizer warnings / errors

**Solutions:**
```python
# 1. Use fast tokenizer
tokenizer = AutoTokenizer.from_pretrained("model-name", use_fast=True)

# 2. Set padding token
tokenizer.pad_token = tokenizer.eos_token

# 3. Trust remote code if required
tokenizer = AutoTokenizer.from_pretrained("model-name", trust_remote_code=True)
```

---

## Training Issues

### Loss is NaN or Inf

**Solutions:**
```python
# 1. Reduce learning rate
learning_rate = 1e-5  # Start very small

# 2. Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. Check data for anomalies
for batch in dataloader:
    if torch.isnan(batch).any() or torch.isinf(batch).any():
        print("Bad batch detected!")

# 4. Use stable loss functions
from torch.nn import CrossEntropyLoss
loss_fn = CrossEntropyLoss(label_smoothing=0.1)

# 5. Initialize properly
for module in model.modules():
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()
```

### Training stuck / not converging

**Solutions:**
```python
# 1. Check learning rate
# Too high: loss oscillates wildly
# Too low: loss decreases extremely slowly

# 2. Use learning rate finder
from torch_lr_finder import LRFinder
lr_finder = LRFinder(model, optimizer, criterion)
lr_finder.range_test(train_loader, end_lr=10, num_iter=100)
lr_finder.plot()

# 3. Use warmup
from transformers import get_linear_schedule_with_warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=100, 
    num_training_steps=total_steps
)

# 4. Check data loading
for batch in dataloader:
    print(batch.shape, batch.dtype)
    break
```

### LoRA/QLoRA not training

**Solutions:**
```python
# 1. Verify trainable parameters
model.print_trainable_parameters()
# Should show < 1% trainable for LoRA

# 2. Check LoRA config targets correct modules
from peft import LoraConfig
config = LoraConfig(
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Check model architecture
    r=16,
    lora_alpha=32,
)

# 3. Ensure model is in training mode
model.train()

# 4. Check gradients are flowing
for name, param in model.named_parameters():
    if param.requires_grad and param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm()}")
```

---

## Inference Issues

### vLLM errors on DGX Spark

**Symptoms:**
```
RuntimeError: CUDA error: invalid device function
```

**Solutions:**
```bash
# 1. Use NVIDIA vLLM container
docker pull nvcr.io/nvidia/vllm:spark

# 2. Use --enforce-eager flag
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B-Instruct \
    --enforce-eager

# 3. Reduce max model length
--max-model-len 4096
```

### TensorRT-LLM attention errors

**Symptoms:**
```
Error: attention sink not supported
```

**Solutions:**
```python
# Set in config
config = {
    "enable_attention_dp": False,
    # other settings...
}
```

### Slow inference

**Solutions:**
```python
# 1. Use appropriate batch size
# Larger batch = better throughput

# 2. Use continuous batching (vLLM)
# Automatically handles dynamic batching

# 3. Use quantization
model = AutoModelForCausalLM.from_pretrained(
    "model",
    load_in_4bit=True,
    device_map="auto"
)

# 4. Use speculative decoding (SGLang)
# Can provide 2x speedup
```

---

## Networking Issues

### Can't access JupyterLab remotely

**Solutions:**
```bash
# 1. Bind to all interfaces
jupyter lab --ip=0.0.0.0 --port=8888

# 2. Check firewall
sudo ufw allow 8888/tcp

# 3. Use SSH tunnel
ssh -L 8888:localhost:8888 user@dgx-spark-ip
# Then access http://localhost:8888 locally
```

### Container can't reach Ollama

**Solutions:**
```bash
# 1. Use host networking
docker run --net=host ...

# 2. Or use host IP from container
# Inside container:
curl http://172.17.0.1:11434/api/tags  # Docker bridge IP

# 3. Or use host.docker.internal
curl http://host.docker.internal:11434/api/tags
```

---

## Performance Issues

### Slower than expected performance

**Diagnosis:**
```python
# Profile your code
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True
) as prof:
    # Your code here
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

**Common fixes:**
```python
# 1. Use appropriate data types
model = model.to(torch.bfloat16)  # Faster than fp32

# 2. Disable gradient computation for inference
with torch.no_grad():
    output = model(input)

# 3. Use torch.compile (if supported)
model = torch.compile(model)

# 4. Increase batch size (within memory limits)
batch_size = 32  # Find optimal for your model

# 5. Use DataLoader workers
DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
```

### High memory bandwidth utilization

DGX Spark has 273 GB/s bandwidth (lower than HBM GPUs). To optimize:

```python
# 1. Use larger batch sizes to improve compute/memory ratio

# 2. Use quantization to reduce memory transfers
model = AutoModelForCausalLM.from_pretrained("model", load_in_4bit=True)

# 3. Use speculative decoding for inference (compensates for bandwidth)
```

---

## Getting More Help

1. **Check NVIDIA Docs**: [DGX Spark Documentation](https://docs.nvidia.com/dgx/dgx-spark/)
2. **NVIDIA Forums**: [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
3. **Open an Issue**: [GitHub Issues](https://github.com/YOUR_USERNAME/dgx-spark-ai-curriculum/issues)
4. **DGX Spark Playbooks**: [NVIDIA Build](https://build.nvidia.com/spark)

### Collecting Debug Information

When reporting issues, include:

```bash
# System info
nvidia-smi
cat /etc/os-release
python --version

# PyTorch info
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# Container info (if applicable)
docker --version
docker images | grep nvidia

# Error logs
journalctl -u ollama --since "1 hour ago"  # For Ollama
docker logs dgx-spark-dev  # For container
```
