# Module 1.1: DGX Spark Platform Mastery - FAQ

## Setup & Environment

### Q: Why can't I just `pip install torch` on DGX Spark?
**A**: DGX Spark uses ARM64 (aarch64) architecture, not x86_64. Standard PyPI PyTorch wheels are only built for x86_64. NGC containers include PyTorch pre-built for ARM64 with GPU support.

**See also**: [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for import errors

---

### Q: Do I need to pull a new container for every module?
**A**: No. The PyTorch NGC container (`nvcr.io/nvidia/pytorch:25.11-py3`) works for most modules. You only need different containers for:
- RAPIDS/cuML (Module 1.6): Use `nvcr.io/nvidia/rapidsai/base:25.11-py3`
- Some specialized tools

---

### Q: How do I keep my installed packages between container sessions?
**A**: Use a requirements file or bind mount a pip cache:

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

### Q: Why is `--ipc=host` recommended in docker run commands?
**A**: The `--ipc=host` flag shares the host's IPC namespace with the container. This is needed for:
- PyTorch DataLoader with `num_workers > 0` (uses shared memory)
- Multi-process training
- Large batch training

Without it, you may see "DataLoader worker exited unexpectedly" errors.

---

## Hardware Concepts

### Q: What's the difference between unified memory and regular GPU memory?
**A**:

| Aspect | Unified (DGX Spark) | Discrete (RTX/A100) |
|--------|---------------------|---------------------|
| Memory pool | Shared 128GB | Separate (e.g., 24GB GPU + 64GB RAM) |
| Transfers | Automatic, no copies | Explicit `.to(device)` copies |
| Max model size | ~128GB | Limited to GPU VRAM |
| Bandwidth | 273 GB/s shared | GPU: 900+ GB/s, PCIe: ~32 GB/s |

**See also**: [ELI5.md](./ELI5.md) for a simpler explanation

---

### Q: What do NVFP4, FP8, and BF16 mean?
**A**: These are numerical precision formats:

| Format | Bits | Best For | DGX Spark Speed |
|--------|------|----------|-----------------|
| NVFP4 | 4 | Inference only | 1 PFLOP |
| FP8 | 8 | Inference | ~209 TFLOPS |
| BF16 | 16 | Training + Inference | ~100 TFLOPS |
| FP32 | 32 | Debugging, legacy | ~31 TFLOPS |

Use BF16 for training, FP8/NVFP4 for inference when accuracy permits.

---

### Q: How big of a model can I run on DGX Spark?
**A**: Depends on precision and whether you're training or inferring:

| Task | Precision | Approximate Max |
|------|-----------|-----------------|
| Inference | FP16 | 50-55B parameters |
| Inference | FP8 | 90-100B parameters |
| Inference | NVFP4 | ~200B parameters |
| Fine-tuning (full) | FP16 | 12-16B parameters |
| Fine-tuning (QLoRA) | 4-bit | 100-120B parameters |

---

## Ollama & Models

### Q: Why use Ollama API calls instead of the Web UI for benchmarks?
**A**: The Web UI adds rendering overhead that skews timing measurements. Direct API calls give you:
- Exact token counts (`prompt_eval_count`, `eval_count`)
- Precise timing in nanoseconds (`prompt_eval_duration`, `eval_duration`)
- No UI latency

```python
# Direct API gives accurate metrics
response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "llama3.1:8b", "prompt": "Hello", "stream": False}
)
```

---

### Q: How do I unload a model from Ollama to free memory?
**A**: Send a generate request with `keep_alive: 0`:

```bash
curl http://localhost:11434/api/generate -d '{"model": "llama3.1:8b", "keep_alive": 0}'
```

Or wait for the default timeout (5 minutes).

---

### Q: Which Ollama model variants should I download?
**A**: For this module, download at minimum:

| Model | Size | Required? |
|-------|------|-----------|
| `llama3.2:3b` | ~2 GB | Yes - quick tests |
| `llama3.1:8b` | ~5 GB | Yes - main benchmark |
| `llama3.1:70b` | ~45 GB | Recommended - shows unified memory advantage |

---

## Memory Management

### Q: Why do I need to "clear buffer cache" before loading models?
**A**: Linux aggressively caches file data in RAM for faster access. On DGX Spark, this cache competes with GPU allocations. Clearing it:

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

Frees that memory for your model. This is especially important before loading 70B+ models.

---

### Q: Why does `nvidia-smi` show less memory than 128GB?
**A**: You may see ~126-127 GB usable because:
1. Small amount reserved for GPU firmware/ECC
2. Display buffers (if using GUI)
3. System overhead

This is normal. You still have access to unified memory.

---

### Q: My Python session shows memory but I can't allocate more tensors. Why?
**A**: PyTorch caches GPU memory for reuse. Run:

```python
import torch, gc
torch.cuda.empty_cache()
gc.collect()
```

If still stuck, also clear buffer cache from terminal.

---

## Compatibility

### Q: Can I use vLLM on DGX Spark?
**A**: Partial support. vLLM has ARM64 support but requires the `--enforce-eager` flag. Current status:
- Basic inference: Works with `--enforce-eager` flag
- Full features: Some may not work
- Workaround: `vllm serve model --enforce-eager`
- Recommendation: Use Ollama, SGLang, or TensorRT-LLM for production (SGLang is 29-45% faster)

---

### Q: Why is my favorite tool marked as "not compatible"?
**A**: Common reasons:
1. **No ARM64 support**: Tool only has x86 binaries
2. **CUDA version mismatch**: Requires different CUDA than DGX Spark
3. **Architecture assumptions**: Hardcoded x86 assembly or instructions

Check if there's:
- An NGC container version
- A conda-forge ARM64 package
- Source code you can build

---

### Q: Can I run Windows or x86 Linux containers on DGX Spark?
**A**: No. DGX Spark runs ARM64 Linux. You can only run ARM64 Linux containers. This is why NGC containers are essential—they're built for ARM64.

---

## Beyond the Basics

### Q: How does DGX Spark compare to cloud GPU instances?
**A**:

| Aspect | DGX Spark | Cloud (A100 80GB) |
|--------|-----------|-------------------|
| Memory | 128GB unified | 80GB GPU + separate RAM |
| Cost | One-time purchase | Hourly (~$2-4/hr) |
| Max model (inference) | ~100B FP8 | ~40B FP16 |
| Best for | Local development, privacy | Burst capacity, training |

---

### Q: Can I use DGX Spark for training, not just inference?
**A**: Yes, but with limits:
- **Full fine-tuning**: Models up to ~12-16B parameters
- **LoRA/QLoRA**: Models up to ~100B parameters
- **From-scratch training**: Good for experimentation, not production scale

For training large models from scratch, cloud or data center GPUs are more appropriate.

---

### Q: How do I maximize performance on DGX Spark?
**A**:
1. Use BF16 for training (not FP32)
2. Use FP8/NVFP4 quantization for inference
3. Use NGC containers (optimized for ARM64)
4. Clear buffer cache before large operations
5. Use batch sizes that fit in memory (larger = faster per-sample)

---

## Still Have Questions?

- Check [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for error-specific help
- Review [ELI5.md](./ELI5.md) for concept explanations
- See [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) for commands
