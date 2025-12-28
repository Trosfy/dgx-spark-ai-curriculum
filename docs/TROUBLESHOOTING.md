# Troubleshooting Guide

## Common Issues

### PyTorch Not Finding GPU
**Problem:** `torch.cuda.is_available()` returns False

**Solution:** Use NGC containers, not pip-installed PyTorch
```bash
docker pull nvcr.io/nvidia/pytorch:25.11-py3
docker run --gpus all ...
```

### Out of Memory Errors
**Problem:** CUDA OOM despite having 128GB

**Solution:** Clear buffer cache before heavy workloads
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

### vLLM Issues
**Problem:** vLLM errors on DGX Spark

**Solution:** Use NVIDIA container and `--enforce-eager` flag
```bash
docker pull nvcr.io/nvidia/vllm:spark
# Add --enforce-eager to vLLM commands
```

### TensorRT-LLM Attention Errors
**Problem:** Attention sink errors

**Solution:** Set `enable_attention_dp: false` in configuration

## Getting Help

1. Check [NVIDIA DGX Spark Docs](https://docs.nvidia.com/dgx/dgx-spark/)
2. Open an issue on this repository
3. NVIDIA Developer Forums
