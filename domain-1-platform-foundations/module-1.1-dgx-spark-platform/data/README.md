# Module 1.1 Data Files

This directory contains data files and outputs generated during Module 1.1.1: DGX Spark Platform Mastery.

## Generated Files

During the course of completing Module 1.1, the following files will be generated:

### From Lab 1.1.1: System Exploration
- `system_info.json` - Comprehensive system specification in JSON format
- Screenshots of `nvidia-smi` output (optional)

### From Lab 1.1.3: NGC Container Setup
- `docker-compose.yml` - Docker Compose configuration for development
- `start_pytorch.sh` - Shell script launcher for PyTorch container
- `verify_gpu.py` - GPU verification script

### From Lab 1.1.4: Compatibility Matrix
- `compatibility_matrix.md` - Markdown table of tool compatibility
- `compatibility_matrix.json` - JSON format for programmatic use

### From Lab 1.1.5: Ollama Benchmarking
- `benchmark_results_[timestamp].json` - Raw benchmark data
- `benchmark_report.md` - Formatted benchmark report

## Sample Data

No external data files are required for Module 1.1. All exercises use:
- System commands (nvidia-smi, lscpu, free, df)
- PyTorch tensor allocations
- Ollama API for model benchmarking

## Expected Outputs

### Benchmark Results Table (Example)
```
| Model         | Prefill (tok/s) | Decode (tok/s) | Memory (GB) |
|---------------|-----------------|----------------|-------------|
| qwen3:3b      | ~5,000         | ~80            | ~3          |
| qwen3:8b      | ~3,000         | ~45            | ~8          |
| qwen3:32b     | ~1,000         | ~25            | ~20         |
```

### System Specification (Example)
```json
{
  "timestamp": "2025-01-15T10:30:00",
  "hostname": "dgx-spark-001",
  "gpu": {
    "name": "NVIDIA Graphics Device",
    "memory_total": "128 GB",
    "cuda_version": "13.0"
  },
  "cpu": {
    "architecture": "aarch64",
    "cores": "20"
  },
  "memory": {
    "total_gb": "128",
    "unified": true
  }
}
```

## Notes

1. **File Locations**: Generated files are typically saved in the notebook's working directory. Move them here for organization.

2. **Benchmark Variance**: Performance numbers may vary based on:
   - System load
   - Buffer cache state
   - Model quantization level
   - Prompt length

3. **Compatibility Updates**: The compatibility matrix should be updated as the ecosystem evolves. New tool versions may change support status.

## Cleaning Up

To remove generated files and start fresh:
```bash
# Remove generated files (keeps this README)
rm -f system_info.json benchmark_results_*.json compatibility_matrix.json
rm -f benchmark_report.md compatibility_matrix.md
rm -f docker-compose.yml start_pytorch.sh verify_gpu.py
```

Keep benchmark results if you want to track performance over time!
