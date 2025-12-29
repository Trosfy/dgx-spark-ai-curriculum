#!/usr/bin/env python3
"""
DGX Spark Utility Functions
Helper functions for common DGX Spark operations
"""

import subprocess
import os
import torch
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class SystemInfo:
    """DGX Spark system information"""
    gpu_name: str
    gpu_memory_total_gb: float
    gpu_memory_used_gb: float
    gpu_memory_free_gb: float
    cuda_version: str
    driver_version: str
    cpu_model: str
    cpu_cores: int
    ram_total_gb: float
    ram_available_gb: float


def get_system_info() -> SystemInfo:
    """Get comprehensive DGX Spark system information"""
    
    # GPU info via nvidia-smi
    gpu_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    gpu_parts = gpu_result.stdout.strip().split(", ")
    
    # CUDA version
    cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"
    
    # CPU info
    cpu_result = subprocess.run(["lscpu"], capture_output=True, text=True)
    cpu_lines = cpu_result.stdout.split("\n")
    cpu_model = ""
    cpu_cores = 0
    for line in cpu_lines:
        if "Model name" in line:
            cpu_model = line.split(":")[1].strip()
        if "CPU(s):" in line and "NUMA" not in line:
            cpu_cores = int(line.split(":")[1].strip())
    
    # RAM info
    mem_result = subprocess.run(["free", "-g"], capture_output=True, text=True)
    mem_lines = mem_result.stdout.split("\n")
    mem_parts = mem_lines[1].split()
    ram_total = float(mem_parts[1])
    ram_available = float(mem_parts[6])
    
    return SystemInfo(
        gpu_name=gpu_parts[0],
        gpu_memory_total_gb=float(gpu_parts[1]) / 1024,
        gpu_memory_used_gb=float(gpu_parts[2]) / 1024,
        gpu_memory_free_gb=float(gpu_parts[3]) / 1024,
        cuda_version=cuda_version,
        driver_version=gpu_parts[4],
        cpu_model=cpu_model,
        cpu_cores=cpu_cores,
        ram_total_gb=ram_total,
        ram_available_gb=ram_available
    )


def print_system_info():
    """Print formatted system information"""
    info = get_system_info()
    
    print("=" * 60)
    print("DGX Spark System Information")
    print("=" * 60)
    print(f"GPU:           {info.gpu_name}")
    print(f"GPU Memory:    {info.gpu_memory_total_gb:.1f} GB total, {info.gpu_memory_used_gb:.1f} GB used, {info.gpu_memory_free_gb:.1f} GB free")
    print(f"CUDA Version:  {info.cuda_version}")
    print(f"Driver:        {info.driver_version}")
    print(f"CPU:           {info.cpu_model}")
    print(f"CPU Cores:     {info.cpu_cores}")
    print(f"RAM:           {info.ram_total_gb:.0f} GB total, {info.ram_available_gb:.0f} GB available")
    print("=" * 60)


def clear_buffer_cache(require_sudo: bool = True) -> bool:
    """
    Clear Linux buffer cache to free memory for GPU operations.
    
    CRITICAL for DGX Spark unified memory - run before loading large models.
    
    Args:
        require_sudo: Whether to use sudo (default True)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if require_sudo:
            subprocess.run(
                ["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
                check=True
            )
        else:
            subprocess.run(["sync"], check=True)
            with open("/proc/sys/vm/drop_caches", "w") as f:
                f.write("3")
        print("✓ Buffer cache cleared")
        return True
    except Exception as e:
        print(f"✗ Failed to clear buffer cache: {e}")
        return False


def clear_gpu_memory():
    """Clear PyTorch GPU memory cache"""
    import gc
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        
        print(f"✓ GPU memory cleared")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")


def get_gpu_memory_usage() -> Dict[str, float]:
    """Get current GPU memory usage in GB"""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "total": 0}
    
    return {
        "allocated": torch.cuda.memory_allocated() / 1e9,
        "reserved": torch.cuda.memory_reserved() / 1e9,
        "total": torch.cuda.get_device_properties(0).total_memory / 1e9
    }


def check_ngc_container() -> bool:
    """Check if running inside NGC container"""
    # NGC containers have specific environment variables
    ngc_indicators = [
        os.path.exists("/opt/nvidia"),
        "NGC" in os.environ.get("NVIDIA_PRODUCT_NAME", ""),
        os.path.exists("/etc/shinit_v2")
    ]
    return any(ngc_indicators)


def verify_environment() -> Dict[str, Any]:
    """Verify DGX Spark environment is properly configured"""
    results = {
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": None,
        "cuda_version": None,
        "pytorch_version": torch.__version__,
        "ngc_container": check_ngc_container(),
        "huggingface_cache": os.environ.get("HF_HOME", "~/.cache/huggingface"),
        "issues": []
    }
    
    if results["gpu_available"]:
        results["gpu_name"] = torch.cuda.get_device_name(0)
        results["cuda_version"] = torch.version.cuda
    else:
        results["issues"].append("GPU not available - ensure --gpus all flag is used")
    
    if not results["ngc_container"]:
        results["issues"].append("Not running in NGC container - PyTorch may not work correctly")
    
    # Check for common issues
    try:
        import transformers
        results["transformers_version"] = transformers.__version__
    except ImportError:
        results["issues"].append("transformers not installed")
    
    return results


def optimal_batch_size(model_size_b: float, sequence_length: int = 2048) -> int:
    """
    Estimate optimal batch size for a given model size on DGX Spark.
    
    Args:
        model_size_b: Model size in billions of parameters
        sequence_length: Context length
        
    Returns:
        Recommended batch size
    """
    # DGX Spark has 128GB unified memory
    available_memory_gb = 128
    
    # Rough estimate: ~2 bytes per parameter for BF16
    # Plus activation memory which scales with batch size and sequence length
    model_memory_gb = model_size_b * 2
    
    # Reserve memory for model and some overhead
    remaining_gb = available_memory_gb - model_memory_gb - 10  # 10GB overhead
    
    # Activation memory per sample (very rough estimate)
    activation_per_sample_gb = (sequence_length * model_size_b * 0.001)
    
    if activation_per_sample_gb > 0:
        batch_size = int(remaining_gb / activation_per_sample_gb)
        return max(1, min(batch_size, 64))  # Cap at 64
    
    return 8  # Default fallback


def recommended_quantization(model_size_b: float) -> str:
    """
    Recommend quantization strategy based on model size.
    
    Args:
        model_size_b: Model size in billions of parameters
        
    Returns:
        Recommended quantization string
    """
    if model_size_b <= 8:
        return "BF16 or FP8 (full precision feasible)"
    elif model_size_b <= 14:
        return "FP8 or INT8"
    elif model_size_b <= 35:
        return "INT8 or GPTQ 4-bit"
    elif model_size_b <= 70:
        return "GPTQ 4-bit or QLoRA (NF4)"
    else:
        return "GPTQ 4-bit or NVFP4 (Blackwell)"


if __name__ == "__main__":
    print_system_info()
    
    print("\nEnvironment Verification:")
    env = verify_environment()
    for key, value in env.items():
        if key != "issues":
            print(f"  {key}: {value}")
    
    if env["issues"]:
        print("\nIssues detected:")
        for issue in env["issues"]:
            print(f"  ⚠ {issue}")
