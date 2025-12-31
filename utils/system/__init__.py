"""
DGX Spark System Utilities
==========================

System information, environment verification, and hardware utilities
for the NVIDIA DGX Spark platform.

Usage:
    from utils.system import get_system_info, print_system_info, verify_environment

    # Get system information
    info = get_system_info()
    print(f"GPU: {info.gpu_name}")
    print(f"Memory: {info.gpu_memory_total_gb:.1f} GB")

    # Verify environment
    env = verify_environment()
    if env["issues"]:
        print("Issues detected:", env["issues"])

    # Clear caches before loading large models
    clear_buffer_cache()
    clear_gpu_memory()
"""

from .dgx_spark import (
    SystemInfo,
    check_ngc_container,
    clear_buffer_cache,
    clear_gpu_memory,
    get_gpu_memory_usage,
    get_system_info,
    optimal_batch_size,
    print_system_info,
    recommended_quantization,
    verify_environment,
)

__all__ = [
    "SystemInfo",
    "get_system_info",
    "print_system_info",
    "clear_buffer_cache",
    "clear_gpu_memory",
    "get_gpu_memory_usage",
    "check_ngc_container",
    "verify_environment",
    "optimal_batch_size",
    "recommended_quantization",
]
