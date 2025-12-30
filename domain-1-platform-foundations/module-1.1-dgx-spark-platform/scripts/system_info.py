#!/usr/bin/env python3
"""
DGX Spark System Information Utility

This module provides functions to gather comprehensive system information
for NVIDIA DGX Spark systems, including GPU, CPU, memory, and software details.

Usage:
    # As a module
    from system_info import get_system_info, print_system_report
    info = get_system_info()
    print_system_report(info)

    # As a script
    python system_info.py
    python system_info.py --json  # Output as JSON
    python system_info.py --save  # Save to file

Author: Professor SPARK
License: MIT
"""

import subprocess
import platform
import os
import json
import argparse
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


def run_command(cmd: str, timeout: int = 30) -> str:
    """
    Execute a shell command and return its output.

    Args:
        cmd: Shell command to execute
        timeout: Maximum time to wait for command (seconds)

    Returns:
        Command output as string, or error message

    Example:
        >>> result = run_command("echo 'hello'")
        >>> print(result)
        hello
    """
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


def get_gpu_info() -> Dict[str, Any]:
    """
    Gather NVIDIA GPU information using nvidia-smi.

    Returns:
        Dictionary containing GPU details:
        - name: GPU model name
        - memory_total: Total GPU memory
        - memory_free: Available GPU memory
        - memory_used: Used GPU memory
        - driver_version: NVIDIA driver version
        - cuda_version: CUDA version
        - temperature: Current GPU temperature
        - power_draw: Current power consumption
        - compute_capability: CUDA compute capability

    Example:
        >>> gpu = get_gpu_info()
        >>> print(f"GPU: {gpu['name']} with {gpu['memory_total']}")
    """
    gpu_info = {}

    # Basic GPU info
    gpu_info["name"] = run_command(
        "nvidia-smi --query-gpu=name --format=csv,noheader"
    )
    gpu_info["memory_total"] = run_command(
        "nvidia-smi --query-gpu=memory.total --format=csv,noheader"
    )
    gpu_info["memory_free"] = run_command(
        "nvidia-smi --query-gpu=memory.free --format=csv,noheader"
    )
    gpu_info["memory_used"] = run_command(
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader"
    )
    gpu_info["driver_version"] = run_command(
        "nvidia-smi --query-gpu=driver_version --format=csv,noheader"
    )
    gpu_info["cuda_version"] = run_command(
        "nvidia-smi | grep 'CUDA Version' | awk '{print $9}'"
    )
    gpu_info["temperature"] = run_command(
        "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader"
    )
    gpu_info["power_draw"] = run_command(
        "nvidia-smi --query-gpu=power.draw --format=csv,noheader"
    )
    gpu_info["compute_capability"] = run_command(
        "nvidia-smi --query-gpu=compute_cap --format=csv,noheader"
    )

    # DGX Spark specific: Check for tensor cores
    gpu_info["tensor_cores"] = "192 (5th generation)"  # GB10 spec
    gpu_info["cuda_cores"] = "6,144"  # GB10 spec

    return gpu_info


def get_cpu_info() -> Dict[str, Any]:
    """
    Gather CPU information for the Grace ARM processor.

    Returns:
        Dictionary containing CPU details:
        - model: CPU model name
        - architecture: CPU architecture (aarch64 for Grace)
        - cores: Number of CPU cores
        - threads: Number of threads
        - max_frequency: Maximum CPU frequency
        - cache_l1d: L1 data cache size
        - cache_l1i: L1 instruction cache size
        - cache_l2: L2 cache size
        - cache_l3: L3 cache size

    Example:
        >>> cpu = get_cpu_info()
        >>> print(f"CPU: {cpu['model']} with {cpu['cores']} cores")
    """
    cpu_info = {}

    cpu_info["model"] = run_command(
        "lscpu | grep 'Model name' | head -1 | cut -d':' -f2"
    ).strip()
    cpu_info["architecture"] = run_command(
        "lscpu | grep 'Architecture' | cut -d':' -f2"
    ).strip()
    cpu_info["cores"] = run_command("nproc")
    cpu_info["threads"] = run_command(
        "lscpu | grep '^CPU(s):' | cut -d':' -f2"
    ).strip()
    cpu_info["max_frequency"] = run_command(
        "lscpu | grep 'CPU max MHz' | cut -d':' -f2"
    ).strip()

    # Cache information
    cpu_info["cache_l1d"] = run_command(
        "lscpu | grep 'L1d cache' | cut -d':' -f2"
    ).strip()
    cpu_info["cache_l1i"] = run_command(
        "lscpu | grep 'L1i cache' | cut -d':' -f2"
    ).strip()
    cpu_info["cache_l2"] = run_command(
        "lscpu | grep 'L2 cache' | cut -d':' -f2"
    ).strip()
    cpu_info["cache_l3"] = run_command(
        "lscpu | grep 'L3 cache' | cut -d':' -f2"
    ).strip()

    return cpu_info


def get_memory_info() -> Dict[str, Any]:
    """
    Gather unified memory information.

    Returns:
        Dictionary containing memory details:
        - total_gb: Total system memory in GB
        - used_gb: Used memory in GB
        - free_gb: Free memory in GB
        - available_gb: Available memory in GB
        - cached_gb: Cached memory in GB
        - swap_total_gb: Total swap in GB
        - swap_used_gb: Used swap in GB

    Example:
        >>> mem = get_memory_info()
        >>> print(f"Memory: {mem['total_gb']}GB total, {mem['available_gb']}GB available")
    """
    memory_info = {}

    memory_info["total_gb"] = run_command("free -g | grep Mem | awk '{print $2}'")
    memory_info["used_gb"] = run_command("free -g | grep Mem | awk '{print $3}'")
    memory_info["free_gb"] = run_command("free -g | grep Mem | awk '{print $4}'")
    memory_info["available_gb"] = run_command("free -g | grep Mem | awk '{print $7}'")
    memory_info["cached_gb"] = run_command("free -g | grep Mem | awk '{print $6}'")
    memory_info["swap_total_gb"] = run_command("free -g | grep Swap | awk '{print $2}'")
    memory_info["swap_used_gb"] = run_command("free -g | grep Swap | awk '{print $3}'")

    # DGX Spark specific
    memory_info["type"] = "LPDDR5X"
    memory_info["bandwidth"] = "273 GB/s"
    memory_info["unified"] = True

    return memory_info


def get_storage_info() -> Dict[str, Any]:
    """
    Gather storage information.

    Returns:
        Dictionary containing storage details for key mount points

    Example:
        >>> storage = get_storage_info()
        >>> print(f"Root: {storage['root']['total']} total")
    """
    storage_info = {}

    # Root partition
    storage_info["root"] = {
        "mount": "/",
        "total": run_command("df -h / | tail -1 | awk '{print $2}'"),
        "used": run_command("df -h / | tail -1 | awk '{print $3}'"),
        "available": run_command("df -h / | tail -1 | awk '{print $4}'"),
        "use_percent": run_command("df -h / | tail -1 | awk '{print $5}'")
    }

    # Home partition (if separate)
    home_info = run_command("df -h /home 2>/dev/null | tail -1")
    if home_info and "/home" in home_info:
        storage_info["home"] = {
            "mount": "/home",
            "total": run_command("df -h /home | tail -1 | awk '{print $2}'"),
            "used": run_command("df -h /home | tail -1 | awk '{print $3}'"),
            "available": run_command("df -h /home | tail -1 | awk '{print $4}'"),
            "use_percent": run_command("df -h /home | tail -1 | awk '{print $5}'")
        }

    # Model cache locations
    hf_cache = Path.home() / ".cache" / "huggingface"
    ollama_cache = Path.home() / ".ollama"

    if hf_cache.exists():
        storage_info["huggingface_cache"] = run_command(f"du -sh {hf_cache} | awk '{{print $1}}'")

    if ollama_cache.exists():
        storage_info["ollama_cache"] = run_command(f"du -sh {ollama_cache} | awk '{{print $1}}'")

    return storage_info


def get_software_info() -> Dict[str, Any]:
    """
    Gather software version information.

    Returns:
        Dictionary containing software versions

    Example:
        >>> sw = get_software_info()
        >>> print(f"Docker: {sw['docker']}")
    """
    software_info = {}

    software_info["os_name"] = run_command(
        "cat /etc/os-release | grep PRETTY_NAME | cut -d'=' -f2 | tr -d '\"'"
    )
    software_info["kernel"] = platform.release()
    software_info["python"] = platform.python_version()
    software_info["docker"] = run_command(
        "docker --version 2>/dev/null | cut -d' ' -f3 | tr -d ','"
    )
    software_info["ollama"] = run_command("ollama --version 2>/dev/null")
    software_info["nvidia_container_toolkit"] = run_command(
        "nvidia-container-cli --version 2>/dev/null | head -1"
    )

    return software_info


def get_system_info() -> Dict[str, Any]:
    """
    Gather comprehensive system information for DGX Spark.

    Returns:
        Dictionary containing all system information:
        - timestamp: When the info was gathered
        - hostname: System hostname
        - gpu: GPU information
        - cpu: CPU information
        - memory: Memory information
        - storage: Storage information
        - software: Software versions

    Example:
        >>> info = get_system_info()
        >>> print(json.dumps(info, indent=2))
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "hostname": platform.node(),
        "gpu": get_gpu_info(),
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "storage": get_storage_info(),
        "software": get_software_info()
    }


def print_system_report(info: Dict[str, Any]) -> None:
    """
    Print a formatted system report.

    Args:
        info: System information dictionary from get_system_info()

    Example:
        >>> info = get_system_info()
        >>> print_system_report(info)
    """
    print("=" * 70)
    print("                    DGX SPARK SYSTEM SPECIFICATION")
    print("=" * 70)
    print(f"\nGenerated: {info['timestamp']}")
    print(f"Hostname:  {info['hostname']}")

    print("\n" + "-" * 70)
    print("GPU - NVIDIA Blackwell GB10")
    print("-" * 70)
    gpu = info['gpu']
    print(f"  Model:              {gpu['name']}")
    print(f"  Memory Total:       {gpu['memory_total']}")
    print(f"  Memory Free:        {gpu['memory_free']}")
    print(f"  Memory Used:        {gpu['memory_used']}")
    print(f"  CUDA Cores:         {gpu['cuda_cores']}")
    print(f"  Tensor Cores:       {gpu['tensor_cores']}")
    print(f"  Temperature:        {gpu['temperature']}C")
    print(f"  Power Draw:         {gpu['power_draw']}")
    print(f"  Driver Version:     {gpu['driver_version']}")
    print(f"  CUDA Version:       {gpu['cuda_version']}")
    print(f"  Compute Capability: {gpu['compute_capability']}")

    print("\n" + "-" * 70)
    print("CPU - NVIDIA Grace ARM")
    print("-" * 70)
    cpu = info['cpu']
    print(f"  Model:        {cpu['model']}")
    print(f"  Architecture: {cpu['architecture']}")
    print(f"  Cores:        {cpu['cores']}")
    print(f"  Threads:      {cpu['threads']}")
    if cpu['max_frequency']:
        print(f"  Max Freq:     {cpu['max_frequency']} MHz")
    print(f"  L1d Cache:    {cpu['cache_l1d']}")
    print(f"  L1i Cache:    {cpu['cache_l1i']}")
    print(f"  L2 Cache:     {cpu['cache_l2']}")
    print(f"  L3 Cache:     {cpu['cache_l3']}")

    print("\n" + "-" * 70)
    print("MEMORY - Unified LPDDR5X")
    print("-" * 70)
    mem = info['memory']
    print(f"  Total:        {mem['total_gb']} GB")
    print(f"  Available:    {mem['available_gb']} GB")
    print(f"  Used:         {mem['used_gb']} GB")
    print(f"  Cached:       {mem['cached_gb']} GB")
    print(f"  Type:         {mem['type']}")
    print(f"  Bandwidth:    {mem['bandwidth']}")
    print(f"  Unified:      {'Yes' if mem['unified'] else 'No'}")

    print("\n" + "-" * 70)
    print("STORAGE")
    print("-" * 70)
    storage = info['storage']
    root = storage['root']
    print(f"  Root ({root['mount']}):  {root['total']} total, {root['available']} available ({root['use_percent']} used)")
    if 'home' in storage:
        home = storage['home']
        print(f"  Home ({home['mount']}): {home['total']} total, {home['available']} available ({home['use_percent']} used)")
    if 'huggingface_cache' in storage:
        print(f"  HuggingFace Cache: {storage['huggingface_cache']}")
    if 'ollama_cache' in storage:
        print(f"  Ollama Cache:      {storage['ollama_cache']}")

    print("\n" + "-" * 70)
    print("SOFTWARE")
    print("-" * 70)
    sw = info['software']
    print(f"  OS:           {sw['os_name']}")
    print(f"  Kernel:       {sw['kernel']}")
    print(f"  Python:       {sw['python']}")
    print(f"  Docker:       {sw['docker']}")
    print(f"  Ollama:       {sw['ollama']}")

    print("\n" + "=" * 70)


def save_system_info(info: Dict[str, Any], filepath: Optional[str] = None) -> str:
    """
    Save system information to a JSON file.

    Args:
        info: System information dictionary
        filepath: Output file path (default: auto-generated)

    Returns:
        Path to saved file

    Example:
        >>> info = get_system_info()
        >>> path = save_system_info(info)
        >>> print(f"Saved to: {path}")
    """
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"dgx_spark_system_info_{timestamp}.json"

    with open(filepath, 'w') as f:
        json.dump(info, f, indent=2)

    return filepath


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Gather and display DGX Spark system information"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save to file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path (implies --save)"
    )

    args = parser.parse_args()

    info = get_system_info()

    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print_system_report(info)

    if args.save or args.output:
        filepath = save_system_info(info, args.output)
        print(f"\nSaved to: {filepath}")


if __name__ == "__main__":
    main()
