"""
CUDA Utilities for Module 1.3

Core utilities for GPU programming with Numba CUDA.

Example usage:
    >>> from cuda_utils import get_device_info, optimal_block_size
    >>> info = get_device_info()
    >>> print(f"Device: {info['name']}")
    >>> print(f"Optimal block size for 1M elements: {optimal_block_size(1_000_000)}")
"""

from typing import Dict, Optional, Tuple, Any
import gc


def check_cuda_available() -> bool:
    """
    Check if CUDA is available via Numba.

    Returns:
        bool: True if CUDA is available, False otherwise

    Example:
        >>> if check_cuda_available():
        ...     print("GPU ready!")
    """
    try:
        from numba import cuda
        return cuda.is_available()
    except ImportError:
        return False


def get_device_info(device_id: int = 0) -> Dict[str, Any]:
    """
    Get comprehensive information about the CUDA device.

    Args:
        device_id: CUDA device ID (default 0)

    Returns:
        Dictionary with device properties

    Example:
        >>> info = get_device_info()
        >>> print(f"Device: {info['name']}")
        >>> print(f"Compute capability: {info['compute_capability']}")
        >>> print(f"Total memory: {info['total_memory_gb']:.1f} GB")
    """
    from numba import cuda

    if not cuda.is_available():
        return {"error": "CUDA not available"}

    device = cuda.get_current_device()

    return {
        "name": device.name.decode() if isinstance(device.name, bytes) else str(device.name),
        "compute_capability": device.compute_capability,
        "max_threads_per_block": device.MAX_THREADS_PER_BLOCK,
        "max_block_dim_x": device.MAX_BLOCK_DIM_X,
        "max_block_dim_y": device.MAX_BLOCK_DIM_Y,
        "max_block_dim_z": device.MAX_BLOCK_DIM_Z,
        "max_grid_dim_x": device.MAX_GRID_DIM_X,
        "max_grid_dim_y": device.MAX_GRID_DIM_Y,
        "max_grid_dim_z": device.MAX_GRID_DIM_Z,
        "max_shared_memory_per_block": device.MAX_SHARED_MEMORY_PER_BLOCK,
        "total_memory_bytes": device.total_memory if hasattr(device, 'total_memory') else None,
        "total_memory_gb": device.total_memory / 1e9 if hasattr(device, 'total_memory') else None,
        "warp_size": device.WARP_SIZE,
        "multiprocessor_count": device.MULTIPROCESSOR_COUNT,
    }


def get_memory_info() -> Dict[str, float]:
    """
    Get current GPU memory usage.

    Returns:
        Dictionary with memory info in GB

    Example:
        >>> mem = get_memory_info()
        >>> print(f"Used: {mem['used_gb']:.2f} GB / {mem['total_gb']:.2f} GB")
    """
    try:
        from numba import cuda

        # Get memory info from context
        ctx = cuda.current_context()
        free, total = ctx.get_memory_info()

        return {
            "free_bytes": free,
            "total_bytes": total,
            "used_bytes": total - free,
            "free_gb": free / 1e9,
            "total_gb": total / 1e9,
            "used_gb": (total - free) / 1e9,
            "utilization_pct": (total - free) / total * 100,
        }
    except Exception as e:
        return {"error": str(e)}


def clear_gpu_memory() -> None:
    """
    Clear GPU memory by resetting the CUDA context and running garbage collection.

    Warning: This will invalidate all existing GPU arrays!

    Example:
        >>> clear_gpu_memory()
        >>> print("GPU memory cleared")
    """
    from numba import cuda

    # Python garbage collection
    gc.collect()

    # Reset CUDA context
    try:
        cuda.current_context().reset()
    except Exception:
        pass  # Context might not exist


def optimal_block_size(n_elements: int, max_threads: int = 256) -> int:
    """
    Calculate optimal threads per block for a given problem size.

    Args:
        n_elements: Total number of elements to process
        max_threads: Maximum threads per block (default 256)

    Returns:
        Recommended threads per block

    Note:
        - Returns power of 2 for best performance
        - Minimum 32 (warp size)
        - Maximum capped at max_threads

    Example:
        >>> threads = optimal_block_size(1000)
        >>> print(f"Use {threads} threads per block")
    """
    from numba import cuda

    if not cuda.is_available():
        return min(max_threads, 256)

    device = cuda.get_current_device()
    max_allowed = min(max_threads, device.MAX_THREADS_PER_BLOCK)

    # Find largest power of 2 that divides evenly or is <= n_elements
    if n_elements <= 32:
        return 32

    # Start with max and work down
    for threads in [256, 128, 64, 32]:
        if threads <= max_allowed:
            return threads

    return 32  # Minimum warp size


def optimal_grid_size(n_elements: int, threads_per_block: int) -> int:
    """
    Calculate number of blocks needed for a kernel launch.

    Args:
        n_elements: Total number of elements to process
        threads_per_block: Number of threads per block

    Returns:
        Number of blocks needed

    Example:
        >>> threads = 256
        >>> blocks = optimal_grid_size(1_000_000, threads)
        >>> print(f"Launch with {blocks} blocks × {threads} threads")
    """
    return (n_elements + threads_per_block - 1) // threads_per_block


def calculate_occupancy(
    threads_per_block: int,
    shared_mem_per_block: int = 0,
    registers_per_thread: int = 32
) -> Dict[str, Any]:
    """
    Estimate kernel occupancy.

    Args:
        threads_per_block: Number of threads per block
        shared_mem_per_block: Bytes of shared memory per block
        registers_per_thread: Registers used per thread

    Returns:
        Dictionary with occupancy metrics

    Example:
        >>> occ = calculate_occupancy(256, shared_mem_per_block=4096)
        >>> print(f"Estimated occupancy: {occ['occupancy_pct']:.0f}%")
    """
    from numba import cuda

    if not cuda.is_available():
        return {"error": "CUDA not available"}

    device = cuda.get_current_device()

    # Simplified occupancy calculation
    # Real occupancy depends on many factors
    max_threads_per_sm = 2048  # Typical for modern GPUs
    max_blocks_per_sm = 32
    max_shared_per_sm = device.MAX_SHARED_MEMORY_PER_BLOCK * 2  # Approximate

    # Limit by threads
    blocks_by_threads = max_threads_per_sm // threads_per_block

    # Limit by shared memory
    if shared_mem_per_block > 0:
        blocks_by_shared = max_shared_per_sm // shared_mem_per_block
    else:
        blocks_by_shared = max_blocks_per_sm

    # Actual blocks per SM
    blocks_per_sm = min(blocks_by_threads, blocks_by_shared, max_blocks_per_sm)

    # Active threads
    active_threads = blocks_per_sm * threads_per_block
    occupancy = active_threads / max_threads_per_sm

    return {
        "blocks_per_sm": blocks_per_sm,
        "active_threads_per_sm": active_threads,
        "max_threads_per_sm": max_threads_per_sm,
        "occupancy_pct": occupancy * 100,
        "limited_by": "threads" if blocks_by_threads < blocks_by_shared else "shared_memory",
    }


def create_device_array(shape: Tuple[int, ...], dtype="float32"):
    """
    Create an uninitialized array on the GPU.

    Args:
        shape: Array shape
        dtype: Data type (default float32)

    Returns:
        Numba device array

    Example:
        >>> d_array = create_device_array((1000, 1000), dtype="float32")
        >>> print(f"Created {d_array.shape} array on GPU")
    """
    from numba import cuda
    import numpy as np

    return cuda.device_array(shape, dtype=np.dtype(dtype))


def to_device(array):
    """
    Copy a NumPy array to the GPU.

    Args:
        array: NumPy array

    Returns:
        Numba device array

    Example:
        >>> import numpy as np
        >>> host_array = np.random.randn(1000).astype(np.float32)
        >>> device_array = to_device(host_array)
    """
    from numba import cuda
    return cuda.to_device(array)


def to_host(device_array):
    """
    Copy a device array back to the host (CPU).

    Args:
        device_array: Numba device array

    Returns:
        NumPy array

    Example:
        >>> host_array = to_host(device_array)
        >>> print(host_array[:10])
    """
    return device_array.copy_to_host()


def synchronize():
    """
    Synchronize the CUDA device (wait for all operations to complete).

    Example:
        >>> kernel[blocks, threads](data)
        >>> synchronize()  # Wait for kernel to finish
        >>> result = to_host(data)
    """
    from numba import cuda
    cuda.synchronize()


if __name__ == "__main__":
    # Demo
    print("CUDA Utils Demo")
    print("=" * 50)

    if check_cuda_available():
        info = get_device_info()
        print(f"\nDevice: {info['name']}")
        print(f"Compute capability: {info['compute_capability']}")
        print(f"Max threads per block: {info['max_threads_per_block']}")
        print(f"Multiprocessors: {info['multiprocessor_count']}")

        mem = get_memory_info()
        print(f"\nMemory: {mem['used_gb']:.2f} / {mem['total_gb']:.2f} GB")

        print(f"\nOptimal block size for 1M elements: {optimal_block_size(1_000_000)}")
    else:
        print("CUDA not available")
