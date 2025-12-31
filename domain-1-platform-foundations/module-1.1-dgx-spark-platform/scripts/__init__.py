"""
Module 01: DGX Spark Platform - Scripts Package

This package contains utility scripts for the DGX Spark Platform module.

Available modules:
- system_info: Functions to gather system information

For benchmarking and memory utilities, use the consolidated utils package:
    from utils.benchmarks import OllamaBenchmark, quick_benchmark
    from utils.monitoring import MemoryMonitor, get_memory_snapshot
"""

from .system_info import *

# Re-export consolidated utilities for convenience
import sys
from pathlib import Path

# Add project root to path for importing shared utilities
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Track what utilities are available
_UTILS_AVAILABLE = False

try:
    from utils.benchmarks import (
        BenchmarkResult,
        BenchmarkSummary,
        OllamaBenchmark,
        quick_benchmark,
    )
    from utils.monitoring import (
        MemorySnapshot,
        get_memory_snapshot,
        print_memory_status,
        MemoryMonitor,
    )
    _UTILS_AVAILABLE = True
except ImportError:
    # utils package not available when running standalone
    # This is expected when the module is used independently
    BenchmarkResult = None
    BenchmarkSummary = None
    OllamaBenchmark = None
    quick_benchmark = None
    MemorySnapshot = None
    get_memory_snapshot = None
    print_memory_status = None
    MemoryMonitor = None


def is_utils_available() -> bool:
    """Check if the shared utils package is available."""
    return _UTILS_AVAILABLE
