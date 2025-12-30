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
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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
except ImportError:
    # utils not available, running standalone
    pass
