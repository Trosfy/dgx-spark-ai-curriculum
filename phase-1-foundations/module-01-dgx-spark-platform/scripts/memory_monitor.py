#!/usr/bin/env python3
"""
DGX Spark Memory Monitor Utility

Real-time monitoring of GPU and system memory for DGX Spark's unified memory
architecture. Provides continuous updates useful during model loading and inference.

Usage:
    # Monitor every 2 seconds
    python memory_monitor.py

    # Monitor with custom interval
    python memory_monitor.py --interval 5

    # Monitor with GPU process list
    python memory_monitor.py --processes

    # Log to file
    python memory_monitor.py --log memory.csv

Author: Professor SPARK
License: MIT
"""

import subprocess
import time
import argparse
import signal
import sys
from datetime import datetime
from typing import Dict, Optional, List

# ANSI color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def run_command(cmd: str) -> str:
    """Execute shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except Exception:
        return ""


def get_memory_stats() -> Dict[str, float]:
    """
    Get current memory statistics.

    Returns:
        Dictionary with memory metrics in GB:
        - gpu_total: Total GPU memory
        - gpu_used: Used GPU memory
        - gpu_free: Free GPU memory
        - sys_total: Total system memory
        - sys_used: Used system memory
        - sys_available: Available system memory
        - sys_cached: Cached memory
        - gpu_temp: GPU temperature
        - gpu_power: GPU power draw
    """
    stats = {}

    # GPU memory (in MiB from nvidia-smi, convert to GB)
    gpu_total = run_command("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits")
    gpu_used = run_command("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")
    gpu_free = run_command("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits")
    gpu_temp = run_command("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits")
    gpu_power = run_command("nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits")

    try:
        stats['gpu_total'] = float(gpu_total) / 1024  # MiB to GB
        stats['gpu_used'] = float(gpu_used) / 1024
        stats['gpu_free'] = float(gpu_free) / 1024
        stats['gpu_temp'] = float(gpu_temp) if gpu_temp else 0
        stats['gpu_power'] = float(gpu_power) if gpu_power else 0
    except (ValueError, TypeError):
        stats['gpu_total'] = stats['gpu_used'] = stats['gpu_free'] = 0
        stats['gpu_temp'] = stats['gpu_power'] = 0

    # System memory (in kB, convert to GB)
    try:
        meminfo = {}
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(':')
                    value = int(parts[1])  # kB
                    meminfo[key] = value

        stats['sys_total'] = meminfo.get('MemTotal', 0) / (1024 * 1024)
        stats['sys_free'] = meminfo.get('MemFree', 0) / (1024 * 1024)
        stats['sys_available'] = meminfo.get('MemAvailable', 0) / (1024 * 1024)
        stats['sys_cached'] = (meminfo.get('Cached', 0) + meminfo.get('Buffers', 0)) / (1024 * 1024)
        stats['sys_used'] = stats['sys_total'] - stats['sys_free'] - stats['sys_cached']
    except Exception:
        stats['sys_total'] = stats['sys_free'] = stats['sys_available'] = 0
        stats['sys_cached'] = stats['sys_used'] = 0

    return stats


def get_gpu_processes() -> List[Dict[str, str]]:
    """
    Get list of processes using GPU.

    Returns:
        List of dictionaries with process info
    """
    processes = []
    output = run_command("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader")

    if output:
        for line in output.split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    processes.append({
                        'pid': parts[0],
                        'name': parts[1],
                        'memory': parts[2]
                    })

    return processes


def format_bar(value: float, total: float, width: int = 30) -> str:
    """Create a visual bar representation."""
    if total <= 0:
        return '[' + ' ' * width + ']'

    percent = min(value / total, 1.0)
    filled = int(width * percent)

    # Color based on usage level
    if percent < 0.5:
        color = Colors.GREEN
    elif percent < 0.8:
        color = Colors.YELLOW
    else:
        color = Colors.RED

    bar = color + '█' * filled + Colors.RESET + '░' * (width - filled)
    return f'[{bar}]'


def print_memory_status(stats: Dict[str, float], show_processes: bool = False) -> None:
    """Print formatted memory status."""
    # Clear screen and move cursor to top
    print('\033[2J\033[H', end='')

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print(f"{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}         DGX SPARK MEMORY MONITOR - {timestamp}{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 70}{Colors.RESET}\n")

    # GPU Memory
    gpu_percent = (stats['gpu_used'] / stats['gpu_total'] * 100) if stats['gpu_total'] > 0 else 0
    print(f"{Colors.CYAN}GPU MEMORY (Unified){Colors.RESET}")
    print(f"  {format_bar(stats['gpu_used'], stats['gpu_total'])} {gpu_percent:5.1f}%")
    print(f"  Used: {stats['gpu_used']:6.1f} GB / {stats['gpu_total']:.1f} GB  (Free: {stats['gpu_free']:.1f} GB)")

    # GPU Stats
    print(f"\n  Temperature: {stats['gpu_temp']:.0f}°C    Power: {stats['gpu_power']:.1f} W")

    # System Memory
    sys_used_percent = (stats['sys_used'] / stats['sys_total'] * 100) if stats['sys_total'] > 0 else 0
    print(f"\n{Colors.MAGENTA}SYSTEM MEMORY{Colors.RESET}")
    print(f"  {format_bar(stats['sys_used'], stats['sys_total'])} {sys_used_percent:5.1f}%")
    print(f"  Used: {stats['sys_used']:6.1f} GB / {stats['sys_total']:.1f} GB")
    print(f"  Available: {stats['sys_available']:.1f} GB    Cached: {stats['sys_cached']:.1f} GB")

    # Buffer Cache Warning
    if stats['sys_cached'] > 10:  # More than 10GB cached
        print(f"\n{Colors.YELLOW}⚠ Buffer cache is {stats['sys_cached']:.1f} GB - may affect GPU memory{Colors.RESET}")
        print(f"  Clear with: sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'")

    # GPU Processes
    if show_processes:
        processes = get_gpu_processes()
        print(f"\n{Colors.BLUE}GPU PROCESSES{Colors.RESET}")
        if processes:
            print(f"  {'PID':<10} {'Memory':<15} {'Process'}")
            print(f"  {'-' * 50}")
            for proc in processes[:10]:  # Show top 10
                print(f"  {proc['pid']:<10} {proc['memory']:<15} {proc['name'][:30]}")
        else:
            print("  No GPU processes running")

    print(f"\n{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    print(f"Press Ctrl+C to exit")


def log_to_file(filepath: str, stats: Dict[str, float], write_header: bool = False) -> None:
    """Append memory stats to CSV file."""
    mode = 'w' if write_header else 'a'
    with open(filepath, mode) as f:
        if write_header:
            f.write("timestamp,gpu_used_gb,gpu_free_gb,gpu_temp,gpu_power,sys_used_gb,sys_available_gb,sys_cached_gb\n")

        timestamp = datetime.now().isoformat()
        f.write(f"{timestamp},{stats['gpu_used']:.2f},{stats['gpu_free']:.2f},"
                f"{stats['gpu_temp']:.1f},{stats['gpu_power']:.1f},"
                f"{stats['sys_used']:.2f},{stats['sys_available']:.2f},{stats['sys_cached']:.2f}\n")


class MemoryMonitor:
    """
    Continuous memory monitoring class.

    Example:
        >>> monitor = MemoryMonitor(interval=2.0, show_processes=True)
        >>> monitor.run()  # Runs until Ctrl+C
    """

    def __init__(self, interval: float = 2.0, show_processes: bool = False, log_file: Optional[str] = None):
        """
        Initialize the memory monitor.

        Args:
            interval: Update interval in seconds
            show_processes: Whether to show GPU processes
            log_file: Path to CSV log file (optional)
        """
        self.interval = interval
        self.show_processes = show_processes
        self.log_file = log_file
        self.running = False

        # Register signal handler for graceful exit
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals."""
        self.running = False
        print(f"\n\n{Colors.GREEN}Monitoring stopped.{Colors.RESET}")
        sys.exit(0)

    def run(self) -> None:
        """Start the monitoring loop."""
        self.running = True
        first_log = True

        while self.running:
            try:
                stats = get_memory_stats()
                print_memory_status(stats, self.show_processes)

                if self.log_file:
                    log_to_file(self.log_file, stats, write_header=first_log)
                    first_log = False

                time.sleep(self.interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(self.interval)


def snapshot() -> Dict[str, float]:
    """
    Take a single memory snapshot.

    Returns:
        Dictionary with current memory statistics

    Example:
        >>> stats = snapshot()
        >>> print(f"GPU used: {stats['gpu_used']:.1f} GB")
    """
    return get_memory_stats()


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Monitor DGX Spark memory usage in real-time"
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=2.0,
        help="Update interval in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--processes", "-p",
        action="store_true",
        help="Show GPU processes"
    )
    parser.add_argument(
        "--log", "-l",
        type=str,
        help="Log to CSV file"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print once and exit"
    )

    args = parser.parse_args()

    if args.once:
        stats = get_memory_stats()
        print_memory_status(stats, args.processes)
    else:
        monitor = MemoryMonitor(
            interval=args.interval,
            show_processes=args.processes,
            log_file=args.log
        )
        monitor.run()


if __name__ == "__main__":
    main()
