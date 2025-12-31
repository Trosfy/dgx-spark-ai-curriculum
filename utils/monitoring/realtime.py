"""
Real-time Memory Monitoring for DGX Spark

Provides continuous terminal-based monitoring with visual feedback,
GPU process tracking, and logging capabilities.
"""

import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class Colors:
    """ANSI color codes for terminal output."""

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def run_command(cmd: str, timeout: int = 10) -> str:
    """Execute shell command and return output."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
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
    gpu_total = run_command(
        "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits"
    )
    gpu_used = run_command(
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
    )
    gpu_free = run_command(
        "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits"
    )
    gpu_temp = run_command(
        "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits"
    )
    gpu_power = run_command(
        "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits"
    )

    try:
        stats["gpu_total"] = float(gpu_total) / 1024
        stats["gpu_used"] = float(gpu_used) / 1024
        stats["gpu_free"] = float(gpu_free) / 1024
        stats["gpu_temp"] = float(gpu_temp)
        stats["gpu_power"] = float(gpu_power.replace(" W", ""))
    except (ValueError, AttributeError):
        stats["gpu_total"] = 0
        stats["gpu_used"] = 0
        stats["gpu_free"] = 0
        stats["gpu_temp"] = 0
        stats["gpu_power"] = 0

    # System memory
    try:
        mem_output = run_command("free -b")
        lines = mem_output.split("\n")
        mem_parts = lines[1].split()
        stats["sys_total"] = float(mem_parts[1]) / 1e9
        stats["sys_used"] = float(mem_parts[2]) / 1e9
        stats["sys_available"] = float(mem_parts[6]) / 1e9 if len(mem_parts) > 6 else 0
        stats["sys_cached"] = float(mem_parts[5]) / 1e9 if len(mem_parts) > 5 else 0
    except Exception:
        stats["sys_total"] = 0
        stats["sys_used"] = 0
        stats["sys_available"] = 0
        stats["sys_cached"] = 0

    return stats


def get_gpu_processes() -> List[Dict]:
    """
    Get list of processes using the GPU.

    Returns:
        List of dicts with pid, name, memory_mb
    """
    output = run_command(
        "nvidia-smi --query-compute-apps=pid,process_name,used_memory "
        "--format=csv,noheader,nounits"
    )

    processes = []
    for line in output.split("\n"):
        if line.strip():
            try:
                parts = line.split(", ")
                if len(parts) >= 3:
                    processes.append(
                        {
                            "pid": parts[0],
                            "name": parts[1],
                            "memory_mb": float(parts[2]),
                        }
                    )
            except (ValueError, IndexError):
                continue

    return processes


def format_bar(value: float, max_value: float, width: int = 30) -> str:
    """
    Create a visual progress bar.

    Args:
        value: Current value
        max_value: Maximum value
        width: Bar width in characters

    Returns:
        Formatted bar string with color coding
    """
    if max_value <= 0:
        return " " * width

    ratio = min(value / max_value, 1.0)
    filled = int(ratio * width)

    # Color based on utilization
    if ratio < 0.5:
        color = Colors.GREEN
    elif ratio < 0.8:
        color = Colors.YELLOW
    else:
        color = Colors.RED

    bar = color + "█" * filled + Colors.RESET + "░" * (width - filled)
    return bar


class RealtimeMonitor:
    """
    Real-time terminal monitor for DGX Spark.

    Provides continuous updates of memory usage with visual bars,
    optional process listing, and CSV logging.

    Usage:
        monitor = RealtimeMonitor(interval=2.0)
        monitor.run()  # Ctrl+C to stop

        # Or with logging
        monitor = RealtimeMonitor(interval=1.0, log_file="memory.csv")
        monitor.run()
    """

    def __init__(
        self,
        interval: float = 2.0,
        log_file: Optional[str] = None,
        show_processes: bool = False,
    ):
        self.interval = interval
        self.log_file = Path(log_file) if log_file else None
        self.show_processes = show_processes
        self._running = False

        if self.log_file:
            self._init_log_file()

    def _init_log_file(self):
        """Initialize CSV log file with headers."""
        headers = "timestamp,gpu_used_gb,gpu_free_gb,sys_used_gb,sys_available_gb,gpu_temp,gpu_power\n"
        with open(self.log_file, "w") as f:
            f.write(headers)

    def _log_stats(self, stats: Dict[str, float]):
        """Append stats to log file."""
        if self.log_file:
            line = (
                f"{datetime.now().isoformat()},"
                f"{stats['gpu_used']:.2f},{stats['gpu_free']:.2f},"
                f"{stats['sys_used']:.2f},{stats['sys_available']:.2f},"
                f"{stats['gpu_temp']:.0f},{stats['gpu_power']:.1f}\n"
            )
            with open(self.log_file, "a") as f:
                f.write(line)

    def _display(self, stats: Dict[str, float]):
        """Display current stats in terminal."""
        # Clear screen
        print("\033[2J\033[H", end="")

        print(f"{Colors.BOLD}DGX Spark Memory Monitor{Colors.RESET}")
        print(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # GPU Memory
        gpu_pct = (
            (stats["gpu_used"] / stats["gpu_total"] * 100)
            if stats["gpu_total"] > 0
            else 0
        )
        gpu_bar = format_bar(stats["gpu_used"], stats["gpu_total"])
        print(f"\n{Colors.CYAN}GPU Memory{Colors.RESET}")
        print(
            f"  {gpu_bar} {stats['gpu_used']:.1f}/{stats['gpu_total']:.1f} GB ({gpu_pct:.0f}%)"
        )

        # System Memory
        sys_used_pct = (
            (stats["sys_used"] / stats["sys_total"] * 100)
            if stats["sys_total"] > 0
            else 0
        )
        sys_bar = format_bar(stats["sys_used"], stats["sys_total"])
        print(f"\n{Colors.CYAN}System Memory{Colors.RESET}")
        print(
            f"  {sys_bar} {stats['sys_used']:.1f}/{stats['sys_total']:.1f} GB ({sys_used_pct:.0f}%)"
        )
        print(
            f"  Available: {stats['sys_available']:.1f} GB | Cached: {stats['sys_cached']:.1f} GB"
        )

        # GPU Stats
        print(f"\n{Colors.CYAN}GPU Status{Colors.RESET}")
        print(
            f"  Temperature: {stats['gpu_temp']:.0f}°C | Power: {stats['gpu_power']:.1f}W"
        )

        # GPU Processes
        if self.show_processes:
            processes = get_gpu_processes()
            if processes:
                print(f"\n{Colors.CYAN}GPU Processes{Colors.RESET}")
                for p in processes:
                    print(
                        f"  PID {p['pid']}: {p['name'][:30]:<30} {p['memory_mb']:.0f} MB"
                    )

        print(f"\n{Colors.YELLOW}Press Ctrl+C to stop{Colors.RESET}")

    def run(self):
        """Start the real-time monitor. Press Ctrl+C to stop."""
        self._running = True

        def signal_handler(sig, frame):
            self._running = False
            print(f"\n\n{Colors.GREEN}Monitor stopped.{Colors.RESET}")
            if self.log_file:
                print(f"Log saved to: {self.log_file}")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        while self._running:
            stats = get_memory_stats()
            self._display(stats)
            self._log_stats(stats)
            time.sleep(self.interval)


def main():
    """CLI entry point for real-time monitoring."""
    import argparse

    parser = argparse.ArgumentParser(description="DGX Spark Memory Monitor")
    parser.add_argument(
        "--interval", "-i", type=float, default=2.0, help="Update interval in seconds"
    )
    parser.add_argument(
        "--log", "-l", type=str, default=None, help="Log file path (CSV)"
    )
    parser.add_argument(
        "--processes", "-p", action="store_true", help="Show GPU processes"
    )

    args = parser.parse_args()

    monitor = RealtimeMonitor(
        interval=args.interval, log_file=args.log, show_processes=args.processes
    )
    monitor.run()


if __name__ == "__main__":
    main()
