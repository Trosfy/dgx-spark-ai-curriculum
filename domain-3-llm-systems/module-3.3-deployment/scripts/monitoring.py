"""
Monitoring Utilities for LLM Inference Servers

This module provides tools for monitoring inference server health,
performance, and resource utilization on DGX Spark.

Features:
- GPU memory and utilization monitoring
- Inference latency tracking
- Request throughput metrics
- Real-time dashboard for terminal
- Alert thresholds for production use

Example:
    >>> from monitoring import InferenceMonitor, GPUMonitor
    >>>
    >>> # Monitor GPU resources
    >>> gpu = GPUMonitor()
    >>> print(gpu.get_status())
    >>>
    >>> # Monitor inference server
    >>> monitor = InferenceMonitor("http://localhost:8080")
    >>> monitor.start_background_monitoring()
    >>> time.sleep(60)
    >>> monitor.stop()
    >>> print(monitor.get_summary())
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

import requests


@dataclass
class GPUStats:
    """
    GPU statistics snapshot.

    Attributes:
        gpu_index: GPU device index
        name: GPU name (e.g., "NVIDIA GB10")
        memory_used_mb: Memory currently in use (MB)
        memory_total_mb: Total memory available (MB)
        memory_free_mb: Free memory (MB)
        gpu_utilization: GPU compute utilization (0-100%)
        memory_utilization: Memory bandwidth utilization (0-100%)
        temperature_c: GPU temperature in Celsius
        power_draw_w: Current power draw in Watts
        timestamp: When this snapshot was taken
    """
    gpu_index: int
    name: str
    memory_used_mb: int
    memory_total_mb: int
    memory_free_mb: int
    gpu_utilization: int
    memory_utilization: int
    temperature_c: int
    power_draw_w: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def memory_used_gb(self) -> float:
        return self.memory_used_mb / 1024

    @property
    def memory_total_gb(self) -> float:
        return self.memory_total_mb / 1024

    @property
    def memory_used_percent(self) -> float:
        if self.memory_total_mb == 0:
            return 0.0
        return (self.memory_used_mb / self.memory_total_mb) * 100


class GPUMonitor:
    """
    Monitor GPU resources using nvidia-smi.

    Example:
        >>> monitor = GPUMonitor()
        >>> stats = monitor.get_stats()
        >>> print(f"GPU Memory: {stats.memory_used_gb:.1f}/{stats.memory_total_gb:.1f} GB")
        >>> print(f"Utilization: {stats.gpu_utilization}%")
    """

    def __init__(self):
        """Initialize GPU monitor."""
        self._verify_nvidia_smi()

    def _verify_nvidia_smi(self) -> None:
        """Verify nvidia-smi is available."""
        try:
            subprocess.run(
                ["nvidia-smi", "--version"],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("nvidia-smi not found. Is NVIDIA driver installed?")

    def get_stats(self, gpu_index: int = 0) -> GPUStats:
        """
        Get current GPU statistics.

        Args:
            gpu_index: GPU device index (default 0)

        Returns:
            GPUStats with current GPU state
        """
        query = ",".join([
            "gpu_name",
            "memory.used",
            "memory.total",
            "memory.free",
            "utilization.gpu",
            "utilization.memory",
            "temperature.gpu",
            "power.draw"
        ])

        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"nvidia-smi failed: {result.stderr}")

        values = [v.strip() for v in result.stdout.strip().split(",")]

        return GPUStats(
            gpu_index=gpu_index,
            name=values[0],
            memory_used_mb=int(values[1]),
            memory_total_mb=int(values[2]),
            memory_free_mb=int(values[3]),
            gpu_utilization=int(values[4]) if values[4] != "[N/A]" else 0,
            memory_utilization=int(values[5]) if values[5] != "[N/A]" else 0,
            temperature_c=int(values[6]) if values[6] != "[N/A]" else 0,
            power_draw_w=float(values[7]) if values[7] != "[N/A]" else 0.0
        )

    def get_all_gpus(self) -> list[GPUStats]:
        """Get statistics for all GPUs."""
        # First, count GPUs
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True,
            text=True
        )

        gpu_count = len(result.stdout.strip().split("\n"))
        return [self.get_stats(i) for i in range(gpu_count)]

    def get_status(self) -> str:
        """Get a formatted status string."""
        try:
            stats = self.get_stats()
            return f"""
GPU Status: {stats.name}
═══════════════════════════════════════
Memory: {stats.memory_used_gb:.1f} / {stats.memory_total_gb:.1f} GB ({stats.memory_used_percent:.1f}%)
Utilization: {stats.gpu_utilization}%
Temperature: {stats.temperature_c}°C
Power: {stats.power_draw_w:.1f}W
"""
        except Exception as e:
            return f"GPU monitoring error: {e}"

    def watch(
        self,
        callback: Callable[[GPUStats], None],
        interval: float = 1.0,
        duration: Optional[float] = None
    ) -> None:
        """
        Watch GPU stats and call callback on each update.

        Args:
            callback: Function to call with GPUStats
            interval: Seconds between updates
            duration: Total seconds to watch (None = forever)
        """
        start = time.time()
        while True:
            try:
                stats = self.get_stats()
                callback(stats)
            except Exception as e:
                print(f"Monitoring error: {e}")

            time.sleep(interval)

            if duration and (time.time() - start) >= duration:
                break


@dataclass
class RequestMetrics:
    """Metrics for a single inference request."""
    start_time: float
    end_time: float
    status_code: int
    tokens_generated: int
    prompt_tokens: int
    success: bool
    error_message: Optional[str] = None

    @property
    def latency_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def tokens_per_second(self) -> float:
        duration = self.end_time - self.start_time
        if duration <= 0:
            return 0.0
        return self.tokens_generated / duration


@dataclass
class MonitoringSummary:
    """Summary of monitoring results over a time period."""
    start_time: datetime
    end_time: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    min_latency_ms: float
    avg_tokens_per_second: float
    total_tokens_generated: int
    requests_per_second: float
    error_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_seconds": (self.end_time - self.start_time).total_seconds(),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "error_rate": f"{self.error_rate:.2%}",
            "requests_per_second": round(self.requests_per_second, 2),
            "latency": {
                "avg_ms": round(self.avg_latency_ms, 2),
                "p50_ms": round(self.p50_latency_ms, 2),
                "p90_ms": round(self.p90_latency_ms, 2),
                "p99_ms": round(self.p99_latency_ms, 2),
                "min_ms": round(self.min_latency_ms, 2),
                "max_ms": round(self.max_latency_ms, 2)
            },
            "throughput": {
                "avg_tokens_per_second": round(self.avg_tokens_per_second, 2),
                "total_tokens": self.total_tokens_generated
            }
        }

    def __str__(self) -> str:
        return f"""
Monitoring Summary
═══════════════════════════════════════
Duration: {(self.end_time - self.start_time).total_seconds():.1f}s
Total Requests: {self.total_requests} ({self.successful_requests} success, {self.failed_requests} failed)
Error Rate: {self.error_rate:.2%}
Throughput: {self.requests_per_second:.2f} req/s

Latency Distribution:
  - Average: {self.avg_latency_ms:.1f}ms
  - P50: {self.p50_latency_ms:.1f}ms
  - P90: {self.p90_latency_ms:.1f}ms
  - P99: {self.p99_latency_ms:.1f}ms
  - Min/Max: {self.min_latency_ms:.1f}ms / {self.max_latency_ms:.1f}ms

Token Throughput:
  - Avg Speed: {self.avg_tokens_per_second:.1f} tok/s
  - Total Generated: {self.total_tokens_generated:,}
"""


class InferenceMonitor:
    """
    Monitor an inference server's health and performance.

    Example:
        >>> monitor = InferenceMonitor("http://localhost:8080")
        >>>
        >>> # Single health check
        >>> is_healthy = monitor.check_health()
        >>> print(f"Server healthy: {is_healthy}")
        >>>
        >>> # Background monitoring
        >>> monitor.start_background_monitoring(interval=5.0)
        >>> time.sleep(300)  # Monitor for 5 minutes
        >>> monitor.stop()
        >>> print(monitor.get_summary())
    """

    def __init__(self, server_url: str, health_endpoint: str = "/health"):
        """
        Initialize the monitor.

        Args:
            server_url: Base URL of the inference server
            health_endpoint: Health check endpoint path
        """
        self.server_url = server_url.rstrip("/")
        self.health_endpoint = health_endpoint
        self._metrics: deque[RequestMetrics] = deque(maxlen=10000)
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._start_time: Optional[datetime] = None
        self._gpu_monitor = GPUMonitor()
        self._gpu_stats: deque[GPUStats] = deque(maxlen=1000)

    def check_health(self) -> bool:
        """Check if the server is healthy."""
        try:
            response = requests.get(
                f"{self.server_url}{self.health_endpoint}",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

    def send_request(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7
    ) -> RequestMetrics:
        """
        Send a test request and record metrics.

        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            RequestMetrics with timing and token info
        """
        start_time = time.time()
        error_message = None
        status_code = 0
        tokens_generated = 0
        prompt_tokens = 0
        success = False

        try:
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=120
            )
            status_code = response.status_code

            if response.status_code == 200:
                data = response.json()
                usage = data.get("usage", {})
                tokens_generated = usage.get("completion_tokens", 0)
                prompt_tokens = usage.get("prompt_tokens", 0)
                success = True
            else:
                error_message = response.text[:200]

        except Exception as e:
            error_message = str(e)

        end_time = time.time()

        metrics = RequestMetrics(
            start_time=start_time,
            end_time=end_time,
            status_code=status_code,
            tokens_generated=tokens_generated,
            prompt_tokens=prompt_tokens,
            success=success,
            error_message=error_message
        )

        self._metrics.append(metrics)
        return metrics

    def _monitoring_loop(self, interval: float) -> None:
        """Background monitoring loop."""
        while self._monitoring:
            # Health check
            is_healthy = self.check_health()
            if not is_healthy:
                print(f"[{datetime.now().isoformat()}] WARNING: Server unhealthy!")

            # GPU stats
            try:
                gpu_stats = self._gpu_monitor.get_stats()
                self._gpu_stats.append(gpu_stats)
            except Exception as e:
                pass

            time.sleep(interval)

    def start_background_monitoring(self, interval: float = 5.0) -> None:
        """
        Start background health monitoring.

        Args:
            interval: Seconds between health checks
        """
        if self._monitoring:
            return

        self._monitoring = True
        self._start_time = datetime.now()
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        print(f"Started monitoring {self.server_url} (interval: {interval}s)")

    def stop(self) -> None:
        """Stop background monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
        print("Monitoring stopped")

    def get_summary(self) -> MonitoringSummary:
        """Get a summary of all recorded metrics."""
        if not self._metrics:
            now = datetime.now()
            return MonitoringSummary(
                start_time=now,
                end_time=now,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                avg_latency_ms=0,
                p50_latency_ms=0,
                p90_latency_ms=0,
                p99_latency_ms=0,
                max_latency_ms=0,
                min_latency_ms=0,
                avg_tokens_per_second=0,
                total_tokens_generated=0,
                requests_per_second=0,
                error_rate=0
            )

        metrics_list = list(self._metrics)
        successful = [m for m in metrics_list if m.success]
        latencies = sorted([m.latency_ms for m in metrics_list])
        n = len(latencies)

        start_time = self._start_time or datetime.fromtimestamp(metrics_list[0].start_time)
        end_time = datetime.fromtimestamp(metrics_list[-1].end_time)
        duration = (end_time - start_time).total_seconds()

        total_tokens = sum(m.tokens_generated for m in successful)
        token_speeds = [m.tokens_per_second for m in successful if m.tokens_per_second > 0]

        return MonitoringSummary(
            start_time=start_time,
            end_time=end_time,
            total_requests=len(metrics_list),
            successful_requests=len(successful),
            failed_requests=len(metrics_list) - len(successful),
            avg_latency_ms=sum(latencies) / n if n else 0,
            p50_latency_ms=latencies[n // 2] if n else 0,
            p90_latency_ms=latencies[int(n * 0.9)] if n else 0,
            p99_latency_ms=latencies[int(n * 0.99)] if n else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            avg_tokens_per_second=sum(token_speeds) / len(token_speeds) if token_speeds else 0,
            total_tokens_generated=total_tokens,
            requests_per_second=len(metrics_list) / duration if duration > 0 else 0,
            error_rate=(len(metrics_list) - len(successful)) / len(metrics_list) if metrics_list else 0
        )

    def get_gpu_history(self) -> list[dict[str, Any]]:
        """Get recorded GPU statistics history."""
        return [
            {
                "timestamp": s.timestamp.isoformat(),
                "memory_used_gb": s.memory_used_gb,
                "memory_total_gb": s.memory_total_gb,
                "utilization": s.gpu_utilization,
                "temperature": s.temperature_c,
                "power": s.power_draw_w
            }
            for s in self._gpu_stats
        ]


class AlertManager:
    """
    Manage alerts based on monitoring thresholds.

    Example:
        >>> alerts = AlertManager()
        >>> alerts.add_threshold("latency_ms", max_value=500)
        >>> alerts.add_threshold("gpu_memory_percent", max_value=90)
        >>>
        >>> # Check values
        >>> alert = alerts.check("latency_ms", 750)
        >>> if alert:
        ...     print(f"ALERT: {alert}")
    """

    def __init__(self):
        self._thresholds: dict[str, dict[str, float]] = {}
        self._alerts: list[dict[str, Any]] = []

    def add_threshold(
        self,
        metric_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> None:
        """
        Add a threshold for a metric.

        Args:
            metric_name: Name of the metric
            min_value: Alert if value falls below this
            max_value: Alert if value exceeds this
        """
        self._thresholds[metric_name] = {
            "min": min_value,
            "max": max_value
        }

    def check(self, metric_name: str, value: float) -> Optional[str]:
        """
        Check if a value triggers an alert.

        Args:
            metric_name: Name of the metric
            value: Current value

        Returns:
            Alert message if threshold exceeded, None otherwise
        """
        if metric_name not in self._thresholds:
            return None

        threshold = self._thresholds[metric_name]
        alert = None

        if threshold["min"] is not None and value < threshold["min"]:
            alert = f"{metric_name}={value:.2f} below minimum {threshold['min']:.2f}"

        if threshold["max"] is not None and value > threshold["max"]:
            alert = f"{metric_name}={value:.2f} exceeds maximum {threshold['max']:.2f}"

        if alert:
            self._alerts.append({
                "timestamp": datetime.now().isoformat(),
                "metric": metric_name,
                "value": value,
                "message": alert
            })

        return alert

    def get_alerts(self) -> list[dict[str, Any]]:
        """Get all recorded alerts."""
        return self._alerts.copy()

    def clear_alerts(self) -> None:
        """Clear all recorded alerts."""
        self._alerts.clear()


def print_live_dashboard(
    server_url: str,
    refresh_interval: float = 2.0,
    duration: Optional[float] = None
) -> None:
    """
    Print a live monitoring dashboard to the terminal.

    Args:
        server_url: Inference server URL
        refresh_interval: Seconds between updates
        duration: Total duration in seconds (None = run forever)
    """
    gpu_monitor = GPUMonitor()
    start_time = time.time()

    def clear_screen() -> None:
        os.system('clear' if os.name == 'posix' else 'cls')

    print("Starting live dashboard... (Ctrl+C to stop)")

    try:
        while True:
            clear_screen()

            # Get GPU stats
            try:
                gpu = gpu_monitor.get_stats()
                gpu_status = f"""
GPU: {gpu.name}
Memory: {gpu.memory_used_gb:.1f}/{gpu.memory_total_gb:.1f} GB ({gpu.memory_used_percent:.1f}%)
Utilization: {gpu.gpu_utilization}%
Temperature: {gpu.temperature_c}°C
Power: {gpu.power_draw_w:.1f}W
"""
            except Exception as e:
                gpu_status = f"GPU monitoring error: {e}"

            # Check server health
            try:
                response = requests.get(f"{server_url}/health", timeout=5)
                if response.status_code == 200:
                    health = response.json()
                    server_status = f"""
Server: {server_url}
Status: {health.get('status', 'unknown')}
Engine: {health.get('engine', 'unknown')}
Model: {health.get('model', 'unknown')}
Uptime: {health.get('uptime_seconds', 0):.1f}s
Requests: {health.get('total_requests', 0)}
Active: {health.get('active_requests', 0)}
"""
                else:
                    server_status = f"Server unhealthy (status {response.status_code})"
            except Exception as e:
                server_status = f"Server unreachable: {e}"

            # Print dashboard
            elapsed = time.time() - start_time
            print(f"""
╔══════════════════════════════════════════════════════════════╗
║           LLM Inference Monitoring Dashboard                  ║
║           Elapsed: {elapsed:.0f}s                                        ║
╠══════════════════════════════════════════════════════════════╣
║ GPU STATUS                                                    ║
╠══════════════════════════════════════════════════════════════╣
{gpu_status}
╠══════════════════════════════════════════════════════════════╣
║ SERVER STATUS                                                 ║
╠══════════════════════════════════════════════════════════════╣
{server_status}
╚══════════════════════════════════════════════════════════════╝
Press Ctrl+C to stop
""")

            time.sleep(refresh_interval)

            if duration and (time.time() - start_time) >= duration:
                break

    except KeyboardInterrupt:
        print("\nDashboard stopped")


if __name__ == "__main__":
    # Demo: Print GPU status
    print("GPU Monitoring Demo")
    print("=" * 50)

    try:
        monitor = GPUMonitor()
        print(monitor.get_status())
    except Exception as e:
        print(f"GPU monitoring not available: {e}")

    print("\nTo run the live dashboard:")
    print("  python monitoring.py http://localhost:8080")
