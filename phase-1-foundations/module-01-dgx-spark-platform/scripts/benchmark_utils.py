#!/usr/bin/env python3
"""
DGX Spark Benchmark Utilities

Comprehensive benchmarking tools for measuring LLM inference performance,
memory usage, and throughput on NVIDIA DGX Spark systems.

This module provides utilities for:
- Ollama API benchmarking
- Token throughput measurement (prefill and decode)
- Memory monitoring during inference
- Result logging and comparison

Usage:
    # As a module
    from benchmark_utils import OllamaBenchmark, run_benchmark_suite

    bench = OllamaBenchmark()
    results = bench.benchmark_model("llama3.1:8b")
    print(results)

    # As a script
    python benchmark_utils.py --model llama3.1:8b
    python benchmark_utils.py --suite  # Run full benchmark suite

Author: Professor SPARK
License: MIT
"""

import requests
import time
import json
import argparse
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
import statistics


@dataclass
class BenchmarkResult:
    """
    Container for benchmark results.

    Attributes:
        model: Model name/tag
        prompt_tokens: Number of tokens in the prompt
        generated_tokens: Number of tokens generated
        prefill_tps: Prompt processing speed (tokens/second)
        decode_tps: Generation speed (tokens/second)
        total_time_s: Total request time in seconds
        memory_gb: GPU memory used (if available)
        timestamp: When the benchmark was run
        error: Error message if benchmark failed
    """
    model: str
    prompt_tokens: int = 0
    generated_tokens: int = 0
    prefill_tps: float = 0.0
    decode_tps: float = 0.0
    total_time_s: float = 0.0
    memory_gb: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.error:
            return f"{self.model}: ERROR - {self.error}"
        return (
            f"{self.model}: "
            f"Prefill: {self.prefill_tps:.1f} tok/s, "
            f"Decode: {self.decode_tps:.1f} tok/s, "
            f"Memory: {self.memory_gb:.1f} GB"
        )


class OllamaBenchmark:
    """
    Benchmark utility for Ollama models.

    Uses direct API calls to measure accurate performance metrics,
    avoiding the latency overhead of web UIs or CLI tools.

    Example:
        >>> bench = OllamaBenchmark()
        >>> result = bench.benchmark_model("llama3.2:3b")
        >>> print(f"Decode speed: {result.decode_tps:.1f} tokens/sec")
    """

    # Standard prompts for benchmarking
    DEFAULT_PROMPTS = {
        "short": "Hello, how are you?",
        "medium": "Explain the concept of machine learning in simple terms. What are the main types of machine learning and how do they differ?",
        "long": """You are an expert AI researcher. Please provide a comprehensive analysis of the following topics:

1. The evolution of transformer architectures from the original 2017 paper to modern LLMs
2. Key innovations that enabled scaling to billions of parameters
3. The role of attention mechanisms in capturing long-range dependencies
4. Trade-offs between model size, training compute, and inference efficiency
5. Future directions for more efficient architectures

Please structure your response with clear sections and provide specific examples where relevant."""
    }

    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize the benchmark utility.

        Args:
            base_url: Ollama API base URL
        """
        self.base_url = base_url.rstrip('/')
        self._verify_connection()

    def _verify_connection(self) -> bool:
        """Verify Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def list_models(self) -> List[str]:
        """
        List available models in Ollama.

        Returns:
            List of model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []

    def get_gpu_memory(self) -> float:
        """
        Get current GPU memory usage in GB.

        Returns:
            GPU memory used in GB, or 0 if unavailable
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Memory is in MiB, convert to GB
                return float(result.stdout.strip()) / 1024
        except Exception:
            pass
        return 0.0

    def benchmark_model(
        self,
        model: str,
        prompt: str = None,
        max_tokens: int = 100,
        num_runs: int = 3,
        warmup: bool = True
    ) -> BenchmarkResult:
        """
        Benchmark a single model.

        Args:
            model: Model name (e.g., "llama3.1:8b")
            prompt: Custom prompt (uses default if None)
            max_tokens: Maximum tokens to generate
            num_runs: Number of runs to average
            warmup: Whether to do a warmup run first

        Returns:
            BenchmarkResult with averaged metrics
        """
        if prompt is None:
            prompt = self.DEFAULT_PROMPTS["medium"]

        # Warmup run (discarded)
        if warmup:
            try:
                self._single_run(model, "Hello", max_tokens=10)
            except Exception:
                pass

        results = []
        memory_readings = []

        for i in range(num_runs):
            try:
                # Get memory before
                mem_before = self.get_gpu_memory()

                # Run benchmark
                result = self._single_run(model, prompt, max_tokens)

                # Get memory after
                mem_after = self.get_gpu_memory()
                memory_readings.append(max(mem_before, mem_after))

                if result:
                    results.append(result)

            except Exception as e:
                return BenchmarkResult(model=model, error=str(e))

        if not results:
            return BenchmarkResult(model=model, error="No successful runs")

        # Average the results
        avg_result = BenchmarkResult(
            model=model,
            prompt_tokens=results[0]["prompt_tokens"],
            generated_tokens=int(statistics.mean(r["generated_tokens"] for r in results)),
            prefill_tps=statistics.mean(r["prefill_tps"] for r in results),
            decode_tps=statistics.mean(r["decode_tps"] for r in results),
            total_time_s=statistics.mean(r["total_time_s"] for r in results),
            memory_gb=statistics.mean(memory_readings) if memory_readings else 0.0
        )

        return avg_result

    def _single_run(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 100
    ) -> Optional[Dict[str, Any]]:
        """
        Perform a single benchmark run.

        Args:
            model: Model name
            prompt: Input prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with timing metrics, or None on failure
        """
        start_time = time.time()

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens
                    }
                },
                timeout=300  # 5 minutes max
            )

            total_time = (time.time() - start_time) * 1000  # ms

            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code}")

            data = response.json()

            # Ollama returns timing in nanoseconds
            prompt_eval_duration = data.get("prompt_eval_duration", 0) / 1e9  # to seconds
            eval_duration = data.get("eval_duration", 0) / 1e9  # to seconds

            prompt_tokens = data.get("prompt_eval_count", 0)
            generated_tokens = data.get("eval_count", 0)

            # Calculate tokens per second
            prefill_tps = prompt_tokens / prompt_eval_duration if prompt_eval_duration > 0 else 0
            decode_tps = generated_tokens / eval_duration if eval_duration > 0 else 0

            return {
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "prefill_tps": prefill_tps,
                "decode_tps": decode_tps,
                "total_time_s": total_time / 1000  # Convert ms to seconds
            }

        except requests.exceptions.Timeout:
            raise Exception("Request timed out")
        except Exception as e:
            raise Exception(f"Request failed: {e}")

    def benchmark_suite(
        self,
        models: List[str] = None,
        prompt_sizes: List[str] = None
    ) -> List[BenchmarkResult]:
        """
        Run a full benchmark suite across multiple models and prompt sizes.

        Args:
            models: List of models to test (uses available models if None)
            prompt_sizes: List of prompt sizes ("short", "medium", "long")

        Returns:
            List of BenchmarkResult objects
        """
        if models is None:
            models = self.list_models()

        if prompt_sizes is None:
            prompt_sizes = ["medium"]

        results = []

        for model in models:
            print(f"\nBenchmarking {model}...")

            for size in prompt_sizes:
                prompt = self.DEFAULT_PROMPTS.get(size, self.DEFAULT_PROMPTS["medium"])
                print(f"  Prompt size: {size}")

                result = self.benchmark_model(model, prompt)
                result.model = f"{model} ({size})"
                results.append(result)

                print(f"    {result}")

        return results


def format_results_table(results: List[BenchmarkResult]) -> str:
    """
    Format benchmark results as a markdown table.

    Args:
        results: List of BenchmarkResult objects

    Returns:
        Formatted markdown table string
    """
    lines = [
        "| Model | Prefill (tok/s) | Decode (tok/s) | Memory (GB) | Status |",
        "|-------|-----------------|----------------|-------------|--------|"
    ]

    for r in results:
        if r.error:
            lines.append(f"| {r.model} | - | - | - | {r.error[:20]} |")
        else:
            lines.append(
                f"| {r.model} | {r.prefill_tps:.1f} | {r.decode_tps:.1f} | "
                f"{r.memory_gb:.1f} | OK |"
            )

    return "\n".join(lines)


def save_results(results: List[BenchmarkResult], filepath: str) -> None:
    """
    Save benchmark results to a JSON file.

    Args:
        results: List of BenchmarkResult objects
        filepath: Output file path
    """
    data = {
        "timestamp": datetime.now().isoformat(),
        "results": [r.to_dict() for r in results]
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to: {filepath}")


def run_benchmark_suite(
    models: List[str] = None,
    output_file: str = None
) -> List[BenchmarkResult]:
    """
    Convenience function to run a full benchmark suite.

    Args:
        models: List of models to benchmark (None for all available)
        output_file: Optional file to save results

    Returns:
        List of BenchmarkResult objects
    """
    bench = OllamaBenchmark()

    if not bench._verify_connection():
        print("Error: Cannot connect to Ollama. Is it running?")
        print("Start with: ollama serve")
        return []

    print("=" * 60)
    print("DGX Spark Ollama Benchmark Suite")
    print("=" * 60)

    available_models = bench.list_models()
    print(f"\nAvailable models: {', '.join(available_models) or 'None'}")

    if models:
        # Filter to only available models
        models = [m for m in models if m in available_models]
        if not models:
            print("Error: None of the specified models are available")
            return []
    else:
        models = available_models

    if not models:
        print("No models available. Pull some models first:")
        print("  ollama pull llama3.2:3b")
        print("  ollama pull llama3.1:8b")
        return []

    results = bench.benchmark_suite(models)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(format_results_table(results))

    if output_file:
        save_results(results, output_file)

    return results


class PyTorchBenchmark:
    """
    Benchmark utility for PyTorch operations on DGX Spark.

    Measures GPU compute performance for common AI operations.
    """

    def __init__(self):
        """Initialize PyTorch benchmark."""
        try:
            import torch
            self.torch = torch
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA not available. Ensure you're running inside an NGC container with GPU access:\n"
                    "  docker run --gpus all -it nvcr.io/nvidia/pytorch:25.11-py3 python your_script.py"
                )
            self.device = torch.device("cuda")
        except ImportError:
            raise RuntimeError(
                "PyTorch not installed. On DGX Spark, you must run inside an NGC container:\n"
                "  docker run --gpus all -it nvcr.io/nvidia/pytorch:25.11-py3 bash\n"
                "Then run your script inside the container."
            )

    def benchmark_matmul(
        self,
        sizes: List[int] = None,
        dtype = None,
        num_runs: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark matrix multiplication.

        Args:
            sizes: List of matrix sizes to test
            dtype: Data type (default: bfloat16)
            num_runs: Number of runs for averaging

        Returns:
            Dictionary with TFLOPS for each size
        """
        if sizes is None:
            sizes = [1024, 2048, 4096, 8192]

        if dtype is None:
            dtype = self.torch.bfloat16

        results = {}

        for size in sizes:
            # Create matrices
            a = self.torch.randn(size, size, dtype=dtype, device=self.device)
            b = self.torch.randn(size, size, dtype=dtype, device=self.device)

            # Warmup
            for _ in range(3):
                c = self.torch.matmul(a, b)
            self.torch.cuda.synchronize()

            # Benchmark
            times = []
            for _ in range(num_runs):
                self.torch.cuda.synchronize()
                start = time.time()
                c = self.torch.matmul(a, b)
                self.torch.cuda.synchronize()
                times.append(time.time() - start)

            avg_time = statistics.mean(times)

            # Calculate TFLOPS (2 * N^3 for matmul)
            flops = 2 * size ** 3
            tflops = flops / avg_time / 1e12

            results[f"{size}x{size}"] = {
                "time_ms": avg_time * 1000,
                "tflops": tflops
            }

            # Cleanup
            del a, b, c
            self.torch.cuda.empty_cache()

        return results

    def benchmark_memory_bandwidth(
        self,
        size_gb: float = 1.0,
        num_runs: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark memory bandwidth.

        Args:
            size_gb: Size of data to copy in GB
            num_runs: Number of runs for averaging

        Returns:
            Dictionary with bandwidth in GB/s
        """
        num_elements = int(size_gb * 1e9 / 4)  # float32 = 4 bytes

        src = self.torch.randn(num_elements, dtype=self.torch.float32, device=self.device)
        dst = self.torch.empty_like(src)

        # Warmup
        for _ in range(3):
            dst.copy_(src)
        self.torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(num_runs):
            self.torch.cuda.synchronize()
            start = time.time()
            dst.copy_(src)
            self.torch.cuda.synchronize()
            times.append(time.time() - start)

        avg_time = statistics.mean(times)
        bandwidth_gb_s = (size_gb * 2) / avg_time  # Read + Write

        del src, dst
        self.torch.cuda.empty_cache()

        return {
            "size_gb": size_gb,
            "time_ms": avg_time * 1000,
            "bandwidth_gb_s": bandwidth_gb_s
        }


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Benchmark AI models on DGX Spark"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Specific model to benchmark"
    )
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Run full benchmark suite"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )
    parser.add_argument(
        "--pytorch",
        action="store_true",
        help="Run PyTorch benchmarks"
    )

    args = parser.parse_args()

    if args.list:
        bench = OllamaBenchmark()
        models = bench.list_models()
        print("Available models:")
        for m in models:
            print(f"  - {m}")
        return

    if args.pytorch:
        try:
            pt_bench = PyTorchBenchmark()
            print("PyTorch Matrix Multiplication Benchmark")
            print("=" * 50)
            results = pt_bench.benchmark_matmul()
            for size, metrics in results.items():
                print(f"{size}: {metrics['tflops']:.1f} TFLOPS ({metrics['time_ms']:.2f} ms)")

            print("\nMemory Bandwidth Benchmark")
            print("=" * 50)
            bw = pt_bench.benchmark_memory_bandwidth()
            print(f"{bw['size_gb']:.1f} GB: {bw['bandwidth_gb_s']:.1f} GB/s")
        except Exception as e:
            print(f"Error: {e}")
        return

    models = [args.model] if args.model else None
    run_benchmark_suite(models=models, output_file=args.output)


if __name__ == "__main__":
    main()
