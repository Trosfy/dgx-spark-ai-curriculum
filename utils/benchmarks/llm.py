"""
LLM Inference Benchmarking Utilities

Specialized benchmarking for Large Language Model inference on DGX Spark.
Supports Ollama and llama.cpp backends.
"""

import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from .base import (
    BaseBenchmark,
    BenchmarkResult,
    BenchmarkSummary,
    format_results_table,
    get_gpu_memory_gb,
)

# Standard prompts for benchmarking
DEFAULT_PROMPTS = {
    "short": "Hello, how are you?",
    "medium": (
        "Explain the concept of machine learning in simple terms. "
        "What are the main types of machine learning and how do they differ?"
    ),
    "long": (
        "You are an expert AI researcher. Please provide a comprehensive analysis "
        "of transformer architectures, including their key innovations, limitations, "
        "and recent advances in efficient attention mechanisms. Include examples of "
        "how these have been applied in modern language models."
    ),
}


class OllamaBenchmark(BaseBenchmark):
    """
    Benchmark utility for Ollama models.

    Uses direct API calls to measure accurate performance metrics,
    avoiding the latency overhead of web UIs or CLI tools.

    Example:
        >>> bench = OllamaBenchmark()
        >>> result = bench.benchmark("qwen3:8b", prompt, runs=5)
        >>> print(f"Decode speed: {result.avg_decode_tps:.1f} tokens/sec")
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        super().__init__(name="ollama")
        self.base_url = base_url

    def is_available(self) -> bool:
        """Check if Ollama is running."""
        if not HAS_REQUESTS:
            return False
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List available models."""
        if not HAS_REQUESTS:
            return []
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def warmup(self, model: str, prompt: str = "Hello", **kwargs) -> None:
        """Warmup model to ensure it's loaded."""
        if not HAS_REQUESTS:
            return
        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=60,
            )
        except Exception:
            pass

    def benchmark_single(
        self, model: str, prompt: str, max_tokens: int = 128, **kwargs
    ) -> BenchmarkResult:
        """Run a single benchmark iteration."""
        if not HAS_REQUESTS:
            return BenchmarkResult(model=model, error="requests library not available")

        start_time = time.perf_counter()

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens},
                },
                timeout=300,
            )
            response.raise_for_status()
        except Exception as e:
            return BenchmarkResult(model=model, backend="ollama", error=str(e))

        total_time = time.perf_counter() - start_time
        data = response.json()

        # Ollama returns timing in nanoseconds
        prompt_tokens = data.get("prompt_eval_count", 0)
        prompt_duration_ns = data.get("prompt_eval_duration", 1)
        gen_tokens = data.get("eval_count", 0)
        gen_duration_ns = data.get("eval_duration", 1)

        prefill_time = prompt_duration_ns / 1e9
        decode_time = gen_duration_ns / 1e9

        prefill_tps = prompt_tokens / prefill_time if prefill_time > 0 else 0
        decode_tps = gen_tokens / decode_time if decode_time > 0 else 0

        # Extract quantization from model name
        quant = self._detect_quantization(model)

        return BenchmarkResult(
            model=model,
            backend="ollama",
            quantization=quant,
            prompt_tokens=prompt_tokens,
            generated_tokens=gen_tokens,
            prefill_time_s=prefill_time,
            decode_time_s=decode_time,
            total_time_s=total_time,
            prefill_tps=prefill_tps,
            decode_tps=decode_tps,
            memory_gb=get_gpu_memory_gb(),
        )

    def _detect_quantization(self, model: str) -> str:
        """Detect quantization from model name."""
        model_lower = model.lower()
        if ":q4" in model_lower or "q4_" in model_lower:
            return "Q4"
        elif ":q8" in model_lower or "q8_" in model_lower:
            return "Q8"
        elif ":fp16" in model_lower:
            return "FP16"
        elif ":bf16" in model_lower:
            return "BF16"
        return "default"


class LlamaCppBenchmark(BaseBenchmark):
    """
    Benchmark utility for llama.cpp using llama-bench.

    Example:
        >>> bench = LlamaCppBenchmark("/path/to/llama.cpp")
        >>> result = bench.benchmark("model.gguf", runs=5)
    """

    def __init__(self, llama_cpp_path: str = "./llama.cpp"):
        super().__init__(name="llama.cpp")
        self.llama_cpp_path = Path(llama_cpp_path)
        self.bench_path = self.llama_cpp_path / "llama-bench"

    def is_available(self) -> bool:
        """Check if llama-bench is available."""
        return self.bench_path.exists()

    def benchmark_single(
        self, model_path: str, prompt_tokens: int = 512, gen_tokens: int = 128, **kwargs
    ) -> BenchmarkResult:
        """Run a single benchmark using llama-bench."""
        if not self.is_available():
            return BenchmarkResult(model=model_path, error="llama-bench not found")

        cmd = [
            str(self.bench_path),
            "-m",
            model_path,
            "-p",
            str(prompt_tokens),
            "-n",
            str(gen_tokens),
            "-r",
            "1",
            "-o",
            "json",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except Exception as e:
            return BenchmarkResult(model=model_path, error=str(e))

        # Parse JSON output
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            lines = result.stdout.strip().split("\n")
            data = {}
            for line in lines:
                if line.startswith("{"):
                    try:
                        data = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue

        prefill_tps = data.get("t_pp_avg", 0) or data.get("pp", 0)
        decode_tps = data.get("t_tg_avg", 0) or data.get("tg", 0)

        return BenchmarkResult(
            model=model_path,
            backend="llama.cpp",
            quantization=self._detect_quantization(model_path),
            prompt_tokens=prompt_tokens,
            generated_tokens=gen_tokens,
            prefill_time_s=prompt_tokens / prefill_tps if prefill_tps > 0 else 0,
            decode_time_s=gen_tokens / decode_tps if decode_tps > 0 else 0,
            total_time_s=0,
            prefill_tps=prefill_tps,
            decode_tps=decode_tps,
            memory_gb=get_gpu_memory_gb(),
        )

    def _detect_quantization(self, model_path: str) -> str:
        """Detect quantization from filename."""
        path_lower = model_path.lower()
        for quant in [
            "q2_k",
            "q3_k",
            "q4_0",
            "q4_k",
            "q5_0",
            "q5_k",
            "q6_k",
            "q8_0",
            "f16",
            "f32",
        ]:
            if quant in path_lower:
                return quant.upper()
        return "unknown"


class BenchmarkSuite:
    """
    Complete benchmark suite for DGX Spark LLM inference.

    Runs comprehensive benchmarks across multiple models and backends.

    Example:
        >>> suite = BenchmarkSuite()
        >>> results = suite.run_ollama_suite(["qwen3:8b", "qwen3:4b"])
        >>> suite.print_results()
        >>> suite.save_results("benchmark_results.json")
    """

    # NVIDIA published baselines for DGX Spark
    NVIDIA_BASELINES = {
        "qwen3:8b": {"prefill": 3000, "decode": 45},
        "qwen3:32b": {"prefill": 500, "decode": 15},
        "gpt-oss-20b": {"prefill": 4500, "decode": 59},
    }

    def __init__(self):
        self.ollama = OllamaBenchmark()
        self.llamacpp = LlamaCppBenchmark()
        self.results: List[BenchmarkSummary] = []

    def run_ollama_suite(
        self,
        models: List[str],
        prompt: str = DEFAULT_PROMPTS["medium"],
        max_tokens: int = 128,
        runs: int = 5,
    ) -> List[BenchmarkSummary]:
        """Run benchmarks on multiple Ollama models."""
        if not self.ollama.is_available():
            print("Error: Ollama is not running. Start with: ollama serve")
            return []

        results = []
        for model in models:
            print(f"\nBenchmarking {model}...")
            try:
                summary = self.ollama.benchmark(
                    model=model, prompt=prompt, max_tokens=max_tokens, runs=runs
                )
                results.append(summary)
                self.results.append(summary)
            except Exception as e:
                print(f"  Error: {e}")

        return results

    def print_results(self, results: Optional[List[BenchmarkSummary]] = None):
        """Print formatted benchmark results."""
        results = results or self.results
        print(format_results_table(results, "DGX Spark LLM Benchmark Results"))

    def save_results(self, filepath: str = "benchmark_results.json"):
        """Save results to JSON file."""
        import json

        output = [summary.to_dict() for summary in self.results]
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {filepath}")

    def compare_with_baseline(self, results: Optional[List[BenchmarkSummary]] = None):
        """Compare results with NVIDIA published baselines."""
        results = results or self.results

        print("\n" + "=" * 70)
        print("Comparison with NVIDIA Baselines")
        print("=" * 70)

        for r in results:
            baseline = self.NVIDIA_BASELINES.get(r.model.lower())
            if baseline:
                prefill_ratio = r.avg_prefill_tps / baseline["prefill"] * 100
                decode_ratio = r.avg_decode_tps / baseline["decode"] * 100
                print(f"{r.model}:")
                print(
                    f"  Prefill: {r.avg_prefill_tps:.1f} tok/s ({prefill_ratio:.0f}% of baseline)"
                )
                print(
                    f"  Decode:  {r.avg_decode_tps:.1f} tok/s ({decode_ratio:.0f}% of baseline)"
                )
            else:
                print(f"{r.model}: No baseline available")


def quick_benchmark(
    model: str = "qwen3:8b", runs: int = 3
) -> Optional[BenchmarkSummary]:
    """
    Quick benchmark for a single Ollama model.

    Args:
        model: Model name (Ollama format)
        runs: Number of benchmark runs

    Returns:
        BenchmarkSummary or None if Ollama not available
    """
    benchmark = OllamaBenchmark()

    if not benchmark.is_available():
        print("Error: Ollama is not running. Start with: ollama serve")
        return None

    print(f"Benchmarking {model} ({runs} runs)...")
    summary = benchmark.benchmark(
        model=model, prompt=DEFAULT_PROMPTS["medium"], max_tokens=128, runs=runs
    )

    print(f"\nResults:")
    print(
        f"  Prefill: {summary.avg_prefill_tps:.1f} ± {summary.std_prefill_tps:.1f} tok/s"
    )
    print(
        f"  Decode:  {summary.avg_decode_tps:.1f} ± {summary.std_decode_tps:.1f} tok/s"
    )
    print(f"  Memory:  {summary.avg_memory_gb:.1f} GB")

    return summary
