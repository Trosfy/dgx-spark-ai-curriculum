#!/usr/bin/env python3
"""
LLM Benchmark Utilities for DGX Spark
Direct backend benchmarking for accurate performance metrics
"""

import time
import json
import subprocess
import statistics
import requests
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
import torch


@dataclass
class BenchmarkResult:
    """Single benchmark run result"""
    model: str
    backend: str
    quantization: str
    prompt_tokens: int
    generated_tokens: int
    prefill_time_s: float
    decode_time_s: float
    total_time_s: float
    prefill_tps: float  # tokens per second
    decode_tps: float   # tokens per second
    memory_gb: float
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class BenchmarkSummary:
    """Summary of multiple benchmark runs"""
    model: str
    backend: str
    quantization: str
    runs: int
    avg_prefill_tps: float
    std_prefill_tps: float
    avg_decode_tps: float
    std_decode_tps: float
    avg_memory_gb: float
    results: List[BenchmarkResult] = field(default_factory=list)


def get_gpu_memory_gb() -> float:
    """Get current GPU memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    
    # Fallback to nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        return float(result.stdout.strip()) / 1024
    except:
        return 0.0


class OllamaBenchmark:
    """Benchmark Ollama models using direct API calls"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
    
    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """List available models"""
        response = requests.get(f"{self.base_url}/api/tags")
        data = response.json()
        return [m["name"] for m in data.get("models", [])]
    
    def warmup(self, model: str, prompt: str = "Hello"):
        """Warmup model to ensure it's loaded"""
        requests.post(
            f"{self.base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
    
    def benchmark_single(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 128
    ) -> BenchmarkResult:
        """Run a single benchmark iteration"""
        
        start_time = time.perf_counter()
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens}
            }
        )
        
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
        quant = "unknown"
        if ":q4" in model.lower() or "q4_" in model.lower():
            quant = "Q4"
        elif ":q8" in model.lower() or "q8_" in model.lower():
            quant = "Q8"
        elif ":fp16" in model.lower():
            quant = "FP16"
        else:
            quant = "default"
        
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
            memory_gb=get_gpu_memory_gb()
        )
    
    def benchmark(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 128,
        runs: int = 5,
        warmup_runs: int = 1
    ) -> BenchmarkSummary:
        """Run multiple benchmark iterations and return summary"""
        
        # Warmup
        for _ in range(warmup_runs):
            self.warmup(model, prompt[:100])
        
        # Benchmark runs
        results = []
        for i in range(runs):
            result = self.benchmark_single(model, prompt, max_tokens)
            results.append(result)
            print(f"  Run {i+1}/{runs}: prefill={result.prefill_tps:.1f} tok/s, decode={result.decode_tps:.1f} tok/s")
        
        prefill_values = [r.prefill_tps for r in results]
        decode_values = [r.decode_tps for r in results]
        memory_values = [r.memory_gb for r in results]
        
        return BenchmarkSummary(
            model=model,
            backend="ollama",
            quantization=results[0].quantization,
            runs=runs,
            avg_prefill_tps=statistics.mean(prefill_values),
            std_prefill_tps=statistics.stdev(prefill_values) if runs > 1 else 0,
            avg_decode_tps=statistics.mean(decode_values),
            std_decode_tps=statistics.stdev(decode_values) if runs > 1 else 0,
            avg_memory_gb=statistics.mean(memory_values),
            results=results
        )


class LlamaCppBenchmark:
    """Benchmark llama.cpp using llama-bench"""
    
    def __init__(self, llama_cpp_path: str = "./llama.cpp"):
        self.llama_cpp_path = Path(llama_cpp_path)
        self.bench_path = self.llama_cpp_path / "llama-bench"
    
    def is_available(self) -> bool:
        """Check if llama-bench is available"""
        return self.bench_path.exists()
    
    def benchmark(
        self,
        model_path: str,
        prompt_tokens: int = 512,
        gen_tokens: int = 128,
        runs: int = 5
    ) -> BenchmarkSummary:
        """Run llama-bench and parse results"""
        
        cmd = [
            str(self.bench_path),
            "-m", model_path,
            "-p", str(prompt_tokens),
            "-n", str(gen_tokens),
            "-r", str(runs),
            "-o", "json"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse JSON output
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            # Try to parse line by line
            lines = result.stdout.strip().split("\n")
            data = [json.loads(line) for line in lines if line.startswith("{")]
            data = data[-1] if data else {}
        
        # Extract metrics
        prefill_tps = data.get("t_pp_avg", 0) or data.get("pp", 0)
        decode_tps = data.get("t_tg_avg", 0) or data.get("tg", 0)
        
        results = [BenchmarkResult(
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
            memory_gb=get_gpu_memory_gb()
        )]
        
        return BenchmarkSummary(
            model=model_path,
            backend="llama.cpp",
            quantization=results[0].quantization,
            runs=runs,
            avg_prefill_tps=prefill_tps,
            std_prefill_tps=0,
            avg_decode_tps=decode_tps,
            std_decode_tps=0,
            avg_memory_gb=results[0].memory_gb,
            results=results
        )
    
    def _detect_quantization(self, model_path: str) -> str:
        """Detect quantization from filename"""
        path_lower = model_path.lower()
        for quant in ["q2_k", "q3_k", "q4_0", "q4_k", "q5_0", "q5_k", "q6_k", "q8_0", "f16", "f32"]:
            if quant in path_lower:
                return quant.upper()
        return "unknown"


class BenchmarkSuite:
    """Complete benchmark suite for DGX Spark"""
    
    def __init__(self):
        self.ollama = OllamaBenchmark()
        self.llamacpp = LlamaCppBenchmark()
        self.results: List[BenchmarkSummary] = []
    
    def run_ollama_suite(
        self,
        models: List[str],
        prompt: str,
        max_tokens: int = 128,
        runs: int = 5
    ) -> List[BenchmarkSummary]:
        """Run benchmarks on multiple Ollama models"""
        
        if not self.ollama.is_available():
            print("Error: Ollama is not running")
            return []
        
        results = []
        
        for model in models:
            print(f"\nBenchmarking {model}...")
            try:
                summary = self.ollama.benchmark(model, prompt, max_tokens, runs)
                results.append(summary)
                self.results.append(summary)
            except Exception as e:
                print(f"  Error: {e}")
        
        return results
    
    def print_results(self, results: Optional[List[BenchmarkSummary]] = None):
        """Print formatted benchmark results"""
        
        results = results or self.results
        
        if not results:
            print("No results to display")
            return
        
        print("\n" + "=" * 90)
        print("DGX Spark LLM Benchmark Results")
        print("=" * 90)
        print(f"{'Model':<30} {'Backend':<12} {'Quant':<8} {'Prefill':<15} {'Decode':<15} {'Memory':<10}")
        print(f"{'':30} {'':12} {'':8} {'(tok/s)':<15} {'(tok/s)':<15} {'(GB)':<10}")
        print("-" * 90)
        
        for r in results:
            prefill_str = f"{r.avg_prefill_tps:.1f} ± {r.std_prefill_tps:.1f}"
            decode_str = f"{r.avg_decode_tps:.1f} ± {r.std_decode_tps:.1f}"
            print(f"{r.model:<30} {r.backend:<12} {r.quantization:<8} {prefill_str:<15} {decode_str:<15} {r.avg_memory_gb:<10.1f}")
        
        print("=" * 90)
    
    def save_results(self, filepath: str = "benchmark_results.json"):
        """Save results to JSON file"""
        
        output = []
        for summary in self.results:
            output.append({
                "model": summary.model,
                "backend": summary.backend,
                "quantization": summary.quantization,
                "runs": summary.runs,
                "avg_prefill_tps": summary.avg_prefill_tps,
                "std_prefill_tps": summary.std_prefill_tps,
                "avg_decode_tps": summary.avg_decode_tps,
                "std_decode_tps": summary.std_decode_tps,
                "avg_memory_gb": summary.avg_memory_gb,
                "individual_runs": [r.to_dict() for r in summary.results]
            })
        
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def compare_with_baseline(
        self,
        results: List[BenchmarkSummary],
        baseline_file: str = "nvidia_baseline.json"
    ):
        """Compare results with NVIDIA published baselines"""
        
        # NVIDIA published baselines for DGX Spark
        nvidia_baselines = {
            "llama3.1:8b": {"prefill": 3000, "decode": 45},
            "llama3.1:70b": {"prefill": 500, "decode": 15},
            "gpt-oss-20b": {"prefill": 4500, "decode": 59},
        }
        
        print("\n" + "=" * 70)
        print("Comparison with NVIDIA Baselines")
        print("=" * 70)
        
        for r in results:
            baseline = nvidia_baselines.get(r.model.lower())
            if baseline:
                prefill_ratio = r.avg_prefill_tps / baseline["prefill"] * 100
                decode_ratio = r.avg_decode_tps / baseline["decode"] * 100
                print(f"{r.model}:")
                print(f"  Prefill: {r.avg_prefill_tps:.1f} tok/s ({prefill_ratio:.0f}% of baseline)")
                print(f"  Decode:  {r.avg_decode_tps:.1f} tok/s ({decode_ratio:.0f}% of baseline)")
            else:
                print(f"{r.model}: No baseline available")


def quick_benchmark(model: str = "llama3.1:8b", runs: int = 3):
    """Quick benchmark for a single model"""
    
    prompt = """Explain the concept of machine learning in simple terms. 
    Include examples of how it's used in everyday life."""
    
    benchmark = OllamaBenchmark()
    
    if not benchmark.is_available():
        print("Error: Ollama is not running. Start with: ollama serve")
        return None
    
    print(f"Benchmarking {model} ({runs} runs)...")
    summary = benchmark.benchmark(model, prompt, max_tokens=128, runs=runs)
    
    print(f"\nResults:")
    print(f"  Prefill: {summary.avg_prefill_tps:.1f} ± {summary.std_prefill_tps:.1f} tok/s")
    print(f"  Decode:  {summary.avg_decode_tps:.1f} ± {summary.std_decode_tps:.1f} tok/s")
    print(f"  Memory:  {summary.avg_memory_gb:.1f} GB")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DGX Spark LLM Benchmark")
    parser.add_argument("--models", nargs="+", default=["llama3.1:8b"],
                       help="Models to benchmark")
    parser.add_argument("--runs", type=int, default=5,
                       help="Number of benchmark runs")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                       help="Output file path")
    
    args = parser.parse_args()
    
    prompt = """You are a helpful AI assistant. Please provide a detailed explanation 
    of how neural networks learn through backpropagation. Include the mathematical 
    foundations and practical considerations for training deep networks."""
    
    suite = BenchmarkSuite()
    results = suite.run_ollama_suite(args.models, prompt, max_tokens=128, runs=args.runs)
    suite.print_results(results)
    suite.save_results(args.output)
