"""
Benchmark Utilities for Inference Engine Comparison

This module provides tools for benchmarking different LLM inference engines
on DGX Spark. Supports Ollama, llama.cpp, vLLM, TensorRT-LLM, and SGLang.

Example:
    >>> from benchmark_utils import InferenceBenchmark, BenchmarkResult
    >>> benchmark = InferenceBenchmark(engine="ollama", model="qwen3:8b")
    >>> result = benchmark.run_single("What is 2+2?", max_tokens=50)
    >>> print(f"TTFT: {result.time_to_first_token:.3f}s, Total: {result.total_time:.3f}s")
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Optional

import aiohttp
import requests


class InferenceEngine(Enum):
    """Supported inference engines."""
    OLLAMA = "ollama"
    LLAMA_CPP = "llama.cpp"
    VLLM = "vllm"
    TENSORRT_LLM = "tensorrt-llm"
    SGLANG = "sglang"


@dataclass
class BenchmarkResult:
    """
    Results from a single inference benchmark.

    Attributes:
        prompt: The input prompt used
        output: The generated text
        time_to_first_token: Seconds until first token received (TTFT)
        total_time: Total generation time in seconds
        tokens_generated: Number of tokens in the output
        tokens_per_second: Decode throughput
        prompt_tokens: Number of tokens in the prompt
        prefill_time: Time to process the prompt (prefill phase)
        decode_time: Time spent generating tokens (decode phase)
        memory_used_gb: GPU memory used during generation
        engine: Which inference engine was used
        model: Model identifier
        timestamp: When the benchmark was run
        error: Error message if benchmark failed
    """
    prompt: str
    output: str = ""
    time_to_first_token: float = 0.0
    total_time: float = 0.0
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    prompt_tokens: int = 0
    prefill_time: float = 0.0
    decode_time: float = 0.0
    memory_used_gb: float = 0.0
    engine: str = ""
    model: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt": self.prompt[:100] + "..." if len(self.prompt) > 100 else self.prompt,
            "output_preview": self.output[:200] + "..." if len(self.output) > 200 else self.output,
            "time_to_first_token": round(self.time_to_first_token, 4),
            "total_time": round(self.total_time, 4),
            "tokens_generated": self.tokens_generated,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "prompt_tokens": self.prompt_tokens,
            "prefill_time": round(self.prefill_time, 4),
            "decode_time": round(self.decode_time, 4),
            "memory_used_gb": round(self.memory_used_gb, 2),
            "engine": self.engine,
            "model": self.model,
            "timestamp": self.timestamp,
            "error": self.error
        }


@dataclass
class BatchBenchmarkResult:
    """
    Results from a batch benchmark run.

    Attributes:
        results: Individual benchmark results
        total_time: Total time for all requests
        throughput_rps: Requests per second
        avg_ttft: Average time to first token
        avg_tokens_per_second: Average decode speed
        p50_latency: 50th percentile latency
        p90_latency: 90th percentile latency
        p99_latency: 99th percentile latency
        concurrency: Number of concurrent requests
        successful: Number of successful requests
        failed: Number of failed requests
    """
    results: list[BenchmarkResult] = field(default_factory=list)
    total_time: float = 0.0
    throughput_rps: float = 0.0
    avg_ttft: float = 0.0
    avg_tokens_per_second: float = 0.0
    p50_latency: float = 0.0
    p90_latency: float = 0.0
    p99_latency: float = 0.0
    concurrency: int = 1
    successful: int = 0
    failed: int = 0

    def compute_stats(self) -> None:
        """Compute aggregate statistics from individual results."""
        if not self.results:
            return

        successful_results = [r for r in self.results if r.error is None]
        self.successful = len(successful_results)
        self.failed = len(self.results) - self.successful

        if not successful_results:
            return

        latencies = sorted([r.total_time for r in successful_results])
        ttfts = [r.time_to_first_token for r in successful_results]
        tps_values = [r.tokens_per_second for r in successful_results if r.tokens_per_second > 0]

        self.avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0
        self.avg_tokens_per_second = sum(tps_values) / len(tps_values) if tps_values else 0

        n = len(latencies)
        self.p50_latency = latencies[n // 2] if n > 0 else 0
        self.p90_latency = latencies[int(n * 0.9)] if n > 0 else 0
        self.p99_latency = latencies[int(n * 0.99)] if n > 0 else 0

        if self.total_time > 0:
            self.throughput_rps = self.successful / self.total_time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_time": round(self.total_time, 3),
            "throughput_rps": round(self.throughput_rps, 2),
            "avg_ttft": round(self.avg_ttft, 4),
            "avg_tokens_per_second": round(self.avg_tokens_per_second, 2),
            "p50_latency": round(self.p50_latency, 4),
            "p90_latency": round(self.p90_latency, 4),
            "p99_latency": round(self.p99_latency, 4),
            "concurrency": self.concurrency,
            "successful": self.successful,
            "failed": self.failed,
            "total_requests": len(self.results)
        }

    def summary(self) -> str:
        """Return a human-readable summary."""
        return f"""
Batch Benchmark Results
=======================
Total Requests: {len(self.results)} (Success: {self.successful}, Failed: {self.failed})
Total Time: {self.total_time:.2f}s
Throughput: {self.throughput_rps:.2f} req/s
Concurrency: {self.concurrency}

Latency Statistics:
  - Average TTFT: {self.avg_ttft*1000:.1f}ms
  - P50 Latency: {self.p50_latency*1000:.1f}ms
  - P90 Latency: {self.p90_latency*1000:.1f}ms
  - P99 Latency: {self.p99_latency*1000:.1f}ms

Performance:
  - Avg Tokens/Second: {self.avg_tokens_per_second:.1f}
"""


class BaseInferenceClient(ABC):
    """Abstract base class for inference engine clients."""

    def __init__(self, base_url: str, model: str):
        """
        Initialize the inference client.

        Args:
            base_url: The API endpoint URL
            model: The model identifier
        """
        self.base_url = base_url.rstrip("/")
        self.model = model

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = False
    ) -> BenchmarkResult:
        """Generate text from a prompt."""
        pass

    @abstractmethod
    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = False
    ) -> BenchmarkResult:
        """Generate text asynchronously."""
        pass


class OllamaClient(BaseInferenceClient):
    """Client for Ollama inference server."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen3:8b"):
        super().__init__(base_url, model)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = True
    ) -> BenchmarkResult:
        """
        Generate text using Ollama.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response

        Returns:
            BenchmarkResult with timing and output data
        """
        result = BenchmarkResult(
            prompt=prompt,
            engine="ollama",
            model=self.model
        )

        try:
            start_time = time.perf_counter()
            first_token_time = None
            output_chunks = []

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature
                    },
                    "stream": stream
                },
                stream=stream,
                timeout=120
            )
            response.raise_for_status()

            if stream:
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if first_token_time is None and chunk.get("response"):
                            first_token_time = time.perf_counter()
                        output_chunks.append(chunk.get("response", ""))

                        if chunk.get("done"):
                            result.prompt_tokens = chunk.get("prompt_eval_count", 0)
                            result.tokens_generated = chunk.get("eval_count", 0)
            else:
                first_token_time = time.perf_counter()
                data = response.json()
                output_chunks.append(data.get("response", ""))
                result.prompt_tokens = data.get("prompt_eval_count", 0)
                result.tokens_generated = data.get("eval_count", 0)

            end_time = time.perf_counter()

            result.output = "".join(output_chunks)
            result.total_time = end_time - start_time
            result.time_to_first_token = (first_token_time - start_time) if first_token_time else result.total_time
            result.prefill_time = result.time_to_first_token
            result.decode_time = result.total_time - result.prefill_time

            if result.decode_time > 0 and result.tokens_generated > 0:
                result.tokens_per_second = result.tokens_generated / result.decode_time

        except Exception as e:
            result.error = str(e)
            result.total_time = time.perf_counter() - start_time if 'start_time' in locals() else 0

        return result

    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = True
    ) -> BenchmarkResult:
        """Async version of generate."""
        result = BenchmarkResult(
            prompt=prompt,
            engine="ollama",
            model=self.model
        )

        try:
            start_time = time.perf_counter()
            first_token_time = None
            output_chunks = []

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": temperature
                        },
                        "stream": stream
                    },
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    response.raise_for_status()

                    if stream:
                        async for line in response.content:
                            if line:
                                chunk = json.loads(line)
                                if first_token_time is None and chunk.get("response"):
                                    first_token_time = time.perf_counter()
                                output_chunks.append(chunk.get("response", ""))

                                if chunk.get("done"):
                                    result.prompt_tokens = chunk.get("prompt_eval_count", 0)
                                    result.tokens_generated = chunk.get("eval_count", 0)
                    else:
                        first_token_time = time.perf_counter()
                        data = await response.json()
                        output_chunks.append(data.get("response", ""))
                        result.prompt_tokens = data.get("prompt_eval_count", 0)
                        result.tokens_generated = data.get("eval_count", 0)

            end_time = time.perf_counter()

            result.output = "".join(output_chunks)
            result.total_time = end_time - start_time
            result.time_to_first_token = (first_token_time - start_time) if first_token_time else result.total_time
            result.prefill_time = result.time_to_first_token
            result.decode_time = result.total_time - result.prefill_time

            if result.decode_time > 0 and result.tokens_generated > 0:
                result.tokens_per_second = result.tokens_generated / result.decode_time

        except Exception as e:
            result.error = str(e)
            result.total_time = time.perf_counter() - start_time if 'start_time' in locals() else 0

        return result


class OpenAICompatibleClient(BaseInferenceClient):
    """Client for OpenAI-compatible APIs (vLLM, SGLang, TensorRT-LLM)."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        engine_name: str = "vllm"
    ):
        super().__init__(base_url, model)
        self.engine_name = engine_name

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = True
    ) -> BenchmarkResult:
        """Generate using OpenAI-compatible API."""
        result = BenchmarkResult(
            prompt=prompt,
            engine=self.engine_name,
            model=self.model
        )

        try:
            start_time = time.perf_counter()
            first_token_time = None
            output_chunks = []

            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": stream
                },
                stream=stream,
                timeout=120
            )
            response.raise_for_status()

            if stream:
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode("utf-8")
                        if line_str.startswith("data: "):
                            data_str = line_str[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    if first_token_time is None:
                                        first_token_time = time.perf_counter()
                                    output_chunks.append(content)
                            except json.JSONDecodeError:
                                continue
            else:
                first_token_time = time.perf_counter()
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                output_chunks.append(content)
                usage = data.get("usage", {})
                result.prompt_tokens = usage.get("prompt_tokens", 0)
                result.tokens_generated = usage.get("completion_tokens", 0)

            end_time = time.perf_counter()

            result.output = "".join(output_chunks)
            result.total_time = end_time - start_time
            result.time_to_first_token = (first_token_time - start_time) if first_token_time else result.total_time
            result.prefill_time = result.time_to_first_token
            result.decode_time = result.total_time - result.prefill_time

            # Estimate tokens if not provided
            if result.tokens_generated == 0:
                result.tokens_generated = int(len(result.output.split()) * 1.3)  # Rough estimate

            if result.decode_time > 0 and result.tokens_generated > 0:
                result.tokens_per_second = result.tokens_generated / result.decode_time

        except Exception as e:
            result.error = str(e)
            result.total_time = time.perf_counter() - start_time if 'start_time' in locals() else 0

        return result

    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = True
    ) -> BenchmarkResult:
        """Async version of generate."""
        result = BenchmarkResult(
            prompt=prompt,
            engine=self.engine_name,
            model=self.model
        )

        try:
            start_time = time.perf_counter()
            first_token_time = None
            output_chunks = []

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "stream": stream
                    },
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    response.raise_for_status()

                    if stream:
                        async for line in response.content:
                            if line:
                                line_str = line.decode("utf-8").strip()
                                if line_str.startswith("data: "):
                                    data_str = line_str[6:]
                                    if data_str == "[DONE]":
                                        break
                                    try:
                                        chunk = json.loads(data_str)
                                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            if first_token_time is None:
                                                first_token_time = time.perf_counter()
                                            output_chunks.append(content)
                                    except json.JSONDecodeError:
                                        continue
                    else:
                        first_token_time = time.perf_counter()
                        data = await response.json()
                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        output_chunks.append(content)
                        usage = data.get("usage", {})
                        result.prompt_tokens = usage.get("prompt_tokens", 0)
                        result.tokens_generated = usage.get("completion_tokens", 0)

            end_time = time.perf_counter()

            result.output = "".join(output_chunks)
            result.total_time = end_time - start_time
            result.time_to_first_token = (first_token_time - start_time) if first_token_time else result.total_time
            result.prefill_time = result.time_to_first_token
            result.decode_time = result.total_time - result.prefill_time

            if result.tokens_generated == 0:
                result.tokens_generated = int(len(result.output.split()) * 1.3)  # Rough estimate

            if result.decode_time > 0 and result.tokens_generated > 0:
                result.tokens_per_second = result.tokens_generated / result.decode_time

        except Exception as e:
            result.error = str(e)
            result.total_time = time.perf_counter() - start_time if 'start_time' in locals() else 0

        return result


class InferenceBenchmark:
    """
    Main benchmark orchestrator for comparing inference engines.

    Example:
        >>> benchmark = InferenceBenchmark(engine="ollama", model="qwen3:8b")
        >>>
        >>> # Single request benchmark
        >>> result = benchmark.run_single("What is AI?")
        >>> print(f"TTFT: {result.time_to_first_token:.3f}s")
        >>>
        >>> # Batch benchmark with concurrency
        >>> prompts = ["Hello", "What is Python?", "Explain ML"]
        >>> batch_result = benchmark.run_batch(prompts, concurrency=3)
        >>> print(batch_result.summary())
    """

    CLIENTS = {
        "ollama": OllamaClient,
        "vllm": lambda url, model: OpenAICompatibleClient(url, model, "vllm"),
        "sglang": lambda url, model: OpenAICompatibleClient(url, model, "sglang"),
        "tensorrt-llm": lambda url, model: OpenAICompatibleClient(url, model, "tensorrt-llm"),
    }

    DEFAULT_URLS = {
        "ollama": "http://localhost:11434",
        "vllm": "http://localhost:8000",
        "sglang": "http://localhost:30000",
        "tensorrt-llm": "http://localhost:8000",
    }

    def __init__(
        self,
        engine: str = "ollama",
        model: str = "qwen3:8b",
        base_url: Optional[str] = None
    ):
        """
        Initialize benchmark for a specific engine.

        Args:
            engine: One of "ollama", "vllm", "sglang", "tensorrt-llm"
            model: Model identifier
            base_url: Custom API endpoint (uses default if None)
        """
        if engine not in self.CLIENTS:
            raise ValueError(f"Unknown engine: {engine}. Supported: {list(self.CLIENTS.keys())}")

        self.engine = engine
        self.model = model
        self.base_url = base_url or self.DEFAULT_URLS.get(engine)

        client_factory = self.CLIENTS[engine]
        if engine == "ollama":
            self.client = client_factory(self.base_url, model)
        else:
            self.client = client_factory(self.base_url, model)

    def run_single(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = True
    ) -> BenchmarkResult:
        """Run a single benchmark request."""
        return self.client.generate(prompt, max_tokens, temperature, stream)

    async def run_single_async(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = True
    ) -> BenchmarkResult:
        """Run a single benchmark request asynchronously."""
        return await self.client.generate_async(prompt, max_tokens, temperature, stream)

    def run_batch(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        concurrency: int = 1,
        stream: bool = True
    ) -> BatchBenchmarkResult:
        """
        Run batch benchmark with specified concurrency.

        Args:
            prompts: List of prompts to benchmark
            max_tokens: Maximum tokens per response
            temperature: Sampling temperature
            concurrency: Number of concurrent requests
            stream: Whether to stream responses

        Returns:
            BatchBenchmarkResult with aggregate statistics
        """
        batch_result = BatchBenchmarkResult(concurrency=concurrency)

        async def run_all():
            semaphore = asyncio.Semaphore(concurrency)

            async def limited_generate(prompt: str) -> BenchmarkResult:
                async with semaphore:
                    return await self.run_single_async(prompt, max_tokens, temperature, stream)

            start_time = time.perf_counter()
            tasks = [limited_generate(p) for p in prompts]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            batch_result.total_time = time.perf_counter() - start_time

            for r in results:
                if isinstance(r, Exception):
                    error_result = BenchmarkResult(
                        prompt="",
                        engine=self.engine,
                        model=self.model,
                        error=str(r)
                    )
                    batch_result.results.append(error_result)
                else:
                    batch_result.results.append(r)

        # Handle case where event loop is already running (e.g., Jupyter notebook)
        try:
            loop = asyncio.get_running_loop()
            # We're inside an async context (like Jupyter), need special handling
            import nest_asyncio
            nest_asyncio.apply()
            asyncio.run(run_all())
        except RuntimeError:
            # No running loop, safe to use asyncio.run directly
            asyncio.run(run_all())
        except ImportError:
            # nest_asyncio not installed, try direct run (may fail in Jupyter)
            try:
                asyncio.run(run_all())
            except RuntimeError as e:
                if "cannot be called from a running event loop" in str(e):
                    print("⚠️ Running in Jupyter notebook without nest_asyncio.")
                    print("   Install with: pip install nest_asyncio")
                    print("   Or use run_single() in a loop instead of run_batch()")
                raise

        batch_result.compute_stats()

        return batch_result

    def warmup(self, num_requests: int = 3) -> None:
        """
        Warm up the inference engine with a few requests.

        This helps ensure the first real benchmark isn't affected
        by model loading or JIT compilation.
        """
        print(f"Warming up {self.engine} with {num_requests} requests...")
        for i in range(num_requests):
            self.run_single("Hello, how are you?", max_tokens=50, stream=False)
        print("Warmup complete!")


def load_benchmark_prompts(
    filepath: str | Path,
    categories: Optional[list[str]] = None
) -> list[dict[str, Any]]:
    """
    Load benchmark prompts from a JSON file.

    Args:
        filepath: Path to the benchmark prompts JSON file
        categories: Optional list of categories to include

    Returns:
        List of prompt dictionaries
    """
    with open(filepath) as f:
        data = json.load(f)

    prompts = []
    categories = categories or list(data.keys())

    for category in categories:
        if category in data:
            for item in data[category]:
                if isinstance(item, dict):
                    if "text" in item:
                        prompts.append(item)
                    elif "messages" in item:
                        # Convert chat format to single prompt
                        messages = item["messages"]
                        prompt_text = "\n".join(
                            f"{m['role']}: {m['content']}" for m in messages
                        )
                        item["text"] = prompt_text
                        prompts.append(item)

    return prompts


def compare_engines(
    engines: list[dict[str, str]],
    prompts: list[str],
    max_tokens: int = 256,
    concurrency: int = 1,
    warmup_requests: int = 3
) -> dict[str, BatchBenchmarkResult]:
    """
    Compare multiple inference engines on the same prompts.

    Args:
        engines: List of dicts with "engine", "model", and optionally "base_url"
        prompts: List of prompts to benchmark
        max_tokens: Maximum tokens per response
        concurrency: Number of concurrent requests
        warmup_requests: Number of warmup requests per engine

    Returns:
        Dictionary mapping engine names to BatchBenchmarkResult

    Example:
        >>> engines = [
        ...     {"engine": "ollama", "model": "qwen3:8b"},
        ...     {"engine": "vllm", "model": "meta-llama/Llama-3.1-8B-Instruct"},
        ... ]
        >>> prompts = ["Hello!", "What is Python?"]
        >>> results = compare_engines(engines, prompts)
        >>> for name, result in results.items():
        ...     print(f"{name}: {result.avg_tokens_per_second:.1f} tok/s")
    """
    results = {}

    for engine_config in engines:
        engine_name = engine_config["engine"]
        model = engine_config["model"]
        base_url = engine_config.get("base_url")

        print(f"\nBenchmarking {engine_name} ({model})...")

        try:
            benchmark = InferenceBenchmark(
                engine=engine_name,
                model=model,
                base_url=base_url
            )

            if warmup_requests > 0:
                benchmark.warmup(warmup_requests)

            batch_result = benchmark.run_batch(
                prompts,
                max_tokens=max_tokens,
                concurrency=concurrency
            )

            results[f"{engine_name}:{model}"] = batch_result
            print(f"  - Throughput: {batch_result.throughput_rps:.2f} req/s")
            print(f"  - Avg TTFT: {batch_result.avg_ttft*1000:.1f}ms")
            print(f"  - Avg Speed: {batch_result.avg_tokens_per_second:.1f} tok/s")

        except Exception as e:
            print(f"  - Error: {e}")
            results[f"{engine_name}:{model}"] = BatchBenchmarkResult()

    return results


def get_gpu_memory_usage() -> float:
    """
    Get current GPU memory usage in GB.

    Note:
        On DGX Spark with unified memory (128GB shared CPU+GPU),
        this returns GPU-allocated memory, not total system memory usage.
        The actual memory available may be higher as CPU and GPU share
        the same physical memory pool.

    Returns:
        Memory usage in GB, or 0.0 if unable to query
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            memory_mb = int(result.stdout.strip().split("\n")[0])
            return memory_mb / 1024.0
    except Exception:
        pass
    return 0.0


def format_comparison_table(results: dict[str, BatchBenchmarkResult]) -> str:
    """
    Format benchmark results as a markdown table.

    Args:
        results: Dictionary of engine names to BatchBenchmarkResult

    Returns:
        Markdown-formatted comparison table
    """
    lines = [
        "| Engine | Throughput (req/s) | Avg TTFT (ms) | Avg Speed (tok/s) | P50 (ms) | P99 (ms) | Success |",
        "|--------|-------------------|---------------|-------------------|----------|----------|---------|"
    ]

    for name, result in results.items():
        lines.append(
            f"| {name} | {result.throughput_rps:.2f} | "
            f"{result.avg_ttft*1000:.1f} | {result.avg_tokens_per_second:.1f} | "
            f"{result.p50_latency*1000:.1f} | {result.p99_latency*1000:.1f} | "
            f"{result.successful}/{len(result.results)} |"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick demo
    print("Benchmark Utils Demo")
    print("=" * 50)

    # Test with Ollama if available
    try:
        benchmark = InferenceBenchmark(engine="ollama", model="qwen3:8b")
        result = benchmark.run_single("What is the capital of France?", max_tokens=50)
        print(f"\nSingle Request Results:")
        print(f"  TTFT: {result.time_to_first_token*1000:.1f}ms")
        print(f"  Total Time: {result.total_time:.3f}s")
        print(f"  Tokens/Second: {result.tokens_per_second:.1f}")
        print(f"  Output: {result.output[:100]}...")
    except Exception as e:
        print(f"Could not connect to Ollama: {e}")
        print("Make sure Ollama is running with: ollama serve")
