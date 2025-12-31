"""
Pipeline Utilities for Hugging Face Transformers

This module provides helper functions for working with Hugging Face pipelines,
including batch inference, latency measurement, and demonstrations.

Example usage:
    from scripts.pipeline_utils import (
        create_pipeline_demo,
        batch_inference,
        measure_pipeline_latency,
        PipelineDemo
    )

    # Create a demo for multiple pipelines
    demo = PipelineDemo()
    demo.add_pipeline("sentiment", "sentiment-analysis")
    demo.run_all("This is amazing!")

    # Measure latency
    stats = measure_pipeline_latency("sentiment-analysis", ["Test 1", "Test 2"])
"""

from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass
import torch
import time
import statistics
from transformers import pipeline


@dataclass
class LatencyStats:
    """Statistics from latency measurements."""
    pipeline_name: str
    num_samples: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    throughput_per_sec: float


@dataclass
class PipelineResult:
    """Result from a pipeline inference."""
    input: str
    output: Any
    latency_ms: float


def get_device() -> int:
    """Get the device index for pipeline (-1 for CPU, 0+ for GPU)."""
    return 0 if torch.cuda.is_available() else -1


def create_pipeline(
    task: str,
    model: Optional[str] = None,
    device: Optional[int] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    **kwargs
) -> Any:
    """
    Create a Hugging Face pipeline with DGX Spark optimizations.

    Args:
        task: Pipeline task type (e.g., "sentiment-analysis", "text-generation")
        model: Optional model name (uses task default if not specified)
        device: Device index (-1 for CPU, 0+ for GPU)
        torch_dtype: Data type for model (bfloat16 recommended for DGX Spark)
        **kwargs: Additional pipeline arguments

    Returns:
        Configured pipeline instance.

    Example:
        >>> pipe = create_pipeline("sentiment-analysis")
        >>> result = pipe("I love this!")
        >>> print(result)
    """
    if device is None:
        device = get_device()

    pipe_kwargs = {
        "task": task,
        "device": device,
        **kwargs
    }

    if model:
        pipe_kwargs["model"] = model

    # Only add torch_dtype for GPU
    if device >= 0:
        pipe_kwargs["torch_dtype"] = torch_dtype

    return pipeline(**pipe_kwargs)


def batch_inference(
    pipe,
    inputs: List[str],
    batch_size: int = 8,
    show_progress: bool = True
) -> List[PipelineResult]:
    """
    Run batch inference with a pipeline.

    Args:
        pipe: Hugging Face pipeline
        inputs: List of input texts
        batch_size: Batch size for processing
        show_progress: Show progress indicator

    Returns:
        List of PipelineResult objects.

    Example:
        >>> pipe = create_pipeline("sentiment-analysis")
        >>> results = batch_inference(pipe, ["Great!", "Bad!", "OK"])
        >>> for r in results:
        ...     print(f"{r.input}: {r.output}")
    """
    results = []
    total = len(inputs)

    for i in range(0, total, batch_size):
        batch = inputs[i:i+batch_size]

        if show_progress:
            progress = min(i + batch_size, total)
            print(f"\rProcessing: {progress}/{total}", end="", flush=True)

        start = time.time()
        outputs = pipe(batch)
        batch_time = (time.time() - start) * 1000

        # Average time per sample in batch
        time_per_sample = batch_time / len(batch)

        for inp, out in zip(batch, outputs):
            results.append(PipelineResult(
                input=inp,
                output=out,
                latency_ms=time_per_sample
            ))

    if show_progress:
        print()  # New line after progress

    return results


def measure_pipeline_latency(
    task_or_pipe: Union[str, Any],
    test_inputs: List[str],
    warmup_runs: int = 3,
    model: Optional[str] = None
) -> LatencyStats:
    """
    Measure pipeline latency statistics.

    Args:
        task_or_pipe: Pipeline task string or pipeline instance
        test_inputs: List of test inputs
        warmup_runs: Number of warmup runs (not counted)
        model: Optional model name (only if task string provided)

    Returns:
        LatencyStats with timing statistics.

    Example:
        >>> stats = measure_pipeline_latency(
        ...     "sentiment-analysis",
        ...     ["Test 1", "Test 2", "Test 3"] * 10
        ... )
        >>> print(f"Mean latency: {stats.mean_ms:.2f}ms")
    """
    # Create pipeline if task string provided
    if isinstance(task_or_pipe, str):
        pipe = create_pipeline(task_or_pipe, model=model)
        task_name = task_or_pipe
    else:
        pipe = task_or_pipe
        task_name = str(type(pipe).__name__)

    # Warmup
    for _ in range(warmup_runs):
        pipe(test_inputs[0])

    # Measure
    latencies = []

    for inp in test_inputs:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        pipe(inp)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        latencies.append((time.time() - start) * 1000)

    return LatencyStats(
        pipeline_name=task_name,
        num_samples=len(test_inputs),
        mean_ms=statistics.mean(latencies),
        std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        min_ms=min(latencies),
        max_ms=max(latencies),
        throughput_per_sec=1000 / statistics.mean(latencies)
    )


class PipelineDemo:
    """
    Class for demonstrating multiple Hugging Face pipelines.

    Example:
        >>> demo = PipelineDemo()
        >>> demo.add_pipeline("sentiment", "sentiment-analysis")
        >>> demo.add_pipeline("ner", "ner")
        >>> demo.run_all("Apple is looking at buying U.K. startup for $1 billion")
    """

    def __init__(self, device: Optional[int] = None):
        """Initialize the demo."""
        self.device = device if device is not None else get_device()
        self.pipelines: Dict[str, Any] = {}
        self.results: Dict[str, List[Any]] = {}

    def add_pipeline(
        self,
        name: str,
        task: str,
        model: Optional[str] = None,
        **kwargs
    ) -> "PipelineDemo":
        """
        Add a pipeline to the demo.

        Args:
            name: Display name for the pipeline
            task: Pipeline task type
            model: Optional specific model
            **kwargs: Additional pipeline arguments

        Returns:
            self for chaining
        """
        self.pipelines[name] = create_pipeline(
            task=task,
            model=model,
            device=self.device,
            **kwargs
        )
        self.results[name] = []
        return self

    def run_pipeline(self, name: str, text: str, **kwargs) -> Any:
        """
        Run a specific pipeline.

        Args:
            name: Pipeline name
            text: Input text
            **kwargs: Additional arguments

        Returns:
            Pipeline output
        """
        if name not in self.pipelines:
            raise ValueError(f"Pipeline '{name}' not found. Available: {list(self.pipelines.keys())}")

        result = self.pipelines[name](text, **kwargs)
        self.results[name].append({"input": text, "output": result})
        return result

    def run_all(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Run all pipelines on the same input.

        Args:
            text: Input text
            **kwargs: Additional arguments per pipeline

        Returns:
            Dictionary of results by pipeline name
        """
        results = {}

        print(f"\nInput: {text[:100]}{'...' if len(text) > 100 else ''}\n")
        print("-" * 60)

        for name, pipe in self.pipelines.items():
            try:
                start = time.time()
                result = pipe(text, **kwargs.get(name, {}))
                latency = (time.time() - start) * 1000

                results[name] = result
                self.results[name].append({"input": text, "output": result})

                # Pretty print result
                print(f"\n{name}:")
                self._print_result(result)
                print(f"  (latency: {latency:.1f}ms)")

            except Exception as e:
                print(f"\n{name}: ERROR - {str(e)}")
                results[name] = None

        return results

    def _print_result(self, result: Any) -> None:
        """Pretty print a pipeline result."""
        if isinstance(result, list):
            for item in result[:5]:  # Limit to first 5
                if isinstance(item, dict):
                    if "label" in item:
                        print(f"  - {item['label']}: {item.get('score', 0):.4f}")
                    elif "word" in item:
                        print(f"  - {item.get('entity_group', item.get('entity', 'ENT'))}: "
                              f"'{item['word']}' ({item.get('score', 0):.4f})")
                    elif "summary_text" in item:
                        print(f"  {item['summary_text'][:200]}...")
                    elif "generated_text" in item:
                        print(f"  {item['generated_text'][:200]}...")
                    elif "answer" in item:
                        print(f"  Answer: {item['answer']} (score: {item.get('score', 0):.4f})")
                    else:
                        print(f"  - {item}")
                else:
                    print(f"  - {item}")
        elif isinstance(result, dict):
            for k, v in result.items():
                print(f"  {k}: {v}")
        else:
            print(f"  {result}")

    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get a summary of all results."""
        return {
            name: {
                "num_runs": len(results),
                "results": results
            }
            for name, results in self.results.items()
        }

    def cleanup(self) -> None:
        """Clean up pipelines and free memory."""
        for pipe in self.pipelines.values():
            del pipe
        self.pipelines.clear()

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Pre-configured pipeline examples
PIPELINE_EXAMPLES = {
    "sentiment-analysis": {
        "task": "sentiment-analysis",
        "examples": [
            "I absolutely love this product! Best purchase ever!",
            "This is terrible. Complete waste of money.",
            "It's okay, nothing special but gets the job done."
        ]
    },
    "text-generation": {
        "task": "text-generation",
        "model": "gpt2",
        "examples": [
            "Once upon a time in a land far away",
            "The future of artificial intelligence is",
            "The best way to learn programming is"
        ]
    },
    "ner": {
        "task": "ner",
        "examples": [
            "Apple CEO Tim Cook announced the new iPhone in Cupertino.",
            "Elon Musk founded SpaceX and leads Tesla in the United States.",
            "Marie Curie won the Nobel Prize in Physics in 1903."
        ]
    },
    "question-answering": {
        "task": "question-answering",
        "examples": [
            {
                "question": "What is the capital of France?",
                "context": "Paris is the capital and largest city of France. It is located on the Seine River."
            },
            {
                "question": "Who founded Microsoft?",
                "context": "Microsoft was founded by Bill Gates and Paul Allen in 1975 in Albuquerque, New Mexico."
            }
        ]
    },
    "summarization": {
        "task": "summarization",
        "examples": [
            """The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building,
            and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side.
            During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest
            man-made structure in the world, a title it held for 41 years until the Chrysler Building in
            New York City was finished in 1930."""
        ]
    }
}


def run_pipeline_showcase(
    pipelines_to_run: Optional[List[str]] = None,
    custom_input: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a showcase of multiple pipelines.

    Args:
        pipelines_to_run: List of pipeline names (None for all)
        custom_input: Optional custom input text

    Returns:
        Dictionary of results
    """
    if pipelines_to_run is None:
        pipelines_to_run = ["sentiment-analysis", "ner", "summarization"]

    demo = PipelineDemo()
    results = {}

    for name in pipelines_to_run:
        if name not in PIPELINE_EXAMPLES:
            print(f"Unknown pipeline: {name}")
            continue

        config = PIPELINE_EXAMPLES[name]
        print(f"\n{'='*60}")
        print(f"Pipeline: {name}")
        print(f"{'='*60}")

        # Create pipeline
        demo.add_pipeline(
            name=name,
            task=config["task"],
            model=config.get("model")
        )

        # Run examples
        examples = [custom_input] if custom_input else config["examples"]

        for example in examples:
            if isinstance(example, dict):
                # QA format
                result = demo.run_pipeline(name, **example)
            else:
                result = demo.run_pipeline(name, example)

            results[name] = result

        print()

    demo.cleanup()
    return results


if __name__ == "__main__":
    # Demo
    print("Pipeline Utilities Demo")
    print("=" * 50)

    # Simple demo
    demo = PipelineDemo()
    demo.add_pipeline("sentiment", "sentiment-analysis")

    result = demo.run_pipeline("sentiment", "This is amazing!")
    print(f"\nResult: {result}")

    # Latency test
    print("\nMeasuring latency...")
    stats = measure_pipeline_latency(
        "sentiment-analysis",
        ["Test input"] * 10
    )
    print(f"Mean: {stats.mean_ms:.2f}ms")
    print(f"Throughput: {stats.throughput_per_sec:.1f}/sec")

    demo.cleanup()
