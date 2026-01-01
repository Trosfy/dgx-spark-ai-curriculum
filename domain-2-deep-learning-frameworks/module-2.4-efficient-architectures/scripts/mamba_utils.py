"""
Mamba (State Space Models) Utilities for DGX Spark

This module provides utilities for loading, running, and analyzing Mamba models.
Mamba offers O(n) complexity vs transformers' O(n^2), enabling efficient long-context processing.

Key Features:
- Model loading with DGX Spark memory optimization
- Inference benchmarking with timing and memory tracking
- State evolution visualization for understanding the selective scan
- Comparison utilities with transformer baselines

Example Usage:
    from scripts.mamba_utils import load_mamba_model, benchmark_mamba_inference

    model, tokenizer = load_mamba_model("state-spaces/mamba-2.8b-hf")
    results = benchmark_mamba_inference(model, tokenizer, context_lengths=[4096, 16384])
    print(f"Tokens/sec at 16K context: {results[16384]['tokens_per_second']:.2f}")

Requirements:
    - transformers >= 4.46.0 (for Mamba support)
    - torch with CUDA support
    - mamba-ssm (optional, for advanced features)

DGX Spark Notes:
    - Mamba's constant memory usage shines with 128GB unified memory
    - No KV cache means 100K+ token contexts are feasible
    - Use bfloat16 for native Blackwell support
"""

import gc
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed. Run: pip install transformers>=4.39.0")


@dataclass
class MambaInferenceResult:
    """Container for Mamba inference benchmark results.

    Attributes:
        context_length: Number of input tokens
        generation_length: Number of tokens generated
        total_time_seconds: Total inference time
        time_to_first_token: Time to generate first token
        tokens_per_second: Generation throughput
        memory_used_gb: Peak GPU memory usage
        memory_allocated_gb: Memory allocated by PyTorch
    """
    context_length: int
    generation_length: int
    total_time_seconds: float
    time_to_first_token: float
    tokens_per_second: float
    memory_used_gb: float
    memory_allocated_gb: float


def load_mamba_model(
    model_name: str = "state-spaces/mamba-2.8b-hf",
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    use_flash_attention: bool = False,
) -> Tuple["AutoModelForCausalLM", "AutoTokenizer"]:
    """
    Load a Mamba model optimized for DGX Spark.

    Args:
        model_name: HuggingFace model identifier or local path
            Options: "state-spaces/mamba-130m-hf", "state-spaces/mamba-1.4b-hf",
                     "state-spaces/mamba-2.8b-hf"
        dtype: Model precision (bfloat16 recommended for Blackwell)
        device_map: Device placement strategy ("auto" for DGX Spark)
        use_flash_attention: Not applicable to Mamba (parameter kept for API consistency)

    Returns:
        Tuple of (model, tokenizer)

    Example:
        >>> model, tokenizer = load_mamba_model("state-spaces/mamba-2.8b-hf")
        >>> print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
        Model loaded with 2,768,000,000 parameters

    DGX Spark Notes:
        - 128GB unified memory easily fits Mamba-2.8B (~6GB)
        - Multiple Mamba models can be loaded simultaneously
        - bfloat16 is native to Blackwell architecture
    """
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers>=4.39.0 required for Mamba support")

    print(f"Loading Mamba model: {model_name}")
    print(f"Using dtype: {dtype}, device_map: {device_map}")

    # Clear cache before loading
    torch.cuda.empty_cache()
    gc.collect()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with DGX Spark optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,  # Some Mamba variants require this
    )

    # Report memory usage
    memory_gb = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded. GPU memory allocated: {memory_gb:.2f} GB")

    return model, tokenizer


def generate_with_mamba(
    model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    return_timing: bool = False,
) -> Union[str, Tuple[str, Dict]]:
    """
    Generate text using a Mamba model.

    Args:
        model: Loaded Mamba model
        tokenizer: Corresponding tokenizer
        prompt: Input text to continue
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling threshold
        do_sample: Whether to use sampling (False = greedy)
        return_timing: If True, return timing information

    Returns:
        Generated text, or tuple of (text, timing_dict) if return_timing=True

    Example:
        >>> text = generate_with_mamba(model, tokenizer, "The key to AI is", max_new_tokens=50)
        >>> print(text)
        The key to AI is understanding that machines can learn patterns...
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    # Time generation
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    # Decode output
    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if return_timing:
        timing = {
            "total_time": end_time - start_time,
            "input_tokens": input_length,
            "output_tokens": len(generated_tokens),
            "tokens_per_second": len(generated_tokens) / (end_time - start_time),
        }
        return full_text, timing

    return full_text


def benchmark_mamba_inference(
    model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
    context_lengths: List[int] = [1024, 4096, 8192, 16384, 32768],
    generation_length: int = 100,
    warmup_runs: int = 2,
    benchmark_runs: int = 3,
) -> Dict[int, MambaInferenceResult]:
    """
    Benchmark Mamba inference across different context lengths.

    This demonstrates Mamba's O(n) scaling advantage - memory usage
    stays constant regardless of context length (no KV cache!).

    Args:
        model: Loaded Mamba model
        tokenizer: Corresponding tokenizer
        context_lengths: List of context sizes to test
        generation_length: Tokens to generate per test
        warmup_runs: Number of warmup iterations
        benchmark_runs: Number of timed iterations

    Returns:
        Dictionary mapping context_length to MambaInferenceResult

    Example:
        >>> results = benchmark_mamba_inference(model, tokenizer, [4096, 16384])
        >>> for ctx_len, result in results.items():
        ...     print(f"{ctx_len}: {result.tokens_per_second:.1f} tok/s, {result.memory_used_gb:.2f} GB")
        4096: 45.2 tok/s, 5.82 GB
        16384: 44.8 tok/s, 5.83 GB  # Notice: memory barely changes!

    DGX Spark Notes:
        - 128GB allows testing up to 100K+ token contexts
        - Mamba's memory is independent of context length
        - Contrast with transformers where KV cache grows with sequence
    """
    results = {}

    for ctx_len in context_lengths:
        print(f"\nBenchmarking context length: {ctx_len:,} tokens")

        # Generate synthetic input of specified length
        # Use a repeating pattern to create meaningful tokens
        base_text = "The quick brown fox jumps over the lazy dog. " * 100
        tokens = tokenizer.encode(base_text, add_special_tokens=False)

        # Repeat tokens to reach desired length
        while len(tokens) < ctx_len:
            tokens = tokens + tokens
        tokens = tokens[:ctx_len]

        input_ids = torch.tensor([tokens], device=model.device)

        # Warmup runs
        print(f"  Warmup ({warmup_runs} runs)...")
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model.generate(
                    input_ids,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

        # Clear cache between warmup and benchmark
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Benchmark runs
        print(f"  Benchmarking ({benchmark_runs} runs)...")
        times = []
        ttft_times = []

        for run in range(benchmark_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=generation_length,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            torch.cuda.synchronize()
            end = time.perf_counter()

            times.append(end - start)
            # Approximate TTFT (time to first token)
            # In practice, this would need callback-based measurement
            ttft_times.append((end - start) / generation_length * 2)

        avg_time = np.mean(times)
        avg_ttft = np.mean(ttft_times)
        memory_used = torch.cuda.max_memory_allocated() / 1e9
        memory_allocated = torch.cuda.memory_allocated() / 1e9

        results[ctx_len] = MambaInferenceResult(
            context_length=ctx_len,
            generation_length=generation_length,
            total_time_seconds=avg_time,
            time_to_first_token=avg_ttft,
            tokens_per_second=generation_length / avg_time,
            memory_used_gb=memory_used,
            memory_allocated_gb=memory_allocated,
        )

        print(f"  Results: {generation_length / avg_time:.1f} tok/s, "
              f"{memory_used:.2f} GB peak memory")

    return results


def get_mamba_memory_usage(model: "AutoModelForCausalLM") -> Dict[str, float]:
    """
    Get detailed memory usage breakdown for a Mamba model.

    Args:
        model: Loaded Mamba model

    Returns:
        Dictionary with memory statistics in GB

    Example:
        >>> memory_info = get_mamba_memory_usage(model)
        >>> print(f"Model weights: {memory_info['model_params_gb']:.2f} GB")
        >>> print(f"Total allocated: {memory_info['allocated_gb']:.2f} GB")
    """
    # Calculate model parameter memory
    param_memory = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / 1e9

    # Get CUDA memory stats
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9

    return {
        "model_params_gb": param_memory,
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "max_allocated_gb": max_allocated,
        "available_gb": 128.0 - reserved,  # DGX Spark has 128GB
    }


def visualize_state_evolution(
    model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
    text: str,
    layer_idx: int = 0,
    state_dim_slice: slice = slice(0, 16),
) -> Tuple[np.ndarray, List[str]]:
    """
    Visualize how Mamba's hidden state evolves across a sequence.

    This helps understand the "selective" part of Selective State Spaces -
    how the model chooses what information to retain at each step.

    Args:
        model: Loaded Mamba model
        tokenizer: Corresponding tokenizer
        text: Input text to analyze
        layer_idx: Which Mamba layer to visualize
        state_dim_slice: Which state dimensions to extract

    Returns:
        Tuple of (state_array, token_strings)
        - state_array: Shape [seq_len, state_dim] showing state evolution
        - token_strings: List of token strings for labeling

    Example:
        >>> states, tokens = visualize_state_evolution(
        ...     model, tokenizer, "The cat sat on the mat", layer_idx=0
        ... )
        >>> import matplotlib.pyplot as plt
        >>> plt.imshow(states.T, aspect='auto')
        >>> plt.xticks(range(len(tokens)), tokens, rotation=45)
        >>> plt.xlabel('Token')
        >>> plt.ylabel('State Dimension')
        >>> plt.title('Mamba State Evolution')

    Note:
        This is a simplified visualization. Full state analysis requires
        hooks into the Mamba layer internals.
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0]]

    # For a proper implementation, we'd need to hook into Mamba internals
    # This is a simplified version using output hidden states
    model.config.output_hidden_states = True

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    # Get hidden states from specified layer
    hidden_states = outputs.hidden_states[layer_idx + 1]  # +1 because index 0 is embeddings

    # Extract state representation
    states = hidden_states[0].cpu().numpy()[:, state_dim_slice]

    return states, tokens


def compare_with_transformer(
    mamba_model: "AutoModelForCausalLM",
    mamba_tokenizer: "AutoTokenizer",
    transformer_model: "AutoModelForCausalLM",
    transformer_tokenizer: "AutoTokenizer",
    context_lengths: List[int] = [1024, 4096, 8192],
    generation_length: int = 50,
) -> Dict[str, Dict]:
    """
    Compare Mamba vs Transformer on memory and speed.

    This demonstrates Mamba's O(n) advantage over Transformer's O(n^2).

    Args:
        mamba_model: Loaded Mamba model
        mamba_tokenizer: Mamba tokenizer
        transformer_model: Loaded transformer model (e.g., Llama, GPT)
        transformer_tokenizer: Transformer tokenizer
        context_lengths: Context sizes to compare
        generation_length: Tokens to generate

    Returns:
        Dictionary with comparison results

    Example:
        >>> comparison = compare_with_transformer(
        ...     mamba_model, mamba_tok, transformer_model, transformer_tok
        ... )
        >>> for ctx_len, results in comparison.items():
        ...     speedup = results['transformer_time'] / results['mamba_time']
        ...     print(f"{ctx_len}: Mamba is {speedup:.1f}x faster")
    """
    results = {}

    for ctx_len in context_lengths:
        print(f"\nComparing at {ctx_len:,} tokens...")

        # Prepare input for Mamba
        mamba_input = torch.randint(
            100, mamba_tokenizer.vocab_size - 100,
            (1, ctx_len), device=mamba_model.device
        )

        # Prepare input for Transformer
        transformer_input = torch.randint(
            100, transformer_tokenizer.vocab_size - 100,
            (1, ctx_len), device=transformer_model.device
        )

        # Benchmark Mamba
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        mamba_start = time.perf_counter()

        with torch.no_grad():
            _ = mamba_model.generate(
                mamba_input,
                max_new_tokens=generation_length,
                do_sample=False,
                pad_token_id=mamba_tokenizer.pad_token_id,
            )

        torch.cuda.synchronize()
        mamba_time = time.perf_counter() - mamba_start
        mamba_memory = torch.cuda.max_memory_allocated() / 1e9

        # Benchmark Transformer
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        transformer_start = time.perf_counter()

        with torch.no_grad():
            _ = transformer_model.generate(
                transformer_input,
                max_new_tokens=generation_length,
                do_sample=False,
                pad_token_id=transformer_tokenizer.pad_token_id,
            )

        torch.cuda.synchronize()
        transformer_time = time.perf_counter() - transformer_start
        transformer_memory = torch.cuda.max_memory_allocated() / 1e9

        results[ctx_len] = {
            "mamba_time": mamba_time,
            "mamba_memory_gb": mamba_memory,
            "mamba_tokens_per_sec": generation_length / mamba_time,
            "transformer_time": transformer_time,
            "transformer_memory_gb": transformer_memory,
            "transformer_tokens_per_sec": generation_length / transformer_time,
            "speedup": transformer_time / mamba_time,
            "memory_savings": transformer_memory / mamba_memory,
        }

        print(f"  Mamba: {mamba_time:.2f}s, {mamba_memory:.2f}GB")
        print(f"  Transformer: {transformer_time:.2f}s, {transformer_memory:.2f}GB")
        print(f"  Speedup: {results[ctx_len]['speedup']:.2f}x")

    return results


if __name__ == "__main__":
    # Quick test when run directly
    print("Testing Mamba utilities...")

    # Check transformers version
    try:
        import transformers
        print(f"transformers version: {transformers.__version__}")

        if tuple(map(int, transformers.__version__.split('.')[:2])) < (4, 46):
            print("Warning: Mamba support requires transformers >= 4.46.0")
            print("Run: pip install --upgrade transformers")
    except ImportError:
        print("transformers not installed")

    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
