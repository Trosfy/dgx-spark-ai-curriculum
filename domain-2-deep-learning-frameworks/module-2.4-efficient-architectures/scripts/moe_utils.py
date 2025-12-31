"""
Mixture of Experts (MoE) Utilities for DGX Spark

This module provides utilities for loading, analyzing, and understanding
MoE models like Mixtral and DeepSeekMoE.

Key Concepts:
- MoE models have many experts but only activate a few per token
- Router/gating network decides which experts process each token
- This enables massive parameter counts with efficient compute

Example Usage:
    from scripts.moe_utils import load_moe_model, extract_expert_activations

    model, tokenizer = load_moe_model("deepseek-ai/deepseek-moe-16b-base")
    activations = extract_expert_activations(model, tokenizer, "Write Python code")
    print(f"Experts used: {activations['experts_used']}")

DGX Spark Advantage:
- 128GB unified memory can load full MoE models (no need for offloading)
- DeepSeekMoE-16B: ~32GB, fits easily
- Mixtral 8x7B: ~90GB, fits with room for inference

Requirements:
    - transformers >= 4.36.0 (for Mixtral/MoE support)
    - torch with CUDA support
"""

import gc
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import numpy as np

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class ExpertActivation:
    """Container for expert activation analysis.

    Attributes:
        token_idx: Position in sequence
        token_str: Decoded token string
        experts_selected: List of expert indices activated
        expert_weights: Routing weights for each selected expert
        router_logits: Raw router output logits
    """
    token_idx: int
    token_str: str
    experts_selected: List[int]
    expert_weights: List[float]
    router_logits: Optional[np.ndarray] = None


@dataclass
class MoEModelInfo:
    """Information about an MoE model's architecture.

    Attributes:
        total_params: Total parameter count
        active_params: Parameters active per forward pass
        num_experts: Number of experts per layer
        experts_per_token: How many experts route to each token
        num_layers: Number of MoE layers
        hidden_size: Model hidden dimension
        memory_gb: Approximate memory usage in GB
    """
    total_params: int
    active_params: int
    num_experts: int
    experts_per_token: int
    num_layers: int
    hidden_size: int
    memory_gb: float


# Known MoE model configurations
MOE_CONFIGS = {
    "mixtral-8x7b": MoEModelInfo(
        total_params=46_700_000_000,
        active_params=12_900_000_000,
        num_experts=8,
        experts_per_token=2,
        num_layers=32,
        hidden_size=4096,
        memory_gb=93.4,  # FP16
    ),
    "deepseek-moe-16b": MoEModelInfo(
        total_params=16_000_000_000,
        active_params=2_500_000_000,
        num_experts=64,
        experts_per_token=6,
        num_layers=28,
        hidden_size=2048,
        memory_gb=32.0,  # FP16
    ),
    "qwen-moe-a2.7b": MoEModelInfo(
        total_params=14_300_000_000,
        active_params=2_700_000_000,
        num_experts=60,
        experts_per_token=4,
        num_layers=24,
        hidden_size=2048,
        memory_gb=28.6,  # FP16
    ),
}


def load_moe_model(
    model_name: str = "deepseek-ai/deepseek-moe-16b-base",
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> Tuple["AutoModelForCausalLM", "AutoTokenizer"]:
    """
    Load an MoE model optimized for DGX Spark.

    Args:
        model_name: HuggingFace model identifier
            Options: "mistralai/Mixtral-8x7B-v0.1",
                     "deepseek-ai/deepseek-moe-16b-base",
                     "Qwen/Qwen1.5-MoE-A2.7B"
        dtype: Model precision (bfloat16 recommended)
        device_map: Device placement ("auto" recommended)
        load_in_8bit: Use 8-bit quantization (saves memory)
        load_in_4bit: Use 4-bit quantization (saves more memory)

    Returns:
        Tuple of (model, tokenizer)

    Example:
        >>> model, tokenizer = load_moe_model("deepseek-ai/deepseek-moe-16b-base")
        >>> # On DGX Spark: Uses ~32GB of 128GB available
        >>> print(f"Memory used: {torch.cuda.memory_allocated()/1e9:.1f}GB")
        Memory used: 31.8GB

    DGX Spark Notes:
        - Mixtral 8x7B (~90GB) fits comfortably in 128GB
        - DeepSeekMoE-16B (~32GB) leaves room for large batch inference
        - With 8-bit, even larger models become feasible
    """
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers required. Install: pip install transformers>=4.36.0")

    print(f"Loading MoE model: {model_name}")

    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare loading kwargs
    load_kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
    }

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
        )
        print("Loading with 4-bit quantization")
    elif load_in_8bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        print("Loading with 8-bit quantization")
    else:
        load_kwargs["torch_dtype"] = dtype

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # Report memory
    memory_gb = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded. GPU memory: {memory_gb:.1f} GB")

    return model, tokenizer


def get_moe_info(model: "AutoModelForCausalLM") -> Dict[str, Any]:
    """
    Extract MoE architecture information from a loaded model.

    Args:
        model: Loaded MoE model

    Returns:
        Dictionary with model architecture details

    Example:
        >>> info = get_moe_info(model)
        >>> print(f"Experts per layer: {info['num_experts']}")
        >>> print(f"Experts per token: {info['experts_per_token']}")
    """
    config = model.config

    info = {
        "model_type": getattr(config, "model_type", "unknown"),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_layers": getattr(config, "num_hidden_layers", None),
        "vocab_size": getattr(config, "vocab_size", None),
    }

    # Extract MoE-specific config (varies by model)
    if hasattr(config, "num_local_experts"):
        info["num_experts"] = config.num_local_experts
    elif hasattr(config, "num_experts"):
        info["num_experts"] = config.num_experts
    elif hasattr(config, "n_routed_experts"):
        info["num_experts"] = config.n_routed_experts

    if hasattr(config, "num_experts_per_tok"):
        info["experts_per_token"] = config.num_experts_per_tok
    elif hasattr(config, "top_k"):
        info["experts_per_token"] = config.top_k
    elif hasattr(config, "num_experts_per_token"):
        info["experts_per_token"] = config.num_experts_per_token

    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    info["total_params"] = total_params
    info["total_params_billions"] = total_params / 1e9

    return info


def get_router_weights(
    model: "AutoModelForCausalLM",
    layer_idx: int = 0,
) -> torch.Tensor:
    """
    Extract router (gate) weights from an MoE layer.

    The router is a linear layer that maps hidden states to expert scores.
    Analyzing these weights reveals what "patterns" each expert specializes in.

    Args:
        model: Loaded MoE model
        layer_idx: Which layer to extract from

    Returns:
        Router weight tensor of shape [num_experts, hidden_size]

    Example:
        >>> router_weights = get_router_weights(model, layer_idx=0)
        >>> print(f"Router shape: {router_weights.shape}")
        >>> # [64, 2048] for DeepSeekMoE with 64 experts, 2048 hidden
    """
    # Navigate model structure (varies by architecture)
    try:
        # Mixtral structure
        if hasattr(model, "model"):
            layers = model.model.layers
        else:
            layers = model.layers

        layer = layers[layer_idx]

        # Try different MoE access patterns
        if hasattr(layer, "block_sparse_moe"):
            # Mixtral
            router = layer.block_sparse_moe.gate
        elif hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            # Some DeepSeek variants
            router = layer.mlp.gate
        elif hasattr(layer, "moe"):
            # Other variants
            router = layer.moe.gate
        else:
            raise AttributeError(f"Cannot find router in layer {layer_idx}")

        return router.weight.data.clone()

    except Exception as e:
        print(f"Error extracting router weights: {e}")
        print("Model structure may differ from expected.")
        return None


def extract_expert_activations(
    model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
    text: str,
    return_logits: bool = False,
) -> List[ExpertActivation]:
    """
    Analyze which experts activate for each token in the input.

    This reveals expert specialization patterns - do certain experts
    handle code? Math? Creative writing?

    Args:
        model: Loaded MoE model
        tokenizer: Corresponding tokenizer
        text: Input text to analyze
        return_logits: If True, include raw router logits

    Returns:
        List of ExpertActivation objects, one per token

    Example:
        >>> activations = extract_expert_activations(
        ...     model, tokenizer, "def fibonacci(n):"
        ... )
        >>> for act in activations:
        ...     print(f"Token '{act.token_str}': experts {act.experts_selected}")
        Token 'def': experts [12, 45]
        Token ' fib': experts [12, 23]
        Token 'onacci': experts [23, 45]

    Note:
        This function hooks into the model's forward pass to capture
        router decisions. Results depend on the specific MoE architecture.
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"][0]

    # Storage for captured activations
    captured_routing = []

    def capture_router_hook(module, input, output):
        """Hook to capture router outputs."""
        # Router output format varies by model
        if isinstance(output, tuple):
            routing_weights = output[0]
        else:
            routing_weights = output
        captured_routing.append(routing_weights.detach().cpu())

    # Find and hook into router
    hooks = []
    try:
        if hasattr(model, "model"):
            layers = model.model.layers
        else:
            layers = model.layers

        # Hook first layer's router (representative)
        layer = layers[0]
        if hasattr(layer, "block_sparse_moe"):
            router = layer.block_sparse_moe.gate
        elif hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            router = layer.mlp.gate
        else:
            print("Warning: Could not find router to hook")
            router = None

        if router:
            hook = router.register_forward_hook(capture_router_hook)
            hooks.append(hook)

    except Exception as e:
        print(f"Error setting up hooks: {e}")

    # Forward pass
    with torch.no_grad():
        _ = model(inputs["input_ids"])

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Process captured routing
    activations = []

    if captured_routing:
        routing_weights = captured_routing[0]  # [batch, seq_len, num_experts] or similar

        # Get model config for top-k
        config = model.config
        top_k = getattr(config, "num_experts_per_tok",
                       getattr(config, "top_k",
                              getattr(config, "num_experts_per_token", 2)))

        for i, token_id in enumerate(input_ids):
            token_str = tokenizer.decode([token_id])

            # Extract this token's routing
            if len(routing_weights.shape) == 3:
                token_routing = routing_weights[0, i]  # [num_experts]
            elif len(routing_weights.shape) == 2:
                token_routing = routing_weights[i]
            else:
                token_routing = routing_weights

            # Get top-k experts
            if torch.is_tensor(token_routing):
                values, indices = torch.topk(token_routing, min(top_k, len(token_routing)))
                experts = indices.tolist()
                weights = torch.softmax(values, dim=0).tolist()
            else:
                experts = list(range(top_k))
                weights = [1.0 / top_k] * top_k

            activation = ExpertActivation(
                token_idx=i,
                token_str=token_str,
                experts_selected=experts,
                expert_weights=weights,
                router_logits=token_routing.numpy() if return_logits and torch.is_tensor(token_routing) else None,
            )
            activations.append(activation)
    else:
        # Fallback: create placeholder activations
        for i, token_id in enumerate(input_ids):
            activations.append(ExpertActivation(
                token_idx=i,
                token_str=tokenizer.decode([token_id]),
                experts_selected=[],
                expert_weights=[],
            ))

    return activations


def analyze_expert_specialization(
    model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
    prompts: Dict[str, List[str]],
) -> Dict[str, Dict[int, float]]:
    """
    Analyze which experts specialize in different domains.

    Args:
        model: Loaded MoE model
        tokenizer: Corresponding tokenizer
        prompts: Dictionary mapping domain names to lists of example prompts
            Example: {"code": ["def foo():", "class Bar:"],
                      "math": ["2 + 2 =", "integral of x"]}

    Returns:
        Dictionary mapping domain to expert activation frequencies

    Example:
        >>> prompts = {
        ...     "code": ["def fibonacci(n):", "import numpy as np"],
        ...     "math": ["The derivative of x^2", "Solve for x:"],
        ... }
        >>> specialization = analyze_expert_specialization(model, tokenizer, prompts)
        >>> print("Code experts:", sorted(specialization["code"].items(),
        ...                               key=lambda x: -x[1])[:5])
        Code experts: [(12, 0.45), (23, 0.38), (7, 0.22), ...]
    """
    domain_expert_counts = {}

    for domain, domain_prompts in prompts.items():
        expert_counts = defaultdict(int)
        total_tokens = 0

        for prompt in domain_prompts:
            activations = extract_expert_activations(model, tokenizer, prompt)

            for act in activations:
                for expert_idx in act.experts_selected:
                    expert_counts[expert_idx] += 1
                total_tokens += 1

        # Normalize to frequencies
        if total_tokens > 0:
            domain_expert_counts[domain] = {
                expert: count / total_tokens
                for expert, count in expert_counts.items()
            }
        else:
            domain_expert_counts[domain] = {}

    return domain_expert_counts


def visualize_expert_distribution(
    activations: List[ExpertActivation],
    num_experts: int = 64,
) -> np.ndarray:
    """
    Create a visualization matrix of expert activations across tokens.

    Args:
        activations: List of ExpertActivation from extract_expert_activations
        num_experts: Total number of experts in the model

    Returns:
        2D array of shape [num_tokens, num_experts] showing activation weights

    Example:
        >>> activations = extract_expert_activations(model, tokenizer, text)
        >>> matrix = visualize_expert_distribution(activations, num_experts=64)
        >>> import matplotlib.pyplot as plt
        >>> plt.imshow(matrix, aspect='auto', cmap='viridis')
        >>> plt.xlabel('Expert Index')
        >>> plt.ylabel('Token Position')
        >>> plt.colorbar(label='Activation Weight')
    """
    num_tokens = len(activations)
    matrix = np.zeros((num_tokens, num_experts))

    for i, act in enumerate(activations):
        for expert_idx, weight in zip(act.experts_selected, act.expert_weights):
            if expert_idx < num_experts:
                matrix[i, expert_idx] = weight

    return matrix


def calculate_load_balance(
    activations: List[ExpertActivation],
    num_experts: int,
) -> Dict[str, float]:
    """
    Calculate load balancing metrics for expert utilization.

    Imbalanced load is a common problem in MoE - some experts may be
    overused while others are neglected. This function measures balance.

    Args:
        activations: List of ExpertActivation objects
        num_experts: Total number of experts

    Returns:
        Dictionary with load balance metrics:
        - expert_utilization: Dict mapping expert_id to usage count
        - balance_score: 0 to 1, higher is more balanced
        - max_load: Highest expert load
        - min_load: Lowest expert load
        - std_load: Standard deviation of loads

    Example:
        >>> balance = calculate_load_balance(activations, num_experts=64)
        >>> print(f"Balance score: {balance['balance_score']:.3f}")
        >>> print(f"Most used expert: {max(balance['expert_utilization'], key=balance['expert_utilization'].get)}")
    """
    expert_counts = defaultdict(int)
    total_activations = 0

    for act in activations:
        for expert_idx in act.experts_selected:
            expert_counts[expert_idx] += 1
            total_activations += 1

    # Ensure all experts are represented
    for i in range(num_experts):
        if i not in expert_counts:
            expert_counts[i] = 0

    counts = list(expert_counts.values())
    mean_load = np.mean(counts)
    std_load = np.std(counts)

    # Balance score: 1 - (std / mean), capped at 0
    if mean_load > 0:
        balance_score = max(0, 1 - (std_load / mean_load))
    else:
        balance_score = 0

    return {
        "expert_utilization": dict(expert_counts),
        "balance_score": balance_score,
        "max_load": max(counts),
        "min_load": min(counts),
        "mean_load": mean_load,
        "std_load": std_load,
        "total_activations": total_activations,
    }


def compute_moe_efficiency(model_info: MoEModelInfo) -> Dict[str, float]:
    """
    Compute efficiency metrics for an MoE model.

    Args:
        model_info: MoEModelInfo dataclass

    Returns:
        Dictionary with efficiency metrics

    Example:
        >>> info = MOE_CONFIGS["deepseek-moe-16b"]
        >>> efficiency = compute_moe_efficiency(info)
        >>> print(f"Sparsity ratio: {efficiency['sparsity_ratio']:.2f}x")
        Sparsity ratio: 6.40x
    """
    sparsity_ratio = model_info.total_params / model_info.active_params

    return {
        "sparsity_ratio": sparsity_ratio,
        "compute_savings_percent": (1 - 1/sparsity_ratio) * 100,
        "memory_per_active_param_gb": model_info.memory_gb / (model_info.active_params / 1e9),
        "params_per_expert_millions": (model_info.total_params / model_info.num_experts) / 1e6,
    }


if __name__ == "__main__":
    print("MoE Utilities Module")
    print("=" * 50)

    # Print known configurations
    print("\nKnown MoE Configurations:")
    for name, config in MOE_CONFIGS.items():
        print(f"\n{name}:")
        print(f"  Total params: {config.total_params / 1e9:.1f}B")
        print(f"  Active params: {config.active_params / 1e9:.1f}B")
        print(f"  Sparsity: {config.total_params / config.active_params:.1f}x")
        print(f"  Experts: {config.num_experts} (top-{config.experts_per_token} per token)")
        print(f"  Memory (FP16): {config.memory_gb:.1f}GB")

    # Check CUDA
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {total_mem:.1f}GB")
        print(f"Can fit Mixtral 8x7B: {'Yes' if total_mem > 95 else 'No'}")
