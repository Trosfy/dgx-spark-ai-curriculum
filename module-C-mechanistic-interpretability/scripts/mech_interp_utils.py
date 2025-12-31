"""
Mechanistic Interpretability Utilities
======================================

Production-quality utilities for mechanistic interpretability research on DGX Spark.

This module provides:
- Model loading with memory optimization
- Activation caching helpers
- Patching utilities
- Common interpretability operations

Author: Professor SPARK
Hardware: Optimized for DGX Spark (128GB unified memory)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Tuple, Callable, Union, Any
from dataclasses import dataclass
from functools import partial
import gc


@dataclass
class PatchingResult:
    """Container for activation patching results.

    Attributes:
        layer: Layer index where patching was performed
        component: Component type (resid, attn, mlp, etc.)
        head: Head index if applicable (None for non-attention components)
        effect: Normalized effect of the patch (0 = no effect, 1 = full effect)
        clean_prob: Probability on clean run
        patched_prob: Probability after patching
    """
    layer: int
    component: str
    head: Optional[int]
    effect: float
    clean_prob: float
    patched_prob: float


def clear_gpu_memory() -> Dict[str, float]:
    """Clear GPU memory and return memory statistics.

    Use this before loading large models or after intensive operations.

    Returns:
        Dictionary with memory statistics in GB

    Example:
        >>> stats = clear_gpu_memory()
        >>> print(f"Free memory: {stats['free_gb']:.2f} GB")
        Free memory: 120.45 GB
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        free = total - reserved

        return {
            "total_gb": total,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "free_gb": free
        }
    return {"total_gb": 0, "allocated_gb": 0, "reserved_gb": 0, "free_gb": 0}


def load_hooked_model(
    model_name: str = "gpt2-small",
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    center_unembed: bool = True,
    center_writing_weights: bool = True,
    fold_ln: bool = True
) -> Any:
    """Load a HookedTransformer model with recommended settings.

    This function loads models with interpretability-friendly settings:
    - Centered unembedding for cleaner logit analysis
    - Centered writing weights for cleaner residual stream
    - Folded layer norms for simpler activation analysis

    Args:
        model_name: Name of the model to load (e.g., "gpt2-small", "gpt2-medium")
        device: Device to load model on ("cuda" or "cpu")
        dtype: Data type for model weights
        center_unembed: Whether to center the unembedding matrix
        center_writing_weights: Whether to center attention/MLP output weights
        fold_ln: Whether to fold layer norms into weights

    Returns:
        HookedTransformer model ready for interpretability research

    Example:
        >>> model = load_hooked_model("gpt2-small")
        >>> print(f"Model has {model.cfg.n_layers} layers")
        Model has 12 layers

    DGX Spark Note:
        With 128GB unified memory, you can load larger models:
        - gpt2-small (124M): ~0.5GB
        - gpt2-medium (355M): ~1.5GB
        - gpt2-large (774M): ~3GB
        - gpt2-xl (1.5B): ~6GB
        - pythia-2.8b: ~11GB
    """
    try:
        from transformer_lens import HookedTransformer
    except ImportError:
        raise ImportError(
            "TransformerLens not installed. Install with: pip install transformer-lens"
        )

    # Clear memory before loading
    clear_gpu_memory()

    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=dtype,
        center_unembed=center_unembed,
        center_writing_weights=center_writing_weights,
        fold_ln=fold_ln
    )

    return model


def get_activation_cache(
    model: Any,
    tokens: torch.Tensor,
    names_filter: Optional[Callable] = None
) -> Tuple[torch.Tensor, Any]:
    """Run model and cache activations with optional filtering.

    Args:
        model: HookedTransformer model
        tokens: Input token tensor [batch, seq]
        names_filter: Optional function to filter which activations to cache

    Returns:
        Tuple of (logits, activation_cache)

    Example:
        >>> tokens = model.to_tokens("Hello world")
        >>> logits, cache = get_activation_cache(model, tokens)
        >>> residual = cache["resid_post", 5]  # Layer 5 residual
        >>> print(residual.shape)
        torch.Size([1, 2, 768])
    """
    if names_filter is not None:
        logits, cache = model.run_with_cache(tokens, names_filter=names_filter)
    else:
        logits, cache = model.run_with_cache(tokens)

    return logits, cache


def compute_logit_diff(
    logits: torch.Tensor,
    correct_token: int,
    incorrect_token: int,
    position: int = -1
) -> float:
    """Compute logit difference between correct and incorrect answers.

    The logit difference is a key metric in interpretability research.
    It measures how much the model prefers the correct answer.

    Args:
        logits: Model output logits [batch, seq, vocab]
        correct_token: Token ID of correct answer
        incorrect_token: Token ID of incorrect answer
        position: Sequence position to evaluate (-1 for last)

    Returns:
        Logit difference (positive = prefers correct)

    Example:
        >>> # For "The capital of France is" -> "Paris" vs "London"
        >>> paris_id = model.to_single_token(" Paris")
        >>> london_id = model.to_single_token(" London")
        >>> diff = compute_logit_diff(logits, paris_id, london_id)
        >>> print(f"Logit diff: {diff:.2f}")
        Logit diff: 3.45
    """
    logits_at_pos = logits[0, position, :]
    correct_logit = logits_at_pos[correct_token].item()
    incorrect_logit = logits_at_pos[incorrect_token].item()

    return correct_logit - incorrect_logit


def get_top_predictions(
    model: Any,
    logits: torch.Tensor,
    k: int = 10,
    position: int = -1
) -> List[Tuple[str, float]]:
    """Get top-k token predictions with probabilities.

    Args:
        model: HookedTransformer model (for tokenizer)
        logits: Model output logits
        k: Number of top predictions to return
        position: Sequence position to evaluate

    Returns:
        List of (token_string, probability) tuples

    Example:
        >>> preds = get_top_predictions(model, logits, k=5)
        >>> for token, prob in preds:
        ...     print(f"{token!r}: {prob:.2%}")
        ' Paris': 45.23%
        ' Berlin': 12.45%
    """
    probs = torch.softmax(logits[0, position, :], dim=-1)
    top_probs, top_indices = torch.topk(probs, k)

    results = []
    for idx, prob in zip(top_indices, top_probs):
        token_str = model.tokenizer.decode(idx.item())
        results.append((token_str, prob.item()))

    return results


def create_patching_hook(
    source_cache: Any,
    hook_name: str,
    head_index: Optional[int] = None
) -> Callable:
    """Create a hook function for activation patching.

    Args:
        source_cache: Cache containing source activations
        hook_name: Name of the activation to patch
        head_index: If patching attention, which head to patch (None = all)

    Returns:
        Hook function for use with model.run_with_hooks

    Example:
        >>> _, corrupted_cache = model.run_with_cache(corrupted_tokens)
        >>> hook_fn = create_patching_hook(corrupted_cache, "blocks.5.hook_resid_post")
        >>> patched_logits = model.run_with_hooks(
        ...     clean_tokens,
        ...     fwd_hooks=[("blocks.5.hook_resid_post", hook_fn)]
        ... )
    """
    def patch_hook(activation: torch.Tensor, hook: Any) -> torch.Tensor:
        source_activation = source_cache[hook_name]

        if head_index is not None and len(activation.shape) == 4:
            # Patching specific attention head: [batch, head, seq, head_dim]
            activation[:, head_index, :, :] = source_activation[:, head_index, :, :]
        else:
            # Patch entire activation
            return source_activation

        return activation

    return patch_hook


def run_activation_patching(
    model: Any,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    answer_token: int,
    components: List[str] = ["resid_post", "attn_out", "mlp_out"]
) -> List[PatchingResult]:
    """Run activation patching experiment across model components.

    This patches activations from corrupted run into clean run,
    measuring the effect on predicting the answer token.

    Args:
        model: HookedTransformer model
        clean_tokens: Tokens for clean prompt
        corrupted_tokens: Tokens for corrupted prompt
        answer_token: Token ID we're measuring
        components: Which activation types to patch

    Returns:
        List of PatchingResult for each layer and component

    Example:
        >>> clean = model.to_tokens("John gave a book to Mary")
        >>> corrupted = model.to_tokens("Mary gave a book to John")
        >>> answer = model.to_single_token(" Mary")
        >>> results = run_activation_patching(model, clean, corrupted, answer)
        >>> for r in results:
        ...     if r.effect > 0.1:
        ...         print(f"Layer {r.layer} {r.component}: {r.effect:.2%}")
    """
    # Get baseline probabilities
    clean_logits = model(clean_tokens)
    clean_prob = torch.softmax(clean_logits[0, -1, :], dim=-1)[answer_token].item()

    corrupted_logits = model(corrupted_tokens)
    corrupted_prob = torch.softmax(corrupted_logits[0, -1, :], dim=-1)[answer_token].item()

    # Cache corrupted activations
    _, corrupted_cache = model.run_with_cache(corrupted_tokens)

    results = []

    for layer in range(model.cfg.n_layers):
        for component in components:
            hook_name = f"blocks.{layer}.hook_{component}"

            # Create patching hook
            def patch_fn(activation, hook, layer=layer, component=component):
                return corrupted_cache[f"blocks.{layer}.hook_{component}"]

            # Run with patch
            patched_logits = model.run_with_hooks(
                clean_tokens,
                fwd_hooks=[(hook_name, patch_fn)]
            )
            patched_prob = torch.softmax(patched_logits[0, -1, :], dim=-1)[answer_token].item()

            # Calculate normalized effect
            # Effect of 1 means patching fully recovers corrupted behavior
            effect = (clean_prob - patched_prob) / (clean_prob - corrupted_prob + 1e-10)

            results.append(PatchingResult(
                layer=layer,
                component=component,
                head=None,
                effect=effect,
                clean_prob=clean_prob,
                patched_prob=patched_prob
            ))

    return results


def run_head_patching(
    model: Any,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    answer_token: int
) -> np.ndarray:
    """Run activation patching on individual attention heads.

    Returns a matrix of effects where result[layer, head] shows
    how much patching that head affects the output.

    Args:
        model: HookedTransformer model
        clean_tokens: Tokens for clean prompt
        corrupted_tokens: Tokens for corrupted prompt
        answer_token: Token ID we're measuring

    Returns:
        Array of shape [n_layers, n_heads] with patching effects

    Example:
        >>> effects = run_head_patching(model, clean, corrupted, answer)
        >>> important_heads = np.argwhere(effects > 0.1)
        >>> print(f"Found {len(important_heads)} important heads")
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Baseline
    clean_logits = model(clean_tokens)
    clean_prob = torch.softmax(clean_logits[0, -1, :], dim=-1)[answer_token].item()

    corrupted_logits = model(corrupted_tokens)
    corrupted_prob = torch.softmax(corrupted_logits[0, -1, :], dim=-1)[answer_token].item()

    _, corrupted_cache = model.run_with_cache(corrupted_tokens)

    effects = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        for head in range(n_heads):
            def patch_head(activation, hook, layer=layer, head=head):
                # activation: [batch, seq, n_heads, head_dim]
                corrupted_act = corrupted_cache[f"blocks.{layer}.attn.hook_z"]
                activation[:, :, head, :] = corrupted_act[:, :, head, :]
                return activation

            patched_logits = model.run_with_hooks(
                clean_tokens,
                fwd_hooks=[(f"blocks.{layer}.attn.hook_z", patch_head)]
            )
            patched_prob = torch.softmax(patched_logits[0, -1, :], dim=-1)[answer_token].item()

            effects[layer, head] = (clean_prob - patched_prob) / (clean_prob - corrupted_prob + 1e-10)

    return effects


def compute_attention_to_position(
    cache: Any,
    query_position: int,
    key_position: int
) -> np.ndarray:
    """Get attention weights from query to key position across all heads.

    Args:
        cache: Activation cache from model.run_with_cache
        query_position: Position doing the attending
        key_position: Position being attended to

    Returns:
        Array of shape [n_layers, n_heads] with attention weights

    Example:
        >>> # How much does the last token attend to position 5?
        >>> attn_weights = compute_attention_to_position(cache, -1, 5)
        >>> strongest = np.unravel_index(attn_weights.argmax(), attn_weights.shape)
        >>> print(f"Strongest attention at layer {strongest[0]}, head {strongest[1]}")
    """
    n_layers = len([k for k in cache.keys() if "pattern" in str(k)])
    pattern = cache["pattern", 0]
    n_heads = pattern.shape[1]

    result = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        pattern = cache["pattern", layer][0]  # [n_heads, seq, seq]
        result[layer, :] = pattern[:, query_position, key_position].cpu().numpy()

    return result


def get_residual_stream_norms(
    cache: Any,
    position: int = -1
) -> np.ndarray:
    """Get L2 norms of residual stream at each layer.

    Useful for understanding how information accumulates.

    Args:
        cache: Activation cache
        position: Sequence position to analyze

    Returns:
        Array of norms at each layer

    Example:
        >>> norms = get_residual_stream_norms(cache)
        >>> print(f"Norm increases from {norms[0]:.1f} to {norms[-1]:.1f}")
    """
    norms = []
    layer = 0
    while True:
        try:
            resid = cache["resid_post", layer][0, position, :]
            norms.append(resid.norm().item())
            layer += 1
        except KeyError:
            break

    return np.array(norms)


class ActivationSteerer:
    """Tool for steering model activations at runtime.

    This allows adding/subtracting directions from activations
    to modify model behavior.

    Example:
        >>> steerer = ActivationSteerer(model)
        >>> # Find a "positive sentiment" direction
        >>> direction = compute_sentiment_direction(model)
        >>> steerer.set_steering_vector(5, direction, strength=2.0)
        >>> output = steerer.generate("This movie was")
        >>> # Output will be more positive
    """

    def __init__(self, model: Any):
        """Initialize the steerer.

        Args:
            model: HookedTransformer model
        """
        self.model = model
        self.steering_hooks = {}

    def set_steering_vector(
        self,
        layer: int,
        direction: torch.Tensor,
        strength: float = 1.0,
        position: Optional[int] = None
    ) -> None:
        """Set a steering vector for a layer.

        Args:
            layer: Layer to apply steering
            direction: Direction vector (should match d_model)
            strength: How much to steer (can be negative)
            position: Which position to steer (None = all)
        """
        direction = direction.to(self.model.cfg.device)
        direction = direction / direction.norm() * strength

        def steering_hook(activation, hook):
            if position is None:
                activation = activation + direction
            else:
                activation[:, position, :] = activation[:, position, :] + direction
            return activation

        self.steering_hooks[layer] = (f"blocks.{layer}.hook_resid_post", steering_hook)

    def clear_steering(self) -> None:
        """Remove all steering vectors."""
        self.steering_hooks = {}

    def run_with_steering(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run model with steering applied.

        Args:
            tokens: Input tokens

        Returns:
            Logits with steering applied
        """
        hooks = list(self.steering_hooks.values())
        return self.model.run_with_hooks(tokens, fwd_hooks=hooks)


def create_ioi_dataset(
    n_samples: int = 100,
    names: Optional[List[str]] = None
) -> List[Dict[str, str]]:
    """Create Indirect Object Identification dataset.

    IOI is a key benchmark for circuit analysis. The task is:
    "When [Name1] and [Name2] went to [Place], [Name1] gave [Object] to [Name2]"
    The model should predict Name2 at the end.

    Args:
        n_samples: Number of samples to generate
        names: Optional list of names to use

    Returns:
        List of dicts with 'clean', 'corrupted', 'answer' keys

    Example:
        >>> dataset = create_ioi_dataset(10)
        >>> print(dataset[0])
        {
            'clean': 'John and Mary went to the store. John gave a book to',
            'corrupted': 'John and Mary went to the store. Mary gave a book to',
            'answer': ' Mary',
            'wrong_answer': ' John'
        }
    """
    if names is None:
        names = ["John", "Mary", "Alice", "Bob", "James", "Emma",
                 "Michael", "Sarah", "David", "Lisa"]

    places = ["store", "park", "beach", "library", "cafe", "museum"]
    objects = ["book", "ball", "key", "letter", "gift", "phone"]

    import random

    dataset = []
    for _ in range(n_samples):
        name1, name2 = random.sample(names, 2)
        place = random.choice(places)
        obj = random.choice(objects)

        clean = f"{name1} and {name2} went to the {place}. {name1} gave a {obj} to"
        corrupted = f"{name1} and {name2} went to the {place}. {name2} gave a {obj} to"

        dataset.append({
            'clean': clean,
            'corrupted': corrupted,
            'answer': f" {name2}",
            'wrong_answer': f" {name1}"
        })

    return dataset


def create_induction_dataset(
    n_samples: int = 100,
    seq_length: int = 50,
    vocab_start: int = 1000,
    vocab_end: int = 10000
) -> List[torch.Tensor]:
    """Create dataset for testing induction heads.

    Induction heads complete patterns: [A][B]...[A] -> [B]
    We create sequences with repeated random tokens.

    Args:
        n_samples: Number of sequences
        seq_length: Half the total sequence length
        vocab_start: Start of vocab range (avoid special tokens)
        vocab_end: End of vocab range

    Returns:
        List of token tensors [1, seq_length * 2]

    Example:
        >>> dataset = create_induction_dataset(10, seq_length=30)
        >>> tokens = dataset[0]
        >>> # tokens[30:60] == tokens[0:30] (repeated)
    """
    dataset = []
    for _ in range(n_samples):
        first_half = torch.randint(vocab_start, vocab_end, (1, seq_length))
        full_seq = torch.cat([first_half, first_half], dim=1)
        dataset.append(full_seq)

    return dataset


if __name__ == "__main__":
    # Quick test
    print("Mechanistic Interpretability Utils loaded successfully!")
    print(f"GPU Memory: {clear_gpu_memory()}")
