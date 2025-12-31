"""
Circuit Analysis Utilities for Mechanistic Interpretability
============================================================

Tools for discovering and analyzing circuits in transformer models.

This module provides:
- Circuit discovery helpers
- Composition analysis (Q, K, V circuits)
- Path patching utilities
- Automated circuit finding

Author: Professor SPARK
Hardware: Optimized for DGX Spark (128GB unified memory)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Tuple, Callable, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import warnings


@dataclass
class CircuitComponent:
    """Represents a component in a circuit.

    Attributes:
        layer: Layer index
        component_type: "attention", "mlp", or "resid"
        head: Head index for attention (None otherwise)
        importance: How important this component is (0-1)
        role: Functional role (e.g., "name_mover", "backup", "inhibition")
    """
    layer: int
    component_type: str
    head: Optional[int] = None
    importance: float = 0.0
    role: str = "unknown"

    @property
    def name(self) -> str:
        """Generate component name."""
        if self.component_type == "attention" and self.head is not None:
            return f"L{self.layer}H{self.head}"
        elif self.component_type == "mlp":
            return f"L{self.layer}_MLP"
        else:
            return f"L{self.layer}_{self.component_type}"


@dataclass
class Circuit:
    """Represents a discovered circuit.

    Attributes:
        name: Circuit name (e.g., "IOI", "induction")
        components: List of CircuitComponent objects
        connections: List of (source_name, target_name, strength) tuples
        performance: How well this circuit explains the behavior (0-1)
        description: Human-readable description
    """
    name: str
    components: List[CircuitComponent] = field(default_factory=list)
    connections: List[Tuple[str, str, float]] = field(default_factory=list)
    performance: float = 0.0
    description: str = ""

    def add_component(self, component: CircuitComponent) -> None:
        """Add a component to the circuit."""
        self.components.append(component)

    def add_connection(self, source: str, target: str, strength: float) -> None:
        """Add a connection between components."""
        self.connections.append((source, target, strength))

    def get_component(self, name: str) -> Optional[CircuitComponent]:
        """Get component by name."""
        for comp in self.components:
            if comp.name == name:
                return comp
        return None

    def summary(self) -> str:
        """Generate circuit summary."""
        lines = [
            f"Circuit: {self.name}",
            f"Performance: {self.performance:.1%}",
            f"Components: {len(self.components)}",
            f"Connections: {len(self.connections)}",
            "",
            "Key Components:"
        ]
        for comp in sorted(self.components, key=lambda x: -x.importance)[:5]:
            lines.append(f"  {comp.name}: {comp.role} (importance={comp.importance:.2f})")

        return "\n".join(lines)


def find_induction_heads(
    model: Any,
    seq_length: int = 50,
    threshold: float = 0.1,
    n_samples: int = 10
) -> List[CircuitComponent]:
    """Find induction heads in a model.

    Induction heads complete patterns: [A][B]...[A] -> [B]
    They are identified by high attention to (position - sequence_half + 1).

    Args:
        model: HookedTransformer model
        seq_length: Half the sequence length for pattern
        threshold: Minimum score to be considered an induction head
        n_samples: Number of random sequences to average over

    Returns:
        List of CircuitComponent objects for identified induction heads

    Example:
        >>> induction_heads = find_induction_heads(model, threshold=0.2)
        >>> for head in induction_heads:
        ...     print(f"{head.name}: score={head.importance:.2f}")
    """
    device = model.cfg.device
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    scores = np.zeros((n_layers, n_heads))

    for _ in range(n_samples):
        # Create repeated sequence
        first_half = torch.randint(1000, 10000, (1, seq_length), device=device)
        repeated = torch.cat([first_half, first_half], dim=1)

        # Get attention patterns
        _, cache = model.run_with_cache(repeated)

        for layer in range(n_layers):
            pattern = cache["pattern", layer][0]  # [heads, seq, seq]

            for head in range(n_heads):
                # Measure attention to "position after previous occurrence"
                # For position i in second half, this is position (i - seq_length + 1)
                score = 0
                count = 0
                for i in range(seq_length, 2 * seq_length):
                    target = i - seq_length + 1
                    if target > 0:
                        score += pattern[head, i, target].item()
                        count += 1
                scores[layer, head] += score / count if count > 0 else 0

    scores /= n_samples

    # Create components for heads above threshold
    components = []
    for layer in range(n_layers):
        for head in range(n_heads):
            if scores[layer, head] > threshold:
                comp = CircuitComponent(
                    layer=layer,
                    component_type="attention",
                    head=head,
                    importance=scores[layer, head],
                    role="induction"
                )
                components.append(comp)

    return sorted(components, key=lambda x: -x.importance)


def find_previous_token_heads(
    model: Any,
    seq_length: int = 20,
    threshold: float = 0.3,
    n_samples: int = 10
) -> List[CircuitComponent]:
    """Find previous token heads (attend to position - 1).

    Previous token heads are essential for induction circuits.
    They move information about token[i] to position[i+1].

    Args:
        model: HookedTransformer model
        seq_length: Sequence length to test
        threshold: Minimum score
        n_samples: Number of samples

    Returns:
        List of CircuitComponent objects

    Example:
        >>> prev_token_heads = find_previous_token_heads(model)
        >>> print(f"Found {len(prev_token_heads)} previous token heads")
    """
    device = model.cfg.device
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    scores = np.zeros((n_layers, n_heads))

    for _ in range(n_samples):
        tokens = torch.randint(1000, 10000, (1, seq_length), device=device)
        _, cache = model.run_with_cache(tokens)

        for layer in range(n_layers):
            pattern = cache["pattern", layer][0]  # [heads, seq, seq]

            for head in range(n_heads):
                # Measure attention to previous position
                score = 0
                for i in range(1, seq_length):
                    score += pattern[head, i, i - 1].item()
                scores[layer, head] += score / (seq_length - 1)

    scores /= n_samples

    components = []
    for layer in range(n_layers):
        for head in range(n_heads):
            if scores[layer, head] > threshold:
                comp = CircuitComponent(
                    layer=layer,
                    component_type="attention",
                    head=head,
                    importance=scores[layer, head],
                    role="previous_token"
                )
                components.append(comp)

    return sorted(components, key=lambda x: -x.importance)


def analyze_attention_composition(
    model: Any,
    cache: Any,
    source_layer: int,
    source_head: int,
    target_layer: int,
    target_head: int
) -> Dict[str, float]:
    """Analyze how two attention heads compose.

    This checks if the output of source_head gets read by target_head
    through Q-composition, K-composition, or V-composition.

    Args:
        model: HookedTransformer model
        cache: Activation cache
        source_layer: Layer of source head
        source_head: Source head index
        target_layer: Layer of target head (must be > source_layer)
        target_head: Target head index

    Returns:
        Dict with 'q_composition', 'k_composition', 'v_composition' scores

    Example:
        >>> _, cache = model.run_with_cache(tokens)
        >>> comp = analyze_attention_composition(model, cache, 0, 3, 5, 9)
        >>> print(f"Q-composition: {comp['q_composition']:.3f}")
    """
    if target_layer <= source_layer:
        raise ValueError("Target layer must be after source layer")

    d_head = model.cfg.d_head
    d_model = model.cfg.d_model

    # Get OV circuit output direction from source head
    W_O = model.W_O[source_layer, source_head]  # [d_head, d_model]
    W_V = model.W_V[source_layer, source_head]  # [d_model, d_head]
    OV_circuit = W_V @ W_O  # [d_model, d_model]

    # Get QK matrices from target head
    W_Q = model.W_Q[target_layer, target_head]  # [d_model, d_head]
    W_K = model.W_K[target_layer, target_head]  # [d_model, d_head]
    W_V_target = model.W_V[target_layer, target_head]

    # Q-composition: does OV output get used by target's queries?
    # Measure: norm(W_Q.T @ OV_circuit) / norm(W_Q.T)
    q_composed = (W_Q.T @ OV_circuit).norm().item()
    q_baseline = W_Q.T.norm().item()
    q_composition = q_composed / (q_baseline * OV_circuit.norm().item() + 1e-10)

    # K-composition: does OV output get used by target's keys?
    k_composed = (W_K.T @ OV_circuit).norm().item()
    k_baseline = W_K.T.norm().item()
    k_composition = k_composed / (k_baseline * OV_circuit.norm().item() + 1e-10)

    # V-composition: does OV output get used by target's values?
    v_composed = (W_V_target.T @ OV_circuit).norm().item()
    v_baseline = W_V_target.T.norm().item()
    v_composition = v_composed / (v_baseline * OV_circuit.norm().item() + 1e-10)

    return {
        'q_composition': q_composition,
        'k_composition': k_composition,
        'v_composition': v_composition,
        'total': q_composition + k_composition + v_composition
    }


def path_patching(
    model: Any,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    answer_token: int,
    sender_layer: int,
    sender_head: int,
    receiver_layer: int,
    receiver_head: int,
    receiver_input: str = "q"
) -> float:
    """Perform path patching between two attention heads.

    Path patching isolates the effect of a specific path through the model.
    It patches only the connection from sender to receiver, not the full
    sender output.

    Args:
        model: HookedTransformer model
        clean_tokens: Clean input tokens
        corrupted_tokens: Corrupted input tokens
        answer_token: Token we're measuring
        sender_layer: Layer of sender head
        sender_head: Sender head index
        receiver_layer: Layer of receiver head
        receiver_head: Receiver head index
        receiver_input: Which input to patch ("q", "k", or "v")

    Returns:
        Path effect (normalized)

    Example:
        >>> effect = path_patching(
        ...     model, clean, corrupted, answer,
        ...     sender_layer=0, sender_head=3,
        ...     receiver_layer=5, receiver_head=9,
        ...     receiver_input="k"
        ... )
        >>> print(f"Path L0H3 -> L5H9 via keys: {effect:.2%}")
    """
    if receiver_layer <= sender_layer:
        raise ValueError("Receiver must be in a later layer than sender")

    # Baselines
    clean_logits = model(clean_tokens)
    clean_prob = torch.softmax(clean_logits[0, -1, :], dim=-1)[answer_token].item()

    corrupted_logits = model(corrupted_tokens)
    corrupted_prob = torch.softmax(corrupted_logits[0, -1, :], dim=-1)[answer_token].item()

    # Get corrupted cache
    _, corrupted_cache = model.run_with_cache(corrupted_tokens)

    # Hook to patch specific path
    hook_name = f"blocks.{receiver_layer}.attn.hook_{receiver_input}_input"

    def path_patch_hook(activation, hook):
        # Get the contribution from sender head to this position
        # This is complex - we need to compute what the sender head wrote
        # and how it affects this specific input

        # Simplified: patch the residual stream component from sender
        sender_output = corrupted_cache[f"blocks.{sender_layer}.attn.hook_result"]
        sender_head_output = sender_output[:, :, sender_head, :]  # [batch, seq, d_head]

        # The sender's contribution to residual is W_O @ head_output
        W_O = model.W_O[sender_layer, sender_head]
        sender_contribution = sender_head_output @ W_O  # [batch, seq, d_model]

        # Patch: replace this contribution in the input to receiver
        # This is an approximation - full path patching is more complex
        return activation + sender_contribution

    # Run with hook
    patched_logits = model.run_with_hooks(
        clean_tokens,
        fwd_hooks=[(hook_name, path_patch_hook)]
    )
    patched_prob = torch.softmax(patched_logits[0, -1, :], dim=-1)[answer_token].item()

    effect = (clean_prob - patched_prob) / (clean_prob - corrupted_prob + 1e-10)
    return effect


def find_ioi_circuit(
    model: Any,
    n_samples: int = 50,
    effect_threshold: float = 0.02
) -> Circuit:
    """Automatically discover the IOI (Indirect Object Identification) circuit.

    This replicates the analysis from "Interpretability in the Wild"
    to find heads responsible for the IOI task.

    Args:
        model: HookedTransformer model
        n_samples: Number of IOI examples to test
        effect_threshold: Minimum effect to include component

    Returns:
        Circuit object with discovered components

    Example:
        >>> circuit = find_ioi_circuit(model, n_samples=100)
        >>> print(circuit.summary())
    """
    from .mech_interp_utils import create_ioi_dataset, run_head_patching

    # Create dataset
    dataset = create_ioi_dataset(n_samples)

    # Aggregate head effects
    all_effects = np.zeros((model.cfg.n_layers, model.cfg.n_heads))

    for sample in dataset:
        clean_tokens = model.to_tokens(sample['clean'])
        corrupted_tokens = model.to_tokens(sample['corrupted'])
        answer_token = model.to_single_token(sample['answer'])

        effects = run_head_patching(model, clean_tokens, corrupted_tokens, answer_token)
        all_effects += effects

    all_effects /= n_samples

    # Create circuit
    circuit = Circuit(
        name="IOI",
        description="Indirect Object Identification circuit"
    )

    # Add significant heads
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            if abs(all_effects[layer, head]) > effect_threshold:
                # Classify role based on effect direction and layer
                if all_effects[layer, head] > 0:
                    if layer < model.cfg.n_layers // 3:
                        role = "name_mover_early"
                    else:
                        role = "name_mover"
                else:
                    role = "negative_head"

                comp = CircuitComponent(
                    layer=layer,
                    component_type="attention",
                    head=head,
                    importance=abs(all_effects[layer, head]),
                    role=role
                )
                circuit.add_component(comp)

    # Estimate circuit performance (how much of behavior is explained)
    circuit.performance = min(1.0, sum(c.importance for c in circuit.components))

    return circuit


def ablate_component(
    model: Any,
    tokens: torch.Tensor,
    layer: int,
    component_type: str,
    head: Optional[int] = None,
    method: str = "zero"
) -> torch.Tensor:
    """Ablate a component and return modified logits.

    Args:
        model: HookedTransformer model
        tokens: Input tokens
        layer: Layer to ablate
        component_type: "attention" or "mlp"
        head: Head index if ablating attention
        method: "zero" or "mean"

    Returns:
        Logits with ablation applied

    Example:
        >>> ablated_logits = ablate_component(model, tokens, layer=5,
        ...                                   component_type="attention", head=3)
        >>> # Compare predictions
    """
    if component_type == "attention":
        if head is None:
            hook_name = f"blocks.{layer}.attn.hook_result"
        else:
            hook_name = f"blocks.{layer}.attn.hook_z"
    else:
        hook_name = f"blocks.{layer}.hook_mlp_out"

    def ablation_hook(activation, hook):
        if method == "zero":
            if head is not None and "hook_z" in hook.name:
                activation[:, :, head, :] = 0
            else:
                activation = torch.zeros_like(activation)
        elif method == "mean":
            if head is not None and "hook_z" in hook.name:
                activation[:, :, head, :] = activation[:, :, head, :].mean()
            else:
                activation = activation.mean(dim=-1, keepdim=True).expand_as(activation)
        return activation

    return model.run_with_hooks(tokens, fwd_hooks=[(hook_name, ablation_hook)])


def compute_direct_effect(
    model: Any,
    cache: Any,
    layer: int,
    head: int,
    position: int = -1
) -> torch.Tensor:
    """Compute direct effect of an attention head on logits.

    The direct effect bypasses all later layers and goes
    straight to the unembedding.

    Args:
        model: HookedTransformer model
        cache: Activation cache
        layer: Layer index
        head: Head index
        position: Sequence position

    Returns:
        Logit contribution from this head [vocab_size]

    Example:
        >>> _, cache = model.run_with_cache(tokens)
        >>> direct_logits = compute_direct_effect(model, cache, layer=5, head=9)
        >>> top_tokens = direct_logits.topk(5)
    """
    # Get head output
    head_output = cache[f"blocks.{layer}.attn.hook_z"][0, position, head, :]  # [d_head]

    # Apply W_O to get contribution to residual stream
    W_O = model.W_O[layer, head]  # [d_head, d_model]
    residual_contribution = head_output @ W_O  # [d_model]

    # Apply final layer norm
    if hasattr(model, 'ln_final'):
        # Need to normalize with the actual residual stream norm
        residual_contribution = model.ln_final(residual_contribution)

    # Apply unembedding
    W_U = model.W_U  # [d_model, vocab]
    logit_contribution = residual_contribution @ W_U  # [vocab]

    return logit_contribution


def trace_token_through_model(
    model: Any,
    cache: Any,
    source_position: int,
    target_position: int
) -> Dict[str, np.ndarray]:
    """Trace how information flows from one position to another.

    Args:
        model: HookedTransformer model
        cache: Activation cache
        source_position: Position we're tracking from
        target_position: Position we're tracking to

    Returns:
        Dict with attention flow per layer/head

    Example:
        >>> # How does "John" info reach the final prediction?
        >>> flow = trace_token_through_model(model, cache, source_position=0, target_position=-1)
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Attention from target to source at each layer
    direct_attention = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        pattern = cache["pattern", layer][0]  # [heads, seq, seq]
        direct_attention[layer, :] = pattern[:, target_position, source_position].cpu().numpy()

    return {
        'direct_attention': direct_attention,
        'total_by_layer': direct_attention.sum(axis=1),
        'total_by_head': direct_attention.sum(axis=0)
    }


class CircuitFinder:
    """Automated circuit discovery tool.

    This class provides methods to systematically find circuits
    for specific behaviors.

    Example:
        >>> finder = CircuitFinder(model)
        >>> circuit = finder.find_circuit_for_task(
        ...     clean_prompts=["The cat sat on the"],
        ...     target_tokens=[" mat"],
        ...     description="Simple next-word prediction"
        ... )
    """

    def __init__(self, model: Any, verbose: bool = True):
        """Initialize circuit finder.

        Args:
            model: HookedTransformer model
            verbose: Whether to print progress
        """
        self.model = model
        self.verbose = verbose

    def _log(self, message: str) -> None:
        """Log message if verbose."""
        if self.verbose:
            print(message)

    def find_circuit_for_task(
        self,
        clean_prompts: List[str],
        corrupted_prompts: Optional[List[str]] = None,
        target_tokens: Optional[List[str]] = None,
        description: str = "",
        threshold: float = 0.05
    ) -> Circuit:
        """Find circuit for a given task.

        Args:
            clean_prompts: Prompts that demonstrate the behavior
            corrupted_prompts: Optional corrupted versions
            target_tokens: Expected completions
            description: Task description
            threshold: Component importance threshold

        Returns:
            Discovered Circuit
        """
        circuit = Circuit(name="custom", description=description)

        self._log("Finding important components...")

        # If no corrupted prompts, use random tokens
        if corrupted_prompts is None:
            corrupted_prompts = clean_prompts  # Placeholder

        # If no target tokens, use top prediction
        if target_tokens is None:
            target_tokens = []
            for prompt in clean_prompts:
                tokens = self.model.to_tokens(prompt)
                logits = self.model(tokens)
                top_token = logits[0, -1, :].argmax()
                target_tokens.append(self.model.tokenizer.decode(top_token.item()))

        # Run patching analysis
        from .mech_interp_utils import run_head_patching

        all_effects = np.zeros((self.model.cfg.n_layers, self.model.cfg.n_heads))

        for clean, corrupted, target in zip(clean_prompts, corrupted_prompts, target_tokens):
            clean_tokens = self.model.to_tokens(clean)
            corrupted_tokens = self.model.to_tokens(corrupted)
            target_token = self.model.to_single_token(target)

            effects = run_head_patching(
                self.model, clean_tokens, corrupted_tokens, target_token
            )
            all_effects += np.abs(effects)

        all_effects /= len(clean_prompts)

        # Add significant components
        for layer in range(self.model.cfg.n_layers):
            for head in range(self.model.cfg.n_heads):
                if all_effects[layer, head] > threshold:
                    comp = CircuitComponent(
                        layer=layer,
                        component_type="attention",
                        head=head,
                        importance=all_effects[layer, head]
                    )
                    circuit.add_component(comp)

        self._log(f"Found {len(circuit.components)} significant heads")

        # Estimate performance
        circuit.performance = min(1.0, sum(c.importance for c in circuit.components))

        return circuit


if __name__ == "__main__":
    print("Circuit Analysis utilities loaded successfully!")
