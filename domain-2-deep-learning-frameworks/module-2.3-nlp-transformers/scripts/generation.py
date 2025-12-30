"""
Text Generation Strategies

This module provides various decoding strategies for autoregressive
text generation with language models.

Example usage:
    >>> from generation import top_p_sampling, beam_search
    >>> from transformers import GPT2LMHeadModel, GPT2Tokenizer
    >>>
    >>> model = GPT2LMHeadModel.from_pretrained("gpt2")
    >>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    >>>
    >>> text = top_p_sampling(
    ...     model, tokenizer, "Once upon a time",
    ...     max_length=50, p=0.9
    ... )

Author: DGX Spark AI Curriculum
License: MIT
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Callable
import warnings


def greedy_decode(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 50,
    device: str = 'cpu'
) -> str:
    """
    Generate text using greedy decoding.

    At each step, selects the token with highest probability.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Starting text
        max_length: Maximum tokens to generate
        device: Device to run on

    Returns:
        Generated text

    Example:
        >>> text = greedy_decode(model, tokenizer, "Hello", max_length=20)
    """
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids[0].tolist()
    eos_token_id = tokenizer.eos_token_id

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(torch.tensor([generated]).to(device))
            logits = outputs.logits[0, -1, :]

        next_token = logits.argmax().item()

        if next_token == eos_token_id:
            break

        generated.append(next_token)

    return tokenizer.decode(generated)


def sample_with_temperature(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 50,
    temperature: float = 1.0,
    device: str = 'cpu'
) -> str:
    """
    Generate text using temperature sampling.

    Temperature controls randomness:
    - temperature < 1.0: More deterministic
    - temperature > 1.0: More random

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Starting text
        max_length: Maximum tokens to generate
        temperature: Sampling temperature
        device: Device to run on

    Returns:
        Generated text

    Example:
        >>> text = sample_with_temperature(
        ...     model, tokenizer, "Hello",
        ...     temperature=0.7
        ... )
    """
    if temperature <= 0:
        warnings.warn("Temperature must be > 0. Using greedy decoding.")
        return greedy_decode(model, tokenizer, prompt, max_length, device)

    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids[0].tolist()
    eos_token_id = tokenizer.eos_token_id

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(torch.tensor([generated]).to(device))
            logits = outputs.logits[0, -1, :]

        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        if next_token == eos_token_id:
            break

        generated.append(next_token)

    return tokenizer.decode(generated)


def top_k_sampling(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 50,
    k: int = 50,
    temperature: float = 1.0,
    device: str = 'cpu'
) -> str:
    """
    Generate text using top-k sampling.

    Only considers the k most likely tokens at each step.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Starting text
        max_length: Maximum tokens to generate
        k: Number of top tokens to consider
        temperature: Sampling temperature
        device: Device to run on

    Returns:
        Generated text

    Example:
        >>> text = top_k_sampling(
        ...     model, tokenizer, "Hello",
        ...     k=40, temperature=0.8
        ... )
    """
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids[0].tolist()
    eos_token_id = tokenizer.eos_token_id

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(torch.tensor([generated]).to(device))
            logits = outputs.logits[0, -1, :]

        scaled_logits = logits / temperature
        top_k_logits, top_k_indices = scaled_logits.topk(k)

        probs = F.softmax(top_k_logits, dim=-1)
        selected_idx = torch.multinomial(probs, num_samples=1).item()
        next_token = top_k_indices[selected_idx].item()

        if next_token == eos_token_id:
            break

        generated.append(next_token)

    return tokenizer.decode(generated)


def top_p_sampling(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 50,
    p: float = 0.9,
    temperature: float = 1.0,
    device: str = 'cpu'
) -> str:
    """
    Generate text using nucleus (top-p) sampling.

    Includes the smallest set of tokens whose cumulative probability >= p.
    This adapts the number of candidates based on the probability distribution.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Starting text
        max_length: Maximum tokens to generate
        p: Cumulative probability threshold
        temperature: Sampling temperature
        device: Device to run on

    Returns:
        Generated text

    Example:
        >>> text = top_p_sampling(
        ...     model, tokenizer, "Hello",
        ...     p=0.9, temperature=0.8
        ... )
    """
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids[0].tolist()
    eos_token_id = tokenizer.eos_token_id

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(torch.tensor([generated]).to(device))
            logits = outputs.logits[0, -1, :]

        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)

        sorted_probs, sorted_indices = probs.sort(descending=True)
        cumsum = sorted_probs.cumsum(dim=-1)

        # Find cutoff
        cutoff_idx = (cumsum >= p).nonzero()
        if len(cutoff_idx) > 0:
            cutoff_idx = cutoff_idx[0].item() + 1
        else:
            cutoff_idx = len(sorted_probs)

        nucleus_probs = sorted_probs[:cutoff_idx]
        nucleus_indices = sorted_indices[:cutoff_idx]

        # Re-normalize
        nucleus_probs = nucleus_probs / nucleus_probs.sum()

        selected_idx = torch.multinomial(nucleus_probs, num_samples=1).item()
        next_token = nucleus_indices[selected_idx].item()

        if next_token == eos_token_id:
            break

        generated.append(next_token)

    return tokenizer.decode(generated)


def top_p_with_repetition_penalty(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 50,
    p: float = 0.9,
    temperature: float = 1.0,
    repetition_penalty: float = 1.2,
    device: str = 'cpu'
) -> str:
    """
    Generate text using top-p sampling with repetition penalty.

    Penalizes tokens that have already appeared in the sequence.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Starting text
        max_length: Maximum tokens to generate
        p: Cumulative probability threshold
        temperature: Sampling temperature
        repetition_penalty: Penalty for repeated tokens (>1 discourages)
        device: Device to run on

    Returns:
        Generated text

    Example:
        >>> text = top_p_with_repetition_penalty(
        ...     model, tokenizer, "Hello",
        ...     p=0.9, repetition_penalty=1.5
        ... )
    """
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids[0].tolist()
    eos_token_id = tokenizer.eos_token_id

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(torch.tensor([generated]).to(device))
            logits = outputs.logits[0, -1, :].clone()

        # Apply repetition penalty
        for token_id in set(generated):
            if logits[token_id] > 0:
                logits[token_id] = logits[token_id] / repetition_penalty
            else:
                logits[token_id] = logits[token_id] * repetition_penalty

        # Top-p sampling
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)

        sorted_probs, sorted_indices = probs.sort(descending=True)
        cumsum = sorted_probs.cumsum(dim=-1)

        cutoff_idx = (cumsum >= p).nonzero()
        if len(cutoff_idx) > 0:
            cutoff_idx = cutoff_idx[0].item() + 1
        else:
            cutoff_idx = len(sorted_probs)

        nucleus_probs = sorted_probs[:cutoff_idx]
        nucleus_indices = sorted_indices[:cutoff_idx]
        nucleus_probs = nucleus_probs / nucleus_probs.sum()

        selected_idx = torch.multinomial(nucleus_probs, num_samples=1).item()
        next_token = nucleus_indices[selected_idx].item()

        if next_token == eos_token_id:
            break

        generated.append(next_token)

    return tokenizer.decode(generated)


def beam_search(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 50,
    beam_width: int = 5,
    length_penalty: float = 1.0,
    device: str = 'cpu'
) -> str:
    """
    Generate text using beam search.

    Maintains multiple hypotheses at each step and returns the best one.

    Algorithm:
        1. Start with the prompt as the initial beam
        2. For each beam, compute next-token probabilities
        3. Expand each beam with top-k next tokens
        4. Keep only the top beam_width candidates by score
        5. Repeat until max_length or all beams hit EOS
        6. Return the highest-scoring completed sequence

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Starting text
        max_length: Maximum tokens to generate
        beam_width: Number of beams (hypotheses) to maintain
        length_penalty: Penalty for longer sequences (>1 encourages longer)
        device: Device to run on

    Returns:
        Generated text

    Example:
        >>> text = beam_search(
        ...     model, tokenizer, "Hello",
        ...     beam_width=5, length_penalty=0.8
        ... )
    """
    model.eval()

    # Step 1: Initialize with the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    eos_token_id = tokenizer.eos_token_id

    # Each beam is a tuple: (token_sequence, cumulative_log_probability)
    # We track log probs (not probs) to avoid numerical underflow
    beams = [(input_ids[0].tolist(), 0.0)]

    # Store completed sequences (those that hit EOS)
    completed = []

    # Step 2: Iterate up to max_length tokens
    for _ in range(max_length):
        # Collect all candidate expansions from all current beams
        candidates = []

        # Step 3: Expand each beam with its top-k next tokens
        for seq, score in beams:
            # Get model predictions for the next token
            with torch.no_grad():
                outputs = model(torch.tensor([seq]).to(device))
                logits = outputs.logits[0, -1, :]  # Last position logits

            # Convert to log probabilities for numerical stability
            log_probs = F.log_softmax(logits, dim=-1)

            # Get top beam_width candidates for this beam
            top_log_probs, top_indices = log_probs.topk(beam_width)

            # Create new candidate sequences
            for log_prob, idx in zip(top_log_probs, top_indices):
                new_seq = seq + [idx.item()]
                # Accumulate log probability (equivalent to multiplying probs)
                new_score = score + log_prob.item()

                if idx.item() == eos_token_id:
                    # Sequence is complete - apply length normalization
                    # This prevents bias toward shorter sequences
                    normalized_score = new_score / (len(new_seq) ** length_penalty)
                    completed.append((new_seq, normalized_score))
                else:
                    # Sequence continues - add to candidates
                    candidates.append((new_seq, new_score))

        # Step 4: Prune to keep only top beam_width candidates
        # Sort by score (descending) and keep the best
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]

        # Early exit if all beams are exhausted
        if not beams:
            break

    # Step 5: Add any remaining incomplete beams to completed
    # (they didn't hit EOS but we ran out of generation steps)
    for seq, score in beams:
        normalized_score = score / (len(seq) ** length_penalty)
        completed.append((seq, normalized_score))

    # Step 6: Return the highest-scoring sequence
    if completed:
        best_seq, _ = max(completed, key=lambda x: x[1])
        return tokenizer.decode(best_seq)
    else:
        # Fallback: return first beam if nothing completed
        return tokenizer.decode(beams[0][0])


def contrastive_search(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 50,
    k: int = 5,
    alpha: float = 0.6,
    device: str = 'cpu'
) -> str:
    """
    Generate text using contrastive search.

    Balances likelihood with diversity to avoid repetitive text.

    score(token) = (1-alpha) * log_prob - alpha * max_similarity

    Args:
        model: Language model (must have hidden states access)
        tokenizer: Tokenizer
        prompt: Starting text
        max_length: Maximum tokens to generate
        k: Number of top candidates to consider
        alpha: Balance between likelihood (0) and diversity (1)
        device: Device to run on

    Returns:
        Generated text

    Note:
        This is a simplified version. Full implementation requires
        access to model hidden states.
    """
    # Simplified version using top-k with diversity penalty
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids[0].tolist()
    eos_token_id = tokenizer.eos_token_id

    # Track recent tokens for diversity
    recent_tokens = set()
    recent_window = 20

    for step in range(max_length):
        with torch.no_grad():
            outputs = model(torch.tensor([generated]).to(device))
            logits = outputs.logits[0, -1, :]

        # Get top k
        top_k_logits, top_k_indices = logits.topk(k)
        probs = F.softmax(top_k_logits, dim=-1)

        # Apply diversity penalty
        diversity_scores = torch.ones_like(probs)
        for i, idx in enumerate(top_k_indices):
            if idx.item() in recent_tokens:
                diversity_scores[i] = 1 - alpha

        # Combined score
        combined_scores = (1 - alpha) * probs + alpha * diversity_scores
        combined_scores = combined_scores / combined_scores.sum()

        selected_idx = combined_scores.argmax().item()
        next_token = top_k_indices[selected_idx].item()

        if next_token == eos_token_id:
            break

        generated.append(next_token)

        # Update recent tokens
        recent_tokens.add(next_token)
        if len(recent_tokens) > recent_window:
            oldest = generated[-recent_window - 1] if len(generated) > recent_window else generated[0]
            recent_tokens.discard(oldest)

    return tokenizer.decode(generated)


def generate_multiple(
    model,
    tokenizer,
    prompt: str,
    num_samples: int = 5,
    max_length: int = 50,
    sampling_fn: Callable = None,
    device: str = 'cpu',
    **kwargs
) -> List[str]:
    """
    Generate multiple samples for the same prompt.

    Useful for selecting the best output or for diversity.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Starting text
        num_samples: Number of samples to generate
        max_length: Maximum tokens per sample
        sampling_fn: Sampling function to use (default: top_p_sampling)
        device: Device to run on
        **kwargs: Additional arguments for sampling function

    Returns:
        List of generated texts

    Example:
        >>> samples = generate_multiple(
        ...     model, tokenizer, "Hello",
        ...     num_samples=5, p=0.9
        ... )
    """
    if sampling_fn is None:
        sampling_fn = top_p_sampling

    results = []
    for _ in range(num_samples):
        text = sampling_fn(
            model, tokenizer, prompt,
            max_length=max_length, device=device, **kwargs
        )
        results.append(text)

    return results


if __name__ == "__main__":
    print("Generation utilities module loaded.")
    print("Run with actual model and tokenizer for testing.")
    print("\nAvailable functions:")
    print("  - greedy_decode")
    print("  - sample_with_temperature")
    print("  - top_k_sampling")
    print("  - top_p_sampling")
    print("  - top_p_with_repetition_penalty")
    print("  - beam_search")
    print("  - contrastive_search")
    print("  - generate_multiple")
