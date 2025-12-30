"""
Perplexity Calculation Utilities

This module provides functions for calculating perplexity - the standard
metric for evaluating language model quality.

Lower perplexity = better model quality.

Example:
    >>> from perplexity import calculate_perplexity
    >>>
    >>> texts = ["Hello, how are you?", "The quick brown fox..."]
    >>> ppl = calculate_perplexity(model, tokenizer, texts)
    >>> print(f"Perplexity: {ppl:.2f}")
"""

import torch
import math
from typing import List, Optional, Union, Generator
from tqdm import tqdm


def calculate_perplexity(
    model,
    tokenizer,
    texts: Union[str, List[str]],
    max_length: int = 512,
    stride: int = 256,
    batch_size: int = 1,
    show_progress: bool = True,
    device: Optional[str] = None
) -> float:
    """
    Calculate perplexity of a model on given texts.

    Perplexity = exp(average negative log-likelihood)
    Lower is better!

    Args:
        model: The language model
        tokenizer: The tokenizer
        texts: Single text or list of texts
        max_length: Maximum sequence length
        stride: Stride for sliding window (for long texts)
        batch_size: Batch size for processing
        show_progress: Show progress bar
        device: Device to use (defaults to model device)

    Returns:
        Perplexity score (float)

    Example:
        >>> ppl = calculate_perplexity(model, tokenizer, eval_texts)
        >>> print(f"Perplexity: {ppl:.2f}")

    Note:
        For quantized models, typical acceptable perplexity increases:
        - <0.1: Excellent (nearly lossless)
        - <0.5: Good (production-ready)
        - <1.0: Acceptable
        - >1.0: May affect downstream tasks
    """
    if isinstance(texts, str):
        texts = [texts]

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    iterator = tqdm(texts, desc="Perplexity", disable=not show_progress)

    with torch.no_grad():
        for text in iterator:
            # Tokenize
            encodings = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
                padding=False
            )

            input_ids = encodings.input_ids.to(device)
            seq_len = input_ids.size(1)

            # Skip very short sequences
            if seq_len < 2:
                continue

            # For long sequences, use sliding window
            if seq_len > max_length:
                for begin in range(0, seq_len - 1, stride):
                    end = min(begin + max_length, seq_len)
                    chunk = input_ids[:, begin:end]

                    target_len = chunk.size(1) - 1
                    if begin > 0:
                        # Only count non-overlapping tokens
                        target_len = min(stride, chunk.size(1) - 1)

                    outputs = model(chunk, labels=chunk)
                    loss = outputs.loss.item()

                    total_loss += loss * target_len
                    total_tokens += target_len
            else:
                # Direct calculation for short sequences
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss.item()
                num_tokens = seq_len - 1  # Exclude first token

                total_loss += loss * num_tokens
                total_tokens += num_tokens

    if total_tokens == 0:
        import warnings
        warnings.warn(
            "No tokens processed - all texts were too short (< 2 tokens). "
            "Returning infinity. Consider using longer evaluation texts.",
            UserWarning
        )
        return float('inf')

    avg_loss = total_loss / total_tokens

    # Protect against overflow (exp(100) overflows to inf)
    if avg_loss > 100:
        import warnings
        warnings.warn(
            f"Loss too high ({avg_loss:.2f}), model may be broken or not properly loaded. "
            "Returning infinity.",
            UserWarning
        )
        return float('inf')

    perplexity = math.exp(avg_loss)

    return perplexity


def calculate_perplexity_batch(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int = 8,
    max_length: int = 512,
    show_progress: bool = True
) -> float:
    """
    Calculate perplexity using batched processing.

    More efficient for many short texts.

    Args:
        model: The language model
        tokenizer: The tokenizer
        texts: List of texts
        batch_size: Batch size
        max_length: Maximum sequence length
        show_progress: Show progress bar

    Returns:
        Perplexity score
    """
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    total_tokens = 0

    # Process in batches
    num_batches = (len(texts) + batch_size - 1) // batch_size

    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=num_batches, desc="Perplexity (batched)")

    with torch.no_grad():
        for i in iterator:
            batch_texts = texts[i:i + batch_size]

            # Tokenize batch
            encodings = tokenizer(
                batch_texts,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
                padding=True
            )

            input_ids = encodings.input_ids.to(device)
            attention_mask = encodings.attention_mask.to(device)

            # Get model outputs
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            # Count actual tokens (excluding padding)
            batch_tokens = attention_mask.sum().item() - len(batch_texts)

            total_loss += outputs.loss.item() * batch_tokens
            total_tokens += batch_tokens

    if total_tokens == 0:
        import warnings
        warnings.warn(
            "No tokens processed in batched evaluation. Returning infinity.",
            UserWarning
        )
        return float('inf')

    avg_loss = total_loss / total_tokens

    # Protect against overflow
    if avg_loss > 100:
        import warnings
        warnings.warn(
            f"Loss too high ({avg_loss:.2f}), model may be broken. Returning infinity.",
            UserWarning
        )
        return float('inf')

    return math.exp(avg_loss)


def calculate_word_perplexity(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 512
) -> float:
    """
    Calculate word-level perplexity (normalized by word count).

    Useful for comparing models with different tokenizers.

    Args:
        model: The language model
        tokenizer: The tokenizer
        texts: List of texts
        max_length: Maximum sequence length

    Returns:
        Word-level perplexity
    """
    # Calculate token perplexity
    token_ppl = calculate_perplexity(
        model, tokenizer, texts, max_length, show_progress=False
    )

    # Count tokens and words
    total_tokens = 0
    total_words = 0

    for text in texts:
        tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
        total_tokens += len(tokens)
        total_words += len(text.split())

    if total_words == 0:
        return float('inf')

    # Adjust perplexity by token/word ratio
    ratio = total_tokens / total_words
    word_ppl = token_ppl ** ratio

    return word_ppl


def perplexity_by_domain(
    model,
    tokenizer,
    domain_texts: dict,
    max_length: int = 512
) -> dict:
    """
    Calculate perplexity for different text domains.

    Useful for understanding where quantization affects quality.

    Args:
        model: The language model
        tokenizer: The tokenizer
        domain_texts: Dict mapping domain names to text lists
        max_length: Maximum sequence length

    Returns:
        Dict mapping domain names to perplexity scores

    Example:
        >>> domains = {
        ...     'code': code_samples,
        ...     'science': science_texts,
        ...     'casual': chat_messages
        ... }
        >>> results = perplexity_by_domain(model, tokenizer, domains)
        >>> print(results)  # {'code': 12.3, 'science': 15.6, 'casual': 8.9}
    """
    results = {}

    for domain, texts in domain_texts.items():
        print(f"Evaluating {domain}...")
        ppl = calculate_perplexity(
            model, tokenizer, texts, max_length, show_progress=False
        )
        results[domain] = ppl

    return results


def compare_perplexity(
    baseline_model,
    quantized_model,
    tokenizer,
    texts: List[str],
    max_length: int = 512
) -> dict:
    """
    Compare perplexity between baseline and quantized models.

    Args:
        baseline_model: Original FP16 model
        quantized_model: Quantized model
        tokenizer: Shared tokenizer
        texts: Evaluation texts
        max_length: Maximum sequence length

    Returns:
        Dict with comparison metrics
    """
    print("Evaluating baseline model...")
    baseline_ppl = calculate_perplexity(
        baseline_model, tokenizer, texts, max_length
    )

    print("Evaluating quantized model...")
    quantized_ppl = calculate_perplexity(
        quantized_model, tokenizer, texts, max_length
    )

    delta = quantized_ppl - baseline_ppl
    ratio = quantized_ppl / baseline_ppl

    return {
        'baseline_ppl': baseline_ppl,
        'quantized_ppl': quantized_ppl,
        'absolute_delta': delta,
        'relative_increase': (ratio - 1) * 100,  # Percentage
        'quality_retained': (1 / ratio) * 100,   # Percentage
    }


if __name__ == "__main__":
    print("Perplexity Utils Demo")
    print("=" * 50)

    # Example usage (requires model)
    print("\nTo use this module:")
    print("  >>> from transformers import AutoModelForCausalLM, AutoTokenizer")
    print("  >>> model = AutoModelForCausalLM.from_pretrained('gpt2')")
    print("  >>> tokenizer = AutoTokenizer.from_pretrained('gpt2')")
    print("  >>> texts = ['Hello world', 'How are you?']")
    print("  >>> ppl = calculate_perplexity(model, tokenizer, texts)")
    print("  >>> print(f'Perplexity: {ppl:.2f}')")

    print("\nInterpretation guide:")
    print("  PPL < 10:   Excellent")
    print("  PPL 10-20:  Good")
    print("  PPL 20-50:  Acceptable")
    print("  PPL > 50:   Poor")

    print("\nQuantization quality thresholds:")
    print("  PPL increase < 0.1: Nearly lossless")
    print("  PPL increase < 0.5: Production-ready")
    print("  PPL increase < 1.0: Acceptable")
    print("  PPL increase > 1.0: May affect tasks")
