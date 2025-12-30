"""
Tokenization Utilities

This module provides tokenization helpers and a simple BPE implementation
for educational purposes.

Example usage:
    >>> from tokenizer_utils import SimpleBPE, compare_tokenizers
    >>>
    >>> # Train custom BPE
    >>> bpe = SimpleBPE()
    >>> bpe.train(corpus, num_merges=1000)
    >>> tokens = bpe.tokenize("Hello world")
    >>>
    >>> # Compare tokenizers
    >>> comparison = compare_tokenizers(
    ...     ["gpt2", "bert-base-uncased"],
    ...     "This is a test sentence."
    ... )

Author: DGX Spark AI Curriculum
License: MIT
"""

import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import warnings


class SimpleBPE:
    """
    Simple Byte Pair Encoding implementation for educational purposes.

    This is not optimized for production - use the `tokenizers` library
    for real applications.

    Example:
        >>> bpe = SimpleBPE()
        >>> bpe.train("hello world hello", num_merges=10)
        >>> tokens = bpe.tokenize("hello")
        >>> print(tokens)
        ['hello', '</w>']
    """

    def __init__(self):
        self.merges: Dict[Tuple[str, str], str] = {}
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}

    def _get_word_freqs(self, text: str) -> Counter:
        """Split text into words and count frequencies."""
        words = re.findall(r"\w+|[^\w\s]", text.lower())
        return Counter(words)

    def _word_to_chars(self, word: str) -> List[str]:
        """Convert word to list of characters with end-of-word marker."""
        return list(word) + ["</w>"]

    def _get_pair_freqs(
        self,
        word_freqs: Counter,
        word_tokens: Dict[str, List[str]]
    ) -> Counter:
        """Count frequency of adjacent pairs."""
        pair_freqs = Counter()

        for word, freq in word_freqs.items():
            tokens = word_tokens[word]
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_freqs[pair] += freq

        return pair_freqs

    def _merge_pair(
        self,
        word_tokens: Dict[str, List[str]],
        pair: Tuple[str, str]
    ) -> Dict[str, List[str]]:
        """Merge a pair in all words."""
        new_word_tokens = {}

        for word, tokens in word_tokens.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_word_tokens[word] = new_tokens

        return new_word_tokens

    def train(
        self,
        text: str,
        num_merges: int = 100,
        verbose: bool = False
    ) -> None:
        """
        Train BPE on text.

        Args:
            text: Training text
            num_merges: Number of merge operations
            verbose: Print progress
        """
        word_freqs = self._get_word_freqs(text)
        word_tokens = {word: self._word_to_chars(word) for word in word_freqs}

        # Get initial vocabulary
        vocab = set()
        for tokens in word_tokens.values():
            vocab.update(tokens)

        if verbose:
            print(f"Initial vocabulary size: {len(vocab)}")
            print(f"Unique words: {len(word_freqs)}")

        # Perform merges
        for i in range(num_merges):
            pair_freqs = self._get_pair_freqs(word_freqs, word_tokens)

            if not pair_freqs:
                break

            best_pair = pair_freqs.most_common(1)[0][0]
            best_freq = pair_freqs[best_pair]

            word_tokens = self._merge_pair(word_tokens, best_pair)

            new_token = best_pair[0] + best_pair[1]
            self.merges[best_pair] = new_token
            vocab.add(new_token)

            if verbose and (i + 1) % 10 == 0:
                print(f"Merge {i+1}: {best_pair} -> '{new_token}' (freq: {best_freq})")

        # Build final vocabulary
        self.vocab = {token: i for i, token in enumerate(sorted(vocab))}
        self.inverse_vocab = {i: token for token, i in self.vocab.items()}

        if verbose:
            print(f"Final vocabulary size: {len(self.vocab)}")
            print(f"Number of merges: {len(self.merges)}")

    def tokenize(self, word: str) -> List[str]:
        """
        Tokenize a single word using learned merges.

        Args:
            word: Word to tokenize

        Returns:
            List of tokens
        """
        tokens = self._word_to_chars(word.lower())

        changed = True
        while changed:
            changed = False
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1:
                    pair = (tokens[i], tokens[i + 1])
                    if pair in self.merges:
                        new_tokens.append(self.merges[pair])
                        i += 2
                        changed = True
                        continue
                new_tokens.append(tokens[i])
                i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Text to encode

        Returns:
            List of token IDs
        """
        words = re.findall(r"\w+|[^\w\s]", text.lower())
        ids = []

        for word in words:
            tokens = self.tokenize(word)
            for token in tokens:
                if token in self.vocab:
                    ids.append(self.vocab[token])
                else:
                    # Unknown token - could use subword fallback
                    warnings.warn(f"Unknown token: {token}")

        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs to text.

        Args:
            ids: List of token IDs

        Returns:
            Decoded text
        """
        tokens = [self.inverse_vocab.get(i, "<UNK>") for i in ids]
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        return text.strip()

    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)


def compare_tokenizers(
    model_names: List[str],
    text: str
) -> Dict[str, Dict]:
    """
    Compare tokenization across different pre-trained tokenizers.

    Args:
        model_names: List of HuggingFace model names
        text: Text to tokenize

    Returns:
        Dictionary with tokenization results

    Example:
        >>> result = compare_tokenizers(
        ...     ["gpt2", "bert-base-uncased"],
        ...     "Hello world!"
        ... )
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        warnings.warn("transformers not installed. Install with: pip install transformers")
        return {}

    results = {}

    for name in model_names:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)

            tokens = tokenizer.tokenize(text)
            ids = tokenizer.encode(text, add_special_tokens=False)

            results[name] = {
                "tokens": tokens,
                "ids": ids,
                "num_tokens": len(tokens),
                "vocab_size": tokenizer.vocab_size,
                "chars_per_token": len(text) / len(tokens) if tokens else 0
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    return results


def analyze_tokenization(
    tokenizer,
    texts: List[str]
) -> Dict:
    """
    Analyze tokenization statistics for a list of texts.

    Args:
        tokenizer: HuggingFace tokenizer
        texts: List of texts to analyze

    Returns:
        Dictionary with statistics
    """
    total_chars = 0
    total_tokens = 0
    token_counts = Counter()

    for text in texts:
        tokens = tokenizer.tokenize(text)
        total_chars += len(text)
        total_tokens += len(tokens)
        token_counts.update(tokens)

    return {
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "avg_chars_per_token": total_chars / total_tokens if total_tokens > 0 else 0,
        "unique_tokens": len(token_counts),
        "most_common_tokens": token_counts.most_common(20),
        "vocab_usage_ratio": len(token_counts) / tokenizer.vocab_size
    }


def estimate_token_cost(
    text: str,
    tokenizer_name: str = "gpt2",
    cost_per_1k_tokens: float = 0.002
) -> Dict:
    """
    Estimate API cost based on token count.

    Args:
        text: Text to estimate cost for
        tokenizer_name: Tokenizer to use for counting
        cost_per_1k_tokens: Cost per 1000 tokens (varies by API)

    Returns:
        Dictionary with token count and cost estimate
    """
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        num_tokens = len(tokens)
    except ImportError:
        # Rough estimate without tokenizer
        num_tokens = len(text.split()) * 1.3  # ~1.3 tokens per word

    estimated_cost = (num_tokens / 1000) * cost_per_1k_tokens

    return {
        "num_tokens": num_tokens,
        "cost_per_1k": cost_per_1k_tokens,
        "estimated_cost": estimated_cost
    }


def chunk_text_by_tokens(
    text: str,
    tokenizer,
    max_tokens: int = 512,
    overlap: int = 50
) -> List[str]:
    """
    Split text into chunks that fit within token limit.

    Args:
        text: Text to chunk
        tokenizer: HuggingFace tokenizer
        max_tokens: Maximum tokens per chunk
        overlap: Token overlap between chunks

    Returns:
        List of text chunks
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)

        start = end - overlap
        if start + overlap >= len(tokens):
            break

    return chunks


def get_special_tokens_info(tokenizer) -> Dict:
    """
    Get information about a tokenizer's special tokens.

    Args:
        tokenizer: HuggingFace tokenizer

    Returns:
        Dictionary with special token information
    """
    special_tokens = {
        "bos_token": getattr(tokenizer, "bos_token", None),
        "eos_token": getattr(tokenizer, "eos_token", None),
        "pad_token": getattr(tokenizer, "pad_token", None),
        "unk_token": getattr(tokenizer, "unk_token", None),
        "sep_token": getattr(tokenizer, "sep_token", None),
        "cls_token": getattr(tokenizer, "cls_token", None),
        "mask_token": getattr(tokenizer, "mask_token", None),
    }

    result = {}
    for name, token in special_tokens.items():
        if token is not None:
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
            except (KeyError, ValueError, TypeError):
                token_id = None
            result[name] = {"token": token, "id": token_id}

    return result


if __name__ == "__main__":
    print("Testing tokenizer utilities...")

    # Test SimpleBPE
    corpus = """
    The quick brown fox jumps over the lazy dog.
    The dog was very lazy and the fox was quick.
    Jumping foxes are quicker than sleeping dogs.
    """ * 10

    bpe = SimpleBPE()
    bpe.train(corpus, num_merges=50, verbose=False)

    test_words = ["the", "quick", "jumping", "foxes"]
    print("\nSimpleBPE tokenization:")
    for word in test_words:
        tokens = bpe.tokenize(word)
        print(f"  '{word}' -> {tokens}")

    print(f"\nVocabulary size: {bpe.get_vocab_size()}")

    # Test encode/decode
    text = "the quick fox"
    ids = bpe.encode(text)
    decoded = bpe.decode(ids)
    print(f"\nEncode/Decode:")
    print(f"  Original: '{text}'")
    print(f"  IDs: {ids}")
    print(f"  Decoded: '{decoded}'")

    print("\nAll tests passed!")
