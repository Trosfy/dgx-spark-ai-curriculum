"""
Safety Classifier using Llama Guard

This module provides utilities for classifying text as safe or unsafe
using Llama Guard 3 or compatible safety models.

Example usage:
    >>> from safety_classifier import SafetyClassifier
    >>> classifier = SafetyClassifier()
    >>> result = classifier.classify("Hello, how are you?")
    >>> print(result.is_safe)
    True
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    import ollama
except ImportError:
    ollama = None


@dataclass
class SafetyResult:
    """
    Result of a safety classification.

    Attributes:
        is_safe: Whether the content is classified as safe
        category: The violation category code (e.g., "S1", "S9") if unsafe
        category_name: Human-readable name of the violation category
        raw_response: Raw response from the classifier
        latency_ms: Time taken for classification in milliseconds
    """
    is_safe: bool
    category: Optional[str] = None
    category_name: Optional[str] = None
    raw_response: str = ""
    latency_ms: float = 0.0


# Llama Guard 3 safety categories
SAFETY_CATEGORIES = {
    "S1": "Violent Crimes",
    "S2": "Non-Violent Crimes",
    "S3": "Sex-Related Crimes",
    "S4": "Child Sexual Exploitation",
    "S5": "Defamation",
    "S6": "Specialized Advice",
    "S7": "Privacy",
    "S8": "Intellectual Property",
    "S9": "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Suicide & Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections",
    "S14": "Code Interpreter Abuse",
}


class SafetyClassifier:
    """
    A safety classifier using Llama Guard or compatible models.

    This classifier can check both user inputs and model outputs
    for safety violations across 14 categories.

    Example:
        >>> classifier = SafetyClassifier(model="llama-guard3:8b")
        >>>
        >>> # Check input
        >>> result = classifier.check_input("How do I cook pasta?")
        >>> print(f"Safe: {result.is_safe}")
        Safe: True
        >>>
        >>> # Check output
        >>> result = classifier.check_output(
        ...     user_message="How do I cook pasta?",
        ...     assistant_message="Boil water, add pasta, cook for 8-10 minutes."
        ... )
        >>> print(f"Safe: {result.is_safe}")
        Safe: True
    """

    def __init__(
        self,
        model: str = "llama-guard3:8b",
        fail_closed: bool = True
    ):
        """
        Initialize the safety classifier.

        Args:
            model: The Ollama model to use for classification
            fail_closed: If True, errors result in "unsafe" classification
        """
        if ollama is None:
            raise ImportError(
                "ollama package is required. Install with: pip install ollama"
            )

        self.model = model
        self.fail_closed = fail_closed
        self._stats = {
            "total_checks": 0,
            "blocked": 0,
            "allowed": 0,
            "errors": 0,
            "total_latency_ms": 0
        }

    def classify(
        self,
        user_message: str,
        assistant_message: Optional[str] = None
    ) -> SafetyResult:
        """
        Classify content for safety violations.

        Args:
            user_message: The user's message to classify
            assistant_message: Optional assistant response to also classify

        Returns:
            SafetyResult with classification details
        """
        # Build conversation string
        conversation = f"User: {user_message}"
        if assistant_message:
            conversation += f"\nAssistant: {assistant_message}"

        # Format prompt for Llama Guard
        prompt = f"""Task: Check if there is unsafe content in the conversation below according to our safety policy.

<BEGIN CONVERSATION>
{conversation}
<END CONVERSATION>

Provide your safety assessment for the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories."""

        start_time = time.time()

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            latency_ms = (time.time() - start_time) * 1000

            result_text = response["message"]["content"].strip()

            # Parse response
            lines = result_text.lower().split("\n")
            is_safe = lines[0].strip().startswith("safe")

            category = None
            category_name = None

            if not is_safe and len(lines) > 1:
                # Extract category codes
                for code in SAFETY_CATEGORIES.keys():
                    if code.lower() in result_text.lower():
                        category = code
                        category_name = SAFETY_CATEGORIES[code]
                        break

            result = SafetyResult(
                is_safe=is_safe,
                category=category,
                category_name=category_name,
                raw_response=result_text,
                latency_ms=latency_ms
            )

        except Exception as e:
            self._stats["errors"] += 1
            result = SafetyResult(
                is_safe=not self.fail_closed,
                raw_response=f"Error: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000
            )

        # Update stats
        self._stats["total_checks"] += 1
        self._stats["total_latency_ms"] += result.latency_ms
        if result.is_safe:
            self._stats["allowed"] += 1
        else:
            self._stats["blocked"] += 1

        return result

    def check_input(self, user_message: str) -> SafetyResult:
        """
        Check if user input is safe.

        Args:
            user_message: The user's message to check

        Returns:
            SafetyResult with classification
        """
        return self.classify(user_message)

    def check_output(
        self,
        user_message: str,
        assistant_message: str
    ) -> SafetyResult:
        """
        Check if assistant output is safe in context.

        Args:
            user_message: The original user message
            assistant_message: The assistant's response to check

        Returns:
            SafetyResult with classification
        """
        return self.classify(user_message, assistant_message)

    def check_batch(
        self,
        messages: List[str],
        show_progress: bool = True
    ) -> List[SafetyResult]:
        """
        Check multiple messages for safety.

        Args:
            messages: List of messages to check
            show_progress: Whether to print progress

        Returns:
            List of SafetyResult objects
        """
        results = []

        for i, msg in enumerate(messages, 1):
            if show_progress:
                print(f"\rChecking {i}/{len(messages)}...", end="")
            results.append(self.classify(msg))

        if show_progress:
            print(" Done!")

        return results

    def get_stats(self) -> Dict:
        """
        Get classification statistics.

        Returns:
            Dictionary with stats including total checks, block rate, avg latency
        """
        total = self._stats["total_checks"]
        return {
            **self._stats,
            "block_rate": self._stats["blocked"] / max(total, 1),
            "error_rate": self._stats["errors"] / max(total, 1),
            "avg_latency_ms": self._stats["total_latency_ms"] / max(total, 1)
        }

    def reset_stats(self):
        """Reset classification statistics."""
        self._stats = {
            "total_checks": 0,
            "blocked": 0,
            "allowed": 0,
            "errors": 0,
            "total_latency_ms": 0
        }


def classify_safety(
    text: str,
    model: str = "llama-guard3:8b"
) -> SafetyResult:
    """
    Convenience function to classify a single text.

    Args:
        text: The text to classify
        model: The model to use for classification

    Returns:
        SafetyResult with classification

    Example:
        >>> result = classify_safety("Hello, how are you?")
        >>> print(result.is_safe)
        True
    """
    classifier = SafetyClassifier(model=model)
    return classifier.classify(text)


if __name__ == "__main__":
    # Demo usage
    print("Safety Classifier Demo")
    print("=" * 40)

    classifier = SafetyClassifier()

    test_inputs = [
        "What's the weather like today?",
        "How do I bake a cake?",
        "How do I hack into a computer?",  # Should be blocked
    ]

    for inp in test_inputs:
        result = classifier.classify(inp)
        status = "SAFE" if result.is_safe else f"UNSAFE ({result.category_name})"
        print(f"Input: {inp[:40]}...")
        print(f"  -> {status}")
        print()

    print("Stats:", classifier.get_stats())
