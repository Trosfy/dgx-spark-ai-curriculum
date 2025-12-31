"""
Dataset Utilities for LLM Fine-Tuning
=====================================

This module provides utilities for preparing and processing datasets
for instruction tuning and preference optimization.

Author: DGX Spark AI Curriculum
Module: 10 - Large Language Model Fine-Tuning
"""

__all__ = [
    'ChatMessage',
    'Conversation',
    'ChatTemplateFormatter',
    'DatasetConverter',
    'DataCleaner',
    'DatasetSplitter',
    'PreferenceDataGenerator',
    'KTODataGenerator',
    'DataQualityFilter',
    'save_dataset',
    'load_dataset',
]

import json
import re
import random
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable


@dataclass
class ChatMessage:
    """Represents a single message in a conversation."""
    role: str  # 'system', 'user', or 'assistant'
    content: str


@dataclass
class Conversation:
    """Represents a full conversation."""
    messages: List[ChatMessage] = field(default_factory=list)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation."""
        self.messages.append(ChatMessage(role=role, content=content))

    def to_dict(self) -> List[Dict[str, str]]:
        """Convert to list of dictionaries."""
        return [{"role": m.role, "content": m.content} for m in self.messages]


class ChatTemplateFormatter:
    """
    Format conversations for different model chat templates.

    Supports: ChatML, Llama 3.1, Llama 2, Mistral formats.

    Example:
        >>> conv = Conversation()
        >>> conv.add_message("user", "Hello!")
        >>> conv.add_message("assistant", "Hi there!")
        >>> print(ChatTemplateFormatter.to_llama3(conv))
    """

    @staticmethod
    def to_chatml(conversation: Conversation) -> str:
        """
        Format conversation in ChatML format.

        Used by: OpenAI models, some open-source models.
        """
        formatted = ""
        for msg in conversation.messages:
            formatted += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
        return formatted.strip()

    @staticmethod
    def to_llama3(conversation: Conversation) -> str:
        """
        Format conversation in Llama 3.1 format.

        Used by: Llama 3, Llama 3.1 models.
        """
        formatted = "<|begin_of_text|>"
        for msg in conversation.messages:
            formatted += f"<|start_header_id|>{msg.role}<|end_header_id|>\n\n"
            formatted += f"{msg.content}<|eot_id|>"
        return formatted

    @staticmethod
    def to_llama2(conversation: Conversation) -> str:
        """
        Format conversation in Llama 2 format.

        Used by: Llama 2 models.
        """
        formatted = "<s>"
        system_msg = None

        for msg in conversation.messages:
            if msg.role == "system":
                system_msg = msg.content
            elif msg.role == "user":
                if system_msg:
                    formatted += f"[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n"
                    system_msg = None
                else:
                    formatted += "[INST] "
                formatted += f"{msg.content} [/INST]"
            elif msg.role == "assistant":
                formatted += f" {msg.content} </s>"

        return formatted

    @staticmethod
    def to_mistral(conversation: Conversation) -> str:
        """
        Format conversation in Mistral/Mixtral format.

        Used by: Mistral, Mixtral models.
        """
        formatted = "<s>"
        for msg in conversation.messages:
            if msg.role == "user":
                formatted += f"[INST] {msg.content} [/INST]"
            elif msg.role == "assistant":
                formatted += f" {msg.content}</s>"
        return formatted


class DatasetConverter:
    """
    Convert between different dataset formats.

    Supports: Alpaca, ShareGPT, and Conversation formats.

    Example:
        >>> alpaca = {"instruction": "Hi", "input": "", "output": "Hello!"}
        >>> conv = DatasetConverter.alpaca_to_conversation(alpaca)
        >>> print(conv.messages[1].content)  # "Hi"
    """

    @staticmethod
    def alpaca_to_conversation(
        alpaca_data: Dict,
        system_prompt: str = "You are a helpful assistant.",
    ) -> Conversation:
        """
        Convert Alpaca format to Conversation format.

        Args:
            alpaca_data: Dict with 'instruction', 'input' (optional), 'output'
            system_prompt: System message to prepend
        """
        conv = Conversation()
        conv.add_message("system", system_prompt)

        user_message = alpaca_data["instruction"]
        if alpaca_data.get("input", "").strip():
            user_message += f"\n\n{alpaca_data['input']}"

        conv.add_message("user", user_message)
        conv.add_message("assistant", alpaca_data["output"])

        return conv

    @staticmethod
    def sharegpt_to_conversation(sharegpt_data: Dict) -> Conversation:
        """
        Convert ShareGPT format to Conversation format.

        Args:
            sharegpt_data: Dict with 'conversations' list
        """
        conv = Conversation()

        role_mapping = {
            "system": "system",
            "human": "user",
            "user": "user",
            "gpt": "assistant",
            "assistant": "assistant",
        }

        for turn in sharegpt_data["conversations"]:
            role = role_mapping.get(turn["from"], turn["from"])
            conv.add_message(role, turn["value"])

        return conv

    @staticmethod
    def conversation_to_alpaca(conv: Conversation) -> List[Dict]:
        """
        Convert Conversation to Alpaca format.

        Multi-turn conversations become multiple examples.
        """
        examples = []
        system_msg = ""
        current_instruction = ""

        for msg in conv.messages:
            if msg.role == "system":
                system_msg = msg.content
            elif msg.role == "user":
                current_instruction = msg.content
            elif msg.role == "assistant":
                examples.append({
                    "instruction": current_instruction,
                    "input": f"System: {system_msg}" if system_msg else "",
                    "output": msg.content,
                })

        return examples


class DataCleaner:
    """
    Clean and filter training data for quality.

    Args:
        min_instruction_length: Minimum instruction character length
        max_instruction_length: Maximum instruction character length
        min_output_length: Minimum output character length
        max_output_length: Maximum output character length
        remove_duplicates: Whether to filter duplicate examples

    Example:
        >>> cleaner = DataCleaner(min_output_length=50)
        >>> clean_data, stats = cleaner.process_dataset(raw_data)
        >>> print(f"Kept {stats['passed']} examples")
    """

    def __init__(
        self,
        min_instruction_length: int = 10,
        max_instruction_length: int = 1000,
        min_output_length: int = 20,
        max_output_length: int = 4000,
        remove_duplicates: bool = True,
    ):
        self.min_instruction_length = min_instruction_length
        self.max_instruction_length = max_instruction_length
        self.min_output_length = min_output_length
        self.max_output_length = max_output_length
        self.remove_duplicates = remove_duplicates
        self.seen_hashes = set()

    def clean_text(self, text: str) -> str:
        """
        Clean a single text string.

        Performs the following cleaning operations:
        - Removes excessive whitespace
        - Normalizes line endings
        - Strips leading/trailing whitespace

        Args:
            text: The input text to clean

        Returns:
            The cleaned text string
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        # Fix common issues
        text = text.replace('\r\n', '\n')
        text = text.replace('\r', '\n')
        return text

    def is_valid_example(self, example: Dict) -> Tuple[bool, str]:
        """
        Check if an example meets quality criteria.

        Returns:
            Tuple of (is_valid, reason if invalid)
        """
        instruction = example.get("instruction", "")
        output = example.get("output", "")

        # Check instruction length
        if len(instruction) < self.min_instruction_length:
            return False, f"Instruction too short ({len(instruction)} chars)"
        if len(instruction) > self.max_instruction_length:
            return False, f"Instruction too long ({len(instruction)} chars)"

        # Check output length
        if len(output) < self.min_output_length:
            return False, f"Output too short ({len(output)} chars)"
        if len(output) > self.max_output_length:
            return False, f"Output too long ({len(output)} chars)"

        # Check for duplicate
        if self.remove_duplicates:
            content_hash = hashlib.md5(
                (instruction + output).encode()
            ).hexdigest()
            if content_hash in self.seen_hashes:
                return False, "Duplicate example"
            self.seen_hashes.add(content_hash)

        # Check for empty or placeholder content
        placeholder_patterns = [
            r'^TODO',
            r'^TBD',
            r'^\[.*\]$',
            r'^N/A$',
        ]
        for pattern in placeholder_patterns:
            if re.match(pattern, output, re.IGNORECASE):
                return False, "Placeholder content detected"

        return True, ""

    def clean_example(self, example: Dict) -> Dict:
        """
        Clean a single training example.

        Applies text cleaning to the instruction, input, and output fields.

        Args:
            example: Dictionary with 'instruction', 'input', and 'output' keys

        Returns:
            Dictionary with cleaned text in all fields
        """
        return {
            "instruction": self.clean_text(example.get("instruction", "")),
            "input": self.clean_text(example.get("input", "")),
            "output": self.clean_text(example.get("output", "")),
        }

    def process_dataset(
        self,
        data: List[Dict],
        verbose: bool = True,
    ) -> Tuple[List[Dict], Dict]:
        """
        Process and clean a full dataset.

        Returns:
            Tuple of (cleaned_data, statistics)
        """
        cleaned = []
        stats = {
            "total": len(data),
            "passed": 0,
            "failed": 0,
            "failure_reasons": {},
        }

        for example in data:
            cleaned_example = self.clean_example(example)
            is_valid, reason = self.is_valid_example(cleaned_example)

            if is_valid:
                cleaned.append(cleaned_example)
                stats["passed"] += 1
            else:
                stats["failed"] += 1
                stats["failure_reasons"][reason] = (
                    stats["failure_reasons"].get(reason, 0) + 1
                )

        if verbose:
            print(f"Dataset Processing Results:")
            print(f"  Total: {stats['total']}")
            if stats['total'] > 0:
                print(f"  Passed: {stats['passed']} ({100*stats['passed']/stats['total']:.1f}%)")
            else:
                print(f"  Passed: {stats['passed']} (N/A - empty dataset)")
            print(f"  Failed: {stats['failed']}")

        return cleaned, stats


class DatasetSplitter:
    """
    Split datasets into train/validation/test sets.

    Example:
        >>> train, val, test = DatasetSplitter.split(data, 0.8, 0.1, 0.1)
        >>> print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    """

    @staticmethod
    def split(
        data: List[Dict],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True,
        seed: int = 42,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split data into train/val/test sets.

        Args:
            data: List of examples
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            shuffle: Whether to shuffle before splitting
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train, validation, test) lists
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        data = data.copy()
        if shuffle:
            random.seed(seed)
            random.shuffle(data)

        n = len(data)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train = data[:train_end]
        val = data[train_end:val_end]
        test = data[val_end:]

        return train, val, test


class PreferenceDataGenerator:
    """
    Generate preference pairs for DPO training.

    Example:
        >>> pairs = PreferenceDataGenerator.from_quality_scores(
        ...     scored_responses, min_score_diff=0.5
        ... )
    """

    @staticmethod
    def from_quality_scores(
        responses: List[Dict],
        min_score_diff: float = 0.5,
    ) -> List[Dict]:
        """
        Create preference pairs from responses with quality scores.

        Input format: [{"prompt": str, "response": str, "score": float}, ...]

        Returns:
            List of {"prompt": str, "chosen": str, "rejected": str}
        """
        # Group by prompt
        by_prompt = {}
        for r in responses:
            prompt = r['prompt']
            if prompt not in by_prompt:
                by_prompt[prompt] = []
            by_prompt[prompt].append(r)

        pairs = []
        for prompt, prompt_responses in by_prompt.items():
            # Sort by score (highest first)
            prompt_responses.sort(key=lambda x: x['score'], reverse=True)

            # Create pairs
            for i, high in enumerate(prompt_responses):
                for low in prompt_responses[i + 1:]:
                    if high['score'] - low['score'] >= min_score_diff:
                        pairs.append({
                            'prompt': prompt,
                            'chosen': high['response'],
                            'rejected': low['response'],
                        })

        return pairs


class KTODataGenerator:
    """
    Generate KTO (Kahneman-Tversky Optimization) training data.

    KTO uses binary feedback (thumbs up/down) instead of paired comparisons.
    Based on Prospect Theory from behavioral economics.

    Example:
        >>> data = KTODataGenerator.from_binary_feedback(feedback_data)
        >>> print(f"Desirable: {len(data['desirable'])}")
    """

    @staticmethod
    def from_binary_feedback(
        data: List[Dict],
        prompt_field: str = "prompt",
        response_field: str = "response",
        feedback_field: str = "feedback",  # "good"/"bad" or True/False
    ) -> Dict[str, List[Dict]]:
        """
        Convert binary feedback data to KTO format.

        Args:
            data: List of dicts with prompt, response, and feedback
            prompt_field: Key for prompt
            response_field: Key for response
            feedback_field: Key for feedback (bool or "good"/"bad")

        Returns:
            Dict with 'desirable' and 'undesirable' lists
        """
        desirable = []
        undesirable = []

        for item in data:
            entry = {
                "prompt": item[prompt_field],
                "completion": item[response_field],
            }

            feedback = item[feedback_field]
            is_good = feedback in [True, "good", "positive", 1, "1", "yes", "thumbs_up"]

            if is_good:
                entry["label"] = True
                desirable.append(entry)
            else:
                entry["label"] = False
                undesirable.append(entry)

        return {
            "desirable": desirable,
            "undesirable": undesirable,
            "total": len(data),
            "desirable_ratio": len(desirable) / max(len(data), 1),
        }

    @staticmethod
    def from_ratings(
        data: List[Dict],
        threshold: float = 3.5,
        prompt_field: str = "prompt",
        response_field: str = "response",
        rating_field: str = "rating",
    ) -> Dict[str, List[Dict]]:
        """
        Convert rated data to KTO format using threshold.

        Args:
            data: List of dicts with ratings
            threshold: Rating threshold for desirable/undesirable split
            prompt_field: Key for prompt
            response_field: Key for response
            rating_field: Key for rating (numeric)

        Returns:
            Dict with 'desirable' and 'undesirable' lists
        """
        desirable = []
        undesirable = []

        for item in data:
            entry = {
                "prompt": item[prompt_field],
                "completion": item[response_field],
            }

            rating = float(item[rating_field])
            if rating >= threshold:
                entry["label"] = True
                desirable.append(entry)
            else:
                entry["label"] = False
                undesirable.append(entry)

        return {
            "desirable": desirable,
            "undesirable": undesirable,
            "total": len(data),
            "desirable_ratio": len(desirable) / max(len(data), 1),
        }

    @staticmethod
    def balance_dataset(
        kto_data: Dict[str, List[Dict]],
        target_ratio: float = 1.0,
        seed: int = 42,
    ) -> Dict[str, List[Dict]]:
        """
        Balance desirable/undesirable samples.

        Args:
            kto_data: Output from from_binary_feedback or from_ratings
            target_ratio: Desired ratio of desirable:undesirable
            seed: Random seed for reproducibility

        Returns:
            Balanced dataset
        """
        random.seed(seed)
        desirable = kto_data["desirable"].copy()
        undesirable = kto_data["undesirable"].copy()

        if len(desirable) > len(undesirable) * target_ratio:
            # Downsample desirable
            target_count = int(len(undesirable) * target_ratio)
            desirable = random.sample(desirable, min(target_count, len(desirable)))
        elif len(undesirable) > len(desirable) / target_ratio:
            # Downsample undesirable
            target_count = int(len(desirable) / target_ratio)
            undesirable = random.sample(undesirable, min(target_count, len(undesirable)))

        return {
            "desirable": desirable,
            "undesirable": undesirable,
            "total": len(desirable) + len(undesirable),
            "desirable_ratio": len(desirable) / max(len(desirable) + len(undesirable), 1),
        }


class DataQualityFilter:
    """
    Advanced data quality filtering for LLM training.

    Applies multiple quality checks including:
    - Length filtering
    - Repetition detection
    - Language detection (basic)
    - Content quality heuristics

    Example:
        >>> filter = DataQualityFilter()
        >>> clean_data = filter.filter_dataset(raw_data)
    """

    def __init__(
        self,
        min_chars: int = 50,
        max_chars: int = 10000,
        min_words: int = 10,
        max_words: int = 2000,
        max_repetition_ratio: float = 0.3,
        min_alpha_ratio: float = 0.5,
    ):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.min_words = min_words
        self.max_words = max_words
        self.max_repetition_ratio = max_repetition_ratio
        self.min_alpha_ratio = min_alpha_ratio
        self.stats = {"total": 0, "passed": 0, "filtered": {}}

    def check_length(self, text: str) -> Tuple[bool, Optional[str]]:
        """Check text length constraints."""
        if len(text) < self.min_chars:
            return False, "too_short_chars"
        if len(text) > self.max_chars:
            return False, "too_long_chars"

        words = text.split()
        if len(words) < self.min_words:
            return False, "too_few_words"
        if len(words) > self.max_words:
            return False, "too_many_words"

        return True, None

    def check_repetition(self, text: str) -> Tuple[bool, Optional[str]]:
        """Check for excessive word repetition."""
        words = text.lower().split()
        if len(words) < 10:
            return True, None

        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < self.max_repetition_ratio:
            return False, "too_repetitive"

        return True, None

    def check_quality(self, text: str) -> Tuple[bool, Optional[str]]:
        """Check general quality heuristics."""
        # Alpha character ratio
        alpha_chars = sum(c.isalpha() for c in text)
        if len(text) > 0 and alpha_chars / len(text) < self.min_alpha_ratio:
            return False, "too_many_special_chars"

        # Excessive caps
        upper_chars = sum(c.isupper() for c in text)
        alpha_chars = max(alpha_chars, 1)
        if upper_chars / alpha_chars > 0.7:
            return False, "excessive_caps"

        # Refusal patterns
        refusal_patterns = [
            r"i cannot",
            r"i'm sorry",
            r"i am unable",
            r"as an ai",
            r"i don't have access",
        ]
        text_lower = text.lower()
        for pattern in refusal_patterns:
            if re.search(pattern, text_lower):
                return False, "contains_refusal"

        return True, None

    def filter_example(
        self,
        text: str
    ) -> Tuple[bool, Optional[str]]:
        """Apply all filters to a single text."""
        checks = [
            self.check_length,
            self.check_repetition,
            self.check_quality,
        ]

        for check in checks:
            passed, reason = check(text)
            if not passed:
                return False, reason

        return True, None

    def filter_dataset(
        self,
        data: List[Dict],
        text_field: str = "text",
    ) -> List[Dict]:
        """
        Filter a dataset for quality.

        Args:
            data: List of examples
            text_field: Field containing text to check

        Returns:
            Filtered list of examples
        """
        self.stats = {"total": len(data), "passed": 0, "filtered": {}}
        filtered = []

        for item in data:
            text = item.get(text_field, "")
            if isinstance(text, list):
                text = " ".join(str(t) for t in text)

            passed, reason = self.filter_example(text)

            if passed:
                filtered.append(item)
                self.stats["passed"] += 1
            else:
                self.stats["filtered"][reason] = self.stats["filtered"].get(reason, 0) + 1

        return filtered

    def get_report(self) -> str:
        """Get filtering report as string."""
        lines = [
            "Data Quality Report",
            "=" * 40,
            f"Total: {self.stats['total']}",
            f"Passed: {self.stats['passed']} ({100*self.stats['passed']/max(self.stats['total'], 1):.1f}%)",
            f"Filtered: {self.stats['total'] - self.stats['passed']}",
            "",
            "Reasons:",
        ]
        for reason, count in sorted(self.stats["filtered"].items(), key=lambda x: -x[1]):
            lines.append(f"  {reason}: {count}")
        return "\n".join(lines)


def save_dataset(
    data: List[Dict],
    filepath: Union[str, Path],
    format: str = "jsonl",
) -> None:
    """
    Save dataset to file.

    Args:
        data: List of examples
        filepath: Output file path
        format: 'jsonl' or 'json'
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        with open(filepath, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    else:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def load_dataset(filepath: Union[str, Path]) -> List[Dict]:
    """
    Load dataset from file.

    Automatically detects JSON vs JSONL format.
    """
    filepath = Path(filepath)

    with open(filepath, 'r') as f:
        content = f.read().strip()

    if content.startswith('['):
        # JSON format
        return json.loads(content)
    else:
        # JSONL format
        return [json.loads(line) for line in content.split('\n') if line.strip()]


if __name__ == "__main__":
    print("Dataset Utilities Demo")
    print("=" * 50)

    # Create sample data
    sample_data = [
        {"instruction": "What is ML?", "input": "", "output": "Machine learning..."},
        {"instruction": "Short", "input": "", "output": "Too brief"},  # Will be filtered
    ]

    # Clean data
    cleaner = DataCleaner(min_output_length=15)
    clean_data, stats = cleaner.process_dataset(sample_data)

    print(f"\nCleaned data: {len(clean_data)} examples")

    # Convert to Llama 3 format
    conv = DatasetConverter.alpaca_to_conversation(clean_data[0])
    formatted = ChatTemplateFormatter.to_llama3(conv)
    print(f"\nFormatted example:\n{formatted[:200]}...")

    print("\nDemo complete!")
