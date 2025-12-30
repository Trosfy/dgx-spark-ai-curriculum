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
