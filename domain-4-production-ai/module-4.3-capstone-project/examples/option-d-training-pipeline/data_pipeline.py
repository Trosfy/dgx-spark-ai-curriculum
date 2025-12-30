#!/usr/bin/env python3
"""
Data Pipeline for Training

Data collection, curation, and preparation for LLM training.
This is a starting point - extend this for your capstone!
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import hashlib


@dataclass
class DataSample:
    """A single training sample."""
    instruction: str
    input: str
    output: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Generate unique ID from content."""
        content = f"{self.instruction}{self.input}{self.output}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "source": self.source,
            "metadata": self.metadata
        }


@dataclass
class PreferenceSample:
    """A preference pair for DPO training."""
    prompt: str
    chosen: str
    rejected: str
    source: str


class DataCollector:
    """
    Collect data from various sources.

    Sources can include:
    - Hugging Face datasets
    - Custom JSONL files
    - API endpoints
    - Web scraping
    """

    def __init__(self):
        self.sources: Dict[str, Callable] = {}
        self.collected: List[DataSample] = []

    def register_source(self, name: str, loader: Callable):
        """Register a data source."""
        self.sources[name] = loader
        print(f"Registered source: {name}")

    def collect_from_jsonl(self, path: str, source_name: str = "jsonl") -> List[DataSample]:
        """Load samples from JSONL file."""
        samples = []
        try:
            with open(path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    samples.append(DataSample(
                        instruction=data.get("instruction", ""),
                        input=data.get("input", ""),
                        output=data.get("output", ""),
                        source=source_name
                    ))
        except FileNotFoundError:
            print(f"File not found: {path}")
        return samples

    def collect_from_hf(self, dataset_name: str, split: str = "train") -> List[DataSample]:
        """Load samples from Hugging Face dataset (mock)."""
        # In production, use: from datasets import load_dataset
        print(f"Loading from HuggingFace: {dataset_name} ({split})")

        # Mock data
        return [
            DataSample(
                instruction="Explain machine learning",
                input="",
                output="Machine learning is a field of AI...",
                source=f"hf:{dataset_name}"
            ),
            DataSample(
                instruction="Write a Python function",
                input="Calculate factorial",
                output="def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
                source=f"hf:{dataset_name}"
            )
        ]

    def collect_all(self) -> List[DataSample]:
        """Collect from all registered sources."""
        for name, loader in self.sources.items():
            print(f"Collecting from {name}...")
            samples = loader()
            self.collected.extend(samples)
            print(f"  Collected {len(samples)} samples")
        return self.collected


class DataCurator:
    """
    Curate and filter collected data.

    Applies quality filters and deduplication.
    """

    def __init__(self):
        self.filters: List[Callable[[DataSample], bool]] = []
        self.stats: Dict[str, int] = {}

    def add_filter(self, name: str, filter_fn: Callable[[DataSample], bool]):
        """Add a quality filter."""
        self.filters.append((name, filter_fn))
        print(f"Added filter: {name}")

    def length_filter(self, min_len: int = 10, max_len: int = 4096) -> Callable:
        """Filter by output length."""
        def filter_fn(sample: DataSample) -> bool:
            length = len(sample.output)
            return min_len <= length <= max_len
        return filter_fn

    def quality_filter(self) -> Callable:
        """Basic quality filter."""
        def filter_fn(sample: DataSample) -> bool:
            # Check for minimum content
            if not sample.instruction or not sample.output:
                return False
            # Check for obvious issues
            if sample.output.lower().startswith("i cannot"):
                return False
            return True
        return filter_fn

    def deduplicate(self, samples: List[DataSample]) -> List[DataSample]:
        """Remove duplicate samples."""
        seen = set()
        unique = []
        for sample in samples:
            if sample.id not in seen:
                seen.add(sample.id)
                unique.append(sample)
        removed = len(samples) - len(unique)
        print(f"Deduplication: removed {removed} duplicates")
        return unique

    def curate(self, samples: List[DataSample]) -> List[DataSample]:
        """Apply all filters and deduplication."""
        print(f"\nCurating {len(samples)} samples...")
        self.stats["input"] = len(samples)

        # Deduplicate first
        samples = self.deduplicate(samples)
        self.stats["after_dedup"] = len(samples)

        # Apply filters
        for name, filter_fn in self.filters:
            before = len(samples)
            samples = [s for s in samples if filter_fn(s)]
            removed = before - len(samples)
            print(f"  {name}: removed {removed}")
            self.stats[f"after_{name}"] = len(samples)

        self.stats["output"] = len(samples)
        print(f"Final dataset: {len(samples)} samples")
        return samples


class DataFormatter:
    """
    Format data for different training objectives.

    Supports:
    - Alpaca format (instruction tuning)
    - Chat format (conversation)
    - DPO format (preference pairs)
    """

    def to_alpaca(self, samples: List[DataSample]) -> List[Dict]:
        """Convert to Alpaca format."""
        return [
            {
                "instruction": s.instruction,
                "input": s.input,
                "output": s.output
            }
            for s in samples
        ]

    def to_chat(self, samples: List[DataSample]) -> List[Dict]:
        """Convert to chat format."""
        formatted = []
        for s in samples:
            user_content = s.instruction
            if s.input:
                user_content += f"\n\n{s.input}"

            formatted.append({
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": s.output}
                ]
            })
        return formatted

    def to_dpo(self, preferences: List[PreferenceSample]) -> List[Dict]:
        """Convert to DPO format."""
        return [
            {
                "prompt": p.prompt,
                "chosen": p.chosen,
                "rejected": p.rejected
            }
            for p in preferences
        ]

    def save_jsonl(self, data: List[Dict], path: str):
        """Save to JSONL file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"Saved {len(data)} samples to {path}")


# Example usage
if __name__ == "__main__":
    print("Data Pipeline Demo")
    print("=" * 50)

    # 1. Collect data
    print("\n1. DATA COLLECTION")
    print("-" * 40)

    collector = DataCollector()

    # Register mock sources
    collector.register_source(
        "mock_hf",
        lambda: collector.collect_from_hf("teknium/OpenHermes-2.5")
    )

    samples = collector.collect_all()
    print(f"Total collected: {len(samples)}")

    # 2. Curate data
    print("\n2. DATA CURATION")
    print("-" * 40)

    curator = DataCurator()
    curator.add_filter("length", curator.length_filter(10, 2000))
    curator.add_filter("quality", curator.quality_filter())

    curated = curator.curate(samples)

    # 3. Format data
    print("\n3. DATA FORMATTING")
    print("-" * 40)

    formatter = DataFormatter()

    alpaca_data = formatter.to_alpaca(curated)
    print(f"Alpaca format: {len(alpaca_data)} samples")

    chat_data = formatter.to_chat(curated)
    print(f"Chat format: {len(chat_data)} samples")

    # Preview
    print("\nSample (Alpaca format):")
    print(json.dumps(alpaca_data[0], indent=2))
