# Module 3.2: Data Files

## Overview

This directory contains calibration data and evaluation datasets for quantization experiments.

## Calibration Data

Quantization methods like GPTQ, AWQ, and NVFP4 require calibration data to compute optimal scaling factors.

### Requirements for Good Calibration Data

1. **Representative**: Should match your deployment use case
2. **Diverse**: Cover different topics, lengths, styles
3. **Sufficient quantity**: Typically 128-512 samples
4. **Appropriate length**: Match your expected input lengths

### Generating Calibration Data

```python
from datasets import load_dataset

# Option 1: Use WikiText (general text)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
calibration_texts = dataset["text"][:128]

# Option 2: Use C4 (web text)
dataset = load_dataset("c4", "en", split="train", streaming=True)
calibration_texts = [next(iter(dataset))["text"] for _ in range(128)]

# Option 3: Domain-specific data
# Use your own data that matches your deployment scenario
calibration_texts = load_your_domain_data()
```

## Evaluation Datasets

### Perplexity Evaluation

For measuring quantization quality, we use perplexity on held-out text:

```python
# WikiText-2 test set
from datasets import load_dataset
wikitext_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

# Your own evaluation data
eval_texts = [
    "The field of machine learning has grown exponentially...",
    "Artificial intelligence systems can perform complex reasoning...",
    # ... more diverse texts
]
```

### Task-Specific Evaluation

For comprehensive quality assessment, use standardized benchmarks:

| Task | Description | lm-eval name |
|------|-------------|--------------|
| HellaSwag | Common sense | `hellaswag` |
| ARC | Science QA | `arc_easy`, `arc_challenge` |
| MMLU | Multitask | `mmlu` |
| WinoGrande | Coreference | `winogrande` |
| TruthfulQA | Truthfulness | `truthfulqa` |

```python
from lm_eval import evaluator

results = evaluator.simple_evaluate(
    model="hf",
    model_args=f"pretrained={model_path}",
    tasks=["hellaswag", "arc_easy"],
    batch_size=8
)
```

## File Structure

```
data/
├── README.md                    # This file
├── calibration/                 # Calibration datasets (created during exercises)
│   ├── general_text.txt
│   ├── code_samples.txt
│   └── domain_specific.txt
└── evaluation/                  # Evaluation datasets (created during exercises)
    ├── perplexity_texts.txt
    └── task_samples.jsonl
```

## Notes for DGX Spark

With 128GB unified memory, you can use larger calibration datasets:
- Standard: 128 samples
- Enhanced: 512 samples
- Comprehensive: 1024+ samples

More calibration data generally improves quantization quality, especially for NVFP4.

## Downloading Pre-Built Datasets

```bash
# WikiText
huggingface-cli download wikitext wikitext-2-raw-v1

# C4 (large, use streaming)
# Use streaming=True in load_dataset

# MMLU
huggingface-cli download cais/mmlu
```

## Creating Custom Calibration Data

For best results, create calibration data from your actual use case:

```python
import json

def create_calibration_dataset(
    your_data_path: str,
    output_path: str,
    num_samples: int = 128
):
    """Create calibration dataset from your data."""
    with open(your_data_path) as f:
        data = json.load(f)

    # Sample and filter
    samples = []
    for item in data[:num_samples]:
        text = item['text']
        if 50 < len(text.split()) < 500:  # Filter by length
            samples.append(text)

    with open(output_path, 'w') as f:
        for text in samples:
            f.write(text + '\n\n')

    print(f"Created {len(samples)} calibration samples")

# Usage
create_calibration_dataset(
    "your_training_data.json",
    "data/calibration/custom.txt"
)
```

---

*Good calibration data is crucial for quantization quality!*
