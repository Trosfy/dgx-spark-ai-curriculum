# Sample Data for LLM Fine-Tuning

This directory contains sample datasets for Module 3.1: Large Language Model Fine-Tuning.

## Files

### `sample_instruction_data.json`
Sample instruction-following dataset in Alpaca format.
- **Format:** JSON array of objects with `instruction`, `input`, `output` fields
- **Size:** 21 examples (extend for production use)
- **Use:** SFT (Supervised Fine-Tuning), Lab 3.1.2

### `sample_preference_data.json`
Sample preference dataset for DPO training.
- **Format:** JSON array of objects with `prompt`, `chosen`, `rejected` fields
- **Size:** 13 preference pairs (extend for production use)
- **Use:** DPO (Direct Preference Optimization), Lab 3.1.5

### `sample_conversation_data.json`
Sample multi-turn conversation dataset in ShareGPT format.
- **Format:** JSON array with `conversations` containing turn arrays
- **Size:** ~10 conversations
- **Use:** Multi-turn chat fine-tuning

## Data Formats

### Alpaca Format (Instruction Tuning)
```json
{
    "instruction": "Task or question for the model",
    "input": "Optional additional context",
    "output": "Expected model response"
}
```

### Preference Format (DPO)
```json
{
    "prompt": "User prompt/question",
    "chosen": "Preferred response (better)",
    "rejected": "Rejected response (worse)"
}
```

### ShareGPT Format (Conversations)
```json
{
    "conversations": [
        {"from": "system", "value": "System prompt"},
        {"from": "human", "value": "User message"},
        {"from": "gpt", "value": "Assistant response"},
        ...
    ]
}
```

## Creating Your Own Dataset

1. **Choose a domain:** Pick a specific topic (coding, medical, legal, etc.)
2. **Collect examples:** Gather 50-1000 high-quality examples
3. **Format correctly:** Use the appropriate format for your training method
4. **Clean the data:** Use `dataset_utils.py` to clean and validate
5. **Split the data:** Create train/val/test splits (typically 80/10/10)

## Quality Guidelines

- **Instruction length:** 10-1000 characters
- **Output length:** 20-4000 characters
- **Diversity:** Include varied instructions and topics
- **Quality:** Every example should be correct and helpful
- **Consistency:** Use consistent formatting and style

## Loading Data

```python
from scripts.dataset_utils import load_dataset, DataCleaner

# Load data
data = load_dataset('data/sample_instruction_data.json')

# Clean and validate
cleaner = DataCleaner()
clean_data, stats = cleaner.process_dataset(data)

print(f"Loaded {len(clean_data)} clean examples")
```
