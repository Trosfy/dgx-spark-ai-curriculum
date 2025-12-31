# Sample Data for LLM Fine-Tuning

This directory contains sample datasets for Module 3.1: Large Language Model Fine-Tuning.

## Quick Reference

| Format | Use Case | Labs |
|--------|----------|------|
| Alpaca | Instruction tuning (SFT) | 3.1.4, 3.1.5 |
| Preference | DPO/SimPO/ORPO | 3.1.7, 3.1.8 |
| KTO Binary | Binary feedback training | 3.1.9 |
| ShareGPT | Multi-turn conversations | 3.1.6 |

## Files

### `sample_instruction_data.json`
Sample instruction-following dataset in Alpaca format.
- **Format:** JSON array with `instruction`, `input`, `output` fields
- **Size:** 20+ examples (extend for production)
- **Use:** SFT (Supervised Fine-Tuning)

### `sample_preference_data.json`
Sample preference dataset for DPO/SimPO/ORPO training.
- **Format:** JSON array with `prompt`, `chosen`, `rejected` fields
- **Size:** 15+ preference pairs
- **Use:** Preference optimization

### `sample_kto_data.json`
Sample binary feedback dataset for KTO training.
- **Format:** JSON array with `prompt`, `completion`, `label` fields
- **Size:** 20+ examples (mixed desirable/undesirable)
- **Use:** KTO (Kahneman-Tversky Optimization)

### `sample_conversation_data.json`
Sample multi-turn conversation dataset in ShareGPT format.
- **Format:** JSON with `conversations` array
- **Size:** 10+ conversations
- **Use:** Multi-turn chat fine-tuning

## Data Formats

### Alpaca Format (Instruction Tuning)
```json
{
    "instruction": "Task or question for the model",
    "input": "Optional additional context (can be empty)",
    "output": "Expected model response"
}
```

### Preference Format (DPO/SimPO/ORPO)
```json
{
    "prompt": "User prompt/question",
    "chosen": "Preferred response (better quality)",
    "rejected": "Rejected response (worse quality)"
}
```

### KTO Binary Format
```json
{
    "prompt": "User prompt/question",
    "completion": "Model response",
    "label": true  // true = desirable (thumbs up), false = undesirable (thumbs down)
}
```

### ShareGPT Format (Conversations)
```json
{
    "conversations": [
        {"from": "system", "value": "System prompt"},
        {"from": "human", "value": "User message"},
        {"from": "gpt", "value": "Assistant response"},
        {"from": "human", "value": "Follow-up question"},
        {"from": "gpt", "value": "Follow-up response"}
    ]
}
```

### ChatML Format (OpenAI-style)
```json
{
    "messages": [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "Response"}
    ]
}
```

## Dataset Size Guidelines

| Task | Minimum | Recommended | Notes |
|------|---------|-------------|-------|
| Basic SFT | 100 | 1,000-10,000 | More diverse = better |
| Domain-specific SFT | 500 | 5,000-50,000 | Quality over quantity |
| DPO/SimPO/ORPO | 500 | 5,000-20,000 | Need preference pairs |
| KTO | 1,000 | 10,000+ | Can use production logs |

## Quality Guidelines

### Instruction Data
- **Instruction length:** 10-1000 characters
- **Output length:** 20-4000 characters
- **Diversity:** Varied instructions and topics
- **Quality:** Every example should be correct and helpful

### Preference Data
- **Clear differences:** Chosen should be noticeably better
- **Similar length:** Avoid length bias (don't always prefer longer)
- **Consistent judging:** Use same criteria across examples

### KTO Data
- **Balance:** Aim for ~50% desirable / ~50% undesirable
- **Clear signal:** Each label should be unambiguous
- **Real feedback:** Production thumbs up/down works great

## Loading Data

```python
from scripts.dataset_utils import (
    load_dataset,
    DataCleaner,
    KTODataGenerator,
    DataQualityFilter
)

# Load instruction data
data = load_dataset('data/sample_instruction_data.json')

# Clean and validate
cleaner = DataCleaner()
clean_data, stats = cleaner.process_dataset(data)
print(f"Loaded {len(clean_data)} clean examples")

# Load and prepare KTO data
kto_raw = load_dataset('data/sample_kto_data.json')
kto_data = KTODataGenerator.from_binary_feedback(kto_raw)
print(f"Desirable: {len(kto_data['desirable'])}")
print(f"Undesirable: {len(kto_data['undesirable'])}")

# Quality filtering
filter = DataQualityFilter()
filtered = filter.filter_dataset(data)
print(filter.get_report())
```

## Converting Between Formats

```python
from scripts.dataset_utils import DatasetConverter, ChatTemplateFormatter

# Alpaca -> Conversation
conv = DatasetConverter.alpaca_to_conversation(alpaca_example)

# Conversation -> Llama 3 format
formatted = ChatTemplateFormatter.to_llama3(conv)

# Conversation -> ChatML format
chatml = ChatTemplateFormatter.to_chatml(conv)
```

## Recommended Datasets

For production fine-tuning, consider these public datasets:

### Instruction Tuning
- [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst1) - High-quality conversations
- [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k) - 15k instructions
- [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) - 52k GPT-4 generated

### Preference Data
- [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) - Large preference dataset
- [HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer) - NVIDIA preference data
- [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) - Safety-focused

### Specialized Domains
- [Code Alpaca](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) - Programming
- [MedInstruct](https://huggingface.co/datasets/keivalya/MedInstruct-52k) - Medical
- [Legal](https://huggingface.co/datasets/nguha/legalbench) - Legal reasoning
