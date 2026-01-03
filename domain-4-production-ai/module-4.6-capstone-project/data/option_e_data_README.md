# Option E: Training Data

This directory contains training data for the Browser-Deployed Fine-Tuned LLM capstone project.

## Files

| File | Description | Format |
|------|-------------|--------|
| `option_e_matcha_training_data.json` | Training examples (150+) | JSON |
| `option_e_matcha_eval_data.json` | Evaluation examples (20+) | JSON |

## Data Format

Each example uses the **messages format** required for chat model training:

```json
{
  "messages": [
    {"role": "system", "content": "You are a matcha tea expert..."},
    {"role": "user", "content": "What is ceremonial grade matcha?"},
    {"role": "assistant", "content": "Ceremonial grade matcha is..."}
  ],
  "category": "grades",
  "difficulty": "easy"
}
```

## Categories

| Category | Target Count | Description |
|----------|-------------|-------------|
| grades | 25 | Matcha grades and quality levels |
| preparation | 25 | How to prepare matcha |
| health | 20 | Health benefits and nutrition |
| culture | 20 | Japanese tea culture and history |
| recipes | 20 | Matcha recipes and culinary uses |
| quality | 20 | Quality assessment and sourcing |
| storage | 10 | Storage and freshness |
| buying | 10 | Buying guide and recommendations |

## Quality Guidelines

1. **Accuracy**: All information should be factually correct
2. **Completeness**: Responses should be detailed and helpful
3. **Formatting**: Use markdown (lists, headers) for structure
4. **Consistency**: Maintain the expert persona throughout
5. **Balance**: Cover all categories evenly

## Generating More Data

Use the provided script to generate the base dataset:

```bash
python scripts/option_e_dataset_generator.py --output ./data/option_e_matcha_training_data.json
```

Then expand with your own examples following the format above.

## Validation

Validate your dataset:

```python
from datasets import load_dataset

dataset = load_dataset('json', data_files='option_e_matcha_training_data.json')
print(f"Total examples: {len(dataset['train'])}")

# Check format
for example in dataset['train'][:5]:
    assert 'messages' in example
    assert len(example['messages']) == 3
    assert example['messages'][0]['role'] == 'system'
    assert example['messages'][1]['role'] == 'user'
    assert example['messages'][2]['role'] == 'assistant'
```
