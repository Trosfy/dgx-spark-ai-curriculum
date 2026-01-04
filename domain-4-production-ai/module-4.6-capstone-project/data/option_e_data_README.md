# Option E: Troscha Matcha Chatbot Training Data

This directory contains training data for the Troscha Matcha Browser-Deployed Fine-Tuned LLM capstone project.

## Overview

- **Model**: Gemma 3 270M (google/gemma-3-270m-it)
- **Total Examples**: 300 conversations
- **Format**: Messages format with `<preferences>` JSON output
- **Language**: English

## Files

| File | Count | Description |
|------|-------|-------------|
| `option-e-browser-llm/01_product_comparisons.json` | 30 | Pairwise and group product comparisons |
| `option-e-browser-llm/02_product_specific.json` | 24 | Questions about specific products |
| `option-e-browser-llm/03_taste_discovery.json` | 40 | Taste preference discovery |
| `option-e-browser-llm/04_experience_onboarding.json` | 35 | Experience-based recommendations |
| `option-e-browser-llm/05_serving_context.json` | 30 | Serving style guidance |
| `option-e-browser-llm/06_matcha_education.json` | 25 | General matcha education |
| `option-e-browser-llm/07_price_questions.json` | 15 | Price and value questions |
| `option-e-browser-llm/08_common_questions.json` | 20 | Storage, prep, caffeine, etc. |
| `option-e-browser-llm/09_troubleshooting.json` | 18 | Problem-solving conversations |
| `option-e-browser-llm/10_multi_turn.json` | 20 | Multi-turn refinement |
| `option-e-browser-llm/11_out_of_scope.json` | 31 | Out-of-scope handling |
| `option-e-browser-llm/12_brand_identity.json` | 12 | Brand and chatbot identity |
| `option-e-browser-llm/system_prompt.txt` | - | System prompt reference |

## Data Format

Each conversation uses the **messages format** with a `<preferences>` JSON block:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are Troscha's matcha guide.\n\nCATALOG:\n1. YURA (001)..."
    },
    {
      "role": "user",
      "content": "What's the difference between Firu and Zeno?"
    },
    {
      "role": "assistant",
      "content": "Great question! These two are quite different...\n\n<preferences>\n{\"intent\": \"compare\", \"recommended_matcha\": [\"m-003\", \"m-005\"], ...}\n</preferences>"
    }
  ]
}
```

## Product Catalog (Current)

| Product | Styles Available | Price Range |
|---------|------------------|-------------|
| Yura | Latte | Rp 27k |
| Taku | Straight, Latte, Strawberry | Rp 25-40k |
| Firu | Straight, Latte, Miruku, Strawberry | Rp 34-52k |
| Giru | Straight, Latte, Miruku, Strawberry | Rp 39-57k |
| Zeno | Straight, Latte, Miruku, Strawberry | Rp 44-62k |
| Moku | Hojicha Latte | Rp 35k |
| Hiku | Straight, Latte | Rp 79-89k |
| Kiyo | Straight, Latte | Rp 94-104k |

**Addon**: Oat Milk +Rp 5k

## Preferences JSON Schema

Every assistant response ends with a `<preferences>` block:

```json
{
  "intent": "recommend|educate|compare|troubleshoot|out_of_scope",
  "sweetness": "low|medium|high"|null,
  "bitterness": "low|medium|high"|null,
  "umami": "low|medium|high"|null,
  "body": "light|medium|full"|null,
  "serving": "straight|latte|miruku"|null,
  "experience": "beginner|intermediate|enthusiast"|null,
  "recommended_matcha": ["m-001", "m-003", ...]|null,
  "origin_preference": ["shiga", "uji", "nishio"]|null,
  "notes": "string"|null
}
```

## Quality Guidelines

1. **Tone**: Warm, conversational (like a knowledgeable barista)
2. **Length**: 50-150 words per response
3. **Emoji**: 1-2 maximum per response
4. **Questions**: Ask one follow-up when appropriate
5. **Prices**: Mention when relevant to customer

## Validation

```python
import json
from pathlib import Path

data_dir = Path("option-e-browser-llm")
total = 0

for f in sorted(data_dir.glob("*.json")):
    with open(f) as fp:
        data = json.load(fp)
        count = len(data)
        total += count
        print(f"{f.name}: {count} conversations")

print(f"Total: {total} conversations")

# Validate format
sample = data[0]
assert "messages" in sample
assert len(sample["messages"]) == 3
assert sample["messages"][0]["role"] == "system"
assert sample["messages"][1]["role"] == "user"
assert sample["messages"][2]["role"] == "assistant"
assert "<preferences>" in sample["messages"][2]["content"]
```

## Architecture

```
Runtime (Browser)
├── Catalog JSON files (matcha.json, products.json, etc.)
├── generateSystemPrompt(catalog) → Dynamic system prompt
├── Fine-tuned Gemma 3 270M
│   ├── System prompt: Dynamic catalog
│   ├── User message: Customer query
│   └── Output: Response + <preferences> JSON
└── Application Layer
    ├── Parse <preferences> JSON
    ├── Validate against live catalog
    └── Render product cards
```

## Related Files

- **Specification**: `prompts/capstone-project/TRAINING_DATA_GENERATION.md`
- **Knowledge Base**: `prompts/capstone-project/MATCHA_KNOWLEDGE_GUIDE.md`
- **Sources**: `prompts/capstone-project/MATCHA_KNOWLEDGE_SOURCES.md`
