# Module 3.3 Data Files

This directory contains sample data for benchmarking and testing inference engines.

## Files

### `benchmark_prompts.json`
A collection of prompts organized by category for comprehensive benchmarking:
- **short**: Single-sentence prompts for latency testing
- **medium**: Paragraph-length prompts for typical usage
- **long**: Multi-paragraph prompts for context window stress testing
- **code**: Programming-related prompts for code generation benchmarks
- **reasoning**: Complex reasoning tasks for quality evaluation

### `test_requests.json`
Pre-formatted API requests for testing various inference servers:
- OpenAI-compatible chat completion requests
- Streaming vs non-streaming configurations
- Various temperature and sampling settings

## Usage

```python
import json
from pathlib import Path

# Load benchmark prompts
data_dir = Path(__file__).parent
with open(data_dir / "benchmark_prompts.json") as f:
    prompts = json.load(f)

# Access short prompts
for prompt in prompts["short"]:
    print(prompt["text"])
```

## Prompt Categories Explained

| Category | Avg Tokens | Purpose |
|----------|------------|---------|
| short | ~20 | Measure pure latency overhead |
| medium | ~100 | Typical chatbot interactions |
| long | ~500 | Stress test context handling |
| code | ~150 | Code generation performance |
| reasoning | ~200 | Multi-step reasoning tasks |

## Adding Custom Prompts

When adding prompts for benchmarking, include:
1. `id`: Unique identifier
2. `text`: The actual prompt text
3. `expected_tokens`: Approximate expected output length
4. `category`: One of the predefined categories
5. `difficulty`: 1-5 scale for reasoning complexity
