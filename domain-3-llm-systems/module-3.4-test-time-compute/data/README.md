# Data Files for Module 3.4: Test-Time Compute & Reasoning

This directory contains datasets for reasoning experiments.

## Included Sample Files

| File | Description | Problems |
|------|-------------|----------|
| `gsm8k_sample.json` | Sample GSM8K math word problems | 20 problems |
| `test_problems.json` | Multi-category test problems | 40 total (math, code, reasoning) |

### Using Sample Files

```python
import json

# Load GSM8K sample
with open("data/gsm8k_sample.json") as f:
    gsm8k = json.load(f)

print(f"Loaded {len(gsm8k)} math problems")
print(f"Example: {gsm8k[0]['question'][:50]}...")
print(f"Answer: {gsm8k[0]['numerical_answer']}")

# Load test problems by category
with open("data/test_problems.json") as f:
    problems = json.load(f)

print(f"Math: {len(problems['math'])} problems")
print(f"Code: {len(problems['code'])} problems")
print(f"Reasoning: {len(problems['reasoning'])} problems")
```

### Using Script Utilities

```python
from scripts.evaluation_utils import load_gsm8k_sample, load_test_problems

# Load automatically
gsm8k = load_gsm8k_sample()  # Returns list of problem dicts
problems = load_test_problems()  # Returns dict with categories
```

---

## Recommended External Datasets

| Dataset | Source | Size | Use Case |
|---------|--------|------|----------|
| GSM8K (sample) | OpenAI | ~8K problems | Math reasoning benchmark |
| HumanEval | OpenAI | 164 problems | Code generation |
| MATH (sample) | Hendrycks | ~12K problems | Competition math |
| ARC-Challenge | AI2 | 1,172 problems | Science reasoning |

## Loading Datasets

### GSM8K for Math Reasoning

```python
from datasets import load_dataset

# Load GSM8K test set
gsm8k = load_dataset("openai/gsm8k", "main", split="test")

# Example problem
print(gsm8k[0]["question"])
print(gsm8k[0]["answer"])

# The answer includes step-by-step reasoning
# Final numerical answer is after ####
```

### HumanEval for Code Reasoning

```python
from datasets import load_dataset

# Load HumanEval
humaneval = load_dataset("openai/humaneval", split="test")

# Example problem
print(humaneval[0]["prompt"])
print(humaneval[0]["canonical_solution"])
```

## Creating Test Sets

For lab experiments, we recommend creating smaller subsets:

```python
import random

# Create a balanced test set
def create_test_set(n_math=20, n_code=10, n_reasoning=10):
    test_set = []

    # Math problems from GSM8K
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    math_indices = random.sample(range(len(gsm8k)), n_math)
    for i in math_indices:
        test_set.append({
            "type": "math",
            "question": gsm8k[i]["question"],
            "answer": gsm8k[i]["answer"]
        })

    # Code problems from HumanEval
    humaneval = load_dataset("openai/humaneval", split="test")
    code_indices = random.sample(range(len(humaneval)), n_code)
    for i in code_indices:
        test_set.append({
            "type": "code",
            "question": humaneval[i]["prompt"],
            "answer": humaneval[i]["canonical_solution"]
        })

    return test_set
```

## Reward Model Resources

For Best-of-N sampling labs:

| Reward Model | Size | Description |
|--------------|------|-------------|
| ArmoRM-Llama3-8B | 8B | General-purpose reward model |
| Skywork-Reward | 7B | Strong general reward model |
| PRM800K | Various | Process reward model for math |

```python
# Download reward model
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="RLHFlow/ArmoRM-Llama3-8B-v0.1",
    local_dir="./data/reward_models/armo-rm"
)
```

## Memory Considerations

On DGX Spark with 128GB unified memory:
- R1-distill-70B (Q4): ~45GB
- Reward model (8B): ~16GB
- Both together: ~61GB - fits comfortably!

This allows running reasoning model + reward model simultaneously for Best-of-N experiments.
