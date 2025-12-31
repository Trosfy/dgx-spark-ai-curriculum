# Module 3.4: Test-Time Compute & Reasoning - Quick Reference

## 🚀 Essential Commands

### Ollama Setup for Reasoning Models
```bash
# Pull DeepSeek-R1 models
ollama pull deepseek-r1:7b    # Fast testing
ollama pull deepseek-r1:70b   # Best quality

# Pull comparison models
ollama pull llama3.1:8b
ollama pull llama3.1:70b
```

### NGC Container
```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $HOME/.ollama:/root/.ollama \
    --ipc=host \
    --network=host \
    nvcr.io/nvidia/pytorch:25.11-py3
```

## 📊 Key Performance Data

### DGX Spark Reasoning Model Performance
| Model | Quantization | Memory | Speed (tok/s) |
|-------|--------------|--------|---------------|
| R1-distill-7B | FP16 | ~14GB | ~50 |
| R1-distill-32B | Q4 | ~20GB | ~30 |
| R1-distill-70B | Q4 | ~45GB | ~20 |
| Llama 3.1 70B | Q4 | ~45GB | ~25 |

### CoT Accuracy Improvements
| Dataset | Without CoT | With CoT | Improvement |
|---------|-------------|----------|-------------|
| GSM8K (math) | ~55% | ~75% | +20% |
| SVAMP (word problems) | ~60% | ~80% | +20% |
| ARC-C (science) | ~70% | ~80% | +10% |

## 🔧 Common Patterns

### Pattern: Zero-Shot Chain-of-Thought
```python
# Just add this magic phrase!
prompt = f"""
{question}

Let's think step by step:
"""

response = ollama.chat(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": prompt}]
)
```

### Pattern: Few-Shot Chain-of-Thought
```python
few_shot_prompt = """
Q: If there are 3 cars in the parking lot and 2 more arrive, how many cars are there?
A: Let's think step by step.
1. Start with 3 cars
2. 2 more cars arrive
3. Total = 3 + 2 = 5 cars
The answer is 5.

Q: If there are 5 apples and I eat 2, how many are left?
A: Let's think step by step.
1. Start with 5 apples
2. Eat 2 apples
3. Remaining = 5 - 2 = 3 apples
The answer is 3.

Q: {question}
A: Let's think step by step.
"""
```

### Pattern: Self-Consistency
```python
import collections

def self_consistency(model, prompt, n_samples=5, temperature=0.7):
    answers = []

    for _ in range(n_samples):
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature}
        )
        answer = extract_answer(response['message']['content'])
        answers.append(answer)

    # Majority vote
    counter = collections.Counter(answers)
    best_answer, count = counter.most_common(1)[0]
    confidence = count / n_samples

    return best_answer, confidence
```

### Pattern: DeepSeek-R1 Usage
```python
import ollama

response = ollama.chat(
    model="deepseek-r1:70b",
    messages=[{
        "role": "user",
        "content": "What is the derivative of x³ + 2x² - 5x + 3?"
    }]
)

# Response includes <think>...</think> blocks
print(response['message']['content'])

# Extract just the answer
import re
answer = re.sub(r'<think>.*?</think>', '', response['message']['content'], flags=re.DOTALL)
print("Clean answer:", answer.strip())
```

### Pattern: Best-of-N with Reward Model
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load reward model
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "RLHFlow/ArmoRM-Llama3-8B-v0.1",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
reward_tokenizer = AutoTokenizer.from_pretrained("RLHFlow/ArmoRM-Llama3-8B-v0.1")

def best_of_n(prompt, n=5, temperature=0.7):
    candidates = []
    for _ in range(n):
        response = ollama.chat(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature}
        )
        candidates.append(response['message']['content'])

    # Score each
    scores = []
    for candidate in candidates:
        inputs = reward_tokenizer(
            prompt + candidate,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to("cuda")
        with torch.no_grad():
            score = reward_model(**inputs).logits[0, 0].item()
        scores.append(score)

    # Return best
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return candidates[best_idx], scores[best_idx]
```

### Pattern: Adaptive Routing
```python
def classify_complexity(question):
    """Simple heuristic for routing."""
    # Complex indicators
    complex_keywords = ["step by step", "explain", "derive", "prove", "calculate"]
    if any(kw in question.lower() for kw in complex_keywords):
        return "complex"
    if len(question.split()) > 50:
        return "complex"
    return "simple"

def adaptive_answer(question):
    complexity = classify_complexity(question)

    if complexity == "simple":
        # Fast model, direct answer
        return ollama.chat(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": question}]
        )
    else:
        # Reasoning model, more compute
        return ollama.chat(
            model="deepseek-r1:70b",
            messages=[{"role": "user", "content": question}]
        )
```

## ⚠️ Common Mistakes

| Mistake | Fix |
|---------|-----|
| Using temperature=0 for self-consistency | Use temperature=0.5-0.8 for diversity |
| Not extracting final answer from CoT | Parse "The answer is X" or similar pattern |
| R1 model too slow | Use smaller distill variant or Q4 quantization |
| Self-consistency on easy questions | Waste of compute; use for hard problems only |
| Ignoring thinking tokens | `<think>` blocks contain useful reasoning |

## 🔗 Quick Links
- [DeepSeek-R1 Paper](https://arxiv.org/abs/2401.02954)
- [Chain-of-Thought Paper](https://arxiv.org/abs/2201.11903)
- [Self-Consistency Paper](https://arxiv.org/abs/2203.11171)
- [Process Reward Models Paper](https://arxiv.org/abs/2305.20050)
- [ArmoRM Reward Model](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1)
