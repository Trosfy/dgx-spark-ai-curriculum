# Module 3.4: Test-Time Compute & Reasoning - Quick Reference

## 🚀 Essential Commands

### Ollama Setup for Reasoning Models (2025)
```bash
# Primary reasoning model (Tier 1)
ollama pull qwq:32b           # SOTA reasoning (~20GB, 79.5% AIME)

# DeepSeek-R1 distillations
ollama pull deepseek-r1:8b    # SOTA 8B reasoning (matches Qwen3-235B!)
ollama pull deepseek-r1:70b   # Frontier reasoning (~45GB)

# Comparison/baseline models
ollama pull qwen3:8b          # Fast general purpose with /think mode
ollama pull qwen3:32b         # Best general purpose
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

### DGX Spark Reasoning Model Performance (2025)
| Model | Quantization | Memory | Decode tok/s | AIME Score |
|-------|--------------|--------|--------------|------------|
| QwQ-32B | Q4_K_M | ~20GB | ~28 | 79.5% |
| DeepSeek-R1-8B | Q4_K_M | ~5GB | ~45 | 72.6% |
| DeepSeek-R1-32B | Q4 | ~20GB | ~30 | 76.8% |
| DeepSeek-R1-70B | Q4 | ~45GB | ~20 | 79.8% |
| Qwen3-32B (/think) | Q4_K_M | ~20GB | ~35 | N/A |

### CoT Accuracy Improvements
| Dataset | Without CoT | With CoT | Improvement |
|---------|-------------|----------|-------------|
| GSM8K (math) | ~55% | ~75% | +20% |
| SVAMP (word problems) | ~60% | ~80% | +20% |
| ARC-C (science) | ~70% | ~80% | +10% |

## 🔧 Common Patterns

### Pattern: Zero-Shot Chain-of-Thought
```python
# Option 1: Traditional CoT prompt
prompt = f"""
{question}

Let's think step by step:
"""

# Option 2: Qwen3 hybrid thinking mode (recommended)
# Just add /think to enable extended reasoning
prompt = f"/think {question}"

response = ollama.chat(
    model="qwen3:8b",  # or qwen3:32b for better quality
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

### Pattern: QwQ/DeepSeek-R1 Reasoning
```python
import ollama
import re

# QwQ-32B - primary reasoning model (always reasons)
response = ollama.chat(
    model="qwq:32b",
    messages=[{
        "role": "user",
        "content": "What is the derivative of x³ + 2x² - 5x + 3?"
    }]
)

# DeepSeek-R1 includes <think>...</think> blocks
response = ollama.chat(
    model="deepseek-r1:8b",  # SOTA 8B, matches larger models!
    messages=[{
        "role": "user",
        "content": "Solve: If 3x + 7 = 22, what is x?"
    }]
)

# Extract just the answer (remove thinking)
answer = re.sub(r'<think>.*?</think>', '', response['message']['content'], flags=re.DOTALL)
print("Clean answer:", answer.strip())
```

### Pattern: Best-of-N with Reward Model
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load reward model (Skywork is current SOTA)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "Skywork/Skywork-Reward-Gemma-2-27B-v0.2",  # or RLHFlow/ArmoRM-Llama3-8B-v0.1
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
reward_tokenizer = AutoTokenizer.from_pretrained("Skywork/Skywork-Reward-Gemma-2-27B-v0.2")

def best_of_n(prompt, n=5, temperature=0.7):
    candidates = []
    for _ in range(n):
        response = ollama.chat(
            model="qwen3:8b",  # Fast generation
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
        # Fast model, direct answer (no thinking)
        return ollama.chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": f"/no_think {question}"}]
        )
    else:
        # Reasoning model, extended compute
        return ollama.chat(
            model="qwq:32b",  # or deepseek-r1:8b for efficiency
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
