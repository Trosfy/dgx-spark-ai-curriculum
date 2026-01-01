# Module 3.4: Test-Time Compute & Reasoning - Quickstart

## ⏱️ Time: ~5 minutes

## 🎯 What You'll Build
Implement Chain-of-Thought prompting and see immediate accuracy improvement.

## ✅ Before You Start
- [ ] DGX Spark with Ollama running
- [ ] Model available (any chat model)

## 🚀 Let's Go!

### Step 1: Start Ollama (if not running)
```bash
ollama serve &
ollama pull qwen3:8b
```

### Step 2: Test WITHOUT Chain-of-Thought
```python
import ollama

question = """
A store sells apples for $2 each and oranges for $3 each.
If I buy 5 apples and some oranges and spend $25 total,
how many oranges did I buy?
"""

# Direct answer (no reasoning)
response = ollama.chat(
    model="qwen3:8b",
    messages=[{"role": "user", "content": question}]
)
print("Direct Answer:", response['message']['content'])
```

### Step 3: Test WITH Chain-of-Thought
```python
cot_prompt = question + "\n\nLet's think step by step:"

response = ollama.chat(
    model="qwen3:8b",
    messages=[{"role": "user", "content": cot_prompt}]
)
print("\nWith Chain-of-Thought:")
print(response['message']['content'])
```

### Step 4: Compare Results

**Expected output**:
```
Direct Answer: 5 oranges

With Chain-of-Thought:
Let's think step by step:

1. First, calculate the cost of 5 apples:
   5 apples × $2 = $10

2. Subtract from total to find orange spending:
   $25 - $10 = $15

3. Calculate number of oranges:
   $15 ÷ $3 = 5 oranges

The answer is 5 oranges.
```

## 🎉 You Did It!

You just used Chain-of-Thought prompting! While this simple example got both right, CoT dramatically improves accuracy on complex problems:
- **GSM8K math**: +15-20% accuracy
- **Code generation**: Better bug-free rate
- **Logic puzzles**: Much higher success

In the full module, you'll learn:
- **Self-consistency**: Multiple reasoning paths + voting
- **DeepSeek-R1**: Models trained to reason with `<think>` tokens
- **Best-of-N**: Reward models to pick best response
- **Adaptive routing**: Send complex queries to reasoning models

## ▶️ Next Steps
1. **Try harder problems**: See [Lab 3.4.1](./labs/lab-3.4.1-cot-workshop.ipynb)
2. **Run DeepSeek-R1**: See [Lab 3.4.3](./labs/lab-3.4.3-deepseek-r1.ipynb)
3. **Full setup**: Start with [LAB_PREP.md](./LAB_PREP.md)
