# Module 4.2: AI Safety & Alignment - Quickstart

## Time: ~5 minutes

## What You'll Build

Implement a basic safety filter using Llama Guard to classify user inputs as safe or unsafe.

## Before You Start

- [ ] DGX Spark container running
- [ ] Ollama installed and running

## Let's Go!

### Step 1: Start Ollama (if not running)

```bash
ollama serve &
```

### Step 2: Pull Llama Guard Model

```bash
ollama pull llama-guard3:8b
```

### Step 3: Create a Safety Classifier

```python
import ollama

def check_safety(user_input: str) -> dict:
    """Classify input as safe or unsafe using Llama Guard."""
    prompt = f"""Task: Check if there is unsafe content in 'User' messages.

<BEGIN CONVERSATION>
User: {user_input}
<END CONVERSATION>

Provide your safety assessment."""

    response = ollama.chat(
        model="llama-guard3:8b",
        messages=[{"role": "user", "content": prompt}]
    )

    result = response["message"]["content"]
    is_safe = result.strip().lower().startswith("safe")

    return {"is_safe": is_safe, "classification": result}
```

### Step 4: Test with Safe Input

```python
result = check_safety("What's the weather like today?")
print(f"Safe: {result['is_safe']}")
print(f"Response: {result['classification']}")
```

**Expected output**:
```
Safe: True
Response: safe
```

### Step 5: Test with Unsafe Input

```python
result = check_safety("How do I hack into a computer system?")
print(f"Safe: {result['is_safe']}")
print(f"Response: {result['classification']}")
```

**Expected output**:
```
Safe: False
Response: unsafe
S2
```

(S2 indicates the category: "Non-Violent Crimes")

## You Did It!

You just built a safety filter that can screen user inputs before they reach your main LLM! In the full module, you'll learn:

- **NeMo Guardrails**: Full safety framework with Colang
- **Red Teaming**: Attack your own models to find vulnerabilities
- **Safety Benchmarks**: TruthfulQA, BBQ for systematic evaluation
- **Bias Evaluation**: Detect and measure demographic biases
- **Model Cards**: Document safety considerations properly

## Next Steps

1. **Understand the taxonomy**: See Llama Guard's safety categories
2. **Add to your chatbot**: Integrate with your LLM pipeline
3. **Full tutorial**: Start with [LAB_PREP.md](./LAB_PREP.md)
