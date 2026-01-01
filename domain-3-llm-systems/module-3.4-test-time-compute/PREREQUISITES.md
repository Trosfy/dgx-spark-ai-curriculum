# Module 3.4: Test-Time Compute & Reasoning - Prerequisites Check

## 🎯 Purpose
This module builds on deployment skills to explore reasoning strategies. Use this self-check.

## ⏱️ Estimated Time
- **If all prerequisites met**: Jump to [QUICKSTART.md](./QUICKSTART.md)
- **If 1-2 gaps**: ~1-2 hours of review

---

## Required Skills

### 1. Ollama: Model Interaction
**Can you do this?**
```python
# Load a model and run inference with Ollama
```

<details>
<summary>✅ Check your answer</summary>

```python
import ollama

response = ollama.chat(
    model="qwen3:8b",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response['message']['content'])
```

</details>

**Not ready?** Review: [Module 3.3: Deployment](../module-3.3-deployment/)

---

### 2. Prompting: Basic Prompt Engineering
**Can you answer this?**
> What's the difference between zero-shot and few-shot prompting?

<details>
<summary>✅ Check your answer</summary>

- **Zero-shot**: No examples, just the instruction
- **Few-shot**: Include 2-5 examples before the actual question

Few-shot helps the model understand the expected format and reasoning pattern.

</details>

---

### 3. Temperature: Sampling Understanding
**Can you answer this?**
> What happens when you increase temperature from 0.0 to 1.0?

<details>
<summary>✅ Check your answer</summary>

- **Temperature 0.0**: Deterministic (always picks most likely token)
- **Temperature 1.0**: More random (samples from full distribution)
- **For reasoning**: Use 0 for consistent answers, 0.5-0.8 for self-consistency diversity

</details>

---

## Ready?
- [ ] I can run Ollama models
- [ ] I understand prompting basics
- [ ] I know how temperature affects generation

**All boxes checked?** → Start with [QUICKSTART.md](./QUICKSTART.md)!
