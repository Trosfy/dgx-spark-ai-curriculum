# Module 3.4: Test-Time Compute & Reasoning - Troubleshooting Guide

## 🔍 Quick Diagnostic
1. Check Ollama: `curl http://localhost:11434/api/tags`
2. Check GPU memory: `nvidia-smi`
3. Check model availability: `ollama list`

---

## 🚨 Common Errors

### Ollama Connection Errors

#### Error: `Connection refused`
**Solution**:
```bash
# Start Ollama server
ollama serve

# Or run in background
nohup ollama serve &> /tmp/ollama.log &
```

---

#### Error: `Model not found`
**Solution**:
```bash
# Pull the model first
ollama pull deepseek-r1:70b

# Check available models
ollama list
```

---

### Reasoning Model Errors

#### Error: R1 model is very slow
**Cause**: Reasoning models generate many thinking tokens.

**Solution**:
```python
# Use smaller model for testing
ollama.chat(model="deepseek-r1:7b", ...)

# Or limit max tokens
ollama.chat(
    model="deepseek-r1:70b",
    messages=[...],
    options={"num_predict": 500}  # Limit thinking
)
```

---

#### Error: `<think>` tokens in output
**Cause**: Normal for reasoning models—they show their work.

**Solution**:
```python
import re

# Remove thinking tokens from output
content = response['message']['content']
clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
print(clean.strip())
```

---

### Self-Consistency Errors

#### Error: Answers never agree
**Cause**: Temperature too high or question too hard.

**Solution**:
```python
# Lower temperature for more consistency
options={"temperature": 0.5}  # Instead of 0.8

# Or increase sample count
n_samples = 10  # Instead of 5
```

---

### Reward Model Errors

#### Error: Reward model OOM
**Solution**:
```python
# Use smaller batch size
# Score one at a time instead of batching

# Or use smaller reward model
reward_model = "OpenAssistant/reward-model-deberta-v3-large"  # Smaller
```

---

## 🔄 Reset Procedures

### Restart Ollama
```bash
pkill ollama
ollama serve &
```

### Clear Memory
```python
import torch, gc
torch.cuda.empty_cache()
gc.collect()
```

---

## ❓ Frequently Asked Questions

**Q: When should I use Chain-of-Thought vs a reasoning model?**

**A**:
- **CoT prompting**: Works with any model, no additional compute for model itself
- **Reasoning model (R1)**: Purpose-built for reasoning, higher quality on hard problems

Use CoT first. If results aren't good enough, try a reasoning model.

---

**Q: How many samples for self-consistency?**

**A**:
- **5 samples**: Good balance (default)
- **3 samples**: Fast, less reliable
- **10+ samples**: High confidence, more compute

Odd numbers work best for majority voting.

---

**Q: Why do reasoning models use `<think>` tokens?**

**A**: The `<think>` tokens are explicit "thinking out loud" markers. The model was trained to:
1. Generate reasoning in `<think>` blocks
2. Then provide the final answer

This makes the reasoning visible and debuggable.

---

**Q: R1 model is too slow. What can I do?**

**A**:
1. Use smaller variant (7B instead of 70B)
2. Use higher quantization (Q4_K_M)
3. Limit max_tokens to cap thinking length
4. For simple questions, use regular model

---

**Q: Self-consistency gives wrong consensus. Why?**

**A**: If all paths lead to wrong answer, self-consistency won't help. Try:
1. Better prompting (more examples)
2. Stronger base model
3. CoT + self-consistency together

---

**Q: Can I combine CoT with reasoning models?**

**A**: Yes! R1 already does CoT-like reasoning, but you can add few-shot examples to guide the format. Generally, R1's built-in reasoning is sufficient.

---

**Q: How do I evaluate reasoning quality?**

**A**:
1. **Accuracy**: Does it get the right answer?
2. **Reasoning validity**: Are the steps correct?
3. **Efficiency**: How many tokens to reach answer?

See Lab 3.4.4 for evaluation framework.

---

## 📞 Still Stuck?
- Check Ollama logs: `cat /tmp/ollama.log`
- Verify model works: `ollama run deepseek-r1:7b "test"`
- See module documentation for additional resources
