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

## 📞 Still Stuck?
- Check Ollama logs: `cat /tmp/ollama.log`
- Verify model works: `ollama run deepseek-r1:7b "test"`
- See [FAQ.md](./FAQ.md) for common questions
