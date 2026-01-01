# Module 4.2: AI Safety & Alignment - Troubleshooting Guide

## Quick Diagnostic

**Before diving into specific errors, check these:**

1. Is Ollama running? `curl http://localhost:11434/api/tags`
2. Are models pulled? `ollama list`
3. Check GPU memory: `nvidia-smi`
4. Restart Ollama: `pkill ollama && ollama serve &`

---

## Ollama Errors

### Error: `Connection refused` or `Ollama not running`

**Symptoms**:
```
httpx.ConnectError: [Errno 111] Connection refused
```

**Solutions**:
```bash
# Solution 1: Start Ollama
ollama serve &

# Solution 2: Check if port is in use
lsof -i :11434

# Solution 3: Kill and restart
pkill ollama
sleep 2
ollama serve &
```

---

### Error: `model not found: llama-guard3:8b`

**Symptoms**:
```
ollama._types.ResponseError: model 'llama-guard3:8b' not found
```

**Solution**:
```bash
ollama pull llama-guard3:8b
```

---

### Error: Ollama OOM during model load

**Symptoms**:
```
CUDA out of memory when loading model
```

**Solutions**:
```python
# Solution 1: Unload other models first
ollama stop llama3.1:8b

# Solution 2: Clear GPU memory
import torch, gc
torch.cuda.empty_cache()
gc.collect()

# Then try again
ollama pull llama-guard3:8b
```

---

## NeMo Guardrails Errors

### Error: `ModuleNotFoundError: No module named 'nemoguardrails'`

**Solution**:
```bash
pip install nemoguardrails>=0.8.0
```

---

### Error: `ValidationError` in config.yaml

**Symptoms**:
```
pydantic.error_wrappers.ValidationError: 1 validation error for RailsConfig
```

**Causes**: YAML syntax or missing required fields

**Solutions**:
```yaml
# Correct config.yaml format:
models:
  - type: main
    engine: ollama
    model: llama3.1:8b

# NOT:
models:
  type: main  # Wrong - should be a list
```

---

### Error: `Colang syntax error`

**Symptoms**:
```
nemoguardrails.colang.v2_x.parser.ColangSyntaxError
```

**Common fixes**:
```colang
# Wrong: Missing quotes
define user ask about hacking
  How do I hack

# Correct: Quoted examples
define user ask about hacking
  "How do I hack"
  "Write malware code"

# Wrong: Indentation
define flow check topic
user ask about hacking
bot refuse

# Correct: Proper indentation
define flow check topic
  user ask about hacking
  bot refuse
```

---

### Error: `Guardrails timeout`

**Symptoms**: Response takes >30 seconds or never returns

**Solutions**:
```python
# Solution 1: Use smaller model
config = RailsConfig.from_content("""
models:
  - type: main
    engine: ollama
    model: llama3.1:8b  # Not 70b
""")

# Solution 2: Increase timeout
config.streaming_timeout = 60
```

---

## Red Teaming Errors

### Error: `promptfoo: command not found`

**Solutions**:
```bash
# Option 1: Install via pip
pip install promptfoo

# Option 2: Install via npm
npm install -g promptfoo
```

---

### Error: Promptfoo tests all failing

**Causes**: Provider configuration wrong

**Solutions**:
```yaml
# Verify provider config
providers:
  - id: ollama:llama3.1:8b
    config:
      apiHost: http://localhost:11434  # Add if needed
      temperature: 0.7

# Test manually first
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Hello"
}'
```

---

### Error: `assert type 'llm-rubric' requires OpenAI`

**Cause**: LLM-based assertions need an evaluator model

**Solutions**:
```yaml
# Use non-LLM assertions
tests:
  - vars:
      prompt: "..."
    assert:
      - type: not-contains  # Works without LLM
        value: "hack"
      - type: contains
        value: "I can't"
      # Remove llm-rubric assertions or configure evaluator
```

---

## Benchmark Errors

### Error: `No module named 'lm_eval'`

**Solution**:
```bash
pip install lm-eval>=0.4.0
```

---

### Error: `Token has not been saved to git credential helper`

**Cause**: Gated model requires HuggingFace login

**Solutions**:
```bash
# Option 1: Login interactively
huggingface-cli login

# Option 2: Set token
export HF_TOKEN="your_token"

# Option 3: In Python
from huggingface_hub import login
login(token="your_token")
```

---

### Error: Benchmark OOM

**Symptoms**:
```
CUDA out of memory during evaluation
```

**Solutions**:
```bash
# Solution 1: Reduce batch size
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=bfloat16 \
    --tasks truthfulqa_mc2 \
    --batch_size 1  # Reduce from 8

# Solution 2: Use 4-bit quantization
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,load_in_4bit=True \
    --tasks truthfulqa_mc2
```

---

### Error: `Task 'xxx' not found`

**Cause**: Wrong task name

**Solutions**:
```bash
# List available tasks
lm_eval --tasks list

# Correct task names
lm_eval --tasks truthfulqa_mc2  # Not truthfulqa
lm_eval --tasks bbq             # Bias benchmark
```

---

## Bias Evaluation Errors

### Error: `No module named 'fairlearn'`

**Solution**:
```bash
pip install fairlearn
```

---

### Error: `MetricFrame requires same length arrays`

**Cause**: Mismatched data lengths

**Solution**:
```python
# Ensure same length
assert len(y_true) == len(y_pred) == len(sensitive_features)

# Debug
print(f"y_true: {len(y_true)}, y_pred: {len(y_pred)}, features: {len(sensitive_features)}")
```

---

## Model Card Errors

### Error: `HfApi.upload_file` permission denied

**Cause**: Token doesn't have write permission

**Solutions**:
```python
# Generate new token with write access at:
# https://huggingface.co/settings/tokens

# Then login
from huggingface_hub import login
login(token="hf_xxx", add_to_git_credential=True)
```

---

### Error: `Repository not found`

**Cause**: Repository doesn't exist

**Solution**:
```python
from huggingface_hub import create_repo

# Create repository first
create_repo("your-username/model-name", exist_ok=True)

# Then push
api.upload_file(...)
```

---

## Reset Procedures

### Full Safety Environment Reset

```bash
# 1. Kill Ollama
pkill ollama

# 2. Clear Ollama cache (if needed)
rm -rf ~/.ollama/models/*

# 3. Restart container
exit
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.ollama:/root/.ollama \
    --ipc=host \
    nvcr.io/nvidia/pytorch:25.11-py3

# 4. Reinstall
pip install nemoguardrails>=0.8.0 lm-eval fairlearn

# 5. Restart Ollama and pull models
ollama serve &
ollama pull llama-guard3:8b
```

### Reset Guardrails Config

```bash
# If config is corrupted
rm -rf /workspace/guardrails_config
mkdir /workspace/guardrails_config

# Recreate minimal config
cat > /workspace/guardrails_config/config.yaml << 'EOF'
models:
  - type: main
    engine: ollama
    model: llama3.1:8b
EOF
```

---

## Still Stuck?

1. **Check Ollama logs**: `ollama logs`
2. **Check NeMo Guardrails docs**: [github.com/NVIDIA/NeMo-Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)
3. **Check lm-eval docs**: [github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
4. **Search with context**: Include "NeMo Guardrails" or "lm-eval" in error searches
