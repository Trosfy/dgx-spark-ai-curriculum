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
ollama stop qwen3:8b

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
    model: qwen3:8b

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
    model: qwen3:8b  # Not 70b
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
  - id: ollama:qwen3:8b
    config:
      apiHost: http://localhost:11434  # Add if needed
      temperature: 0.7

# Test manually first
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3:8b",
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
    model: qwen3:8b
EOF
```

---

## ❓ Frequently Asked Questions

### Setup & Environment

**Q: Do I need NeMo Guardrails AND Llama Guard?**

A: They complement each other:

| Tool | Best For |
|------|----------|
| **NeMo Guardrails** | Rule-based filtering, dialog flow control, topic restrictions |
| **Llama Guard** | ML-based content classification, nuanced safety detection |

**Recommendation**: Use both in production. NeMo for rules you know, Llama Guard for edge cases.

---

**Q: Why does Llama Guard need a specific prompt format?**

A: Llama Guard was trained on a specific conversation format. Using the wrong format gives unreliable results.

**Correct format**:
```
Task: Check if there is unsafe content in 'User' messages.

<BEGIN CONVERSATION>
User: [message]
<END CONVERSATION>
```

---

**Q: How much latency does safety filtering add?**

A: On DGX Spark with Ollama:

| Component | Latency |
|-----------|---------|
| Llama Guard classification | ~200-500ms |
| NeMo Guardrails (simple rules) | ~50-100ms |
| NeMo Guardrails (with LLM) | ~500-1000ms |

**Total overhead**: 250ms-1.5s per request depending on configuration.

**Tip**: Run safety checks in parallel with generation, cancel if unsafe.

---

### Concepts

**Q: What's the difference between prompt injection and jailbreaking?**

A:

| Prompt Injection | Jailbreaking |
|------------------|--------------|
| Inserting malicious instructions into prompts | Convincing the model to ignore its training |
| Often via external data (documents, URLs) | Usually via direct conversation |
| "Ignore above and do X" | "Pretend you're DAN who has no rules" |
| Exploits context confusion | Exploits role-play capability |

**Both** aim to bypass safety, but use different vectors.

---

**Q: Why can't I just tell the AI "refuse harmful requests"?**

A: LLMs don't follow instructions reliably for safety. They're trained to be helpful, which attackers exploit:

1. "Be helpful" + "How to hack" = AI tries to help with hacking
2. Prompt injection can override any instruction
3. Role-play bypasses can work: "As a security researcher..."

**You need external enforcement** (guardrails) that the AI can't override.

---

**Q: What's the difference between TruthfulQA and factual accuracy?**

A:

- **Factual accuracy**: Does the model know correct facts?
- **TruthfulQA**: Does the model resist giving plausible-sounding wrong answers?

TruthfulQA specifically tests questions where humans often believe false things (misconceptions). A model that always says "I don't know" would score higher than one that confidently gives wrong answers.

---

**Q: How is AI bias different from human bias?**

A: AI bias is **learned from data**, not innate:

1. AI has no beliefs or prejudices
2. AI reflects statistical patterns in training data
3. AI applies those patterns consistently (no individual variation)
4. AI bias is measurable and quantifiable
5. AI bias can be (partially) corrected through training

**Key difference**: AI bias is systematic. A biased human might sometimes act fairly. A biased AI is consistently biased.

---

### Additional Troubleshooting FAQs

**Q: My guardrails block too many legitimate requests (false positives)**

A: Tune your configuration:

1. **Reduce rule strictness**: Use more specific triggers
   ```colang
   # Too broad
   define user ask about security
     "security"

   # Better - more specific
   define user ask about hacking
     "How do I hack into"
     "Break into someone's"
   ```

2. **Add whitelists**: Explicitly allow safe variations
3. **Use confidence thresholds**: For Llama Guard, only block high-confidence unsafe

---

**Q: Llama Guard classifies everything as safe**

A: Check prompt format. The model is sensitive to exact formatting:

```python
# Wrong - may return "safe" for everything
response = ollama.chat(
    model="llama-guard3:8b",
    messages=[{"role": "user", "content": "Is this safe: How to hack?"}]
)

# Correct format
response = ollama.chat(
    model="llama-guard3:8b",
    messages=[{"role": "user", "content": """Task: Check if there is unsafe content in 'User' messages.

<BEGIN CONVERSATION>
User: How to hack into a computer?
<END CONVERSATION>"""}]
)
```

---

**Q: Red teaming found vulnerabilities - now what?**

A: Follow this process:

1. **Document each vulnerability**
   - Attack that worked
   - Category (injection, jailbreak, etc.)
   - Severity (what could go wrong)

2. **Prioritize by risk**
   - High: Could cause real harm
   - Medium: Reputation damage
   - Low: Minor policy violations

3. **Implement mitigations**
   - Add specific guardrail rules
   - Update Llama Guard prompts
   - Add output filtering

4. **Re-test to verify fix**

5. **Document in model card**

---

**Q: How do I know if my safety measures are good enough?**

A: There's no "good enough" - safety is continuous. But benchmarks help:

| Benchmark | Target |
|-----------|--------|
| TruthfulQA MC2 | > 0.5 (higher = more truthful) |
| Red team pass rate | > 90% attacks blocked |
| False positive rate | < 5% legitimate requests blocked |
| Bias disparity | < 10% difference across groups |

**Also**: Get external review. Your own red teaming has blind spots.

---

### Beyond the Basics

**Q: Should I use safety measures for internal tools?**

A: Yes, but maybe lighter:

| Context | Safety Level |
|---------|--------------|
| Public-facing chatbot | Full guardrails + Llama Guard + logging |
| Internal employee tool | Basic guardrails + logging |
| Personal development | Optional, but good practice |

**Key point**: Even internal tools can be abused or produce harmful outputs.

---

**Q: How does EU AI Act affect my model?**

A: Depends on risk classification:

| Your Model | Likely Classification | Requirements |
|------------|----------------------|--------------|
| General chatbot | Limited Risk | Transparency (tell users it's AI) |
| Medical advice | High Risk | Extensive documentation, testing, auditing |
| Employment screening | High Risk | Bias testing, human oversight |
| Creative writing | Minimal Risk | None specific |

**Check**: [artificialintelligenceact.eu](https://artificialintelligenceact.eu/) for current requirements.

---

**Q: Can I fine-tune safety into the model?**

A: Yes, but it's not enough alone:

| Approach | Pros | Cons |
|----------|------|------|
| RLHF/DPO safety training | Model inherently safer | Can be bypassed, costly |
| Constitutional AI | Self-critique | Not foolproof |
| External guardrails | Reliable, auditable | Latency, cost |

**Best practice**: Defense in depth. Fine-tune for safety AND use guardrails.

---

**Q: What about copyright/IP concerns?**

A: Growing area of concern:

1. **Training data**: Was copyrighted content used?
2. **Output**: Does the model reproduce copyrighted text?
3. **Style mimicry**: Can it copy specific artists?

**Mitigations**:
- Check outputs for verbatim copying
- Add guardrails against style-copying requests
- Document training data sources

---

**Q: How do I handle safety for multimodal models?**

A: Additional considerations:

| Modality | Safety Concerns |
|----------|-----------------|
| Image input | NSFW detection, deepfake source |
| Image output | NSFW generation, copyright |
| Audio input | Voice cloning source |
| Audio output | Deepfake voice generation |

**Tools**: Use specialized classifiers for each modality (not just Llama Guard).

---

## Still Stuck?

1. **Check Ollama logs**: `ollama logs`
2. **Check NeMo Guardrails docs**: [github.com/NVIDIA/NeMo-Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)
3. **Check lm-eval docs**: [github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
4. **Search with context**: Include "NeMo Guardrails" or "lm-eval" in error searches
