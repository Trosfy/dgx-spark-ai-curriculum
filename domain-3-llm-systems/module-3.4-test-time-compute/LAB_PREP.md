# Module 3.4: Test-Time Compute & Reasoning - Lab Preparation Guide

## ⏱️ Time Estimates
| Lab | Preparation | Execution | Total |
|-----|-------------|-----------|-------|
| Lab 3.4.1: CoT Workshop | 10 min | 2 hr | 2.2 hr |
| Lab 3.4.2: Self-Consistency | 5 min | 1.5 hr | 1.6 hr |
| Lab 3.4.3: DeepSeek-R1 | 15 min | 2 hr | 2.3 hr |
| Lab 3.4.4: R1 Comparison | 10 min | 1.5 hr | 1.7 hr |
| Lab 3.4.5: Best-of-N | 20 min | 2 hr | 2.3 hr |
| Lab 3.4.6: Reasoning Pipeline | 15 min | 2 hr | 2.3 hr |

## 📦 Required Downloads

### Ollama Models (2025)
```bash
# Start Ollama
ollama serve &

# Primary reasoning model (Tier 1)
ollama pull qwq:32b           # SOTA reasoning (~20GB, 79.5% AIME)

# DeepSeek-R1 distillations
ollama pull deepseek-r1:8b    # SOTA 8B (~5GB, matches Qwen3-235B!)
ollama pull deepseek-r1:70b   # Frontier reasoning (~45GB with Q4)

# Comparison/baseline models
ollama pull qwen3:8b          # Fast with hybrid /think mode (~5GB)
ollama pull qwen3:32b         # Primary teaching model (~20GB)
```

### Reward Model (Lab 3.4.5)
```python
# Pre-download reward model
from huggingface_hub import snapshot_download
snapshot_download("RLHFlow/ArmoRM-Llama3-8B-v0.1")
```

## 🔧 Environment Setup

### 1. Start Container
```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $HOME/.ollama:/root/.ollama \
    --ipc=host \
    --network=host \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

### 2. Install Dependencies
```bash
pip install ollama openai sentence-transformers
```

### 3. Verify Setup
```python
import ollama

# Check available models
models = ollama.list()
print("Available models:", [m['name'] for m in models['models']])

# Test reasoning model
response = ollama.chat(
    model="qwq:32b",  # or deepseek-r1:8b for faster testing
    messages=[{"role": "user", "content": "What is 2+2? Think through it."}]
)
print(response['message']['content'])

# Test Qwen3 hybrid thinking mode
response = ollama.chat(
    model="qwen3:8b",
    messages=[{"role": "user", "content": "/think What is 2+2?"}]
)
print(response['message']['content'])
```

## ✅ Pre-Lab Checklists

### Lab 3.4.1-3.4.2: CoT and Self-Consistency
- [ ] Ollama running
- [ ] Any 8B model available
- [ ] Test prompts ready (in `data/`)

### Lab 3.4.3-3.4.4: Reasoning Models (QwQ/R1)
- [ ] QwQ-32B or DeepSeek-R1 model pulled
- [ ] At least 20GB GPU memory free (32B) or 50GB (70B)
- [ ] GSM8K sample data available

### Lab 3.4.5: Best-of-N
- [ ] Reward model downloaded
- [ ] transformers installed
- [ ] At least 25GB GPU memory free

### Lab 3.4.6: Reasoning Pipeline
- [ ] Both fast and reasoning models available
- [ ] Complexity classifier ready
- [ ] Caching mechanism understood

## 📁 Expected File Structure
```
/workspace/module-3.4-test-time-compute/
├── labs/
│   └── lab-3.4.1-cot-workshop.ipynb
├── data/
│   ├── gsm8k_sample.json
│   └── test_problems.json
├── scripts/
│   ├── reasoning_utils.py
│   └── reward_models.py
└── outputs/
```

## ⚡ Quick Start Commands
```bash
cd /workspace
pip install ollama openai
ollama pull qwq:32b          # Primary reasoning model
ollama pull deepseek-r1:8b   # Efficient reasoning alternative
python -c "import ollama; print('✅ Ready!')"
```
