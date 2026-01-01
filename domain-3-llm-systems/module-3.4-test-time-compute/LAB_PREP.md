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

### Ollama Models
```bash
# Start Ollama
ollama serve &

# DeepSeek-R1 models
ollama pull deepseek-r1:7b    # Fast testing (~14GB)
ollama pull deepseek-r1:70b   # Best quality (~45GB with Q4)

# Comparison models
ollama pull llama3.1:8b
ollama pull llama3.1:70b
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
    model="deepseek-r1:7b",
    messages=[{"role": "user", "content": "What is 2+2?"}]
)
print(response['message']['content'])
```

## ✅ Pre-Lab Checklists

### Lab 3.4.1-3.4.2: CoT and Self-Consistency
- [ ] Ollama running
- [ ] Any 8B model available
- [ ] Test prompts ready (in `data/`)

### Lab 3.4.3-3.4.4: DeepSeek-R1
- [ ] DeepSeek-R1 model pulled
- [ ] At least 15GB GPU memory free (7B) or 50GB (70B)
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
ollama pull deepseek-r1:7b
python -c "import ollama; print('✅ Ready!')"
```
