# Module 4.6: Capstone Project - Lab Preparation Guide

## Time Estimates

| Phase | Duration | Key Activities |
|-------|----------|----------------|
| Planning (Lab 4.6.0-4.6.1) | 1 week | Proposal, architecture |
| Foundation (Lab 4.6.2-4.6.5) | 2 weeks | Core implementation |
| Evaluation (Lab 4.6.6) | 1 week | Benchmarks, safety |
| Documentation (Lab 4.6.7) | 1 week | Report, demo, presentation |
| **Total** | **5-6 weeks** | ~40-50 hours |

---

## Project Option Requirements

### Option A: Domain AI Assistant

| Component | Storage | GPU Memory | Time |
|-----------|---------|------------|------|
| Base model (70B 4-bit) | ~40 GB | ~40 GB | - |
| LoRA adapters | ~500 MB | - | 4-6 hrs training |
| RAG database | ~1-5 GB | ~2 GB | - |
| Total | ~50 GB | ~45 GB | 10+ hrs |

### Option B: Document Intelligence

| Component | Storage | GPU Memory | Time |
|-----------|---------|------------|------|
| VLM (Qwen2-VL-7B) | ~15 GB | ~20 GB | - |
| OCR models | ~2 GB | ~4 GB | - |
| Document store | ~5 GB | ~2 GB | - |
| Total | ~25 GB | ~26 GB | 8+ hrs |

### Option C: Agent Swarm

| Component | Storage | GPU Memory | Time |
|-----------|---------|------------|------|
| Base model (8-70B) | ~8-40 GB | ~10-40 GB | - |
| Tools/environments | ~2 GB | ~2 GB | - |
| Safety guardrails | ~500 MB | ~2 GB | - |
| Total | ~15-45 GB | ~14-44 GB | 15+ hrs |

### Option D: Training Pipeline

| Component | Storage | GPU Memory | Time |
|-----------|---------|------------|------|
| Base model (8-70B) | ~8-40 GB | ~40-80 GB | - |
| Training datasets | ~5-20 GB | - | - |
| Checkpoints | ~10-50 GB | - | - |
| Total | ~30-110 GB | ~40-80 GB | 20+ hrs |

---

## Environment Setup

### 1. Start NGC Container (Full Resources)

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v /var/run/docker.sock:/var/run/docker.sock \
    --ipc=host \
    --shm-size=16g \
    -p 5000:5000 \
    -p 7860:7860 \
    -p 8501:8501 \
    -p 8000:8000 \
    nvcr.io/nvidia/pytorch:25.11-py3
```

### 2. Install Core Dependencies

```bash
# Core ML stack
pip install torch transformers accelerate bitsandbytes
pip install peft trl datasets evaluate

# RAG & Agents
pip install langchain langgraph langchain-community
pip install chromadb ollama

# Safety
pip install nemoguardrails

# Demo & MLOps
pip install gradio streamlit mlflow wandb

# Utilities
pip install pypdf python-docx pillow
```

### 3. Download Base Models (2025 Tier 1)

```bash
# For all options - chat model
ollama pull qwen3:8b               # Fast development
ollama pull qwen3:32b              # Production quality
ollama pull qwen3-embedding:8b     # RAG embeddings

# Option A: Large fine-tuning base
huggingface-cli download Qwen/Qwen3-32B-Instruct

# Option B: Vision-language model
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct

# Option C: Agents (with tool calling support)
ollama pull nemotron-3-nano        # Fast agent model
# WARNING: Do NOT use deepseek-r1 for agents - no tool calling!

# Option D: Training-specific
huggingface-cli download Qwen/Qwen3-8B-Instruct
```

### 4. Verify Setup

```python
import torch
import transformers
import gradio
import streamlit
import mlflow

print(f"PyTorch {torch.__version__}")
print(f"Transformers {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")

# Check Ollama
import ollama
models = ollama.list()
print(f"Ollama models: {[m['name'] for m in models['models']]}")
```

**Expected output**:
```
PyTorch 2.x.x
Transformers 4.4x.x
CUDA available: True
GPU: NVIDIA GH200 480GB
Memory: 128 GB
Ollama models: ['qwen3:8b', 'qwen3-embedding:8b']
```

---

## Pre-Lab Checklists

### Lab 4.6.0: Project Kickoff

- [ ] Reviewed all four project options
- [ ] Identified preferred option based on interests
- [ ] Assessed prerequisite completion (Domains 1-4)
- [ ] Reviewed example implementations in `examples/`

**Preparation**:
```bash
# Review example implementations
ls -la /workspace/domain-4-production-ai/module-4.6-capstone-project/examples/

# Read README for each option
cat examples/option-a-assistant/README.md
```

---

### Lab 4.6.1: Project Planning

- [ ] Project option selected
- [ ] Use case defined (specific domain/problem)
- [ ] Architecture diagram drafted
- [ ] Data sources identified
- [ ] Timeline created (5-6 weeks)

**Templates ready**:
- `templates/project-proposal.md`
- `templates/technical-report.md`

---

### Lab 4.6.2-4.6.5: Implementation (Option-Specific)

#### Option A: AI Assistant

- [ ] Domain corpus collected (PDFs, docs, web pages)
- [ ] Fine-tuning dataset prepared (500+ examples)
- [ ] ChromaDB configured
- [ ] NeMo Guardrails config created
- [ ] HuggingFace token set for Llama access

**Quick Test**:
```python
# Verify 70B loads in 4-bit
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",  # Test with 8B first
    quantization_config=bnb_config,
    device_map="auto"
)
print("Model loaded!")
```

#### Option B: Document Intelligence

- [ ] Sample documents collected (PDFs, images)
- [ ] VLM model downloaded (Qwen2-VL)
- [ ] OCR dependencies installed
- [ ] Extraction schema defined
- [ ] Output format specified (JSON, CSV)

**Quick Test**:
```python
from transformers import Qwen2VLForConditionalGeneration
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("VLM loaded!")
```

#### Option C: Agent Swarm

- [ ] Agent roles defined (4+ agents)
- [ ] Tools implemented
- [ ] Communication protocol designed
- [ ] Safety policies documented
- [ ] LangGraph/LangChain installed

**Quick Test**:
```python
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph

llm = Ollama(model="qwen3:8b")  # or nemotron-3-nano for agents
print(llm.invoke("Hello, I'm an agent!"))
```

#### Option D: Training Pipeline

- [ ] Training data collected and cleaned
- [ ] Data format standardized
- [ ] Baseline model selected
- [ ] Evaluation benchmarks chosen
- [ ] MLflow/W&B configured

**Quick Test**:
```python
from datasets import Dataset
from trl import SFTTrainer

# Create dummy dataset
data = [{"text": "Hello\nWorld"}]
dataset = Dataset.from_list(data)
print(f"Dataset ready: {len(dataset)} examples")
```

---

### Lab 4.6.6: Evaluation Framework

- [ ] Benchmark suite selected (MMLU, etc.)
- [ ] Custom evaluation metrics defined
- [ ] Safety test prompts prepared
- [ ] Comparison baselines ready
- [ ] lm-evaluation-harness installed

**Quick Test**:
```bash
# Verify lm-eval works
lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks hellaswag \
    --limit 10
```

---

### Lab 4.6.7: Documentation Guide

- [ ] Technical report outline complete
- [ ] Screenshots captured
- [ ] Demo video recorded
- [ ] Model card drafted
- [ ] Presentation slides started

**Templates ready**:
- `templates/technical-report.md`
- `templates/model-card.md`
- `templates/presentation-outline.md`

---

## Common Setup Mistakes

| Mistake | Consequence | Prevention |
|---------|-------------|------------|
| Scope too large | Won't finish | Start with MVP |
| No baseline metrics | Can't measure improvement | Evaluate base model first |
| Skipping safety | Unsafe system | Include guardrails from start |
| Late documentation | Rushed/incomplete | Document as you build |
| No version control | Lost work | Git commit frequently |
| Wrong model size | OOM or too slow | Test with smaller model first |

---

## Expected Directory Structure

```
/workspace/
├── capstone/
│   ├── README.md                # Project overview
│   ├── requirements.txt         # Dependencies
│   ├── Dockerfile              # For deployment
│   │
│   ├── src/                    # Source code
│   │   ├── __init__.py
│   │   ├── model/              # Model code
│   │   ├── data/               # Data processing
│   │   ├── eval/               # Evaluation
│   │   └── api/                # API endpoints
│   │
│   ├── notebooks/              # Development notebooks
│   │   ├── 01-data-prep.ipynb
│   │   ├── 02-training.ipynb
│   │   └── 03-evaluation.ipynb
│   │
│   ├── demo/                   # Demo application
│   │   ├── app.py              # Gradio/Streamlit app
│   │   └── assets/
│   │
│   ├── configs/                # Configuration files
│   │   ├── model_config.yaml
│   │   └── guardrails/
│   │
│   ├── data/                   # Data (gitignored)
│   │   ├── raw/
│   │   ├── processed/
│   │   └── embeddings/
│   │
│   ├── models/                 # Saved models (gitignored)
│   │   ├── checkpoints/
│   │   └── final/
│   │
│   ├── results/                # Evaluation results
│   │   ├── benchmarks/
│   │   └── safety/
│   │
│   └── docs/                   # Documentation
│       ├── technical-report.md
│       ├── model-card.md
│       └── presentation.pdf
```

---

## Quick Start Commands

```bash
# Create project structure
cd /workspace
mkdir -p capstone/{src/{model,data,eval,api},notebooks,demo/assets,configs/guardrails,data/{raw,processed,embeddings},models/{checkpoints,final},results/{benchmarks,safety},docs}

# Initialize git
cd capstone
git init
echo "data/\nmodels/\n__pycache__/\n*.pyc\n.env" > .gitignore

# Create requirements.txt
cat > requirements.txt << 'EOF'
torch>=2.0.0
transformers>=4.40.0
accelerate>=0.25.0
bitsandbytes>=0.42.0
peft>=0.8.0
trl>=0.7.0
datasets>=2.16.0
langchain>=0.1.0
langgraph>=0.0.20
chromadb>=0.4.22
ollama>=0.1.0
gradio>=4.0.0
streamlit>=1.30.0
mlflow>=2.10.0
wandb>=0.16.0
lm-eval>=0.4.0
nemoguardrails>=0.6.0
EOF

# Install dependencies
pip install -r requirements.txt

# Start Ollama (2025 Tier 1)
ollama serve &
sleep 5
ollama pull qwen3:8b

# Create initial notebook
touch notebooks/01-exploration.ipynb

echo "Project structure ready!"
```

---

## Resource Budgeting

For DGX Spark (128GB unified memory):

| Concurrent Tasks | Memory Allocation |
|-----------------|-------------------|
| 70B inference (4-bit) | ~40 GB |
| RAG embeddings | ~5 GB |
| ChromaDB | ~5 GB |
| Safety guardrails | ~5 GB |
| Demo app overhead | ~3 GB |
| **Available headroom** | ~70 GB |

**Tip**: You can run 70B inference AND RAG simultaneously on DGX Spark.

---

## Milestone Schedule

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1 | Proposal approved | `project-proposal.md` |
| 2 | Data ready, MVP working | Basic demo |
| 3 | Core features complete | Functional system |
| 4 | Integration done | End-to-end working |
| 5 | Evaluation complete | Benchmark results |
| 6 | Documentation done | All deliverables |

---

## Next Steps

After preparation:
1. Complete Lab 4.6.0 (Kickoff)
2. Submit proposal using `templates/project-proposal.md`
3. Begin implementation with your chosen option
