# Module 4.6: Capstone Project - Prerequisites Check

## Purpose

The capstone project synthesizes skills from the entire curriculum. This self-assessment helps you identify any gaps to address before starting.

## Time Estimates

- **If all prerequisites met**: Start immediately
- **If 1-2 gaps**: Review specific modules (2-4 hours each)
- **If multiple gaps**: Complete prerequisite modules first

---

## Domain 1: Platform Foundations

### 1.1 DGX Spark Platform

**Can you do this?**

```python
# Check GPU memory and run a simple tensor operation
import torch

# Get device info
# Write your code here...
```

<details>
<summary>Check your answer</summary>

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")

# Create tensor on GPU
x = torch.randn(1000, 1000, device="cuda")
print(f"Tensor shape: {x.shape}, device: {x.device}")
```

**Key points:**
- Use `torch.cuda.is_available()` to check GPU
- DGX Spark has 128GB unified memory
- Use `device="cuda"` for GPU tensors

</details>

**Not ready?** Review: Module 1.1, Notebook 01

---

### 1.5 Neural Networks

**Can you explain this?**

What happens during backpropagation? Why do we need it for training?

<details>
<summary>Check your answer</summary>

**Backpropagation**:
1. Forward pass: Compute predictions
2. Compute loss: Compare predictions to targets
3. Backward pass: Compute gradients (∂loss/∂weights)
4. Update weights: Move in direction that reduces loss

We need it because:
- Models have millions of parameters
- Manually computing gradients is impossible
- Automatic differentiation makes training feasible

</details>

**Not ready?** Review: Module 1.5, particularly gradient descent concepts

---

## Domain 2: Deep Learning Frameworks

### 2.1 PyTorch

**Can you do this?**

```python
# Load a HuggingFace model with custom config
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Llama-3.1-8B with 4-bit quantization
# Write your code here...
```

<details>
<summary>Check your answer</summary>

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
```

**Key points:**
- Use `BitsAndBytesConfig` for quantization
- `device_map="auto"` for automatic placement
- Always load tokenizer with model

</details>

**Not ready?** Review: Module 2.5 (HuggingFace Ecosystem)

---

### 2.3 Transformers

**Do you understand?**

| Term | Your Definition |
|------|-----------------|
| Attention | |
| Self-attention | |
| Multi-head attention | |
| KV cache | |

<details>
<summary>Check definitions</summary>

| Term | Definition |
|------|------------|
| Attention | Mechanism to focus on relevant parts of input |
| Self-attention | Query, Key, Value all come from same sequence |
| Multi-head attention | Multiple attention heads learn different patterns |
| KV cache | Cache past key-value pairs to avoid recomputation during generation |

</details>

**Not ready?** Review: Module 2.3, ELI5.md for intuitive understanding

---

## Domain 3: LLM Systems

### 3.1 Fine-tuning

**Can you do this?**

```python
# Set up LoRA config for fine-tuning
from peft import LoraConfig

# Create config for Llama model
# Write your code here...
```

<details>
<summary>Check your answer</summary>

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=16,                    # Rank
    lora_alpha=32,           # Scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

**Key points:**
- `r` controls number of trainable parameters
- `target_modules` depends on model architecture
- `lora_alpha / r` is the effective scaling

</details>

**Not ready?** Review: Module 3.1

---

### 3.2 Quantization

**Can you answer?**

| Format | Bits | Typical Memory (8B model) |
|--------|------|---------------------------|
| FP32 | ? | ? |
| FP16/BF16 | ? | ? |
| INT8 | ? | ? |
| INT4/NF4 | ? | ? |

<details>
<summary>Check your answer</summary>

| Format | Bits | Typical Memory (8B model) |
|--------|------|---------------------------|
| FP32 | 32 | ~32 GB |
| FP16/BF16 | 16 | ~16 GB |
| INT8 | 8 | ~8 GB |
| INT4/NF4 | 4 | ~4 GB |

**Rule of thumb**: Memory ≈ Parameters × Bits / 8

</details>

**Not ready?** Review: Module 3.2

---

### 3.5 RAG Systems

**Can you do this?**

```python
# Set up a basic RAG pipeline with ChromaDB
import chromadb

# Create collection, add documents, and query
# Write your code here...
```

<details>
<summary>Check your answer</summary>

```python
import chromadb
import ollama

# Create client and collection
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("my_docs")

# Get embeddings
def get_embedding(text):
    response = ollama.embeddings(model="qwen3-embedding:8b", prompt=text)
    return response["embedding"]

# Add documents
docs = ["DGX Spark has 128GB memory", "CUDA cores: 6144"]
for i, doc in enumerate(docs):
    collection.add(
        documents=[doc],
        embeddings=[get_embedding(doc)],
        ids=[f"doc_{i}"]
    )

# Query
results = collection.query(
    query_embeddings=[get_embedding("How much memory?")],
    n_results=2
)
print(results["documents"])
```

</details>

**Not ready?** Review: Module 3.5

---

### 3.6 AI Agents

**Do you know these terms?**

| Term | Your Definition |
|------|-----------------|
| Tool calling | |
| Agent loop | |
| ReAct pattern | |

<details>
<summary>Check definitions</summary>

| Term | Definition |
|------|------------|
| Tool calling | LLM decides which tools to use and generates arguments |
| Agent loop | Observe → Think → Act → Observe cycle |
| ReAct | Reasoning + Acting: LLM explains reasoning before action |

</details>

**Not ready?** Review: Module 3.6

---

## Domain 4: Production AI

### 4.1 Multimodal

**Can you do this?** (Option B)

```python
# Load a vision-language model
# Write your code here...
```

<details>
<summary>Check your answer</summary>

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
```

</details>

**Not ready?** Review: Module 4.1 (required for Option B)

---

### 4.2 AI Safety

**Can you answer?**

What is NeMo Guardrails? What problems does it solve?

<details>
<summary>Check your answer</summary>

**NeMo Guardrails**:
- Framework for adding safety controls to LLM applications
- Uses Colang DSL for defining conversational flows
- Solves: prompt injection, jailbreaking, harmful outputs

**Key components:**
- Input rails: Filter unsafe user inputs
- Output rails: Filter unsafe model outputs
- Dialog rails: Control conversation flow

</details>

**Not ready?** Review: Module 4.2

---

### 4.3 MLOps

**Can you do this?**

```python
# Log an experiment with MLflow
import mlflow

# Start a run, log parameters and metrics
# Write your code here...
```

<details>
<summary>Check your answer</summary>

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("capstone")

with mlflow.start_run(run_name="experiment-001"):
    # Log parameters
    mlflow.log_params({
        "model": "llama-8b",
        "learning_rate": 2e-5,
        "epochs": 3,
    })

    # Log metrics
    mlflow.log_metrics({
        "accuracy": 0.92,
        "loss": 0.15,
    })

    # Log model
    mlflow.pytorch.log_model(model, "model")
```

</details>

**Not ready?** Review: Module 4.3

---

### 4.4 Containerization

**Can you answer?**

Write a basic Dockerfile for an ML inference server:

<details>
<summary>Check your answer</summary>

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.11-py3

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ /app/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "serve.py"]
```

</details>

**Not ready?** Review: Module 4.4

---

### 4.5 Demo Building

**Can you do this?**

```python
# Create a simple Gradio chat interface
import gradio as gr

# Write your code here...
```

<details>
<summary>Check your answer</summary>

```python
import gradio as gr
import ollama

def chat(message, history):
    messages = []
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})

    response = ollama.chat(model="qwen3:8b", messages=messages)
    return response["message"]["content"]

demo = gr.ChatInterface(fn=chat, title="My Assistant")
demo.launch()
```

</details>

**Not ready?** Review: Module 4.5

---

## Option-Specific Prerequisites

### Option A: Domain AI Assistant

| Skill | Module | Status |
|-------|--------|--------|
| QLoRA fine-tuning | 3.1 | ☐ |
| RAG implementation | 3.5 | ☐ |
| NeMo Guardrails | 4.2 | ☐ |
| Gradio/Streamlit | 4.5 | ☐ |

### Option B: Document Intelligence

| Skill | Module | Status |
|-------|--------|--------|
| Vision-language models | 4.1 | ☐ |
| Document processing | 4.1 | ☐ |
| RAG for multimodal | 3.5, 4.1 | ☐ |
| Structured extraction | 4.1 | ☐ |

### Option C: Agent Swarm

| Skill | Module | Status |
|-------|--------|--------|
| Agent fundamentals | 3.6 | ☐ |
| LangChain/LangGraph | 3.6 | ☐ |
| Safety guardrails | 4.2 | ☐ |
| Tool implementation | 3.6 | ☐ |

### Option D: Training Pipeline

| Skill | Module | Status |
|-------|--------|--------|
| SFT training | 3.1 | ☐ |
| DPO training | 3.1 | ☐ |
| MLflow tracking | 4.3 | ☐ |
| Evaluation benchmarks | 4.3 | ☐ |

---

## Ready?

- [ ] I can complete all the skill checks above
- [ ] I understand the terminology
- [ ] I've chosen my project option
- [ ] My environment is set up (see [LAB_PREP.md](./LAB_PREP.md))

**All boxes checked?** → Start with [QUICKSTART.md](./QUICKSTART.md)!

**Some gaps?** → Review the linked modules first. It's better to spend a few hours reviewing than to struggle for days during the capstone.
