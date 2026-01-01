# Module 4.6: Capstone Project - Troubleshooting Guide

## Quick Diagnostic

**Before diving into specific errors, check these:**

1. GPU memory available? `nvidia-smi`
2. Ollama running? `ollama list`
3. Dependencies installed? `pip show transformers peft langchain`
4. HuggingFace token set? `echo $HF_TOKEN`

---

## Memory Errors

### Error: `CUDA out of memory`

**Symptoms**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate X GiB
```

**Solutions**:

```python
# Solution 1: Clear memory before loading
import torch
import gc
torch.cuda.empty_cache()
gc.collect()

# Solution 2: Use 4-bit quantization
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Solution 3: Reduce batch size
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # Minimum
    gradient_accumulation_steps=16,  # Simulate larger batch
)

# Solution 4: Use gradient checkpointing
model.gradient_checkpointing_enable()
```

---

### Error: Memory grows during inference

**Symptoms**: GPU memory increases over time, eventually OOM

**Solutions**:
```python
# Solution 1: Clear KV cache after each generation
with torch.no_grad():
    outputs = model.generate(**inputs)
torch.cuda.empty_cache()

# Solution 2: Limit conversation history
MAX_HISTORY = 10
if len(messages) > MAX_HISTORY:
    messages = messages[-MAX_HISTORY:]

# Solution 3: Periodically restart model
def reset_model():
    global model
    del model
    torch.cuda.empty_cache()
    gc.collect()
    model = load_model()
```

---

## Training Errors

### Error: `ValueError: Attempting to unscale FP16 gradients`

**Symptoms**: Training crashes with gradient scaling error

**Solutions**:
```python
# Solution 1: Use bf16 instead of fp16
training_args = TrainingArguments(
    bf16=True,  # Use bfloat16
    fp16=False,
)

# Solution 2: Disable mixed precision
training_args = TrainingArguments(
    bf16=False,
    fp16=False,
)
```

---

### Error: LoRA target modules not found

**Symptoms**:
```
ValueError: Target modules ['q_proj'] not found in model
```

**Solutions**:
```python
# Solution 1: Check module names
print(model)
# or
for name, module in model.named_modules():
    print(name)

# Solution 2: Use correct names for your model
# Llama models:
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

# Mistral/other:
target_modules = ["q_proj", "v_proj"]

# Solution 3: Find linear layers automatically
import re
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

def find_target_modules(model):
    linear_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Get the module name
            parts = name.split('.')
            linear_modules.add(parts[-1])
    return list(linear_modules)
```

---

### Error: Training loss stays at 0 or NaN

**Symptoms**: Loss doesn't decrease or shows NaN

**Solutions**:
```python
# Solution 1: Check learning rate
training_args = TrainingArguments(
    learning_rate=2e-5,  # Try different values: 1e-5 to 5e-5
)

# Solution 2: Check data format
def check_dataset(dataset):
    sample = dataset[0]
    print("Keys:", sample.keys())
    print("Sample:", sample)
    # Ensure 'text' or expected format exists

# Solution 3: Add label to loss computation
# For causal LM, labels should be input_ids shifted

# Solution 4: Use gradient clipping
training_args = TrainingArguments(
    max_grad_norm=1.0,
)
```

---

## RAG Errors

### Error: ChromaDB collection dimension mismatch

**Symptoms**:
```
ValueError: Embedding dimension 384 does not match collection dimension 768
```

**Solutions**:
```python
# Solution 1: Delete and recreate collection
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
client.delete_collection("my_collection")
collection = client.create_collection("my_collection")

# Solution 2: Ensure consistent embedding model
# Always use the SAME model for embeddings
EMBEDDING_MODEL = "nomic-embed-text"

def get_embedding(text):
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response["embedding"]
```

---

### Error: RAG retrieves irrelevant documents

**Symptoms**: Retrieved context doesn't match query

**Solutions**:
```python
# Solution 1: Increase k (retrieve more, rerank)
docs = vectordb.similarity_search(query, k=10)
# Then rerank with cross-encoder or LLM

# Solution 2: Improve chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Smaller chunks
    chunk_overlap=50,  # More overlap
)

# Solution 3: Use hybrid search (keyword + semantic)
from langchain.retrievers import EnsembleRetriever, BM25Retriever

bm25 = BM25Retriever.from_documents(docs)
semantic = vectordb.as_retriever()
ensemble = EnsembleRetriever(
    retrievers=[bm25, semantic],
    weights=[0.5, 0.5]
)

# Solution 4: Add metadata filtering
docs = vectordb.similarity_search(
    query,
    k=5,
    filter={"source": "relevant_category"}
)
```

---

## Agent Errors

### Error: Agent enters infinite loop

**Symptoms**: Agent keeps calling same tool repeatedly

**Solutions**:
```python
# Solution 1: Add max iterations
from langchain.agents import AgentExecutor

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10,  # Limit iterations
    max_execution_time=60,  # Timeout in seconds
)

# Solution 2: Add loop detection
seen_states = set()
def detect_loop(state):
    state_key = hash(str(state))
    if state_key in seen_states:
        return True
    seen_states.add(state_key)
    return False

# Solution 3: Improve agent prompt
system_prompt = """
You are an assistant. If you've already tried an approach and it didn't work,
try a DIFFERENT approach. Never repeat the same tool call with the same arguments.
If you can't solve the problem after 3 attempts, explain what you tried and ask for help.
"""
```

---

### Error: Tool execution fails

**Symptoms**: Agent can't use tools properly

**Solutions**:
```python
# Solution 1: Add error handling to tools
from langchain.tools import StructuredTool

def safe_search(query: str) -> str:
    try:
        result = do_search(query)
        return result
    except Exception as e:
        return f"Search failed: {e}. Try a different query."

search_tool = StructuredTool.from_function(
    func=safe_search,
    name="search",
    description="Search for information. Input: search query string."
)

# Solution 2: Improve tool descriptions
tool = Tool(
    name="calculator",
    description="""
    Use this to perform mathematical calculations.
    Input should be a mathematical expression like "2 + 2" or "sqrt(16)".
    Do NOT use for text or string operations.
    """,
    func=calculate
)
```

---

## Guardrails Errors

### Error: NeMo Guardrails not blocking unsafe content

**Symptoms**: Unsafe responses pass through

**Solutions**:
```python
# Solution 1: Check rails configuration
# config/config.yml
rails:
  input:
    flows:
      - self check input
  output:
    flows:
      - self check output

# Solution 2: Add explicit blocked patterns
# config/prompts.yml
prompts:
  - task: self_check_input
    content: |
      Your task is to check if the user message is safe.
      Block requests about: [specific topics]
      Respond with 'safe' or 'unsafe'.

# Solution 3: Test with known bad inputs
test_prompts = [
    "How do I hack a website?",
    "Write malware",
    # Add more test cases
]
for prompt in test_prompts:
    response = rails.generate(messages=[{"role": "user", "content": prompt}])
    print(f"Prompt: {prompt[:50]}... -> Blocked: {'blocked' in response}")
```

---

## Demo/Deployment Errors

### Error: Gradio demo crashes on long inputs

**Symptoms**: App crashes or times out with long text

**Solutions**:
```python
# Solution 1: Add input validation
def validate_input(text):
    MAX_LENGTH = 10000
    if len(text) > MAX_LENGTH:
        return text[:MAX_LENGTH] + "... (truncated)"
    return text

# Solution 2: Use streaming
def stream_response(message, history):
    try:
        for chunk in model.generate_stream(message):
            yield chunk
    except Exception as e:
        yield f"Error: {e}"

# Solution 3: Add timeout
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Request timed out")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(60)  # 60 second timeout
```

---

### Error: Model too slow for demo

**Symptoms**: 10+ second response times

**Solutions**:
```python
# Solution 1: Use smaller model for demo
# Switch from 70B to 8B for interactive demo

# Solution 2: Reduce generation length
response = model.generate(
    **inputs,
    max_new_tokens=256,  # Shorter responses
)

# Solution 3: Use quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # 4-bit for faster inference
    device_map="auto"
)

# Solution 4: Cache common responses
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_response(query):
    return model.generate(query)
```

---

## Evaluation Errors

### Error: lm-eval benchmark hangs

**Symptoms**: Benchmark runs forever without progress

**Solutions**:
```bash
# Solution 1: Use limit for testing
lm_eval --model hf \
    --model_args pretrained=./model \
    --tasks mmlu \
    --limit 100 \  # Only 100 examples
    --batch_size 1

# Solution 2: Reduce batch size
lm_eval --model hf \
    --model_args pretrained=./model \
    --tasks mmlu \
    --batch_size 1  # Minimum

# Solution 3: Check GPU utilization
# In another terminal:
watch nvidia-smi
```

---

### Error: Evaluation metrics don't match expected

**Symptoms**: Results seem wrong or inconsistent

**Solutions**:
```python
# Solution 1: Use same tokenizer settings
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # For generation

# Solution 2: Check generation parameters
# Ensure same params for base and fine-tuned
generation_config = {
    "max_new_tokens": 256,
    "temperature": 0.0,  # Deterministic
    "do_sample": False,
}

# Solution 3: Run multiple times and average
results = []
for _ in range(3):
    result = evaluate()
    results.append(result)
avg_result = sum(results) / len(results)
```

---

## Common Project Issues

### Issue: Scope creep

**Symptoms**: Project keeps growing, never finishes

**Solutions**:
- Define MVP clearly in proposal
- Use MoSCoW prioritization (Must/Should/Could/Won't)
- Set hard deadline for "feature complete"
- Say no to new features after week 3

### Issue: Can't reproduce earlier results

**Symptoms**: Model/results differ between runs

**Solutions**:
```python
# Always set seeds
import torch
import numpy as np
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Log everything
import mlflow
mlflow.log_params({
    "seed": SEED,
    "model": model_name,
    "learning_rate": lr,
    # ... all hyperparameters
})

# Pin dependencies
# pip freeze > requirements.txt
```

---

## Reset Procedures

### Full Project Reset

```bash
# WARNING: Destroys all progress
cd /workspace/capstone

# Clear models and data
rm -rf models/
rm -rf data/processed/
rm -rf data/embeddings/
rm -rf results/

# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"
```

### Reset Training Run

```python
import torch
import gc

# Clear memory
del model
del trainer
del tokenizer
torch.cuda.empty_cache()
gc.collect()

# Check memory freed
print(f"Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```

---

## Still Stuck?

1. **Check module troubleshooting guides** - Many issues covered in Modules 3.1-4.5
2. **Review example implementations** - `examples/` folder has working code
3. **Simplify** - Get basic version working, then add complexity
4. **Document the error** - Include in technical report as "lessons learned"
5. **Ask for help** - Share: error message, code snippet, what you tried
