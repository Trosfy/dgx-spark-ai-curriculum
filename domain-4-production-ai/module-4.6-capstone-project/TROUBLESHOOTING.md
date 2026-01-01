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
EMBEDDING_MODEL = "qwen3-embedding:8b"

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

## ❓ Frequently Asked Questions

### Project Selection

**Q: Which project option should I choose?**

**A**: Choose based on your interests and career goals:

| If you want to... | Choose |
|-------------------|--------|
| Work on LLM applications | Option A: Domain Assistant |
| Work with documents/images | Option B: Multimodal |
| Build complex systems | Option C: Agent Swarm |
| Focus on ML infrastructure | Option D: Training Pipeline |

**Recommendation**: Pick what excites you - you'll spend 40+ hours on it.

---

**Q: Can I combine project options?**

**A**: Yes, but be careful about scope. Good combinations:

| Combination | Works Well | Why |
|-------------|------------|-----|
| A + B | Yes | Document RAG with fine-tuning |
| A + C | Yes | Agents with domain expertise |
| C + Safety | Yes | Required for Option C |
| All options | No | Too broad, won't finish |

Keep core focus on one option with elements from others.

---

**Q: What if I want to do something not listed?**

**A**: Custom projects are allowed if they:

1. Demonstrate equivalent complexity
2. Use DGX Spark capabilities
3. Include required deliverables
4. Have clear evaluation criteria

Get approval before starting a custom project.

---

### Scope and Planning

**Q: How do I know if my scope is too big?**

**A**: Signs your project is too ambitious:

- More than 5 major components
- Requires data you don't have
- Needs external APIs you can't access
- Would take a team months to build
- You can't explain it in 2 sentences

**Rule of thumb**: If you can't build an MVP in week 1, scope down.

---

**Q: How do I know if my scope is too small?**

**A**: Signs your project is too simple:

- Single API call wrapper
- No model training or fine-tuning
- No evaluation beyond "it works"
- Could be done in a weekend
- Doesn't use DGX Spark's unique capabilities

**Rule of thumb**: Should take 40+ hours to complete properly.

---

**Q: What's a realistic timeline?**

**A**: Based on past projects:

| Phase | Expected | Buffer for Issues |
|-------|----------|-------------------|
| Planning | 1 week | + 2 days |
| Foundation | 2 weeks | + 3-4 days |
| Integration | 1 week | + 2 days |
| Optimization | 1 week | + 2 days |
| Documentation | 1 week | + 1 day |

Build in buffer time for unexpected issues.

---

### Technical Questions

**Q: How do I leverage DGX Spark's 128GB memory?**

**A**: Great ways to use the unified memory:

1. **Larger models**: Run 70B+ models without quantization
2. **Bigger batch sizes**: Faster fine-tuning
3. **Multiple models**: Keep several loaded simultaneously
4. **Large documents**: Process entire document sets in memory
5. **Agent parallelism**: Run multiple agents concurrently

Document your memory usage in the technical report.

---

**Q: What evaluation metrics should I use?**

**A**: Depends on project type:

| Project | Metrics |
|---------|---------|
| Option A | Task accuracy, latency, safety pass rate |
| Option B | Extraction F1, comprehension scores |
| Option C | Task success rate, safety violations, latency |
| Option D | Model improvement, pipeline throughput |

Always include:
- Performance metrics (accuracy, F1, etc.)
- Latency measurements
- Safety evaluation (if applicable)
- Cost analysis (if cloud-deployed)

---

**Q: How extensive should safety evaluation be?**

**A**: Minimum requirements by project:

| Project | Safety Requirements |
|---------|---------------------|
| Option A | Guardrails config + test suite + model card |
| Option B | Input validation + output filtering |
| Option C | Agent action limits + human-in-loop + red teaming |
| Option D | Data quality checks + model validation |

Safety is 15% of the grade - don't skip it.

---

**Q: What if I run out of compute resources?**

**A**: Strategies to reduce compute:

1. **Quantize models**: Use 4-bit instead of 16-bit
2. **Smaller models**: 8B instead of 70B for development
3. **Shorter training**: Fewer epochs, validate early
4. **Batch efficiently**: Optimize batch sizes
5. **Use checkpoints**: Resume training, don't restart

---

### Deliverables

**Q: How long should the technical report be?**

**A**: 15-20 pages, focused content:

| Section | Pages |
|---------|-------|
| Introduction/Problem | 1-2 |
| Related Work | 1-2 |
| System Architecture | 3-4 |
| Implementation | 4-5 |
| Evaluation | 3-4 |
| Discussion/Lessons | 2-3 |
| Appendix (optional) | 2-3 |

Quality over quantity - concise is better than padded.

---

**Q: What should the demo showcase?**

**A**: Focus on:

1. **Core functionality**: Main feature working smoothly
2. **Best examples**: Pre-selected inputs that work well
3. **Edge case handling**: Show graceful failure
4. **Speed**: Optimize for demo performance

Don't demo:
- Features that might break
- Slow operations without progress indicators
- Raw error messages

---

**Q: What should be in the model card?**

**A**: Essential sections:

```markdown
# Model Card: [Your Model Name]

## Model Details
- Base model
- Fine-tuning method
- Training data summary

## Intended Use
- Primary use case
- Out-of-scope uses

## Training Data
- Data sources
- Data size
- Data processing

## Evaluation
- Metrics and results
- Safety evaluation results
- Limitations

## Ethical Considerations
- Potential biases
- Mitigation strategies
- Recommendations for use
```

---

**Q: How should I organize my code repository?**

**A**: Recommended structure:

```
capstone/
├── README.md           # Setup instructions
├── requirements.txt    # Dependencies
├── Dockerfile          # Container build
├── src/
│   ├── __init__.py
│   ├── main.py         # Entry point
│   ├── model.py        # Model code
│   ├── data.py         # Data processing
│   └── utils.py        # Utilities
├── tests/
│   ├── test_model.py
│   └── test_data.py
├── notebooks/          # Development notebooks
├── configs/            # Configuration files
├── docs/
│   ├── ARCHITECTURE.md
│   └── MODEL_CARD.md
└── demo/
    └── app.py          # Gradio/Streamlit demo
```

---

### Common Issues

**Q: My project isn't working as expected**

**A**: Debugging strategy:

1. **Isolate the problem**: Which component fails?
2. **Check basics**: Data format, model loading, API calls
3. **Review logs**: Error messages, stack traces
4. **Simplify**: Does minimal example work?
5. **Reference modules**: Did it work in earlier labs?

Document issues and solutions in your report.

---

**Q: I'm running behind schedule**

**A**: Recovery strategies:

1. **Reduce scope**: Cut nice-to-have features
2. **Simplify evaluation**: Fewer metrics, smaller test set
3. **Streamline docs**: Focus on key sections
4. **Use existing code**: Adapt from earlier modules
5. **Ask for help**: Don't struggle alone

What to cut first:
- Extra features
- Extensive ablations
- Perfect documentation

What to keep:
- Core functionality
- Safety evaluation
- Demo that works

---

## Still Stuck?

1. **Check module troubleshooting guides** - Many issues covered in Modules 3.1-4.5
2. **Review example implementations** - `examples/` folder has working code
3. **Simplify** - Get basic version working, then add complexity
4. **Document the error** - Include in technical report as "lessons learned"
5. **Ask for help** - Share: error message, code snippet, what you tried
