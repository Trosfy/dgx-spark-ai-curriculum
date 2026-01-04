# Module 4.6: Capstone Project - Quick Reference

## Project Options Summary

| Option | Focus | Key Tech | Deliverables |
|--------|-------|----------|--------------|
| A | Domain AI Assistant | QLoRA, RAG, Guardrails | Chatbot + API |
| B | Document Intelligence | VLM, OCR, Extraction | Document pipeline |
| C | Agent Swarm | Multi-agent, Safety | Agent system |
| D | Training Pipeline | Data, SFT, DPO | MLOps pipeline |
| E | Browser-Deployed LLM | QLoRA, ONNX, Transformers.js | Static web app |

---

## Essential Commands

### NGC Container Setup (All Options)

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 7860:7860 \
    -p 8501:8501 \
    -p 5000:5000 \
    nvcr.io/nvidia/pytorch:25.11-py3
```

### Common Dependencies

```bash
# Core ML
pip install torch transformers accelerate bitsandbytes
pip install peft trl datasets

# RAG/Agents
pip install langchain langgraph chromadb ollama

# Safety
pip install nemoguardrails

# Demo
pip install gradio streamlit

# MLOps
pip install mlflow wandb
```

---

## Option A: Domain AI Assistant

### Fine-tuning with QLoRA

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-32B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
```

### RAG Setup

```python
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")
vectordb = Chroma(
    collection_name="domain_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Add documents
vectordb.add_texts(texts, metadatas=metadata)

# Retrieve
docs = vectordb.similarity_search(query, k=3)
```

### NeMo Guardrails

```python
from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_path("./config")
rails = LLMRails(config)

response = rails.generate(messages=[{
    "role": "user",
    "content": user_input
}])
```

---

## Option B: Document Intelligence

### PDF Processing

```python
import pypdf
from PIL import Image

def extract_pdf_content(pdf_path):
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    images = []

    for page in reader.pages:
        text += page.extract_text()
        # Extract images if needed
        for img in page.images:
            images.append(img)

    return text, images
```

### Vision-Language Model

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Process image
inputs = processor(
    text="<image>\nDescribe this document.",
    images=image,
    return_tensors="pt"
).to("cuda")

output = model.generate(**inputs, max_new_tokens=256)
```

### Structured Extraction

```python
def extract_structured_data(text, schema):
    prompt = f"""Extract the following fields from the document:

Schema: {schema}

Document:
{text}

Return as JSON:"""

    response = ollama.generate(model="qwen3:8b", prompt=prompt)
    return json.loads(response["response"])
```

---

## Option C: Agent Swarm

### Base Agent Pattern

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate

def create_specialized_agent(name, system_prompt, tools):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}"),
    ])

    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# Create specialized agents
researcher = create_specialized_agent("Researcher", RESEARCH_PROMPT, [search_tool])
coder = create_specialized_agent("Coder", CODE_PROMPT, [code_tool])
reviewer = create_specialized_agent("Reviewer", REVIEW_PROMPT, [review_tool])
```

### Agent Coordination

```python
from langgraph.graph import StateGraph, END

def create_agent_graph():
    workflow = StateGraph(AgentState)

    # Add agent nodes
    workflow.add_node("planner", planner_agent)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("executor", executor_agent)
    workflow.add_node("reviewer", reviewer_agent)

    # Add edges
    workflow.add_edge("planner", "researcher")
    workflow.add_conditional_edges(
        "researcher",
        should_continue,
        {"continue": "executor", "end": END}
    )

    return workflow.compile()
```

### Safety Guardrails for Agents

```python
def safe_tool_execution(tool, args):
    # Check dangerous operations
    dangerous_patterns = ["rm -rf", "sudo", "DROP TABLE"]
    if any(p in str(args) for p in dangerous_patterns):
        return "Blocked: Dangerous operation detected"

    # Human approval for sensitive actions
    if tool.name in ["file_write", "api_call"]:
        if not get_human_approval(tool.name, args):
            return "Blocked: Human approval required"

    return tool.invoke(args)
```

---

## Option D: Training Pipeline

### Data Collection

```python
from datasets import load_dataset, DatasetDict

def prepare_training_data(raw_data):
    def format_example(example):
        return {
            "text": f"<|user|>\n{example['prompt']}\n<|assistant|>\n{example['response']}"
        }

    dataset = raw_data.map(format_example)
    return dataset.train_test_split(test_size=0.1)
```

### SFT Training

```python
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./sft_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    bf16=True,  # Use bfloat16 (native Blackwell support)
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    peft_config=lora_config,
)

trainer.train()
```

### DPO Training

```python
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    output_dir="./dpo_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-6,
    beta=0.1,
    num_train_epochs=1,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    train_dataset=preference_dataset,
    args=dpo_config,
)

trainer.train()
```

---

## Option E: Browser-Deployed Fine-Tuned LLM

### Pipeline Overview

```
Dataset → QLoRA Training → Merge Adapters → ONNX Export → INT4 Quantization → Browser
  (150+)    (DGX Spark)      (BF16)         (FP32)         (~500MB)        (WebGPU)
```

### Dependencies

```bash
# Training (DGX Spark)
pip install -U transformers>=4.50.0 datasets accelerate peft trl bitsandbytes
pip install unsloth mlflow optimum onnx onnxruntime

# Browser (npm)
npm install @huggingface/transformers
```

### Dataset Format (Messages)

```python
# Training data format for chat models
training_example = {
    "messages": [
        {"role": "system", "content": "You are a matcha tea expert."},
        {"role": "user", "content": "What is ceremonial grade matcha?"},
        {"role": "assistant", "content": "Ceremonial grade matcha is..."}
    ]
}

# Save with datasets library
from datasets import Dataset
dataset = Dataset.from_list(training_data)
dataset.push_to_hub("your-username/matcha-dataset")
```

### QLoRA Training with Unsloth

```python
from unsloth import FastLanguageModel
import torch

# Load model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-3-270m-it",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=torch.bfloat16,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# Train with SFTTrainer
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="./matcha-lora",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_steps=100,
    ),
)
trainer.train()
model.save_pretrained("./matcha-lora")
```

### Merge LoRA Adapters (BF16)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model in BF16 (CRITICAL: full precision for quality!)
base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-270m-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Load and merge LoRA
model = PeftModel.from_pretrained(base_model, "./matcha-lora")
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./matcha-merged")
tokenizer.save_pretrained("./matcha-merged")
```

### ONNX Export + INT4 Quantization

```python
from optimum.onnxruntime import ORTModelForCausalLM
from optimum.exporters.onnx import main_export

# Export to ONNX
main_export(
    model_name_or_path="./matcha-merged",
    output="./matcha-onnx",
    task="text-generation-with-past",
)

# INT4 Quantization (browser-compatible!)
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="./matcha-onnx/model.onnx",
    model_output="./matcha-onnx-int4/model_quantized.onnx",
    weight_type=QuantType.QInt4,
)
```

### Transformers.js Browser Integration

```javascript
import { pipeline, env } from '@huggingface/transformers';

// Configure model location
env.localModelPath = 'https://your-bucket.s3.amazonaws.com/models/';
env.allowRemoteModels = true;

// Load model with WebGPU + INT4
const generator = await pipeline(
  'text-generation',
  'matcha-chatbot-int4',
  {
    device: 'webgpu',
    dtype: 'q4',
    progress_callback: (progress) => {
      console.log(`Loading: ${(progress.progress * 100).toFixed(1)}%`);
    },
  }
);

// Generate response
const messages = [
  { role: 'system', content: 'You are a matcha tea expert.' },
  { role: 'user', content: 'What makes matcha green?' }
];

const output = await generator(messages, {
  max_new_tokens: 256,
  temperature: 0.7,
  do_sample: true,
});

console.log(output[0].generated_text);
```

### React Component Pattern

```jsx
import { useState, useCallback } from 'react';
import { pipeline } from '@huggingface/transformers';

function MatchaChatbot() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [generator, setGenerator] = useState(null);

  // Load model on first use
  const loadModel = useCallback(async () => {
    if (generator) return generator;

    setLoading(true);
    const pipe = await pipeline('text-generation', 'matcha-chatbot', {
      device: 'webgpu',
      dtype: 'q4',
    });
    setGenerator(pipe);
    setLoading(false);
    return pipe;
  }, [generator]);

  const sendMessage = async (userMessage) => {
    const pipe = await loadModel();

    const newMessages = [...messages, { role: 'user', content: userMessage }];
    setMessages(newMessages);

    const response = await pipe(newMessages, { max_new_tokens: 256 });
    const assistantMessage = response[0].generated_text.at(-1);

    setMessages([...newMessages, assistantMessage]);
  };

  return (
    <div className="chatbot">
      {loading && <div>Loading model (~500MB, cached after first load)...</div>}
      {/* Chat UI here */}
    </div>
  );
}
```

### S3 CORS Configuration

```json
[
    {
        "AllowedHeaders": ["*"],
        "AllowedMethods": ["GET", "HEAD"],
        "AllowedOrigins": [
            "https://yourdomain.com",
            "http://localhost:5173",
            "http://localhost:3000"
        ],
        "ExposeHeaders": ["Content-Length", "Content-Type", "ETag"],
        "MaxAgeSeconds": 3600
    }
]
```

### Deployment Headers (Required for SharedArrayBuffer)

```javascript
// vercel.json
{
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        { "key": "Cross-Origin-Opener-Policy", "value": "same-origin" },
        { "key": "Cross-Origin-Embedder-Policy", "value": "require-corp" }
      ]
    }
  ]
}
```

### File Size Reference

| Stage | Model Size | File Size |
|-------|-----------|-----------|
| Base (BF16) | 270M params | ~540MB |
| LoRA Adapters | - | ~10-30MB |
| Merged (BF16) | 270M params | ~540MB |
| ONNX (FP32) | 270M params | ~1GB |
| ONNX INT4 | 270M params | ~150-200MB |

### Performance Expectations

| Device | Tokens/Second |
|--------|---------------|
| RTX 4090 (WebGPU) | 40-60 |
| RTX 3080 (WebGPU) | 25-40 |
| M1/M2 Mac (WebGPU) | 15-30 |
| Integrated GPU | 5-15 |
| CPU (WASM fallback) | 1-5 |

---

## Evaluation Patterns

### Benchmark Suite

```bash
lm_eval --model hf \
    --model_args pretrained=./my-finetuned-model \
    --tasks mmlu,hellaswag,truthfulqa_mc2 \
    --batch_size 1 \
    --output_path ./results
```

### Custom Evaluation

```python
def evaluate_model(model, test_cases):
    results = []

    for case in test_cases:
        response = model.generate(case["input"])

        score = {
            "id": case["id"],
            "correct": case["expected"] in response,
            "response": response,
        }
        results.append(score)

    accuracy = sum(r["correct"] for r in results) / len(results)
    return {"accuracy": accuracy, "details": results}
```

### Safety Evaluation

```python
def evaluate_safety(model, attack_prompts):
    results = []

    for prompt in attack_prompts:
        response = model.generate(prompt["text"])
        refused = is_refusal(response)

        results.append({
            "attack_type": prompt["type"],
            "refused": refused,
            "response": response[:200],
        })

    refusal_rate = sum(r["refused"] for r in results) / len(results)
    return {"refusal_rate": refusal_rate, "details": results}
```

---

## Demo Templates

### Gradio RAG Demo

```python
import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("# Domain AI Assistant")

    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot()
            msg = gr.Textbox(placeholder="Ask a question...")
            send = gr.Button("Send")

        with gr.Column():
            with gr.Accordion("Sources", open=False):
                sources = gr.Markdown()

    send.click(rag_chat, [msg, chatbot], [chatbot, sources, msg])

demo.launch()
```

### Streamlit Dashboard

```python
import streamlit as st

st.set_page_config(page_title="Capstone Demo", layout="wide")

tab1, tab2, tab3 = st.tabs(["Chat", "Metrics", "About"])

with tab1:
    # Chat interface
    ...

with tab2:
    # Performance metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "92%")
    with col2:
        st.metric("Safety", "98%")

with tab3:
    st.markdown("""
    ## About This Project
    Built on DGX Spark with 128GB unified memory...
    """)
```

---

## Documentation Templates

### Model Card

```markdown
# Model Card: [Your Model Name]

## Model Details
- **Base Model**: Llama-3.1-70B
- **Fine-tuning**: QLoRA (r=16)
- **Training Data**: [Description]

## Intended Use
- Primary: [Use case]
- Not suitable for: [Limitations]

## Performance
| Benchmark | Score |
|-----------|-------|
| MMLU | XX% |
| Safety | XX% |

## Limitations
- [Known limitations]

## Ethical Considerations
- [Safety measures implemented]
```

### Technical Report Outline

```
1. Introduction (1-2 pages)
2. Related Work (1 page)
3. System Architecture (2-3 pages)
4. Implementation (3-4 pages)
5. Evaluation (3-4 pages)
6. Discussion (1-2 pages)
7. Conclusion (1 page)
8. References
9. Appendix
```

---

## Project Timeline

| Week | Phase | Key Activities |
|------|-------|----------------|
| 35 | Planning | Proposal, architecture design |
| 36-37 | Foundation | Core components, data prep |
| 38 | Integration | Connect components, build API |
| 39 | Optimization | Performance tuning, safety eval |
| 40 | Documentation | Report, demo, presentation |

---

## Common Patterns

### Error Handling

```python
def safe_generate(model, prompt):
    try:
        response = model.generate(prompt)
        return {"success": True, "response": response}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"success": False, "error": "Out of memory"}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### Memory Management

```python
import torch
import gc

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

def log_memory():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
```

### Experiment Tracking

```python
import mlflow

with mlflow.start_run(run_name="capstone-v1"):
    mlflow.log_params(config)

    # Training
    for epoch in range(epochs):
        metrics = train_epoch()
        mlflow.log_metrics(metrics, step=epoch)

    # Evaluation
    results = evaluate()
    mlflow.log_metrics(results)

    # Save model
    mlflow.pytorch.log_model(model, "model")
```

---

## Quick Links

- [Project Templates](./templates/)
- [Example Implementations](./examples/)
- [STUDY_GUIDE.md](./STUDY_GUIDE.md)
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
