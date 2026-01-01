# Module 4.5: Demo Building & Prototyping - Lab Preparation Guide

## Time Estimates

| Lab | Preparation | Execution | Total |
|-----|-------------|-----------|-------|
| 4.5.1 Complete RAG Demo | 15 min | 3h | 3.25h |
| 4.5.2 Agent Playground | 10 min | 3h | 3.2h |
| 4.5.3 Portfolio Demo | 10 min | 2h | 2.2h |

---

## Required Downloads

### Models for Demos (2025 Tier 1)

```bash
# Chat model for all labs (~5GB, hybrid thinking)
ollama pull qwen3:8b

# Embedding model for RAG (~8GB, #1 MTEB)
ollama pull qwen3-embedding:8b

# (Optional) Larger model for better quality (~20GB)
ollama pull qwen3:32b
```

**Total download size**: ~5-10 GB (basic) or ~50 GB (with large model)
**Estimated download time**: 10-60 minutes depending on model choices

---

## Environment Setup

### 1. Start NGC Container with Demo Ports

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 7860:7860 \
    -p 8501:8501 \
    -p 11434:11434 \
    nvcr.io/nvidia/pytorch:25.11-py3
```

### 2. Install Demo Dependencies

```bash
# Core demo libraries
pip install gradio>=4.0.0 streamlit>=1.30.0

# Backend support
pip install ollama chromadb

# For file handling
pip install pypdf python-docx

# For visualizations
pip install plotly altair
```

### 3. Start Ollama

```bash
# Start Ollama server in background
ollama serve &

# Wait for startup
sleep 5

# Pull required models (2025 Tier 1)
ollama pull qwen3:8b
ollama pull qwen3-embedding:8b
```

### 4. Verify Setup

```python
import gradio as gr
import streamlit as st
import ollama

# Check Gradio
print(f"Gradio {gr.__version__} ready!")

# Check Ollama
response = ollama.chat(
    model="qwen3:8b",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(f"Ollama ready! Response: {response['message']['content'][:50]}...")

print("All systems go!")
```

**Expected output**:
```
Gradio 4.x.x ready!
Ollama ready! Response: Hello! How can I help you today?...
All systems go!
```

---

## Pre-Lab Checklists

### Lab 4.5.1: Complete RAG Demo

- [ ] Ollama running with qwen3:8b
- [ ] Ollama running with qwen3-embedding:8b
- [ ] Gradio installed
- [ ] chromadb installed
- [ ] pypdf installed (for PDF upload)
- [ ] Port 7860 available
- [ ] Sample documents ready for testing

**Quick Test**:
```python
import gradio as gr

def greet(name):
    return f"Hello, {name}!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch(server_port=7860)
```

Access http://localhost:7860 - should show Gradio interface.

**Sample Documents** (create or use your own):
```bash
# Create sample directory
mkdir -p /workspace/module-4.5/sample-docs

# Create a simple test document
echo "DGX Spark features 128GB unified memory..." > /workspace/module-4.5/sample-docs/dgx-spark-info.txt
```

---

### Lab 4.5.2: Agent Playground

- [ ] Ollama running with qwen3:8b
- [ ] Streamlit installed
- [ ] Completed basic Streamlit example
- [ ] Port 8501 available
- [ ] Understanding of agent concepts from Domain 3

**Quick Test**:
```python
# test_app.py
import streamlit as st

st.title("Test Agent Playground")
st.write("Streamlit is working!")

if "count" not in st.session_state:
    st.session_state.count = 0

if st.button("Click me"):
    st.session_state.count += 1

st.write(f"Button clicked {st.session_state.count} times")
```

```bash
streamlit run test_app.py --server.port 8501
```

---

### Lab 4.5.3: Portfolio Demo

- [ ] Completed Labs 4.5.1 and 4.5.2
- [ ] Clear idea of what to showcase
- [ ] (Optional) Hugging Face account for Spaces deployment
- [ ] (Optional) GitHub account for Streamlit Cloud
- [ ] Video recording capability (for walkthrough)

**Quick Test**: Can you describe your demo in one sentence?

---

## Platform Accounts (Optional)

### Hugging Face Spaces

For deploying Gradio demos publicly:

1. Create account at [huggingface.co](https://huggingface.co)
2. Create new Space (Gradio SDK)
3. Note: Free tier has limited compute

### Streamlit Cloud

For deploying Streamlit apps publicly:

1. Create account at [streamlit.io/cloud](https://streamlit.io/cloud)
2. Connect your GitHub account
3. Note: Apps sleep after inactivity on free tier

---

## Common Setup Mistakes

| Mistake | Consequence | Prevention |
|---------|-------------|------------|
| Ollama not running | Chat/embeddings fail | Start with `ollama serve &` |
| Wrong port mapped | Can't access demo | Use `-p 7860:7860 -p 8501:8501` |
| No model pulled | Model not found errors | Run `ollama pull qwen3:8b` |
| Missing embedding model | RAG embedding fails | Run `ollama pull qwen3-embedding:8b` |
| ChromaDB not installed | Vector store errors | Run `pip install chromadb` |
| Port already in use | Demo won't start | Kill existing process or use different port |

---

## Expected Directory Structure

```
/workspace/
├── module-4.5/
│   ├── lab-4.5.1-rag-demo/
│   │   ├── app.py           # Main Gradio app
│   │   ├── rag_utils.py     # RAG helper functions
│   │   └── requirements.txt
│   ├── lab-4.5.2-agent-playground/
│   │   ├── Home.py          # Streamlit entry point
│   │   ├── pages/
│   │   │   ├── 1_Chat.py
│   │   │   ├── 2_Tools.py
│   │   │   └── 3_History.py
│   │   └── utils/
│   │       └── agent.py
│   ├── lab-4.5.3-portfolio/
│   │   ├── app.py           # Your capstone demo
│   │   └── README.md        # For Spaces deployment
│   └── sample-docs/
│       └── [test documents]
```

---

## Quick Start Commands

```bash
# Copy-paste to set up everything:
cd /workspace
mkdir -p module-4.5/{lab-4.5.1-rag-demo,lab-4.5.2-agent-playground/pages,lab-4.5.3-portfolio,sample-docs}

# Install dependencies
pip install gradio>=4.0.0 streamlit>=1.30.0 ollama chromadb pypdf python-docx plotly altair

# Start Ollama and pull models (2025 Tier 1)
ollama serve &
sleep 5
ollama pull qwen3:8b
ollama pull qwen3-embedding:8b

# Verify installation
python -c "import gradio as gr; print(f'Gradio {gr.__version__} ready!')"
python -c "import streamlit as st; print(f'Streamlit {st.__version__} ready!')"
python -c "import ollama; print('Ollama ready!')"

echo "Setup complete! Ready for demos."
```

---

## Demo Design Tips

### What Makes a Good Demo

| Good Demo | Bad Demo |
|-----------|----------|
| Works on first try | Crashes immediately |
| Clear purpose | Confusing interface |
| Fast responses | Long waits with no feedback |
| Handles errors gracefully | Crashes on edge cases |
| Professional appearance | Default/ugly styling |

### Pre-Demo Checklist

Before showing your demo:

- [ ] Tested with realistic inputs
- [ ] Tested with edge cases (empty input, very long input)
- [ ] Error messages are user-friendly
- [ ] Loading states show progress
- [ ] Examples pre-populated
- [ ] Theme looks professional
- [ ] Mobile/small screen tested (if sharing publicly)

---

## Storage Requirements

| Component | Size |
|-----------|------|
| Gradio/Streamlit packages | ~500 MB |
| Ollama models (8B) | ~5 GB |
| ChromaDB embeddings | ~100 MB per 1000 docs |
| Demo code | ~50 MB |
| **Total** | ~6-10 GB |

Ensure sufficient disk space before starting.
