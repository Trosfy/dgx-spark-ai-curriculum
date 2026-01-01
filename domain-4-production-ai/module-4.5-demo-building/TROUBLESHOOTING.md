# Module 4.5: Demo Building & Prototyping - Troubleshooting Guide

## Quick Diagnostic

**Before diving into specific errors, check these:**

1. Ollama running? `ollama list` or `curl http://localhost:11434/api/tags`
2. Model available? `ollama list` should show your model
3. Port available? `lsof -i :7860` or `lsof -i :8501`
4. Dependencies installed? `pip show gradio streamlit`

---

## Gradio Errors

### Error: `Connection closed` during streaming

**Symptoms**:
```
Error: Connection closed
```
Response cuts off mid-generation.

**Solutions**:
```python
# Solution 1: Add proper error handling
def chat(message, history):
    try:
        response = ollama.chat(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": message}]
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}. Please try again."

# Solution 2: For streaming, catch generator errors
def stream_chat(message, history):
    try:
        partial = ""
        for chunk in ollama.chat(model="llama3.1:8b", messages=messages, stream=True):
            partial += chunk["message"]["content"]
            yield partial
    except GeneratorExit:
        pass  # User closed connection, that's okay
    except Exception as e:
        yield f"Error: {e}"
```

---

### Error: `AttributeError: 'ChatInterface' has no attribute 'queue'`

**Symptoms**: Old Gradio code doesn't work with new version.

**Solutions**:
```python
# Solution: Update to new Gradio 4.x syntax
# Old (Gradio 3.x):
demo.queue().launch()

# New (Gradio 4.x):
demo.launch()  # Queue is now automatic
```

---

### Error: Gradio interface is very slow on first load

**Symptoms**: 30+ second delay on first request

**Solutions**:
```python
# Solution 1: Preload model at app startup
import gradio as gr
import ollama

# Load model before creating interface
print("Loading model...")
client = ollama.Client()
# Warm up with a small request
client.generate(model="llama3.1:8b", prompt="Hello", options={"num_predict": 1})
print("Model ready!")

def chat(message, history):
    # Model already warm
    ...

demo = gr.ChatInterface(fn=chat)
demo.launch()

# Solution 2: Show loading indicator
with gr.Blocks() as demo:
    gr.Markdown("# Chat (loading model...)")
    # Interface components
```

---

### Error: `share=True` doesn't create public link

**Symptoms**: No public share URL generated

**Solutions**:
```python
# Solution 1: Check firewall/network
# Some corporate networks block Gradio's tunnel service

# Solution 2: Use server_name for local network access
demo.launch(server_name="0.0.0.0", server_port=7860)
# Access via http://<your-local-ip>:7860

# Solution 3: Deploy to Hugging Face Spaces instead
# More reliable for sharing
```

---

### Error: File upload fails

**Symptoms**: Files don't upload or cause errors

**Solutions**:
```python
# Solution 1: Check file types
files = gr.File(
    label="Upload Documents",
    file_count="multiple",
    file_types=[".pdf", ".txt", ".md"]  # Explicit types
)

# Solution 2: Handle upload errors
def process_files(files):
    if files is None:
        return "No files uploaded"

    try:
        results = []
        for file in files:
            with open(file.name, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            results.append(f"Processed: {file.name}")
        return "\n".join(results)
    except Exception as e:
        return f"Error processing files: {e}"

# Solution 3: Increase max file size
demo.launch(max_file_size="100mb")
```

---

### Error: Chatbot history not displaying correctly

**Symptoms**: Messages appear in wrong format or not at all

**Solutions**:
```python
# Solution 1: Return correct format for ChatInterface
def chat(message, history):
    # ChatInterface expects: string response
    return "This is my response"

# Solution 2: For Blocks with Chatbot, return list of tuples
def chat_blocks(message, history):
    # Chatbot expects: list of [user, assistant] tuples
    response = "My response"
    history.append([message, response])
    return history, ""  # Return updated history and clear input

# Solution 3: For streaming
def stream_chat(message, history):
    partial = ""
    for chunk in ollama.chat(..., stream=True):
        partial += chunk["message"]["content"]
        yield partial  # Yield string for ChatInterface
```

---

## Streamlit Errors

### Error: App reruns on every interaction

**Symptoms**: Model reloads, state resets, slow performance

**Solutions**:
```python
# Solution 1: Cache model loading
@st.cache_resource
def load_model():
    """Load model once and cache."""
    import ollama
    return ollama.Client()

client = load_model()  # Only runs once per session

# Solution 2: Use session_state for persistence
if "messages" not in st.session_state:
    st.session_state.messages = []

# Solution 3: Cache expensive computations
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_embeddings(text):
    return ollama.embeddings(model="nomic-embed-text", prompt=text)
```

---

### Error: `DuplicateWidgetID`

**Symptoms**:
```
DuplicateWidgetID: There are multiple identical widgets with key='xxx'
```

**Solutions**:
```python
# Solution 1: Add unique keys
for i, item in enumerate(items):
    st.button(f"Click {item}", key=f"button_{i}")  # Unique key

# Solution 2: Use dynamic keys based on content
st.text_input("Name", key=f"name_{st.session_state.form_id}")

# Solution 3: Reset keys when needed
if st.button("Reset"):
    st.session_state.form_id = uuid.uuid4()
```

---

### Error: Streamlit memory usage grows indefinitely

**Symptoms**: App slows down, eventually crashes

**Solutions**:
```python
# Solution 1: Limit conversation history
MAX_MESSAGES = 100
if len(st.session_state.messages) > MAX_MESSAGES:
    st.session_state.messages = st.session_state.messages[-50:]

# Solution 2: Clear caches periodically
if st.button("Clear Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()

# Solution 3: Use TTL on caches
@st.cache_data(ttl=600)  # Expire after 10 minutes
def fetch_data():
    ...
```

---

### Error: File uploader shows old files

**Symptoms**: Uploaded files persist across sessions or show stale data

**Solutions**:
```python
# Solution 1: Clear uploader with key change
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0

if st.button("Clear"):
    st.session_state.upload_key += 1

uploaded_file = st.file_uploader(
    "Upload",
    key=f"uploader_{st.session_state.upload_key}"
)

# Solution 2: Process immediately and store
if uploaded_file is not None:
    content = uploaded_file.read()
    st.session_state.file_content = content
    # Don't rely on uploaded_file later - use session_state
```

---

### Error: Multi-page app pages not showing

**Symptoms**: Sidebar shows only Home, not other pages

**Solutions**:
```
# Solution: Correct directory structure
my_app/
├── Home.py              # Main entry (must be named Home.py or have emoji)
├── pages/
│   ├── 1_Chat.py        # Prefix with number for ordering
│   ├── 2_Analytics.py
│   └── 3_Settings.py
```

```bash
# Run from parent directory
cd /workspace/my_app
streamlit run Home.py
```

---

## Ollama Errors

### Error: `ConnectionError: Connection refused`

**Symptoms**:
```
requests.exceptions.ConnectionError: Connection refused
```

**Solutions**:
```bash
# Solution 1: Start Ollama
ollama serve &

# Wait for startup
sleep 5

# Solution 2: Check if already running
pgrep ollama

# Solution 3: Use correct host
import ollama
client = ollama.Client(host="http://localhost:11434")
```

---

### Error: `Model 'xxx' not found`

**Symptoms**:
```
ollama._types.ResponseError: model 'llama3.1:8b' not found
```

**Solutions**:
```bash
# Solution 1: Pull the model
ollama pull llama3.1:8b

# Solution 2: Check available models
ollama list

# Solution 3: Check model name spelling
# Common mistakes:
# - llama3:8b (wrong - missing .1)
# - llama-3.1-8b (wrong - dashes not colons)
# - llama3.1:8B (wrong - case sensitive)
```

---

### Error: Ollama responses are slow

**Symptoms**: Long wait times for responses

**Solutions**:
```python
# Solution 1: Check GPU utilization
# Run in terminal: nvidia-smi

# Solution 2: Clear model memory
ollama stop llama3.1:8b  # If supported

# Solution 3: Reduce context length
response = ollama.chat(
    model="llama3.1:8b",
    messages=messages,
    options={"num_ctx": 2048}  # Reduce from default 4096
)

# Solution 4: Limit response length
response = ollama.generate(
    model="llama3.1:8b",
    prompt=prompt,
    options={"num_predict": 256}  # Limit tokens
)
```

---

## ChromaDB Errors

### Error: `Collection not found`

**Symptoms**:
```
chromadb.errors.InvalidCollectionException: Collection xxx does not exist
```

**Solutions**:
```python
# Solution 1: Use get_or_create
import chromadb
client = chromadb.Client()
collection = client.get_or_create_collection("my_docs")

# Solution 2: Check existing collections
print(client.list_collections())

# Solution 3: Persist to disk for durability
client = chromadb.PersistentClient(path="/workspace/chroma_db")
```

---

### Error: Embedding dimension mismatch

**Symptoms**:
```
ValueError: Embedding dimension xxx does not match collection dimension yyy
```

**Solutions**:
```python
# Solution 1: Delete and recreate collection
client.delete_collection("my_docs")
collection = client.create_collection("my_docs")

# Solution 2: Use consistent embedding model
# Always use same model for embeddings:
def get_embedding(text):
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]
```

---

## Deployment Errors

### Error: Hugging Face Spaces deployment fails

**Symptoms**: Space shows error or doesn't start

**Solutions**:
```yaml
# Solution 1: Check README.md header
---
title: My Demo
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0  # Match your Gradio version
app_file: app.py
pinned: true
---

# Solution 2: Check requirements.txt
gradio>=4.0.0
ollama  # Note: Ollama won't work on Spaces - need API

# Solution 3: Use Inference API instead of local Ollama
# Spaces can't run Ollama, use Hugging Face Inference API instead
```

---

### Error: Streamlit Cloud deployment fails

**Symptoms**: App shows error on Streamlit Cloud

**Solutions**:
```toml
# Solution 1: Check requirements.txt format
# Just package names, no paths or git URLs that won't work

# Solution 2: Add secrets properly
# In Streamlit Cloud dashboard, add secrets
# Access with st.secrets["key"]

# Solution 3: Add .streamlit/config.toml
[server]
headless = true
port = 8501
enableCORS = false
```

---

## Reset Procedures

### Reset Gradio

```bash
# Kill Gradio process
fuser -k 7860/tcp

# Or find and kill
lsof -i :7860 | awk 'NR>1 {print $2}' | xargs kill -9
```

### Reset Streamlit

```bash
# Kill Streamlit process
fuser -k 8501/tcp

# Clear cache
rm -rf ~/.streamlit/cache
```

### Reset Ollama

```bash
# Stop Ollama
pkill ollama

# Restart
ollama serve &
sleep 5

# Verify
ollama list
```

### Reset ChromaDB

```python
import chromadb

# Delete persistent data
import shutil
shutil.rmtree("/workspace/chroma_db", ignore_errors=True)

# Or delete in-memory
client = chromadb.Client()
for collection in client.list_collections():
    client.delete_collection(collection.name)
```

---

## Demo Checklist Before Presenting

- [ ] App starts without errors
- [ ] First request works (model warm)
- [ ] Error messages are user-friendly
- [ ] Edge cases handled (empty input, long input)
- [ ] No console errors visible
- [ ] Loading indicators present
- [ ] Examples work correctly
- [ ] Tested on target screen size

---

## Still Stuck?

1. **Check terminal output** - Most errors print details there
2. **Restart everything** - Ollama, then app
3. **Clear caches** - Both Streamlit and ChromaDB
4. **Simplify** - Start with minimal example, add features back
5. **Check versions** - `pip show gradio streamlit`
