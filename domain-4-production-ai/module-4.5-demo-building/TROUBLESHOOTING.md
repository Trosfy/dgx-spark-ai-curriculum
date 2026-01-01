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
            model="qwen3:8b",
            messages=[{"role": "user", "content": message}]
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}. Please try again."

# Solution 2: For streaming, catch generator errors
def stream_chat(message, history):
    try:
        partial = ""
        for chunk in ollama.chat(model="qwen3:8b", messages=messages, stream=True):
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
client.generate(model="qwen3:8b", prompt="Hello", options={"num_predict": 1})
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
    return ollama.embeddings(model="qwen3-embedding:8b", prompt=text)
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
ollama._types.ResponseError: model 'qwen3:8b' not found
```

**Solutions**:
```bash
# Solution 1: Pull the model
ollama pull qwen3:8b

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
ollama stop qwen3:8b  # If supported

# Solution 3: Reduce context length
response = ollama.chat(
    model="qwen3:8b",
    messages=messages,
    options={"num_ctx": 2048}  # Reduce from default 4096
)

# Solution 4: Limit response length
response = ollama.generate(
    model="qwen3:8b",
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
    response = ollama.embeddings(model="qwen3-embedding:8b", prompt=text)
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

## ❓ Frequently Asked Questions

### Framework Selection

**Q: Gradio or Streamlit - which should I learn first?**

**A**: **Gradio for ML demos**, Streamlit for data apps.

| Choose Gradio if | Choose Streamlit if |
|------------------|---------------------|
| Building ML model demo | Building dashboards |
| Need chat interface | Need multi-page app |
| Want 5-minute setup | Want fine control |
| Deploying to HF Spaces | Need complex state |

**Recommendation**: Start with Gradio for this module's focus on ML demos.

---

**Q: Can I use both in one project?**

**A**: Not directly together, but you can:

1. **Separate deployments**: Gradio for demo, Streamlit for dashboard
2. **API backend**: Both can call the same API
3. **Iframe embed**: Embed Gradio in Streamlit (not recommended)

Pick one per application for simplicity.

---

### Gradio Questions

**Q: What's the difference between Interface and Blocks?**

**A**:

| Interface | Blocks |
|-----------|--------|
| Simple, one-function | Complex, multi-component |
| Auto-layout | Manual layout control |
| Quick prototypes | Production demos |
| Limited customization | Full customization |

```python
# Interface - simple
gr.Interface(fn=predict, inputs="text", outputs="text")

# Blocks - flexible
with gr.Blocks() as demo:
    with gr.Row():
        input = gr.Textbox()
        output = gr.Textbox()
    btn = gr.Button()
    btn.click(fn=predict, inputs=input, outputs=output)
```

---

**Q: How do I add streaming to Gradio?**

**A**: Use `yield` instead of `return`:

```python
def stream_response(message, history):
    partial = ""
    for chunk in model.generate(message, stream=True):
        partial += chunk
        yield partial  # Not return!

demo = gr.ChatInterface(fn=stream_response)
```

---

**Q: Why is my Gradio demo slow to load?**

**A**: Model loading on first request. Solutions:

1. **Preload in app startup**:
```python
# Load before creating interface
model = load_model()

def predict(x):
    return model(x)  # Model already loaded

demo = gr.Interface(fn=predict, ...)
```

2. **Use GPU Spaces** for free GPU-accelerated hosting

3. **Show loading state**:
```python
btn.click(fn=slow_fn, outputs=out, show_progress=True)
```

---

### Streamlit Questions

**Q: Why does my Streamlit app rerun everything?**

**A**: Streamlit reruns the entire script on every interaction. Use caching:

```python
@st.cache_resource  # For models, connections
def load_model():
    return heavy_model_load()

@st.cache_data  # For data transformations
def process_data(data):
    return expensive_computation(data)
```

---

**Q: How do I maintain chat history in Streamlit?**

**A**: Use `st.session_state`:

```python
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Add new message
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
```

---

**Q: How do I share state between pages?**

**A**: `st.session_state` is shared across all pages:

```python
# Page 1: Set value
st.session_state.user_model = "qwen3:8b"

# Page 2: Read value
model = st.session_state.get("user_model", "default")
```

---

### Deployment Questions

**Q: How do I deploy to Hugging Face Spaces?**

**A**:

1. **Create Space** at huggingface.co/spaces
2. **Add README.md**:
```yaml
---
title: My Demo
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
---
```
3. **Push code**:
```bash
git add .
git commit -m "Initial demo"
git push
```

Space will auto-build and deploy.

---

**Q: How do I get free GPU on Hugging Face Spaces?**

**A**:

1. Go to Space Settings
2. Select "Hardware"
3. Choose GPU tier (some are free for verified users)

Or use **ZeroGPU** for serverless GPU access.

---

**Q: How do I handle secrets in deployed demos?**

**A**:

**Hugging Face Spaces**:
- Go to Settings > Repository secrets
- Add secrets (e.g., `HF_TOKEN`)
- Access in code: `os.environ.get("HF_TOKEN")`

**Streamlit Cloud**:
- Use `.streamlit/secrets.toml`:
```toml
API_KEY = "your-key"
```
- Access: `st.secrets["API_KEY"]`

---

**Q: My demo works locally but fails on Spaces. Why?**

**A**: Common causes:

| Issue | Solution |
|-------|----------|
| Missing dependencies | Check `requirements.txt` |
| File paths | Use relative paths, not absolute |
| Large models | Use smaller models or external hosting |
| Port conflicts | Gradio uses 7860, let platform handle it |
| Memory limits | Reduce model size or upgrade hardware |

---

### Demo Design

**Q: How polished should a demo be?**

**A**: Depends on audience:

| Audience | Polish Level |
|----------|-------------|
| Personal testing | Minimal - just works |
| Team demo | Medium - clear UI, handles errors |
| Stakeholders | High - polished, no crashes |
| Public/portfolio | Very high - professional quality |

Focus on: error handling, clear instructions, working examples.

---

**Q: What makes a good demo?**

**A**: The 5 Cs:

1. **Clear** - Obvious what it does
2. **Concise** - No unnecessary features
3. **Correct** - Works reliably
4. **Complete** - Handles edge cases gracefully
5. **Compelling** - Shows value immediately

---

**Q: How do I handle errors gracefully?**

**A**: Wrap everything in try/except with friendly messages:

```python
def safe_respond(message):
    try:
        return model.generate(message)
    except ConnectionError:
        return "Model service unavailable. Please try again."
    except Exception as e:
        return f"Something went wrong. Error: {str(e)}"
```

---

## Still Stuck?

1. **Check terminal output** - Most errors print details there
2. **Restart everything** - Ollama, then app
3. **Clear caches** - Both Streamlit and ChromaDB
4. **Simplify** - Start with minimal example, add features back
5. **Check versions** - `pip show gradio streamlit`
