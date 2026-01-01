# Module 4.5: Demo Building - Frequently Asked Questions

## Framework Selection

### Q: Gradio or Streamlit - which should I learn first?

**A**: **Gradio for ML demos**, Streamlit for data apps.

| Choose Gradio if | Choose Streamlit if |
|------------------|---------------------|
| Building ML model demo | Building dashboards |
| Need chat interface | Need multi-page app |
| Want 5-minute setup | Want fine control |
| Deploying to HF Spaces | Need complex state |

**Recommendation**: Start with Gradio for this module's focus on ML demos.

---

### Q: Can I use both in one project?

**A**: Not directly together, but you can:

1. **Separate deployments**: Gradio for demo, Streamlit for dashboard
2. **API backend**: Both can call the same API
3. **Iframe embed**: Embed Gradio in Streamlit (not recommended)

Pick one per application for simplicity.

---

## Gradio Questions

### Q: What's the difference between Interface and Blocks?

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

### Q: How do I add streaming to Gradio?

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

### Q: Why is my Gradio demo slow to load?

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

## Streamlit Questions

### Q: Why does my Streamlit app rerun everything?

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

### Q: How do I maintain chat history in Streamlit?

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

### Q: How do I share state between pages?

**A**: `st.session_state` is shared across all pages:

```python
# Page 1: Set value
st.session_state.user_model = "llama3.1:8b"

# Page 2: Read value
model = st.session_state.get("user_model", "default")
```

---

## Deployment Questions

### Q: How do I deploy to Hugging Face Spaces?

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

### Q: How do I get free GPU on Hugging Face Spaces?

**A**:

1. Go to Space Settings
2. Select "Hardware"
3. Choose GPU tier (some are free for verified users)

Or use **ZeroGPU** for serverless GPU access.

---

### Q: How do I handle secrets in deployed demos?

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

### Q: My demo works locally but fails on Spaces. Why?

**A**: Common causes:

| Issue | Solution |
|-------|----------|
| Missing dependencies | Check `requirements.txt` |
| File paths | Use relative paths, not absolute |
| Large models | Use smaller models or external hosting |
| Port conflicts | Gradio uses 7860, let platform handle it |
| Memory limits | Reduce model size or upgrade hardware |

---

## Demo Design

### Q: How polished should a demo be?

**A**: Depends on audience:

| Audience | Polish Level |
|----------|-------------|
| Personal testing | Minimal - just works |
| Team demo | Medium - clear UI, handles errors |
| Stakeholders | High - polished, no crashes |
| Public/portfolio | Very high - professional quality |

Focus on: error handling, clear instructions, working examples.

---

### Q: What makes a good demo?

**A**: The 5 Cs:

1. **Clear** - Obvious what it does
2. **Concise** - No unnecessary features
3. **Correct** - Works reliably
4. **Complete** - Handles edge cases gracefully
5. **Compelling** - Shows value immediately

---

### Q: How do I handle errors gracefully?

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

## Still Have Questions?

- Check [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) for code patterns
- See [STUDY_GUIDE.md](./STUDY_GUIDE.md) for learning path
- Review module [Resources](./README.md#resources) for official docs
