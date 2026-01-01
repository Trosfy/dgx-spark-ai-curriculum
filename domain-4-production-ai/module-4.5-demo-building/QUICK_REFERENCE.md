# Module 4.5: Demo Building - Quick Reference

## Gradio Essentials

### Simple Chat Interface

```python
import gradio as gr

def respond(message, history):
    return f"You said: {message}"

demo = gr.ChatInterface(fn=respond, title="My Chat")
demo.launch()
```

### Streaming Chat

```python
import gradio as gr
import ollama

def chat(message, history):
    messages = [{"role": "user", "content": message}]
    response = ollama.chat(model="llama3.2:3b", messages=messages, stream=True)

    partial = ""
    for chunk in response:
        partial += chunk["message"]["content"]
        yield partial

demo = gr.ChatInterface(fn=chat)
demo.launch()
```

### Blocks API - Tabs Layout

```python
import gradio as gr

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Chat"):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Message")

        with gr.TabItem("Settings"):
            model = gr.Dropdown(["model-a", "model-b"], label="Model")
            temp = gr.Slider(0, 1, 0.7, label="Temperature")

demo.launch()
```

### Blocks API - Rows and Columns

```python
import gradio as gr

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(label="Input")
        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Output")

demo.launch()
```

### File Upload with Processing

```python
import gradio as gr

def process_file(file):
    return f"Processed: {file.name}"

demo = gr.Interface(
    fn=process_file,
    inputs=gr.File(label="Upload", file_types=[".pdf", ".txt"]),
    outputs="text"
)
demo.launch()
```

### Custom Theme

```python
import gradio as gr

theme = gr.themes.Soft(
    primary_hue="blue",
    font=gr.themes.GoogleFont("Inter")
)

with gr.Blocks(theme=theme) as demo:
    gr.Markdown("# Styled Demo")
    # ... components

demo.launch()
```

---

## Streamlit Essentials

### Basic Chat

```python
import streamlit as st

st.title("Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask something"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    response = "Your response here"
    st.chat_message("assistant").write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
```

### Multi-page App Structure

```
my_app/
├── Home.py              # Entry point
├── pages/
│   ├── 1_Chat.py
│   ├── 2_Settings.py
│   └── 3_About.py
└── .streamlit/
    └── config.toml
```

### Sidebar Navigation

```python
import streamlit as st

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Chat", "Settings", "About"])

if page == "Chat":
    st.title("Chat Page")
elif page == "Settings":
    st.title("Settings Page")
```

### Caching Models

```python
import streamlit as st

@st.cache_resource  # Persists across reruns
def load_model():
    import ollama
    return ollama.Client()

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_embeddings(text):
    client = load_model()
    return client.embeddings(model="nomic-embed-text", prompt=text)
```

### Columns and Metrics

```python
import streamlit as st

col1, col2, col3 = st.columns(3)
col1.metric("Latency", "120ms", "-10ms")
col2.metric("Throughput", "50 req/s", "+5")
col3.metric("GPU", "78%", "+3%")
```

### Expanders and Accordions

```python
import streamlit as st

with st.expander("Show Details"):
    st.write("Hidden content here")
    st.code("print('hello')")
```

### File Upload

```python
import streamlit as st

uploaded = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded:
    content = uploaded.read()
    st.success(f"Uploaded: {uploaded.name}")
```

---

## Deployment Commands

### Hugging Face Spaces

```yaml
# README.md frontmatter
---
title: My Demo
emoji: "robot"
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
---
```

```bash
# Push to Spaces
git remote add spaces https://huggingface.co/spaces/username/my-demo
git push spaces main
```

### Streamlit Cloud

```bash
# requirements.txt
streamlit>=1.30.0
ollama>=0.1.0
```

```toml
# .streamlit/config.toml
[theme]
primaryColor = "#007bff"
backgroundColor = "#ffffff"
```

---

## Common Patterns

### Error Handling for Demos

```python
def safe_respond(message):
    try:
        return model_respond(message)
    except Exception as e:
        return f"Oops! Something went wrong: {str(e)}"
```

### Progressive Disclosure

```python
# Show simple first, complex on demand
with gr.Accordion("Advanced Settings", open=False):
    temperature = gr.Slider(0, 1, 0.7)
    max_tokens = gr.Slider(100, 2000, 500)
```

### Loading Indicators

```python
# Gradio
with gr.Blocks() as demo:
    btn = gr.Button("Generate")
    btn.click(fn=slow_function, outputs=output, show_progress=True)

# Streamlit
with st.spinner("Thinking..."):
    result = slow_function()
```

### Source Citations Display

```python
# Gradio
with gr.Accordion("Sources", open=False):
    sources = gr.Markdown()

# Streamlit
with st.expander("View Sources"):
    for source in sources:
        st.markdown(f"- {source['title']}: {source['content'][:100]}...")
```

---

## Quick Comparison

| Feature | Gradio | Streamlit |
|---------|--------|-----------|
| Chat interface | `gr.ChatInterface` | Manual with `st.chat_message` |
| Streaming | `yield` in function | `st.write_stream` |
| State | `gr.State` | `st.session_state` |
| Tabs | `gr.Tabs()` | `st.tabs()` |
| File upload | `gr.File` | `st.file_uploader` |
| Deployment | HF Spaces (free GPU) | Streamlit Cloud |
| Learning curve | 5 min | 30 min |
