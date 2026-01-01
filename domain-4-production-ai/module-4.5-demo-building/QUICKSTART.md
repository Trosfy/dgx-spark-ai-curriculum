# Module 4.5: Demo Building - Quickstart

## Time: ~5 minutes

## What You'll Build

Create a simple chat demo with Gradio and deploy locally.

## Before You Start

- [ ] Python 3.11+ environment
- [ ] Ollama running with a model

## Let's Go!

### Step 1: Install Gradio

```bash
pip install gradio ollama
```

### Step 2: Create a Chat Demo

```python
# save as demo.py
import gradio as gr
import ollama

def chat(message, history):
    """Chat with the model."""
    messages = []
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})

    response = ollama.chat(
        model="qwen3:4b",
        messages=messages,
        stream=True
    )

    partial = ""
    for chunk in response:
        partial += chunk["message"]["content"]
        yield partial

demo = gr.ChatInterface(
    fn=chat,
    title="My AI Assistant",
    description="Chat with Llama 3.2",
    examples=["Hello!", "Explain AI in one sentence", "Write a haiku"],
)

demo.launch()
```

### Step 3: Run the Demo

```bash
python demo.py
```

### Step 4: Open in Browser

Visit: http://localhost:7860

**You should see:**
- Chat interface with your title
- Example prompts to click
- Streaming responses

## You Did It!

You just built a working AI demo in 5 minutes! In the full module, you'll learn:

- **Gradio Blocks API**: Complex layouts with tabs, rows, columns
- **Streamlit**: Multi-page applications with session state
- **RAG Demos**: Document upload, source citations
- **Agent Playgrounds**: Tool visualization, reasoning display
- **Deployment**: Hugging Face Spaces, Streamlit Cloud

## Next Steps

1. **Add custom theme**: `gr.themes.Soft()` for polished look
2. **Deploy to Spaces**: Share publicly on Hugging Face
3. **Full tutorial**: Start with [STUDY_GUIDE.md](./STUDY_GUIDE.md)
