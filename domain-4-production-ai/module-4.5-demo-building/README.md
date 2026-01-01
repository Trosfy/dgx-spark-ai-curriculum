# Module 4.5: Demo Building & Prototyping

**Domain:** 4 - Production AI
**Duration:** Week 34 (6-8 hours)
**Prerequisites:** Module 4.4 (Containerization)
**Priority:** P2 (Medium - Portfolio Building)

---

## Overview

You've built an amazing AI system. Now you need to *show* it to people - your team, stakeholders, users, or potential employers. This module focuses on building polished, shareable demos that showcase your work effectively.

**Why This Matters:** A working demo is worth a thousand architecture diagrams. The ability to rapidly prototype and present AI systems is a critical skill for AI engineers and researchers.

### The Kitchen Table Explanation

Think of demos like a restaurant's front-of-house. The kitchen (your model training, RAG pipeline, etc.) is where the magic happens, but nobody sees it. The demo is what customers experience - it needs to be clean, intuitive, and make the food (your AI) look delicious. Even a Michelin-star kitchen needs good service.

---

## Learning Outcomes

By the end of this module, you will be able to:

- Build polished demo applications rapidly with Gradio and Streamlit
- Create shareable prototypes for stakeholders
- Integrate multiple ML components into cohesive applications
- Deploy demos to free hosting platforms

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 4.5.1 | Build complex interfaces with Gradio Blocks API | Apply |
| 4.5.2 | Create multi-page Streamlit applications | Apply |
| 4.5.3 | Integrate RAG, agents, and chat in a single demo | Apply |
| 4.5.4 | Deploy demos to Hugging Face Spaces and Streamlit Cloud | Apply |

---

## Topics

### 4.5.1 Gradio Advanced

- **Blocks API for Complex Layouts**
  - Rows, Columns, Tabs
  - Custom component arrangement
  - Responsive design

- **State Management**
  - Session state
  - Global state
  - State persistence

- **Custom Components**
  - Creating custom inputs/outputs
  - Theming and styling
  - JavaScript integration

- **Authentication**
  - Username/password auth
  - OAuth integration
  - Access control

### 4.5.2 Streamlit Advanced

- **Multi-page Applications**
  - Page navigation
  - Shared state across pages
  - Sidebar organization

- **Session State Deep Dive**
  - Conversation history
  - User preferences
  - Form data persistence

- **Caching Strategies**
  - `@st.cache_data` for data
  - `@st.cache_resource` for models
  - Cache invalidation

- **Performance Optimization**
  - Lazy loading
  - Async operations
  - Progress indicators

### 4.5.3 Prototype Patterns

- **Demo ≠ Production**
  - When to cut corners
  - What to polish
  - Error handling for demos

- **Rapid Iteration**
  - Feature prioritization
  - User feedback integration
  - MVP mindset

- **Stakeholder Communication**
  - Technical vs non-technical audiences
  - Highlighting capabilities
  - Managing expectations

### 4.5.4 Deployment Options

- **Hugging Face Spaces**
  - Free GPU-enabled hosting
  - Gradio and Streamlit support
  - Persistent storage

- **Streamlit Cloud**
  - GitHub integration
  - Secrets management
  - Custom domains

- **Other Options**
  - Railway, Render, Heroku
  - Self-hosting considerations

---

## Labs

### Lab 4.5.1: Complete RAG Demo
**Time:** 3 hours

Build a polished Gradio app showcasing your RAG system.

**Instructions:**
1. Create multi-tab Gradio interface using Blocks API
2. Tab 1: Document upload and indexing
   - Multiple file upload
   - Progress indicator during indexing
   - Success/failure feedback
3. Tab 2: Chat interface
   - Conversation history
   - Source citation display
   - Confidence indicators
4. Tab 3: Settings
   - Model selection
   - Retrieval parameters
   - Theme toggle
5. Add custom CSS for polished appearance
6. Implement error handling with user-friendly messages
7. Deploy to Hugging Face Spaces

**Deliverable:** Full-featured RAG demo on Hugging Face Spaces

---

### Lab 4.5.2: Agent Playground
**Time:** 3 hours

Create a Streamlit app to visualize agent reasoning.

**Instructions:**
1. Build multi-page Streamlit app
2. Page 1: Agent Chat
   - Chat interface with agent
   - Tool call visualization
   - Thought process display (thinking tokens)
3. Page 2: Tool Configuration
   - Enable/disable tools
   - Tool documentation
   - Test individual tools
4. Page 3: Session History
   - Past conversations
   - Export functionality
   - Analytics (tool usage, response times)
5. Implement proper caching for model loading
6. Add session persistence
7. Deploy to Streamlit Cloud

**Deliverable:** Agent playground with reasoning visualization

---

### Lab 4.5.3: Portfolio Demo
**Time:** 2 hours

Create the demo for your capstone project.

**Instructions:**
1. Identify key features to showcase
2. Choose Gradio or Streamlit based on needs
3. Create clean, professional interface
4. Add clear explanations and documentation
5. Include example inputs that work well
6. Create short video walkthrough
7. Deploy to appropriate platform

**Deliverable:** Polished capstone demo ready for portfolio

---

## Guidance

### Gradio Blocks API - Complex Layout

```python
import gradio as gr

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 RAG Chat Application")

    with gr.Tabs():
        with gr.TabItem("📁 Documents"):
            with gr.Row():
                with gr.Column(scale=2):
                    files = gr.File(
                        label="Upload Documents",
                        file_count="multiple",
                        file_types=[".pdf", ".txt", ".md"]
                    )
                    index_btn = gr.Button("Index Documents", variant="primary")

                with gr.Column(scale=1):
                    status = gr.Textbox(label="Status", lines=5)
                    doc_count = gr.Number(label="Documents Indexed", value=0)

        with gr.TabItem("💬 Chat"):
            chatbot = gr.Chatbot(
                height=500,
                show_copy_button=True,
                avatar_images=["user.png", "bot.png"]
            )

            with gr.Row():
                msg = gr.Textbox(
                    label="Message",
                    placeholder="Ask about your documents...",
                    scale=4
                )
                send = gr.Button("Send", scale=1, variant="primary")

            with gr.Accordion("📚 Sources", open=False):
                sources = gr.Markdown()

        with gr.TabItem("⚙️ Settings"):
            with gr.Row():
                model = gr.Dropdown(
                    choices=["llama3.1:8b", "llama3.1:70b"],
                    value="llama3.1:8b",
                    label="Model"
                )
                chunks = gr.Slider(1, 10, 3, step=1, label="Retrieved Chunks")
                temperature = gr.Slider(0, 1, 0.7, step=0.1, label="Temperature")

    # Event handlers
    def index_documents(files):
        # Your indexing logic
        return f"Indexed {len(files)} files", len(files)

    def chat_response(message, history):
        # Your RAG logic
        response = "This is the response..."
        sources_text = "**Source:** document.pdf, page 5"
        return history + [[message, response]], sources_text, ""

    index_btn.click(index_documents, [files], [status, doc_count])
    send.click(chat_response, [msg, chatbot], [chatbot, sources, msg])
    msg.submit(chat_response, [msg, chatbot], [chatbot, sources, msg])

demo.launch()
```

### Streamlit Multi-page App Structure

```
my_app/
├── Home.py              # Main entry point
├── pages/
│   ├── 1_💬_Chat.py
│   ├── 2_📊_Analytics.py
│   └── 3_⚙️_Settings.py
├── utils/
│   ├── __init__.py
│   ├── model.py
│   └── rag.py
└── .streamlit/
    └── config.toml
```

### Home.py

```python
import streamlit as st

st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = []

st.title("🤖 AI Assistant")
st.markdown("""
Welcome to the AI Assistant demo! This application showcases:

- **Chat**: Conversational AI with RAG
- **Analytics**: Usage metrics and performance
- **Settings**: Configure the assistant

Use the sidebar to navigate between pages.
""")

# Show quick stats
col1, col2, col3 = st.columns(3)
col1.metric("Documents", len(st.session_state.documents))
col2.metric("Conversations", len(st.session_state.messages))
col3.metric("Model", "Llama 3.1 8B")
```

### Chat Page with Agent Visualization

```python
# pages/1_💬_Chat.py
import streamlit as st
import json

st.title("💬 Chat with Agent")

# Sidebar for tool configuration
with st.sidebar:
    st.subheader("Agent Tools")
    tools_enabled = {
        "search": st.checkbox("Web Search", value=True),
        "calculator": st.checkbox("Calculator", value=True),
        "code": st.checkbox("Code Executor", value=False),
    }

# Chat interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        # Show tool calls if present
        if "tool_calls" in msg:
            with st.expander("🔧 Tool Calls"):
                for tool in msg["tool_calls"]:
                    st.code(json.dumps(tool, indent=2), language="json")

        # Show thinking if present
        if "thinking" in msg:
            with st.expander("💭 Thinking"):
                st.markdown(msg["thinking"])

# User input
if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Your agent logic here
            response = agent_respond(prompt, tools_enabled)

            st.write(response["content"])

            if response.get("tool_calls"):
                with st.expander("🔧 Tool Calls"):
                    for tool in response["tool_calls"]:
                        st.code(json.dumps(tool, indent=2), language="json")

            if response.get("thinking"):
                with st.expander("💭 Thinking"):
                    st.markdown(response["thinking"])

    st.session_state.messages.append(response)
```

### Caching Models Properly

```python
import streamlit as st
import ollama

@st.cache_resource
def load_model():
    """Load model once and cache it."""
    # This runs only once per session
    return ollama.Client()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_embeddings(text: str):
    """Cache embeddings to avoid recomputation."""
    client = load_model()
    response = client.embeddings(
        model="nomic-embed-text",
        prompt=text
    )
    return response["embedding"]

# Usage
client = load_model()  # Cached
embedding = get_embeddings("Hello world")  # Cached
```

### Custom Gradio Theme

```python
import gradio as gr

# Custom theme
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    font=gr.themes.GoogleFont("Inter"),
).set(
    button_primary_background_fill="*primary_500",
    button_primary_text_color="white",
    block_title_text_weight="600",
)

# Custom CSS
css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}

.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
}

.source-citation {
    background-color: #f0f0f0;
    padding: 0.5rem;
    border-left: 3px solid #007bff;
    margin: 0.5rem 0;
}
"""

with gr.Blocks(theme=theme, css=css) as demo:
    # Your interface here
    pass
```

### Hugging Face Spaces Deployment

```yaml
# README.md for Spaces
---
title: My RAG Demo
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: true
license: mit
---
```

```python
# app.py - Entry point for Spaces
import gradio as gr
import os

# Access secrets
HF_TOKEN = os.environ.get("HF_TOKEN")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Your demo code
demo = gr.Interface(...)

if __name__ == "__main__":
    demo.launch()
```

### Streamlit Cloud Deployment

```toml
# .streamlit/config.toml
[theme]
primaryColor = "#007bff"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
```

```toml
# requirements.txt
streamlit>=1.30.0
ollama>=0.1.0
chromadb>=0.4.22
```

---

## Demo Patterns

### Pattern 1: Progressive Disclosure

Show simple interface first, reveal complexity on demand:

```python
# Start simple
chat = gr.ChatInterface(fn=respond, title="Chat")

# Then reveal advanced options
with gr.Accordion("Advanced Settings", open=False):
    # Complex settings here
```

### Pattern 2: Live Preview

Show results as user types:

```python
prompt = gr.Textbox(label="Prompt")
preview = gr.Textbox(label="Preview")

prompt.change(fn=preview_fn, inputs=prompt, outputs=preview)
```

### Pattern 3: Error Handling for Demos

```python
def safe_respond(message):
    try:
        return model_respond(message)
    except Exception as e:
        return f"Oops! Something went wrong. Try again.\n\nError: {str(e)}"
```

---

## Milestone Checklist

- [ ] RAG demo with full features deployed
- [ ] Agent playground with tool visualization
- [ ] Portfolio demo polished and ready
- [ ] At least one demo on Hugging Face Spaces
- [ ] Video walkthrough created

---

## Common Issues

| Issue | Solution |
|-------|----------|
| Gradio slow to load | Preload models in app startup |
| Streamlit reruns everything | Use `st.cache_resource` for models |
| Spaces timeout | Use smaller models or async loading |
| Demo crashes on edge cases | Add try/except with friendly errors |
| Mobile layout broken | Test with `demo.launch(share=True)` on phone |

---

## Next Steps

After completing this module:
1. Polish your capstone demo
2. Share your demos publicly
3. Proceed to [Module 4.6: Capstone Project](../module-4.6-capstone-project/)

---

## Module Navigation

| Previous | Current | Next |
|----------|---------|------|
| [Module 4.4: Containerization](../module-4.4-containerization-deployment/) | **Module 4.5: Demo Building** | [Module 4.6: Capstone Project](../module-4.6-capstone-project/) |

---

## Study Materials

| Document | Purpose |
|----------|---------|
| [QUICKSTART.md](./QUICKSTART.md) | Build a chat demo in 5 minutes |
| [STUDY_GUIDE.md](./STUDY_GUIDE.md) | Learning objectives and 3-lab roadmap |
| [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) | Gradio and Streamlit code patterns |
| [LAB_PREP.md](./LAB_PREP.md) | Environment setup and model downloads |
| [WORKFLOWS.md](./WORKFLOWS.md) | Step-by-step demo building workflows |
| [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) | Gradio, Streamlit, Ollama error solutions, and FAQs |

---

## Resources

- [Gradio Documentation](https://gradio.app/docs/)
- [Gradio Blocks Guide](https://gradio.app/guides/blocks-and-event-listeners)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [Hugging Face Spaces](https://huggingface.co/spaces)
- [Streamlit Cloud](https://streamlit.io/cloud)
- [Gradio Themes](https://gradio.app/guides/theming-guide)
