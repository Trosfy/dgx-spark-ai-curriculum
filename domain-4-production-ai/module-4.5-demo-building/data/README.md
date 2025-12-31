# Data Files for Module 4.5: Demo Building & Prototyping

This directory contains templates, assets, and example code for building demos.

## Demo Templates

### Gradio Templates

```
templates/gradio/
├── basic_chat.py           # Simple chat interface
├── rag_demo.py             # RAG with document upload
├── agent_playground.py     # Agent with tool visualization
├── multimodal_demo.py      # Image + text interface
└── custom_theme.py         # Professional styling
```

### Streamlit Templates

```
templates/streamlit/
├── basic_app/
│   ├── Home.py
│   └── pages/
│       └── Chat.py
├── multi_page_app/
│   ├── Home.py
│   └── pages/
│       ├── 1_Chat.py
│       ├── 2_Analytics.py
│       └── 3_Settings.py
└── agent_app/
    ├── Home.py
    ├── pages/
    │   ├── 1_Agent.py
    │   ├── 2_Tools.py
    │   └── 3_History.py
    └── utils/
        └── agent.py
```

## Sample Assets

### Placeholder Images

```
assets/
├── bot_avatar.png          # Bot avatar for chat
├── user_avatar.png         # User avatar for chat
├── logo.png                # App logo
└── background.png          # Background image
```

### Custom CSS

```css
/* assets/custom.css */

/* Modern card style */
.card {
    background: white;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    padding: 1.5rem;
    margin: 1rem 0;
}

/* Gradient header */
.header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 12px 12px 0 0;
}

/* Source citation style */
.source {
    background: #f8f9fa;
    border-left: 4px solid #007bff;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.9em;
}

/* Tool call visualization */
.tool-call {
    background: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-family: monospace;
}

/* Thinking block */
.thinking {
    background: #e8f4fd;
    border: 1px dashed #0d6efd;
    border-radius: 8px;
    padding: 1rem;
    font-style: italic;
    color: #495057;
}
```

## Example Data for Demos

### Sample Documents for RAG Demo

```
sample_docs/
├── product_manual.pdf      # Technical documentation
├── faq.md                  # Frequently asked questions
├── company_policy.txt      # Policy document
└── api_reference.md        # API documentation
```

### Sample Conversations

```json
// sample_conversations.json
{
  "examples": [
    {
      "user": "What are the return policies?",
      "assistant": "Based on our company policy...",
      "sources": ["company_policy.txt", "section 5"]
    },
    {
      "user": "How do I authenticate to the API?",
      "assistant": "To authenticate, you'll need to...",
      "sources": ["api_reference.md", "Authentication section"]
    }
  ]
}
```

## Hugging Face Spaces Configuration

### Gradio Space README

```markdown
---
title: My RAG Assistant
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.19.0
app_file: app.py
pinned: true
license: mit
models:
  - meta-llama/Llama-3.1-8B-Instruct
tags:
  - rag
  - llm
  - chat
---

# My RAG Assistant

An AI assistant that can answer questions about your documents.

## Features
- Document upload and indexing
- Conversational Q&A with source citations
- Multiple model support

## Usage
1. Upload your documents in the Documents tab
2. Ask questions in the Chat tab
3. View sources for each response
```

### Streamlit Space README

```markdown
---
title: Agent Playground
emoji: 🛠️
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.31.0
app_file: Home.py
pinned: true
---

# Agent Playground

Interactive playground for AI agents with tool visualization.
```

## Deployment Checklists

### Pre-deployment Checklist

- [ ] All secrets stored in environment variables
- [ ] Error handling for all user inputs
- [ ] Loading states for slow operations
- [ ] Mobile-responsive layout tested
- [ ] Example inputs that work reliably
- [ ] Clear error messages for failures
- [ ] Rate limiting considered
- [ ] Memory usage optimized

### Hugging Face Spaces Checklist

- [ ] README.md with proper frontmatter
- [ ] requirements.txt with pinned versions
- [ ] .gitignore for local files
- [ ] Secrets configured in Space settings
- [ ] Hardware (GPU) selected if needed
- [ ] Persistent storage enabled if needed

### Streamlit Cloud Checklist

- [ ] .streamlit/config.toml configured
- [ ] requirements.txt complete
- [ ] secrets.toml for local testing
- [ ] Secrets configured in Streamlit Cloud
- [ ] Memory usage under limit

## Video Walkthrough Guide

### Recording Tips

1. **Script your demo**: Know what you'll show before recording
2. **Use clean example inputs**: Pre-prepare inputs that work well
3. **Show the happy path first**: Demonstrate success before edge cases
4. **Keep it short**: 3-5 minutes is ideal
5. **Add narration**: Explain what you're doing and why

### Recommended Tools

- **OBS Studio**: Free screen recording
- **Loom**: Quick cloud recording with sharing
- **Camtasia**: Professional editing
- **FFmpeg**: Command-line conversion

### Video Outline Template

```
1. Introduction (30s)
   - What the demo does
   - Key features

2. Document Upload (1m)
   - Show file upload
   - Show indexing progress
   - Verify documents indexed

3. Chat Demo (2m)
   - Ask simple question
   - Show source citations
   - Ask follow-up question
   - Show conversation context

4. Advanced Features (1m)
   - Settings configuration
   - Model selection
   - Any unique features

5. Conclusion (30s)
   - Summary of capabilities
   - Where to access the demo
```

## Common Demo Patterns

### Loading State Pattern

```python
# Gradio
with gr.Row():
    status = gr.Textbox(label="Status", value="Ready")

def process_with_status(input):
    yield gr.update(value="Processing...")
    result = do_processing(input)
    yield gr.update(value="Complete!")
    return result

# Streamlit
with st.spinner("Processing..."):
    result = do_processing(input)
st.success("Complete!")
```

### Error Recovery Pattern

```python
def safe_process(input):
    """Process with graceful error handling."""
    try:
        result = risky_operation(input)
        return result, "Success"
    except ModelError as e:
        return None, f"Model error: Please try again. ({e})"
    except TimeoutError:
        return None, "Request timed out. Please try a shorter input."
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"
```

### Caching Pattern

```python
# Streamlit - cache model loading
@st.cache_resource
def load_model():
    return MyModel.load("path/to/model")

# Streamlit - cache data with TTL
@st.cache_data(ttl=3600)
def get_embeddings(text):
    return model.encode(text)

# Gradio - use global state
_model = None
def get_model():
    global _model
    if _model is None:
        _model = MyModel.load("path/to/model")
    return _model
```

## Resources

- [Gradio Blocks Tutorial](https://gradio.app/guides/blocks-and-event-listeners)
- [Streamlit Multi-page Apps](https://docs.streamlit.io/library/get-started/multipage-apps)
- [HF Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-community-cloud)
