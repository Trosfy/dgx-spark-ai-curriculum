# Module 4.5: Demo Building - Workflow Cheatsheets

## Workflow 1: Gradio ChatInterface → HF Spaces

### When to Use
Quick ML chat demo with streaming, deployed publicly.

### Step-by-Step

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Create Chat Function                                │
├─────────────────────────────────────────────────────────────┤
│ □ Import dependencies                                       │
│ □ Create streaming chat function                            │
│                                                             │
│ Code:                                                       │
│ ```python                                                   │
│ import gradio as gr                                         │
│ import ollama                                               │
│                                                             │
│ def chat(message, history):                                 │
│     messages = [{"role": "user", "content": message}]       │
│     response = ollama.chat(                                 │
│         model="qwen3:4b",                                │
│         messages=messages,                                  │
│         stream=True                                         │
│     )                                                       │
│     partial = ""                                            │
│     for chunk in response:                                  │
│         partial += chunk["message"]["content"]              │
│         yield partial                                       │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Function works in Python REPL                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Create Interface                                    │
├─────────────────────────────────────────────────────────────┤
│ □ Wrap in ChatInterface                                     │
│ □ Add title and examples                                    │
│ □ Apply theme                                               │
│                                                             │
│ Code:                                                       │
│ ```python                                                   │
│ demo = gr.ChatInterface(                                    │
│     fn=chat,                                                │
│     title="My AI Assistant",                                │
│     examples=["Hello!", "Explain AI"],                      │
│     theme=gr.themes.Soft()                                  │
│ )                                                           │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: demo.launch() works locally                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Prepare for Deployment                              │
├─────────────────────────────────────────────────────────────┤
│ □ Create requirements.txt                                   │
│ □ Create README.md with frontmatter                         │
│ □ Test locally with minimal dependencies                    │
│                                                             │
│ requirements.txt:                                           │
│ ```                                                         │
│ gradio>=4.0.0                                               │
│ ollama>=0.1.0                                               │
│ ```                                                         │
│                                                             │
│ README.md:                                                  │
│ ```yaml                                                     │
│ ---                                                         │
│ title: My AI Assistant                                      │
│ emoji: "robot"                                              │
│ sdk: gradio                                                 │
│ sdk_version: 4.0.0                                          │
│ app_file: app.py                                            │
│ ---                                                         │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: All files ready                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Deploy to Spaces                                    │
├─────────────────────────────────────────────────────────────┤
│ □ Create new Space on huggingface.co                        │
│ □ Push code to Space                                        │
│ □ Wait for build                                            │
│ □ Test public URL                                           │
│                                                             │
│ Code:                                                       │
│ ```bash                                                     │
│ git init                                                    │
│ git add .                                                   │
│ git commit -m "Initial demo"                                │
│ git remote add spaces https://huggingface.co/spaces/user/demo │
│ git push spaces main                                        │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Demo accessible at public URL                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Workflow 2: Gradio Blocks RAG Demo

### When to Use
Complex demo with document upload, chat, and settings tabs.

### Step-by-Step

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Design Layout                                       │
├─────────────────────────────────────────────────────────────┤
│ □ Plan tabs: Documents, Chat, Settings                      │
│ □ Sketch component layout                                   │
│                                                             │
│ Layout:                                                     │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [Documents] [Chat] [Settings]                           │ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │ Tab Content                                             │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ✓ Checkpoint: Layout designed                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Build Document Tab                                  │
├─────────────────────────────────────────────────────────────┤
│ □ File upload component                                     │
│ □ Index button                                              │
│ □ Status display                                            │
│                                                             │
│ Code:                                                       │
│ ```python                                                   │
│ with gr.Blocks() as demo:                                   │
│     with gr.Tabs():                                         │
│         with gr.TabItem("Documents"):                       │
│             files = gr.File(file_count="multiple")          │
│             index_btn = gr.Button("Index")                  │
│             status = gr.Textbox(label="Status")             │
│             index_btn.click(index_docs, files, status)      │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Document upload works                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Build Chat Tab                                      │
├─────────────────────────────────────────────────────────────┤
│ □ Chatbot component                                         │
│ □ Message input                                             │
│ □ Sources accordion                                         │
│                                                             │
│ Code:                                                       │
│ ```python                                                   │
│ with gr.TabItem("Chat"):                                    │
│     chatbot = gr.Chatbot(height=500)                        │
│     msg = gr.Textbox(label="Ask about your docs")           │
│     with gr.Accordion("Sources", open=False):               │
│         sources = gr.Markdown()                             │
│     msg.submit(rag_chat, [msg, chatbot], [chatbot, sources])│
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Chat with sources works                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Build Settings Tab                                  │
├─────────────────────────────────────────────────────────────┤
│ □ Model selector                                            │
│ □ Temperature slider                                        │
│ □ Chunk count slider                                        │
│                                                             │
│ Code:                                                       │
│ ```python                                                   │
│ with gr.TabItem("Settings"):                                │
│     model = gr.Dropdown(["qwen3:8b", "qwen3:32b"])    │
│     temp = gr.Slider(0, 1, 0.7, label="Temperature")        │
│     chunks = gr.Slider(1, 10, 3, label="Chunks")            │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Settings affect chat behavior                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Polish and Deploy                                   │
├─────────────────────────────────────────────────────────────┤
│ □ Add theme                                                 │
│ □ Add error handling                                        │
│ □ Test all features                                         │
│ □ Deploy to Spaces                                          │
│                                                             │
│ ✓ Checkpoint: Production-ready demo live                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Workflow 3: Streamlit Multi-page App

### When to Use
Dashboard-style app with multiple pages and persistent state.

### Step-by-Step

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Create Project Structure                            │
├─────────────────────────────────────────────────────────────┤
│ □ Create directory structure                                │
│ □ Initialize session state                                  │
│                                                             │
│ Structure:                                                  │
│ ```                                                         │
│ my_app/                                                     │
│ ├── Home.py                                                 │
│ ├── pages/                                                  │
│ │   ├── 1_Chat.py                                           │
│ │   ├── 2_Analytics.py                                      │
│ │   └── 3_Settings.py                                       │
│ └── .streamlit/                                             │
│     └── config.toml                                         │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: streamlit run Home.py shows navigation        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Create Home Page                                    │
├─────────────────────────────────────────────────────────────┤
│ □ Page config                                               │
│ □ Welcome content                                           │
│ □ Initialize shared state                                   │
│                                                             │
│ Home.py:                                                    │
│ ```python                                                   │
│ import streamlit as st                                      │
│                                                             │
│ st.set_page_config(page_title="AI Dashboard", layout="wide")│
│                                                             │
│ if "messages" not in st.session_state:                      │
│     st.session_state.messages = []                          │
│                                                             │
│ st.title("AI Dashboard")                                    │
│ st.write("Welcome! Use sidebar to navigate.")               │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Home page renders                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Create Chat Page                                    │
├─────────────────────────────────────────────────────────────┤
│ □ Chat message display                                      │
│ □ Input handling                                            │
│ □ Model integration                                         │
│                                                             │
│ pages/1_Chat.py:                                            │
│ ```python                                                   │
│ import streamlit as st                                      │
│                                                             │
│ st.title("Chat")                                            │
│                                                             │
│ for msg in st.session_state.messages:                       │
│     st.chat_message(msg["role"]).write(msg["content"])      │
│                                                             │
│ if prompt := st.chat_input():                               │
│     st.session_state.messages.append(                       │
│         {"role": "user", "content": prompt}                 │
│     )                                                       │
│     # Add response logic                                    │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Chat persists across page navigation          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Add Analytics Page                                  │
├─────────────────────────────────────────────────────────────┤
│ □ Metrics display                                           │
│ □ Charts                                                    │
│ □ Data from session state                                   │
│                                                             │
│ pages/2_Analytics.py:                                       │
│ ```python                                                   │
│ import streamlit as st                                      │
│                                                             │
│ st.title("Analytics")                                       │
│                                                             │
│ col1, col2 = st.columns(2)                                  │
│ col1.metric("Messages", len(st.session_state.messages))     │
│ col2.metric("Model", st.session_state.get("model", "N/A"))  │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Analytics shows real data                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Deploy to Streamlit Cloud                           │
├─────────────────────────────────────────────────────────────┤
│ □ Push to GitHub                                            │
│ □ Connect to Streamlit Cloud                                │
│ □ Configure secrets                                         │
│ □ Deploy                                                    │
│                                                             │
│ ✓ Checkpoint: App live at streamlit.app URL                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Workflow 4: Portfolio Demo Polish

### When to Use
Final polish before sharing publicly or adding to portfolio.

### Step-by-Step

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Error Handling                                      │
├─────────────────────────────────────────────────────────────┤
│ □ Wrap all functions in try/except                          │
│ □ Show friendly error messages                              │
│ □ Test edge cases                                           │
│                                                             │
│ Pattern:                                                    │
│ ```python                                                   │
│ def safe_predict(input):                                    │
│     try:                                                    │
│         return model.predict(input)                         │
│     except Exception as e:                                  │
│         return f"Error: {str(e)}. Please try again."        │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: No crashes on any input                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: UI Polish                                           │
├─────────────────────────────────────────────────────────────┤
│ □ Apply professional theme                                  │
│ □ Add clear labels and instructions                         │
│ □ Include example inputs                                    │
│ □ Test on mobile                                            │
│                                                             │
│ ✓ Checkpoint: Looks professional, easy to use               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Documentation                                       │
├─────────────────────────────────────────────────────────────┤
│ □ Clear README with setup instructions                      │
│ □ In-app help text                                          │
│ □ Example use cases                                         │
│                                                             │
│ ✓ Checkpoint: New user can understand in 30 seconds         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Create Video Walkthrough                            │
├─────────────────────────────────────────────────────────────┤
│ □ Record 3-5 minute demo                                    │
│ □ Show key features                                         │
│ □ Explain architecture briefly                              │
│ □ Upload to YouTube/Loom                                    │
│                                                             │
│ ✓ Checkpoint: Video linked in README                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Final Deploy                                        │
├─────────────────────────────────────────────────────────────┤
│ □ Test production deployment                                │
│ □ Verify all features work                                  │
│ □ Share link                                                │
│                                                             │
│ ✓ Checkpoint: Portfolio-ready demo live!                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Demo Type Decision Flowchart

```
                    Start: Build Demo
                           │
                           ▼
              ┌────────────────────────┐
              │ Primary purpose?       │
              └───────────┬────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
     ML Demo         Dashboard        Portfolio
          │               │               │
          ▼               ▼               ▼
  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
  │ Gradio       │ │ Streamlit    │ │ Either +     │
  │ ChatInterface│ │ Multi-page   │ │ Max polish   │
  └──────────────┘ └──────────────┘ └──────────────┘
          │               │               │
          ▼               ▼               ▼
  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
  │ HF Spaces    │ │ Streamlit    │ │ Video +      │
  │ (free GPU)   │ │ Cloud        │ │ Documentation│
  └──────────────┘ └──────────────┘ └──────────────┘
```
