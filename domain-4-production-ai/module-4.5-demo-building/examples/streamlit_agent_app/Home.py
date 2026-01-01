"""
Agent Playground - Streamlit Application

A multi-page Streamlit app for visualizing agent reasoning.

To run:
    pip install streamlit ollama
    streamlit run Home.py
"""

import streamlit as st

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Agent Playground",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tool_calls" not in st.session_state:
    st.session_state.tool_calls = []
if "enabled_tools" not in st.session_state:
    st.session_state.enabled_tools = ["calculator", "datetime", "weather", "web_search"]
if "model" not in st.session_state:
    st.session_state.model = "qwen3:4b"

# Main page content
st.title("🤖 Agent Playground")

st.markdown("""
Welcome to the Agent Playground! This application lets you:

- **Chat** with an AI agent that can use tools
- **Visualize** the agent's reasoning process
- **Configure** which tools are available
- **Analyze** conversation history

---

### Getting Started

1. **Configure Tools** → Go to 🔧 Tools to enable/disable tools
2. **Start Chatting** → Go to 💬 Chat to interact with the agent
3. **Review History** → Go to 📊 History to see analytics

---
""")

# Quick stats
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Messages",
        len(st.session_state.messages),
        help="Total messages in current conversation"
    )

with col2:
    st.metric(
        "Tool Calls",
        len(st.session_state.tool_calls),
        help="Total tool invocations"
    )

with col3:
    st.metric(
        "Active Tools",
        len(st.session_state.enabled_tools),
        help="Number of enabled tools"
    )

# Sidebar
with st.sidebar:
    st.markdown("### Current Settings")
    st.info(f"**Model:** {st.session_state.model}")
    st.info(f"**Tools:** {', '.join(st.session_state.enabled_tools) or 'None'}")

    st.markdown("---")
    st.markdown("### Quick Links")
    st.page_link("pages/1_💬_Chat.py", label="💬 Chat", icon="💬")
    st.page_link("pages/2_🔧_Tools.py", label="🔧 Tools", icon="🔧")
    st.page_link("pages/3_📊_History.py", label="📊 History", icon="📊")

    st.markdown("---")
    st.markdown("*Built with Streamlit | Module 4.5*")
