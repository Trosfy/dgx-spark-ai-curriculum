"""
Sample Multi-Page Streamlit App - Home Page.

This is a complete, runnable example of a multi-page Streamlit application.
Use it as a reference when building your own apps.

Author: Professor SPARK
Module: 4.5 - Demo Building & Prototyping
"""

import streamlit as st

st.set_page_config(
    page_title="AI Demo App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "settings" not in st.session_state:
    st.session_state.settings = {
        "model": "llama3.1:8b",
        "temperature": 0.7,
        "theme": "light"
    }

if "documents" not in st.session_state:
    st.session_state.documents = []

# Main content
st.title("🤖 AI Demo Application")

st.markdown("""
Welcome to the AI Demo Application! This multi-page app demonstrates
best practices for building Streamlit applications.

### Features

- **💬 Chat**: Conversational AI interface with history
- **📊 Analytics**: View usage metrics and session data
- **⚙️ Settings**: Configure the application

### Navigation

Use the sidebar on the left to navigate between pages.
""")

st.markdown("---")

# Quick stats
st.subheader("📈 Quick Stats")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Messages", len(st.session_state.messages))

with col2:
    st.metric("Documents", len(st.session_state.documents))

with col3:
    st.metric("Model", st.session_state.settings["model"].split(":")[0])

with col4:
    st.metric("Temperature", st.session_state.settings["temperature"])

st.markdown("---")

# Quick actions
st.subheader("🚀 Quick Actions")

col1, col2, col3 = st.columns(3)

with col1:
    st.page_link("pages/1_💬_Chat.py", label="Start Chatting", icon="💬")

with col2:
    st.page_link("pages/2_📊_Analytics.py", label="View Analytics", icon="📊")

with col3:
    st.page_link("pages/3_⚙️_Settings.py", label="Configure Settings", icon="⚙️")

# Footer
st.markdown("---")
st.caption("Built with Streamlit | Module 4.5 Demo")
