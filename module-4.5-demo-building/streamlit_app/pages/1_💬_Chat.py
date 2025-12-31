"""Chat Page for the Sample Streamlit App."""

import streamlit as st
import time

st.set_page_config(page_title="Chat", page_icon="💬")

# Ensure session state is initialized
if "messages" not in st.session_state:
    st.session_state.messages = []
if "settings" not in st.session_state:
    st.session_state.settings = {"model": "llama3.1:8b", "temperature": 0.7}

st.title("💬 Chat")

# Sidebar
with st.sidebar:
    st.subheader("Current Settings")
    st.write(f"Model: {st.session_state.settings['model']}")
    st.write(f"Temperature: {st.session_state.settings['temperature']}")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            time.sleep(0.5)
            response = f"[{st.session_state.settings['model']}] Echo: {prompt}"
            st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
