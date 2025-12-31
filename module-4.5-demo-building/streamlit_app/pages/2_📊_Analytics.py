"""Analytics Page for the Sample Streamlit App."""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Analytics", page_icon="📊")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("📊 Analytics")

# Metrics
total = len(st.session_state.messages)
user_msgs = sum(1 for m in st.session_state.messages if m["role"] == "user")
ai_msgs = total - user_msgs

col1, col2, col3 = st.columns(3)
col1.metric("Total Messages", total)
col2.metric("User Messages", user_msgs)
col3.metric("AI Responses", ai_msgs)

st.markdown("---")

# Simulated usage chart
st.subheader("📈 Usage Over Time")
chart_data = pd.DataFrame(np.random.randn(20, 2) * 10 + 50, columns=["Messages", "Tokens"])
st.line_chart(chart_data)

# Chat history
st.markdown("---")
st.subheader("📝 Conversation History")

if st.session_state.messages:
    for i, msg in enumerate(st.session_state.messages):
        with st.expander(f"{msg['role'].title()} ({i+1})"):
            st.write(msg["content"])
else:
    st.info("No messages yet. Start chatting to see history!")
