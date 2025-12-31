"""
Agent Playground - History & Analytics Page

View conversation history and analytics.
"""

import streamlit as st
import json
from collections import Counter
import pandas as pd

st.set_page_config(page_title="History - Agent Playground", page_icon="📊", layout="wide")

st.title("📊 Session History & Analytics")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tool_calls" not in st.session_state:
    st.session_state.tool_calls = []

# Overview metrics
st.markdown("### Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    user_msgs = len(st.session_state.messages)
    st.metric("Conversations", user_msgs)

with col2:
    st.metric("Tool Calls", len(st.session_state.tool_calls))

with col3:
    if st.session_state.tool_calls:
        tool_counts = Counter(tc["tool"] for tc in st.session_state.tool_calls)
        most_used = tool_counts.most_common(1)[0][0] if tool_counts else "N/A"
    else:
        most_used = "N/A"
    st.metric("Most Used Tool", most_used)

with col4:
    avg_tools = (
        len(st.session_state.tool_calls) / len(st.session_state.messages)
        if st.session_state.messages else 0
    )
    st.metric("Avg Tools/Message", f"{avg_tools:.1f}")

st.markdown("---")

# Tool usage chart
st.markdown("### Tool Usage Breakdown")

if st.session_state.tool_calls:
    tool_counts = Counter(tc["tool"] for tc in st.session_state.tool_calls)

    df = pd.DataFrame({
        "Tool": list(tool_counts.keys()),
        "Calls": list(tool_counts.values())
    })

    st.bar_chart(df.set_index("Tool"))
else:
    st.info("No tool calls recorded yet. Chat with the agent to generate data!")

st.markdown("---")

# Conversation history
st.markdown("### Conversation History")

if st.session_state.messages:
    for i, msg in enumerate(st.session_state.messages):
        with st.expander(
            f"{i+1}. User: {msg.get('user', '')[:50]}...",
            expanded=False
        ):
            st.markdown("**User:**")
            st.write(msg.get("user", ""))

            st.markdown("**Assistant:**")
            st.write(msg.get("assistant", ""))

            if msg.get("tool_calls"):
                st.markdown("**Tool Calls:**")
                st.json(msg["tool_calls"])

            if msg.get("thinking"):
                st.markdown("**Thinking:**")
                st.markdown(msg["thinking"])
else:
    st.info("No conversation history yet.")

st.markdown("---")

# Export section
st.markdown("### Export")

export_col1, export_col2 = st.columns(2)

with export_col1:
    if st.session_state.messages:
        json_export = json.dumps({
            "messages": st.session_state.messages,
            "tool_calls": st.session_state.tool_calls
        }, indent=2)

        st.download_button(
            "📥 Download as JSON",
            data=json_export,
            file_name="agent_conversation.json",
            mime="application/json"
        )

with export_col2:
    if st.session_state.messages:
        # Markdown export
        md_lines = ["# Agent Conversation\n"]
        for msg in st.session_state.messages:
            md_lines.append("## User\n")
            md_lines.append(f"{msg.get('user', '')}\n")
            md_lines.append("## Assistant\n")
            md_lines.append(f"{msg.get('assistant', '')}\n")
            if msg.get("tool_calls"):
                md_lines.append("### Tool Calls\n")
                md_lines.append(f"```json\n{json.dumps(msg['tool_calls'], indent=2)}\n```\n")

        st.download_button(
            "📄 Download as Markdown",
            data="\n".join(md_lines),
            file_name="agent_conversation.md",
            mime="text/markdown"
        )

st.markdown("---")

# Clear section
st.markdown("### Clear Data")

if st.button("🗑️ Clear All History", type="primary"):
    st.session_state.messages = []
    st.session_state.tool_calls = []
    st.success("All history cleared!")
    st.rerun()
