"""
Agent Playground - Tools Configuration Page

Configure which tools the agent can use.
"""

import streamlit as st

st.set_page_config(page_title="Tools - Agent Playground", page_icon="🔧", layout="wide")

st.title("🔧 Tool Configuration")

st.markdown("""
Configure which tools the agent can use. Enable tools that match your use case.
""")

# Tool definitions
AVAILABLE_TOOLS = {
    "calculator": {
        "name": "calculator",
        "description": "Perform mathematical calculations. Supports basic operations and functions like sin, cos, sqrt.",
        "example": "What is sqrt(144) + 25?"
    },
    "datetime": {
        "name": "get_datetime",
        "description": "Get the current date and time.",
        "example": "What time is it?"
    },
    "weather": {
        "name": "get_weather",
        "description": "Get weather information for a location (mock data).",
        "example": "What's the weather in Paris?"
    },
    "web_search": {
        "name": "web_search",
        "description": "Search the web for information (mock data).",
        "example": "Search for Python tutorials"
    }
}

# Initialize enabled tools
if "enabled_tools" not in st.session_state:
    st.session_state.enabled_tools = list(AVAILABLE_TOOLS.keys())

# Tool toggles
st.markdown("### Enable/Disable Tools")

col1, col2 = st.columns(2)

tools_list = list(AVAILABLE_TOOLS.items())

for i, (tool_key, tool_info) in enumerate(tools_list):
    col = col1 if i % 2 == 0 else col2

    with col:
        with st.container(border=True):
            enabled = st.checkbox(
                f"**{tool_info['name']}**",
                value=tool_key in st.session_state.enabled_tools,
                key=f"tool_{tool_key}"
            )

            st.caption(tool_info["description"])
            st.markdown(f"*Example: {tool_info['example']}*")

            # Update session state
            if enabled and tool_key not in st.session_state.enabled_tools:
                st.session_state.enabled_tools.append(tool_key)
            elif not enabled and tool_key in st.session_state.enabled_tools:
                st.session_state.enabled_tools.remove(tool_key)

st.markdown("---")

# Quick actions
st.markdown("### Quick Actions")

quick_col1, quick_col2, quick_col3 = st.columns(3)

with quick_col1:
    if st.button("Enable All", use_container_width=True):
        st.session_state.enabled_tools = list(AVAILABLE_TOOLS.keys())
        st.rerun()

with quick_col2:
    if st.button("Disable All", use_container_width=True):
        st.session_state.enabled_tools = []
        st.rerun()

with quick_col3:
    if st.button("Reset Default", use_container_width=True):
        st.session_state.enabled_tools = ["calculator", "datetime"]
        st.rerun()

st.markdown("---")

# Model settings
st.markdown("### Model Settings")

if "model" not in st.session_state:
    st.session_state.model = "qwen3:4b"

model = st.selectbox(
    "LLM Model",
    options=["qwen3:1.7b", "qwen3:4b", "qwen3:8b", "mistral:7b"],
    index=["qwen3:1.7b", "qwen3:4b", "qwen3:8b", "mistral:7b"].index(st.session_state.model)
)
st.session_state.model = model

st.info(f"Current model: **{model}**")

# Summary
st.markdown("---")
st.markdown("### Current Configuration")

st.success(f"**Enabled tools:** {', '.join(st.session_state.enabled_tools) or 'None'}")
st.info(f"**Model:** {st.session_state.model}")
