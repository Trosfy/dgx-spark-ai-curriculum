"""
Agent Playground - Chat Page

Chat interface with tool call visualization.
"""

import streamlit as st
import json
import math
import datetime
import ollama
import re

st.set_page_config(page_title="Chat - Agent Playground", page_icon="💬", layout="wide")

st.title("💬 Agent Chat")


# ===== TOOL DEFINITIONS =====

TOOLS = {
    "calculator": {
        "name": "calculator",
        "description": "Perform mathematical calculations",
        "execute": lambda args: f"Result: {eval(args.get('expression', '0'), {'__builtins__': {}, 'sin': math.sin, 'cos': math.cos, 'sqrt': math.sqrt, 'pi': math.pi})}"
    },
    "datetime": {
        "name": "get_datetime",
        "description": "Get current date and time",
        "execute": lambda args: datetime.datetime.now().strftime("Current: %Y-%m-%d %H:%M:%S")
    },
    "weather": {
        "name": "get_weather",
        "description": "Get weather for a location (mock)",
        "execute": lambda args: f"Weather in {args.get('location', 'Unknown')}: Sunny, 72°F"
    },
    "web_search": {
        "name": "web_search",
        "description": "Search the web (mock)",
        "execute": lambda args: f"Search results for '{args.get('query', '')}': Found 10 results"
    }
}


# ===== AGENT LOGIC =====

def get_agent_response(message: str, history: list, enabled_tools: list, model: str):
    """Get response from agent with tool use."""

    # Build tool descriptions
    tool_desc = "\n".join([
        f"- {TOOLS[t]['name']}: {TOOLS[t]['description']}"
        for t in enabled_tools if t in TOOLS
    ])

    system_prompt = f"""You are a helpful assistant with access to tools.

Available tools:
{tool_desc}

To use a tool, respond with:
<tool>tool_name</tool>
<args>{{"param": "value"}}</args>

After getting results, provide your answer."""

    messages = [{"role": "system", "content": system_prompt}]
    for h in history:
        messages.append({"role": "user", "content": h["user"]})
        messages.append({"role": "assistant", "content": h["assistant"]})
    messages.append({"role": "user", "content": message})

    # First LLM call
    response = ollama.chat(model=model, messages=messages)
    content = response["message"]["content"]

    # Parse tool calls
    tool_calls = []
    tool_pattern = r"<tool>(.*?)</tool>\s*<args>(.*?)</args>"
    matches = re.findall(tool_pattern, content, re.DOTALL)

    thinking = ""
    final_content = content

    if matches:
        # Extract thinking
        first_tool = content.find("<tool>")
        if first_tool > 0:
            thinking = content[:first_tool].strip()

        # Execute tools
        tool_results = []
        for tool_name, args_str in matches:
            tool_name = tool_name.strip()
            try:
                args = json.loads(args_str.strip())
            except:
                args = {}

            if tool_name in [TOOLS[t]["name"] for t in enabled_tools]:
                for key, tool in TOOLS.items():
                    if tool["name"] == tool_name:
                        result = tool["execute"](args)
                        tool_calls.append({
                            "tool": tool_name,
                            "args": args,
                            "result": result
                        })
                        tool_results.append(f"{tool_name}: {result}")
                        break

        # Get final response
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": f"Tool results:\n" + "\n".join(tool_results) + "\n\nProvide your final answer."})

        final_response = ollama.chat(model=model, messages=messages)
        final_content = final_response["message"]["content"]

    return {
        "content": final_content,
        "tool_calls": tool_calls,
        "thinking": thinking
    }


# ===== UI =====

# Get settings
enabled_tools = st.session_state.get("enabled_tools", list(TOOLS.keys()))
model = st.session_state.get("model", "qwen3:4b")

# Layout
chat_col, tool_col = st.columns([2, 1])

with chat_col:
    st.markdown("### Conversation")

    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message("user"):
            st.write(msg.get("user", ""))

        with st.chat_message("assistant"):
            st.write(msg.get("assistant", ""))

            if msg.get("tool_calls"):
                with st.expander("🔧 Tool Calls"):
                    for tc in msg["tool_calls"]:
                        st.code(json.dumps(tc, indent=2), language="json")

            if msg.get("thinking"):
                with st.expander("💭 Thinking"):
                    st.markdown(msg["thinking"])

    # Chat input
    if prompt := st.chat_input("Ask the agent something..."):
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_agent_response(
                    prompt,
                    st.session_state.messages,
                    enabled_tools,
                    model
                )

            st.write(response["content"])

            if response["tool_calls"]:
                with st.expander("🔧 Tool Calls"):
                    for tc in response["tool_calls"]:
                        st.code(json.dumps(tc, indent=2), language="json")

            if response["thinking"]:
                with st.expander("💭 Thinking"):
                    st.markdown(response["thinking"])

        # Store message
        st.session_state.messages.append({
            "user": prompt,
            "assistant": response["content"],
            "tool_calls": response["tool_calls"],
            "thinking": response["thinking"]
        })

        # Store tool calls
        st.session_state.tool_calls.extend(response["tool_calls"])

    # Clear button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.tool_calls = []
        st.rerun()

with tool_col:
    st.markdown("### 🔧 Active Tools")

    for tool in enabled_tools:
        if tool in TOOLS:
            st.markdown(f"- ✅ {TOOLS[tool]['name']}")

    st.markdown("---")

    st.markdown("### Recent Tool Calls")

    if st.session_state.tool_calls:
        for tc in reversed(st.session_state.tool_calls[-5:]):
            with st.container(border=True):
                st.markdown(f"**{tc['tool']}**")
                st.caption(f"Args: {tc.get('args', {})}")
                st.success(tc.get('result', 'N/A')[:80])
    else:
        st.info("No tool calls yet.")

    st.markdown("---")
    st.markdown("**Try asking:**")
    st.markdown("- *What is 25 * 4 + 100?*")
    st.markdown("- *What time is it?*")
    st.markdown("- *Weather in Tokyo?*")
