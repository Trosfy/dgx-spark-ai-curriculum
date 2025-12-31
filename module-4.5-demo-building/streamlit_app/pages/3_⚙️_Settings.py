"""Settings Page for the Sample Streamlit App."""

import streamlit as st

st.set_page_config(page_title="Settings", page_icon="⚙️")

if "settings" not in st.session_state:
    st.session_state.settings = {"model": "llama3.1:8b", "temperature": 0.7, "theme": "light"}

st.title("⚙️ Settings")

st.subheader("Model Configuration")

model = st.selectbox(
    "Select Model",
    ["llama3.1:8b", "llama3.1:70b", "mistral:7b"],
    index=["llama3.1:8b", "llama3.1:70b", "mistral:7b"].index(
        st.session_state.settings["model"]
    )
)

temperature = st.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=st.session_state.settings["temperature"],
    step=0.1
)

st.markdown("---")
st.subheader("Appearance")

theme = st.radio("Theme", ["light", "dark"], index=0 if st.session_state.settings.get("theme") == "light" else 1)

st.markdown("---")

if st.button("💾 Save Settings", type="primary"):
    st.session_state.settings = {"model": model, "temperature": temperature, "theme": theme}
    st.success("✅ Settings saved!")
    st.balloons()

with st.expander("Current Settings"):
    st.json(st.session_state.settings)
