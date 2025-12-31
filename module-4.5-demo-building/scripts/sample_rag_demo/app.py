"""
Sample RAG Demo Application.

A complete, deployable RAG demo for Hugging Face Spaces.
This can be used as a starting point for your own RAG demos.

Author: Professor SPARK
Module: 4.5 - Demo Building & Prototyping
"""

import gradio as gr
import os
from typing import List, Tuple, Dict

# =============================================================================
# CONFIGURATION
# =============================================================================

# Demo mode - when True, uses cached responses for known queries
DEMO_MODE = os.environ.get("DEMO_MODE", "true").lower() == "true"

# Golden path examples with cached responses
DEMO_RESPONSES = {
    "What are the main topics in these documents?":
        "Based on the uploaded documents, the main topics covered include:\n\n"
        "1. **Introduction to Machine Learning** - Fundamental concepts and terminology\n"
        "2. **Neural Networks** - Architecture and training methods\n"
        "3. **Best Practices** - Industry standards and recommendations\n\n"
        "*Sources: sample_ml_guide.txt (pages 1-3)*",

    "Summarize the key findings.":
        "The key findings from the documents are:\n\n"
        "- Machine learning models benefit from large, diverse datasets\n"
        "- Regular validation prevents overfitting\n"
        "- Proper preprocessing improves model performance by up to 30%\n\n"
        "*Sources: research_summary.txt (section 4)*",

    "What recommendations are mentioned?":
        "The documents recommend the following:\n\n"
        "1. Start with simple models before adding complexity\n"
        "2. Use cross-validation for reliable performance estimates\n"
        "3. Document your experiments thoroughly\n"
        "4. Version control your data and models\n\n"
        "*Sources: best_practices.txt (recommendations section)*",
}

# =============================================================================
# RAG BACKEND (Simplified for Demo)
# =============================================================================

class SimpleRAGDemo:
    """Simplified RAG system for demonstration."""

    def __init__(self):
        self.documents = []
        self.indexed = False

    def index_documents(self, files) -> str:
        """Index uploaded documents."""
        if not files:
            return "⚠️ Please upload at least one file."

        self.documents = [f.name for f in files]
        self.indexed = True

        return f"✅ Successfully indexed {len(files)} document(s)!"

    def query(self, question: str) -> str:
        """Answer a question using the indexed documents."""
        # Check demo responses first
        if DEMO_MODE and question in DEMO_RESPONSES:
            return DEMO_RESPONSES[question]

        if not self.indexed:
            return "📚 Please upload and index documents first!"

        # Simulated response
        return (
            f"Based on the indexed documents ({len(self.documents)} files), "
            f"here's what I found about: {question}\n\n"
            "This is a demo response. In production, this would use actual RAG "
            "with embeddings, vector search, and LLM generation."
        )

# Global RAG instance
rag = SimpleRAGDemo()

# =============================================================================
# INTERFACE
# =============================================================================

# Custom theme
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
).set(
    button_primary_background_fill="#2563eb",
    button_primary_text_color="white",
)

# Build interface
with gr.Blocks(theme=theme, title="RAG Document Q&A") as demo:
    gr.Markdown("""
    # 📚 RAG Document Q&A Demo

    Upload your documents and ask questions using Retrieval-Augmented Generation.

    **How to use:**
    1. Upload documents in the **Documents** tab
    2. Click **Index Documents** to process them
    3. Switch to **Chat** tab and ask questions!
    """)

    with gr.Tabs():
        # Documents Tab
        with gr.TabItem("📁 Documents"):
            gr.Markdown("### Upload and Index Documents")

            files = gr.File(
                label="Upload Documents",
                file_count="multiple",
                file_types=[".pdf", ".txt", ".md"],
            )

            index_btn = gr.Button("📥 Index Documents", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)

            index_btn.click(rag.index_documents, [files], [status])

        # Chat Tab
        with gr.TabItem("💬 Chat"):
            gr.Markdown("### Ask Questions")

            chatbot = gr.Chatbot(height=350, show_copy_button=True)

            with gr.Row():
                msg = gr.Textbox(
                    label="Your Question",
                    placeholder="What would you like to know?",
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            gr.Markdown("### Try These Examples")
            examples = gr.Examples(
                examples=[
                    ["What are the main topics in these documents?"],
                    ["Summarize the key findings."],
                    ["What recommendations are mentioned?"],
                ],
                inputs=[msg],
            )

            def respond(message, history):
                response = rag.query(message)
                history = history + [[message, response]]
                return history, ""

            send_btn.click(respond, [msg, chatbot], [chatbot, msg])
            msg.submit(respond, [msg, chatbot], [chatbot, msg])

    # Footer
    gr.Markdown("---")
    gr.Markdown("*Built with Gradio | Module 4.5 Demo*")

# =============================================================================
# LAUNCH
# =============================================================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        show_error=False
    )
