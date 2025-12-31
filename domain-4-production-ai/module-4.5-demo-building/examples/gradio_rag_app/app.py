"""
RAG Chat Demo - Gradio Application

A complete, production-ready RAG demo using Gradio.

Features:
- Multi-document upload and indexing
- Conversational Q&A with source citations
- Configurable settings
- Professional styling

To run:
    pip install gradio chromadb ollama
    python app.py
"""

import gradio as gr
import chromadb
import ollama
from pathlib import Path
from typing import List, Tuple, Dict
import json


# ===== RAG BACKEND =====

class RAGBackend:
    """
    Simple but effective RAG backend.

    Uses ChromaDB for vector storage and Ollama for LLM inference.
    """

    def __init__(self, collection_name: str = "demo_docs"):
        """Initialize the RAG backend."""
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Default settings
        self.llm_model = "llama3.2:3b"
        self.embed_model = "nomic-embed-text"
        self.n_results = 3
        self.temperature = 0.7

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()

            if len(chunk) > 50:
                chunks.append(chunk)

            start = end - overlap

        return chunks

    def index_document(self, text: str, filename: str) -> Tuple[int, str]:
        """Index a document into the vector store."""
        try:
            chunks = self.chunk_text(text)

            if not chunks:
                return 0, "Document too short"

            # Generate embeddings
            embeddings = []
            for chunk in chunks:
                response = ollama.embeddings(
                    model=self.embed_model,
                    prompt=chunk
                )
                embeddings.append(response["embedding"])

            # Store in ChromaDB
            base_id = filename.replace(" ", "_").replace(".", "_")
            ids = [f"{base_id}_{i}" for i in range(len(chunks))]

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=[{"source": filename}] * len(chunks)
            )

            return len(chunks), f"Indexed {len(chunks)} chunks from {filename}"

        except Exception as e:
            return 0, f"Error: {str(e)}"

    def search(self, query: str) -> List[Dict]:
        """Search for relevant chunks."""
        response = ollama.embeddings(
            model=self.embed_model,
            prompt=query
        )

        results = self.collection.query(
            query_embeddings=[response["embedding"]],
            n_results=self.n_results
        )

        chunks = []
        if results["documents"][0]:
            for i in range(len(results["documents"][0])):
                chunks.append({
                    "text": results["documents"][0][i],
                    "source": results["metadatas"][0][i]["source"],
                    "distance": results["distances"][0][i] if "distances" in results else 0
                })

        return chunks

    def chat(self, query: str, history: List[Tuple[str, str]]) -> Tuple[str, str]:
        """Generate a response using RAG."""
        context_chunks = self.search(query)

        if not context_chunks:
            return "No documents indexed. Please upload some documents first!", ""

        context = "\n\n---\n\n".join([c["text"] for c in context_chunks])

        system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Always cite which document your information comes from.
If the context doesn't contain relevant information, say so honestly.
Keep responses concise but complete."""

        messages = [{"role": "system", "content": system_prompt}]

        for user_msg, assistant_msg in history[-5:]:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})

        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\n---\n\nQuestion: {query}"
        })

        response = ollama.chat(
            model=self.llm_model,
            messages=messages,
            options={"temperature": self.temperature}
        )

        answer = response["message"]["content"]

        # Format sources
        sources_md = "**Sources:**\n"
        for i, chunk in enumerate(context_chunks, 1):
            confidence = 1 - chunk["distance"]
            sources_md += f"\n{i}. **{chunk['source']}** ({confidence:.0%})\n"
            sources_md += f"   > {chunk['text'][:100]}...\n"

        return answer, sources_md

    def get_stats(self) -> Dict:
        """Get collection statistics."""
        return {
            "total_chunks": self.collection.count(),
            "llm_model": self.llm_model,
            "embed_model": self.embed_model
        }

    def clear(self):
        """Clear all documents."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )


# ===== GRADIO INTERFACE =====

def create_demo():
    """Create the Gradio demo interface."""

    rag = RAGBackend()

    # Custom theme
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )

    # Custom CSS
    css = """
    .gradio-container { max-width: 1200px !important; margin: auto; }
    .source-box { background: #f0f7ff; padding: 12px; border-radius: 8px; }
    """

    with gr.Blocks(theme=theme, css=css, title="RAG Chat Demo") as demo:
        # Header
        gr.Markdown("""
        # 🤖 RAG Chat Demo

        Upload your documents, then chat with them! Powered by local LLMs via Ollama.
        """)

        with gr.Tabs():
            # Tab 1: Documents
            with gr.TabItem("📁 Documents"):
                gr.Markdown("### Upload & Manage Documents")

                with gr.Row():
                    with gr.Column(scale=2):
                        file_upload = gr.File(
                            label="Upload Documents",
                            file_count="multiple",
                            file_types=[".txt", ".md", ".pdf"]
                        )

                        with gr.Row():
                            index_btn = gr.Button("📥 Index Documents", variant="primary")
                            clear_docs_btn = gr.Button("🗑️ Clear All")

                        status_box = gr.Textbox(label="Status", lines=6)

                    with gr.Column(scale=1):
                        doc_count = gr.Number(label="Chunks Indexed", value=0)
                        gr.Markdown("""
                        **Supported formats:**
                        - 📄 Plain text (.txt)
                        - 📝 Markdown (.md)
                        - 📕 PDF (.pdf)
                        """)

            # Tab 2: Chat
            with gr.TabItem("💬 Chat"):
                gr.Markdown("### Chat with Your Documents")

                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(height=450, show_copy_button=True)

                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder="Ask about your documents...",
                                scale=4,
                                show_label=False
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)

                        clear_chat_btn = gr.Button("🗑️ Clear Chat")

                    with gr.Column(scale=1):
                        gr.Markdown("### 📚 Sources")
                        sources_display = gr.Markdown("*Sources appear here*")

            # Tab 3: Settings
            with gr.TabItem("⚙️ Settings"):
                gr.Markdown("### Configure RAG Settings")

                with gr.Row():
                    with gr.Column():
                        model_select = gr.Dropdown(
                            choices=["llama3.2:3b", "llama3.1:8b", "mistral:7b"],
                            value="llama3.2:3b",
                            label="LLM Model"
                        )

                        chunks_slider = gr.Slider(1, 10, 3, step=1, label="Retrieved Chunks")
                        temp_slider = gr.Slider(0, 1, 0.7, step=0.1, label="Temperature")

                        save_btn = gr.Button("💾 Save Settings", variant="primary")
                        settings_status = gr.Textbox(label="Status")

        # Footer
        gr.Markdown("""
        ---
        *Built with Gradio, ChromaDB, and Ollama | Module 4.5 Demo*
        """)

        # Event Handlers
        def process_files(files, progress=gr.Progress()):
            if not files:
                return "No files uploaded", 0

            results = []
            for file in progress.tqdm(files, desc="Indexing"):
                filename = Path(file.name).name
                try:
                    with open(file.name, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()

                    chunks, msg = rag.index_document(text, filename)
                    results.append(f"✅ {filename}: {msg}")
                except Exception as e:
                    results.append(f"❌ {filename}: {str(e)}")

            return "\n".join(results), rag.collection.count()

        def clear_documents():
            rag.clear()
            return "All documents cleared!", 0

        def chat_respond(message, history):
            if not message:
                return history, "", ""

            response, sources = rag.chat(message, history)
            return history + [[message, response]], sources, ""

        def clear_history():
            return [], "*Sources appear here*"

        def update_settings(model, chunks, temp):
            rag.llm_model = model
            rag.n_results = int(chunks)
            rag.temperature = temp
            return f"Settings saved: model={model}, chunks={chunks}, temp={temp}"

        # Wire up events
        index_btn.click(process_files, [file_upload], [status_box, doc_count])
        clear_docs_btn.click(clear_documents, outputs=[status_box, doc_count])

        send_btn.click(chat_respond, [msg_input, chatbot], [chatbot, sources_display, msg_input])
        msg_input.submit(chat_respond, [msg_input, chatbot], [chatbot, sources_display, msg_input])
        clear_chat_btn.click(clear_history, outputs=[chatbot, sources_display])

        save_btn.click(update_settings, [model_select, chunks_slider, temp_slider], [settings_status])

    return demo


# ===== MAIN =====

if __name__ == "__main__":
    demo = create_demo()
    demo.queue()
    demo.launch()
