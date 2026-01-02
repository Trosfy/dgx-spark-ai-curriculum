# DGX Spark Compatibility Matrix

Last updated: 2026-01-02


| Tool | Category | Status | Notes |
|------|----------|--------|-------|
| NumPy | Data Science | ✅ Full Support | ARM64 wheels available. |
| Pandas | Data Science | ✅ Full Support | Pure Python - works everywhere. |
| RAPIDS (cuDF, cuML) | Data Science | 🐳 NGC Required | GPU-accelerated data science. Full NGC support. |
| Scikit-learn | Data Science | ✅ Full Support | ARM64 wheels available. |
| JAX | Deep Learning Framework | 🐳 NGC Required | NGC container available with CUDA support. |
| PyTorch | Deep Learning Framework | 🐳 NGC Required | Must use NGC container. pip install does NOT work ... |
| TensorFlow | Deep Learning Framework | 🐳 NGC Required | Use NGC container for GPU support. |
| Docker | Development | ✅ Full Support | Pre-installed with NVIDIA runtime. |
| Git | Development | ✅ Full Support | Pre-installed. |
| JupyterLab | Development | ✅ Full Support | Pre-installed on DGX OS. |
| VS Code | Development | ✅ Full Support | ARM64 version available. |
| Ollama | LLM Inference | ✅ Full Support | Native ARM64 support. Pre-installed on DGX OS. Exc... |
| SGLang | LLM Inference | ✅ Full Support | Full Blackwell/Jetson support. 29-45% faster than ... |
| TensorRT-LLM | LLM Inference | 🐳 NGC Required | Requires NGC container or source build. Blackwell ... |
| Text Generation Inference (TGI) | LLM Inference | ⚠️ Partial | HuggingFace server. ARM64 Docker image available. |
| llama.cpp | LLM Inference | ✅ Full Support | Native ARM64+CUDA support. Compile with CUDA flags... |
| vLLM | LLM Inference | ⚠️ Partial | ARM64 support available. Requires --enforce-eager ... |
| MLflow | MLOps | ✅ Full Support | Pure Python - works everywhere. |
| Triton Inference Server | MLOps | 🐳 NGC Required | Full NGC support. |
| Weights & Biases | MLOps | ✅ Full Support | Pure Python - works everywhere. |
| Hugging Face Diffusers | Model Library | 🐳 NGC Required | Works inside NGC PyTorch container. |
| Hugging Face Transformers | Model Library | 🐳 NGC Required | Works inside NGC PyTorch container. |
| LangChain | Model Library | ✅ Full Support | Pure Python - works everywhere. Use with Ollama. |
| LlamaIndex | Model Library | ✅ Full Support | Pure Python - works everywhere. |
| NVIDIA NeMo | Model Library | 🐳 NGC Required | Full support via NGC container. |
| OpenAI API (client) | Model Library | ✅ Full Support | Pure Python - works everywhere. |
| Axolotl | Training | 🐳 NGC Required | Fine-tuning framework. Use with NGC container. |
| DeepSpeed | Training | ⚠️ Partial | Some features may not work. Use NGC container. |
| PEFT (LoRA) | Training | 🐳 NGC Required | Works inside NGC container. |
| Unsloth | Training | ❓ Untested | Fast fine-tuning. Needs testing on ARM64. |
| bitsandbytes | Training | ⚠️ Partial | 4-bit/8-bit quantization. ARM64 support improving. |
| ChromaDB | Vector Database | ✅ Full Support | Pure Python with SQLite. Works everywhere. |
| FAISS | Vector Database | 🐳 NGC Required | GPU version needs NGC container. |
| Milvus | Vector Database | ⚠️ Partial | ARM64 Docker images available. |
| Qdrant | Vector Database | ✅ Full Support | ARM64 Docker images available. |