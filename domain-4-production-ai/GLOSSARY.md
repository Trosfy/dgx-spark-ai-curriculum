# Domain 4: Production AI - Glossary

## Module 4.1: Multimodal AI

| Term | Definition |
|------|------------|
| **CLIP** | Contrastive Language-Image Pre-training. OpenAI model that learns visual concepts from natural language supervision. Matches images to text descriptions. |
| **Diffusion Model** | Generative model that creates images by gradually denoising random noise. Examples: Stable Diffusion, DALL-E. |
| **Document AI** | AI systems that process, understand, and extract information from documents (PDFs, images, forms). |
| **Multimodal** | AI systems that process multiple types of data (text, images, audio, video) in a unified model. |
| **SDXL** | Stable Diffusion XL. High-resolution image generation model from Stability AI. |
| **VLM** | Vision-Language Model. AI that understands both images and text, enabling image captioning, visual Q&A, and more. Examples: LLaVA, Qwen-VL. |
| **Whisper** | OpenAI's speech recognition model for transcription and translation. |

---

## Module 4.2: AI Safety & Guardrails

| Term | Definition |
|------|------------|
| **Colang** | Domain-specific language used by NeMo Guardrails to define conversational flows and safety rules. |
| **Guardrails** | Safety mechanisms that constrain AI behavior to prevent harmful outputs. Can be input filters, output filters, or behavioral rules. |
| **Jailbreak** | Attempt to bypass an AI's safety measures through clever prompting to elicit prohibited responses. |
| **Llama Guard** | Meta's safety classifier model that categorizes inputs/outputs into predefined safety categories. |
| **NeMo Guardrails** | NVIDIA's open-source toolkit for adding programmable guardrails to LLM applications. |
| **OWASP Top 10 for LLMs** | Industry-standard list of top security vulnerabilities in LLM applications (prompt injection, data leakage, etc.). |
| **Prompt Injection** | Attack where malicious instructions are hidden in user input to manipulate the AI's behavior. |
| **Red Teaming** | Adversarial testing where humans attempt to find vulnerabilities and failures in AI systems. |
| **Toxicity** | Harmful, offensive, or inappropriate content in AI outputs. |

---

## Module 4.3: MLOps & Model Management

| Term | Definition |
|------|------------|
| **Artifact** | Any file produced during ML workflow: models, datasets, logs, metrics. Tracked for reproducibility. |
| **Experiment** | A single training run with specific parameters, tracked for comparison and reproducibility. |
| **lm-eval-harness** | EleutherAI's framework for evaluating language models across standardized benchmarks. |
| **MLflow** | Open-source platform for managing ML lifecycle: tracking experiments, packaging models, deployment. |
| **Model Registry** | Centralized store for versioned models with metadata, staging status, and deployment history. |
| **Run** | Single execution of an experiment with specific configuration, logged for tracking. |
| **W&B (Weights & Biases)** | ML experiment tracking platform with visualization, collaboration, and model management. |

---

## Module 4.4: Containerization & Cloud Deployment

| Term | Definition |
|------|------------|
| **Container** | Lightweight, standalone executable package including application code and all dependencies. |
| **Docker** | Platform for building, running, and sharing containers. Standard for ML deployment. |
| **Docker Compose** | Tool for defining multi-container applications with YAML configuration. |
| **ECR** | Elastic Container Registry. AWS's managed Docker registry service. |
| **HPA** | Horizontal Pod Autoscaler. Kubernetes feature that scales pods based on metrics. |
| **Kubernetes (K8s)** | Container orchestration platform for automating deployment, scaling, and management. |
| **NGC** | NVIDIA GPU Cloud. Registry of GPU-optimized containers for AI/ML workloads. |
| **Pod** | Smallest deployable unit in Kubernetes, containing one or more containers. |
| **SageMaker** | AWS's managed ML platform for training, tuning, and deploying models. |
| **Vertex AI** | Google Cloud's managed ML platform for building and deploying models. |

---

## Module 4.5: Demo Building

| Term | Definition |
|------|------------|
| **Blocks API** | Gradio's low-level API for building complex, custom layouts with full control over component arrangement. |
| **ChatInterface** | Gradio's high-level component for creating chat applications with minimal code. |
| **Gradio** | Python library for building ML demos and web interfaces quickly. |
| **HF Spaces** | Hugging Face Spaces. Free hosting platform for ML demos with optional GPU support. |
| **Session State** | Mechanism for persisting data across user interactions in Streamlit (st.session_state). |
| **Streamlit** | Python library for building data apps and dashboards with minimal frontend code. |
| **Streamlit Cloud** | Hosting platform for Streamlit applications with GitHub integration. |

---

## Module 4.6: Capstone Project

| Term | Definition |
|------|------------|
| **Model Card** | Documentation describing a model's intended use, training data, evaluation results, and limitations. |
| **MVP** | Minimum Viable Product. Simplest version of a project that demonstrates core functionality. |
| **Technical Report** | Comprehensive documentation of project design, implementation, evaluation, and lessons learned. |

---

## Cross-Cutting Terms

| Term | Definition |
|------|------------|
| **API** | Application Programming Interface. Way for software to communicate with other software. |
| **Blackwell** | NVIDIA's latest GPU architecture (2024). DGX Spark uses Blackwell GB10 Superchip. |
| **CI/CD** | Continuous Integration/Continuous Deployment. Automated testing and deployment pipelines. |
| **DGX Spark** | NVIDIA's personal AI supercomputer with Blackwell GPU and 128GB unified memory. |
| **Endpoint** | URL that exposes a model for inference via HTTP requests. |
| **Inference** | Using a trained model to make predictions on new data. |
| **Latency** | Time from request to response. Critical metric for production AI. |
| **Throughput** | Number of requests processed per unit time. |
| **Unified Memory** | Shared memory accessible by both CPU and GPU, eliminating data transfer overhead. DGX Spark has 128GB. |

---

## Acronym Reference

| Acronym | Full Form |
|---------|-----------|
| API | Application Programming Interface |
| AWS | Amazon Web Services |
| CI/CD | Continuous Integration/Continuous Deployment |
| CLIP | Contrastive Language-Image Pre-training |
| CPU | Central Processing Unit |
| DGX | Data center GPU (NVIDIA product line) |
| ECR | Elastic Container Registry |
| GCP | Google Cloud Platform |
| GPU | Graphics Processing Unit |
| HF | Hugging Face |
| HPA | Horizontal Pod Autoscaler |
| K8s | Kubernetes |
| LLM | Large Language Model |
| ML | Machine Learning |
| MLOps | Machine Learning Operations |
| MVP | Minimum Viable Product |
| NGC | NVIDIA GPU Cloud |
| OWASP | Open Web Application Security Project |
| RAG | Retrieval-Augmented Generation |
| REST | Representational State Transfer |
| SDXL | Stable Diffusion XL |
| SFT | Supervised Fine-Tuning |
| VLM | Vision-Language Model |
| W&B | Weights & Biases |
