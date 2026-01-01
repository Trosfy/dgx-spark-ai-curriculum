# Module 4.1: Multimodal AI - Quick Reference

## Essential Commands

### NGC Container Setup

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

### Install Dependencies

```bash
# VLM and image
pip install transformers>=4.46.0 accelerate>=0.28.0 bitsandbytes>=0.43.0
pip install pillow>=10.0.0 opencv-python qwen-vl-utils matplotlib

# Image generation
pip install diffusers>=0.27.0 controlnet_aux>=0.0.7

# Multimodal RAG
pip install sentence-transformers>=2.3.0 chromadb>=0.4.22

# Document AI
pip install pdf2image pytesseract pymupdf>=1.23.0

# Audio
pip install openai-whisper soundfile librosa scipy
```

---

## Vision-Language Models (2025)

### Qwen3-VL-8B (Primary - Tier 1)

```python
import ollama
from PIL import Image
import base64
import io

# Using Ollama (recommended)
def describe_image(image_path: str) -> str:
    """Describe an image using Qwen3-VL via Ollama."""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    response = ollama.chat(
        model="qwen3-vl:8b",
        messages=[{
            "role": "user",
            "content": "What's in this image? Describe it in detail.",
            "images": [image_data]
        }]
    )
    return response['message']['content']

# Using HuggingFace (for advanced usage)
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",  # 2025 version: 256K context, design-to-code
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "image.jpg"},
        {"type": "text", "text": "What's in this image?"}
    ]
}]
text = processor.apply_chat_template(messages, tokenize=False)
image_inputs, _ = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=200)
print(processor.batch_decode(output, skip_special_tokens=True)[0])
```

### MiniCPM-V 4.5 (OCR Specialist)

```python
from transformers import AutoModel, AutoTokenizer
import torch

# Best-in-class for document OCR
model = AutoModel.from_pretrained(
    "openbmb/MiniCPM-V-4.5",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-V-4.5", trust_remote_code=True)

# OCR-focused inference
image = Image.open("document.jpg")
response = model.chat(
    image=image,
    msgs=[{"role": "user", "content": "Extract all text from this document."}],
    tokenizer=tokenizer
)
print(response)
```

---

## Image Generation

### SDXL Basic

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.bfloat16
).to("cuda")

# Enable memory optimization
pipe.enable_model_cpu_offload()

image = pipe(
    prompt="A cat astronaut floating in space",
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]
image.save("output.png")
```

### Flux (Higher Quality)

```python
from diffusers import FluxPipeline
import torch

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to("cuda")

image = pipe(
    prompt="A serene mountain landscape at sunset",
    num_inference_steps=50,
    guidance_scale=3.5
).images[0]
```

### ControlNet (Edge-Guided)

```python
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from controlnet_aux import CannyDetector

# Load ControlNet
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.bfloat16
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.bfloat16
).to("cuda")

# Extract edges from reference image
canny = CannyDetector()
edges = canny(reference_image)

# Generate with control
image = pipe(prompt="A house", image=edges).images[0]
```

---

## CLIP Embeddings

### Embed Images and Text

```python
from transformers import CLIPModel, CLIPProcessor
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model = model.to("cuda")

# Embed image
image = Image.open("image.jpg")
inputs = processor(images=image, return_tensors="pt").to("cuda")
image_embedding = model.get_image_features(**inputs)

# Embed text
inputs = processor(text=["a photo of a cat"], return_tensors="pt").to("cuda")
text_embedding = model.get_text_features(**inputs)

# Compute similarity
similarity = torch.cosine_similarity(image_embedding, text_embedding)
```

### Multimodal Vector Store

```python
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

# Create client with CLIP embeddings
embedding_function = OpenCLIPEmbeddingFunction()
client = chromadb.Client()
collection = client.create_collection(
    name="multimodal",
    embedding_function=embedding_function
)

# Add images
collection.add(
    ids=["img1", "img2"],
    images=[image1_array, image2_array],  # numpy arrays
    metadatas=[{"source": "file1.jpg"}, {"source": "file2.jpg"}]
)

# Query with text
results = collection.query(
    query_texts=["sunset over mountains"],
    n_results=5
)
```

---

## Document AI

### PDF to Text with PyMuPDF

```python
import fitz  # pymupdf

doc = fitz.open("document.pdf")
text = ""
for page in doc:
    text += page.get_text()

# Extract images
for page_num, page in enumerate(doc):
    images = page.get_images()
    for img_index, img in enumerate(images):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
```

### OCR with Tesseract

```python
from pdf2image import convert_from_path
import pytesseract

# Convert PDF to images
images = convert_from_path("document.pdf")

# OCR each page
for i, image in enumerate(images):
    text = pytesseract.image_to_string(image)
    print(f"Page {i+1}:\n{text}")
```

---

## Audio Transcription

### Whisper Basic

```python
import whisper

model = whisper.load_model("large-v3")  # base, small, medium, large-v3
result = model.transcribe("audio.mp3")

print(result["text"])  # Full transcription
for segment in result["segments"]:
    print(f"{segment['start']:.2f}s - {segment['end']:.2f}s: {segment['text']}")
```

### Whisper with Timestamps

```python
result = model.transcribe(
    "audio.mp3",
    word_timestamps=True,
    language="en"  # or None for auto-detect
)

for segment in result["segments"]:
    for word in segment["words"]:
        print(f"{word['word']} ({word['start']:.2f}s)")
```

---

## Key Values to Remember

| Model | VRAM | Generation Time |
|-------|------|-----------------|
| LLaVA-7B | ~16GB | ~2-3s/response |
| LLaVA-13B | ~28GB | ~4-5s/response |
| Qwen2-VL-7B | ~18GB | ~3-4s/response |
| SDXL (1024x1024) | ~8GB | ~5-8s |
| Flux (1024x1024) | ~24GB | ~15-20s |
| Whisper-large-v3 | ~4GB | ~0.5x real-time |
| CLIP-ViT-L/14 | ~2GB | ~50ms/image |

---

## Common Patterns

### Pattern: VLM + LLM Pipeline

```python
# Use VLM to describe, LLM to analyze
vlm_description = vlm.generate(image, "Describe this image in detail")
llm_analysis = llm.generate(f"Based on this description: {vlm_description}\n\nWhat emotions does this scene evoke?")
```

### Pattern: Multimodal RAG

```python
# 1. Index images with CLIP
# 2. Query with text
# 3. Pass retrieved images to VLM
# 4. Generate answer with LLM

retrieved_images = vector_db.query(user_question, k=3)
vlm_context = [vlm.describe(img) for img in retrieved_images]
answer = llm.generate(user_question, context=vlm_context)
```

### Pattern: Document Q&A

```python
# 1. Extract text + layout from PDF
# 2. If tables/charts: use VLM to describe
# 3. Chunk and embed
# 4. RAG with context

pages = extract_pdf(document)
for page in pages:
    if has_complex_visuals(page):
        page.text += vlm.describe(page.image)
chunks = chunk_pages(pages)
index = embed_chunks(chunks)
```

---

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| VLM OOM with large images | Resize to 384x384 or use `low_cpu_mem_usage=True` |
| SDXL slow generation | Enable `pipe.enable_model_cpu_offload()` |
| CLIP embeddings not normalized | Use `F.normalize(embeddings, dim=-1)` |
| Whisper wrong language | Set `language="en"` explicitly |
| PDF images not extracted | Use `pymupdf` instead of `pypdf` |
| ControlNet poor results | Adjust `controlnet_conditioning_scale` (0.5-1.0) |

---

## Quick Links

- [LLaVA Project](https://llava-vl.github.io/)
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL)
- [Diffusers Library](https://huggingface.co/docs/diffusers)
- [Whisper](https://github.com/openai/whisper)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
