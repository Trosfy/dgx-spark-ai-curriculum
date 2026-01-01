# Module 4.1: Multimodal AI - Quickstart

## Time: ~5 minutes

## What You'll Build

Get an AI to describe an image in under 5 minutes using a vision-language model on DGX Spark.

## Before You Start

- [ ] DGX Spark container running
- [ ] Internet connection for model download

## Let's Go!

### Step 1: Start the NGC Container

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    nvcr.io/nvidia/pytorch:25.11-py3
```

### Step 2: Install Dependencies

```bash
pip install transformers accelerate pillow requests -q
```

### Step 3: Download a Test Image

```python
import requests
from PIL import Image

# Download a sample image
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image.save("/workspace/test_image.jpg")
print("Image saved!")
```

### Step 4: Load a Vision-Language Model

```python
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
print("Model loaded!")
```

### Step 5: Ask the Model About the Image

```python
from PIL import Image

image = Image.open("/workspace/test_image.jpg")
prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"

inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=200)
response = processor.decode(output[0], skip_special_tokens=True)
print(response)
```

**Expected Output:**
```
USER: Describe this image in detail.
ASSISTANT: The image shows a cat sitting on a surface. The cat appears to be
a tabby with distinctive markings. It has alert eyes and is looking directly
at the camera...
```

## You Did It!

You just ran a vision-language model that can understand images! In the full module, you'll learn:

- **Larger VLMs**: LLaVA-13B, Qwen2-VL-72B with quantization
- **Image Generation**: SDXL, Flux, ControlNet
- **Multimodal RAG**: Search images with natural language
- **Document AI**: Extract information from PDFs
- **Audio**: Whisper transcription and analysis

## Next Steps

1. **Understand what happened**: Read [lab-4.1.1-vision-language-demo.ipynb](./labs/lab-4.1.1-vision-language-demo.ipynb)
2. **Try variations**: Ask different questions about the image
3. **Full tutorial**: Start with [LAB_PREP.md](./LAB_PREP.md)
