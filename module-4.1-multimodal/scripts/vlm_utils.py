"""
Vision-Language Model Utilities for DGX Spark

This module provides utilities for loading and using vision-language models
like LLaVA, Qwen-VL, and CLIP on DGX Spark's 128GB unified memory.

Example:
    >>> from scripts.vlm_utils import load_llava, analyze_image
    >>> model, processor = load_llava("llava-hf/llava-1.5-7b-hf")
    >>> result = analyze_image(model, processor, "image.jpg", "What's in this image?")
    >>> print(result)
"""

import gc
import time
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Any

import torch
from PIL import Image


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available GPU devices.

    Returns:
        Dict containing device information including name, memory, and compute capability.

    Example:
        >>> info = get_device_info()
        >>> print(f"GPU: {info['name']}, Memory: {info['total_memory_gb']:.1f}GB")
        GPU: NVIDIA GB10, Memory: 128.0GB
    """
    if not torch.cuda.is_available():
        return {"available": False, "device": "cpu"}

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    return {
        "available": True,
        "device": f"cuda:{device}",
        "name": props.name,
        "total_memory_gb": props.total_memory / (1024**3),
        "compute_capability": f"{props.major}.{props.minor}",
        "multi_processor_count": props.multi_processor_count,
    }


def clear_gpu_memory() -> float:
    """
    Clear GPU memory cache and return freed memory in GB.

    Returns:
        Amount of memory freed in GB.

    Example:
        >>> freed = clear_gpu_memory()
        >>> print(f"Freed {freed:.2f}GB of GPU memory")
    """
    if not torch.cuda.is_available():
        return 0.0

    before = torch.cuda.memory_allocated() / (1024**3)
    torch.cuda.empty_cache()
    gc.collect()
    after = torch.cuda.memory_allocated() / (1024**3)

    return before - after


def get_memory_usage() -> Dict[str, float]:
    """
    Get current GPU memory usage statistics.

    Returns:
        Dict with allocated, reserved, and free memory in GB.

    Example:
        >>> usage = get_memory_usage()
        >>> print(f"Using {usage['allocated_gb']:.1f}GB of {usage['total_gb']:.1f}GB")
    """
    if not torch.cuda.is_available():
        return {"available": False}

    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)

    return {
        "available": True,
        "total_gb": total,
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "free_gb": total - reserved,
    }


def load_llava(
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    torch_dtype: torch.dtype = torch.bfloat16,
    load_in_4bit: bool = False,
    device_map: str = "auto",
) -> Tuple[Any, Any]:
    """
    Load a LLaVA vision-language model optimized for DGX Spark.

    Args:
        model_name: HuggingFace model name or path.
        torch_dtype: Data type for model weights (bfloat16 recommended for Blackwell).
        load_in_4bit: Whether to load in 4-bit quantization for larger models.
        device_map: Device mapping strategy ("auto" recommended).

    Returns:
        Tuple of (model, processor).

    Example:
        >>> model, processor = load_llava("llava-hf/llava-1.5-7b-hf")
        >>> # Model uses ~16GB of the 128GB available
    """
    from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

    print(f"Loading LLaVA model: {model_name}")
    start_time = time.time()

    # Configure quantization if requested
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        print("  Using 4-bit quantization")

    # Load processor
    processor = AutoProcessor.from_pretrained(model_name)

    # Load model
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
    )

    elapsed = time.time() - start_time
    memory = get_memory_usage()

    print(f"  Loaded in {elapsed:.1f}s")
    print(f"  Memory usage: {memory['allocated_gb']:.1f}GB / {memory['total_gb']:.1f}GB")

    return model, processor


def load_qwen_vl(
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype: torch.dtype = torch.bfloat16,
    load_in_4bit: bool = False,
    device_map: str = "auto",
) -> Tuple[Any, Any]:
    """
    Load a Qwen-VL vision-language model optimized for DGX Spark.

    Args:
        model_name: HuggingFace model name or path.
        torch_dtype: Data type for model weights.
        load_in_4bit: Whether to load in 4-bit quantization.
        device_map: Device mapping strategy.

    Returns:
        Tuple of (model, processor).

    Example:
        >>> model, processor = load_qwen_vl("Qwen/Qwen2-VL-7B-Instruct")
        >>> # For 72B model, use load_in_4bit=True
    """
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig

    print(f"Loading Qwen-VL model: {model_name}")
    start_time = time.time()

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    processor = AutoProcessor.from_pretrained(model_name)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
    )

    elapsed = time.time() - start_time
    print(f"  Loaded in {elapsed:.1f}s")

    return model, processor


def load_clip(
    model_name: str = "openai/clip-vit-large-patch14",
    device: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Load a CLIP model for image-text embeddings.

    Args:
        model_name: CLIP model name from HuggingFace.
        device: Device to load model on (auto-detected if None).

    Returns:
        Tuple of (model, processor).

    Example:
        >>> model, processor = load_clip()
        >>> # CLIP is lightweight - only ~2GB
    """
    from transformers import CLIPModel, CLIPProcessor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading CLIP model: {model_name}")

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    print(f"  Loaded on {device}")

    return model, processor


def load_image(image_source: Union[str, Path, Image.Image]) -> Image.Image:
    """
    Load an image from various sources.

    Args:
        image_source: File path, URL, or PIL Image.

    Returns:
        PIL Image in RGB format.

    Example:
        >>> img = load_image("photo.jpg")
        >>> img = load_image("https://example.com/image.png")
    """
    import requests
    from io import BytesIO

    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")

    source_str = str(image_source)

    if source_str.startswith(("http://", "https://")):
        response = requests.get(source_str, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(source_str)

    return image.convert("RGB")


def analyze_image_llava(
    model: Any,
    processor: Any,
    image: Union[str, Path, Image.Image],
    question: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    """
    Analyze an image using LLaVA and answer a question about it.

    Args:
        model: Loaded LLaVA model.
        processor: LLaVA processor.
        image: Image to analyze.
        question: Question to answer about the image.
        max_new_tokens: Maximum tokens in response.
        temperature: Sampling temperature.

    Returns:
        Model's response as a string.

    Example:
        >>> model, processor = load_llava()
        >>> response = analyze_image_llava(
        ...     model, processor,
        ...     "cat.jpg",
        ...     "What breed is this cat?"
        ... )
        >>> print(response)
    """
    # Load image
    pil_image = load_image(image)

    # Create prompt
    prompt = f"USER: <image>\n{question}\nASSISTANT:"

    # Process inputs
    inputs = processor(text=prompt, images=pil_image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

    # Decode response
    response = processor.decode(output_ids[0], skip_special_tokens=True)

    # Extract assistant response
    if "ASSISTANT:" in response:
        response = response.split("ASSISTANT:")[-1].strip()

    return response


def analyze_image_qwen(
    model: Any,
    processor: Any,
    image: Union[str, Path, Image.Image],
    question: str,
    max_new_tokens: int = 512,
) -> str:
    """
    Analyze an image using Qwen-VL and answer a question about it.

    Args:
        model: Loaded Qwen-VL model.
        processor: Qwen-VL processor.
        image: Image to analyze.
        question: Question to answer about the image.
        max_new_tokens: Maximum tokens in response.

    Returns:
        Model's response as a string.

    Example:
        >>> model, processor = load_qwen_vl()
        >>> response = analyze_image_qwen(
        ...     model, processor,
        ...     "chart.png",
        ...     "What does this chart show?"
        ... )
    """
    from qwen_vl_utils import process_vision_info

    pil_image = load_image(image)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def get_clip_embeddings(
    model: Any,
    processor: Any,
    images: Optional[List[Union[str, Image.Image]]] = None,
    texts: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Get CLIP embeddings for images and/or text.

    Args:
        model: CLIP model.
        processor: CLIP processor.
        images: List of images to embed.
        texts: List of texts to embed.

    Returns:
        Dict with 'image_embeddings' and/or 'text_embeddings'.

    Example:
        >>> model, processor = load_clip()
        >>> embeddings = get_clip_embeddings(
        ...     model, processor,
        ...     images=["cat.jpg", "dog.jpg"],
        ...     texts=["a photo of a cat", "a photo of a dog"]
        ... )
        >>> print(embeddings['image_embeddings'].shape)
        torch.Size([2, 768])
    """
    result = {}
    device = next(model.parameters()).device

    if images is not None:
        pil_images = [load_image(img) for img in images]
        inputs = processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            # Normalize embeddings
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        result["image_embeddings"] = image_features.cpu()

    if texts is not None:
        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        result["text_embeddings"] = text_features.cpu()

    return result


def compute_similarity(
    embeddings1: torch.Tensor,
    embeddings2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cosine similarity between two sets of embeddings.

    Args:
        embeddings1: First set of embeddings [N, D].
        embeddings2: Second set of embeddings [M, D].

    Returns:
        Similarity matrix [N, M].

    Example:
        >>> sim = compute_similarity(image_embeds, text_embeds)
        >>> best_match = sim.argmax(dim=1)
    """
    # Ensure normalized
    embeddings1 = embeddings1 / embeddings1.norm(dim=-1, keepdim=True)
    embeddings2 = embeddings2 / embeddings2.norm(dim=-1, keepdim=True)

    return embeddings1 @ embeddings2.T


def batch_analyze_images(
    model: Any,
    processor: Any,
    images: List[Union[str, Path, Image.Image]],
    question: str,
    model_type: str = "llava",
    batch_size: int = 4,
    show_progress: bool = True,
) -> List[str]:
    """
    Analyze multiple images with the same question.

    Args:
        model: Vision-language model.
        processor: Model processor.
        images: List of images to analyze.
        question: Question to ask about each image.
        model_type: "llava" or "qwen".
        batch_size: Number of images to process at once.
        show_progress: Whether to show progress bar.

    Returns:
        List of responses for each image.

    Example:
        >>> responses = batch_analyze_images(
        ...     model, processor,
        ...     ["img1.jpg", "img2.jpg", "img3.jpg"],
        ...     "Describe this image in one sentence."
        ... )
    """
    from tqdm import tqdm

    analyze_fn = analyze_image_llava if model_type == "llava" else analyze_image_qwen

    results = []
    iterator = tqdm(images) if show_progress else images

    for image in iterator:
        try:
            response = analyze_fn(model, processor, image, question)
            results.append(response)
        except Exception as e:
            results.append(f"Error: {str(e)}")

    return results


def create_image_grid(
    images: List[Image.Image],
    cols: int = 3,
    padding: int = 10,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Create a grid of images for visualization.

    Args:
        images: List of PIL Images.
        cols: Number of columns in grid.
        padding: Padding between images in pixels.
        bg_color: Background color as RGB tuple.

    Returns:
        Grid image.

    Example:
        >>> images = [Image.open(f"img{i}.jpg") for i in range(6)]
        >>> grid = create_image_grid(images, cols=3)
        >>> grid.save("grid.jpg")
    """
    if not images:
        raise ValueError("No images provided")

    # Calculate grid dimensions
    rows = (len(images) + cols - 1) // cols

    # Get maximum image dimensions
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create grid canvas
    grid_width = cols * max_width + (cols + 1) * padding
    grid_height = rows * max_height + (rows + 1) * padding

    grid = Image.new("RGB", (grid_width, grid_height), bg_color)

    # Place images
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols

        x = padding + col * (max_width + padding) + (max_width - img.width) // 2
        y = padding + row * (max_height + padding) + (max_height - img.height) // 2

        grid.paste(img, (x, y))

    return grid


if __name__ == "__main__":
    # Quick test
    print("VLM Utils - DGX Spark Optimized")
    print("=" * 40)

    info = get_device_info()
    print(f"Device: {info.get('name', 'CPU')}")
    print(f"Memory: {info.get('total_memory_gb', 0):.1f}GB")

    usage = get_memory_usage()
    if usage.get("available"):
        print(f"Free Memory: {usage['free_gb']:.1f}GB")
