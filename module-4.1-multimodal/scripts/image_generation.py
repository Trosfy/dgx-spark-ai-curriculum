"""
Image Generation Utilities for DGX Spark

This module provides utilities for generating images using Stable Diffusion,
SDXL, Flux, and ControlNet on DGX Spark's 128GB unified memory.

Example:
    >>> from scripts.image_generation import load_sdxl, generate_image
    >>> pipe = load_sdxl()
    >>> image = generate_image(pipe, "A serene mountain landscape at sunset")
    >>> image.save("landscape.png")
"""

import gc
import time
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Any, Literal

import torch
from PIL import Image
import numpy as np


def get_optimal_dtype() -> torch.dtype:
    """
    Get the optimal dtype for the current GPU.

    Returns:
        torch.bfloat16 for Blackwell/Ampere+, torch.float16 otherwise.

    Example:
        >>> dtype = get_optimal_dtype()
        >>> print(dtype)
        torch.bfloat16
    """
    if not torch.cuda.is_available():
        return torch.float32

    # Check compute capability - Blackwell is 10.0, Ampere is 8.x
    capability = torch.cuda.get_device_capability()
    if capability[0] >= 8:
        return torch.bfloat16
    return torch.float16


def clear_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def load_sdxl(
    model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype: Optional[torch.dtype] = None,
    use_refiner: bool = False,
    enable_vae_slicing: bool = True,
    enable_vae_tiling: bool = False,
) -> Any:
    """
    Load SDXL pipeline optimized for DGX Spark.

    Args:
        model_name: SDXL model from HuggingFace.
        torch_dtype: Data type (auto-detected if None).
        use_refiner: Whether to also load the refiner model.
        enable_vae_slicing: Enable memory-efficient VAE decoding.
        enable_vae_tiling: Enable tiled VAE for very large images.

    Returns:
        SDXL pipeline (or dict with 'base' and 'refiner' if use_refiner=True).

    Example:
        >>> pipe = load_sdxl()
        >>> # Uses ~8GB VRAM - plenty of headroom on DGX Spark!
    """
    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

    if torch_dtype is None:
        torch_dtype = get_optimal_dtype()

    print(f"Loading SDXL: {model_name}")
    start_time = time.time()

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        variant="fp16" if torch_dtype == torch.float16 else None,
    )
    pipe = pipe.to("cuda")

    # Memory optimizations
    if enable_vae_slicing:
        pipe.enable_vae_slicing()
    if enable_vae_tiling:
        pipe.enable_vae_tiling()

    elapsed = time.time() - start_time
    print(f"  Loaded in {elapsed:.1f}s")

    if use_refiner:
        print("Loading SDXL Refiner...")
        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )
        refiner = refiner.to("cuda")
        if enable_vae_slicing:
            refiner.enable_vae_slicing()

        return {"base": pipe, "refiner": refiner}

    return pipe


def load_flux(
    model_name: str = "black-forest-labs/FLUX.1-dev",
    torch_dtype: Optional[torch.dtype] = None,
    enable_model_cpu_offload: bool = False,
) -> Any:
    """
    Load Flux pipeline for state-of-the-art image generation.

    Args:
        model_name: Flux model from HuggingFace.
        torch_dtype: Data type (auto-detected if None).
        enable_model_cpu_offload: Offload to CPU for memory efficiency.

    Returns:
        Flux pipeline.

    Example:
        >>> pipe = load_flux()
        >>> # Uses ~24GB - fits easily on DGX Spark
    """
    from diffusers import FluxPipeline

    if torch_dtype is None:
        torch_dtype = get_optimal_dtype()

    print(f"Loading Flux: {model_name}")
    print("  Note: This may take a few minutes for first download...")
    start_time = time.time()

    pipe = FluxPipeline.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    )

    if enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")

    elapsed = time.time() - start_time
    print(f"  Loaded in {elapsed:.1f}s")

    return pipe


def load_controlnet(
    controlnet_model: str = "diffusers/controlnet-canny-sdxl-1.0",
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype: Optional[torch.dtype] = None,
) -> Tuple[Any, Any]:
    """
    Load ControlNet with SDXL base model.

    Args:
        controlnet_model: ControlNet model from HuggingFace.
        base_model: Base SDXL model.
        torch_dtype: Data type.

    Returns:
        Tuple of (pipeline, controlnet).

    Example:
        >>> pipe, controlnet = load_controlnet()
        >>> # Use with edge detection for precise control
    """
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

    if torch_dtype is None:
        torch_dtype = get_optimal_dtype()

    print(f"Loading ControlNet: {controlnet_model}")
    start_time = time.time()

    controlnet = ControlNetModel.from_pretrained(
        controlnet_model,
        torch_dtype=torch_dtype,
    )

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        torch_dtype=torch_dtype,
        use_safetensors=True,
    )
    pipe = pipe.to("cuda")
    pipe.enable_vae_slicing()

    elapsed = time.time() - start_time
    print(f"  Loaded in {elapsed:.1f}s")

    return pipe, controlnet


def generate_image(
    pipe: Any,
    prompt: str,
    negative_prompt: Optional[str] = None,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    num_images: int = 1,
) -> Union[Image.Image, List[Image.Image]]:
    """
    Generate image(s) from a text prompt.

    Args:
        pipe: Diffusion pipeline (SDXL, Flux, etc.).
        prompt: Text description of desired image.
        negative_prompt: What to avoid in the image.
        width: Image width (must be divisible by 8).
        height: Image height (must be divisible by 8).
        num_inference_steps: Number of denoising steps (more = better quality).
        guidance_scale: How closely to follow the prompt.
        seed: Random seed for reproducibility.
        num_images: Number of images to generate.

    Returns:
        Single image or list of images.

    Example:
        >>> image = generate_image(
        ...     pipe,
        ...     "A majestic lion in a sunset savanna, photorealistic",
        ...     negative_prompt="blurry, low quality",
        ...     num_inference_steps=30
        ... )
        >>> image.save("lion.png")
    """
    # Set up generator for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    # Default negative prompt if not provided
    if negative_prompt is None:
        negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy"

    print(f"Generating image: '{prompt[:50]}...' " if len(prompt) > 50 else f"Generating image: '{prompt}'")
    start_time = time.time()

    # Generate
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        num_images_per_prompt=num_images,
    )

    elapsed = time.time() - start_time
    print(f"  Generated in {elapsed:.1f}s ({elapsed/num_images:.1f}s per image)")

    images = result.images
    return images[0] if num_images == 1 else images


def generate_with_refiner(
    pipes: Dict[str, Any],
    prompt: str,
    negative_prompt: Optional[str] = None,
    width: int = 1024,
    height: int = 1024,
    base_steps: int = 30,
    refiner_steps: int = 20,
    high_noise_frac: float = 0.8,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Generate image using SDXL base + refiner for highest quality.

    Args:
        pipes: Dict with 'base' and 'refiner' pipelines.
        prompt: Text description.
        negative_prompt: What to avoid.
        width: Image width.
        height: Image height.
        base_steps: Steps for base model.
        refiner_steps: Steps for refiner.
        high_noise_frac: Fraction of steps for base model.
        seed: Random seed.

    Returns:
        Refined image.

    Example:
        >>> pipes = load_sdxl(use_refiner=True)
        >>> image = generate_with_refiner(pipes, "A detailed portrait")
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    if negative_prompt is None:
        negative_prompt = "blurry, low quality, distorted"

    print("Stage 1: Base model generation...")
    start_time = time.time()

    # Generate with base
    image = pipes["base"](
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=base_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
        generator=generator,
    ).images

    print("Stage 2: Refiner enhancement...")

    # Refine
    image = pipes["refiner"](
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        num_inference_steps=refiner_steps,
        denoising_start=high_noise_frac,
        generator=generator,
    ).images[0]

    elapsed = time.time() - start_time
    print(f"  Total generation time: {elapsed:.1f}s")

    return image


def generate_with_controlnet(
    pipe: Any,
    prompt: str,
    control_image: Union[str, Path, Image.Image],
    control_type: Literal["canny", "depth", "pose", "scribble"] = "canny",
    negative_prompt: Optional[str] = None,
    controlnet_conditioning_scale: float = 0.5,
    num_inference_steps: int = 30,
    seed: Optional[int] = None,
) -> Tuple[Image.Image, Image.Image]:
    """
    Generate image with ControlNet guidance.

    Args:
        pipe: ControlNet pipeline.
        prompt: Text description.
        control_image: Image to extract control signal from.
        control_type: Type of control signal.
        negative_prompt: What to avoid.
        controlnet_conditioning_scale: Strength of control (0-1).
        num_inference_steps: Number of denoising steps.
        seed: Random seed.

    Returns:
        Tuple of (generated_image, control_image).

    Example:
        >>> pipe, _ = load_controlnet()
        >>> image, control = generate_with_controlnet(
        ...     pipe,
        ...     "A beautiful castle",
        ...     "sketch.png",
        ...     control_type="canny"
        ... )
    """
    import cv2
    from controlnet_aux import CannyDetector, MidasDetector, OpenposeDetector

    # Load control image
    if isinstance(control_image, (str, Path)):
        control_img = Image.open(control_image).convert("RGB")
    else:
        control_img = control_image.convert("RGB")

    # Process control image based on type
    print(f"Processing control image ({control_type})...")

    if control_type == "canny":
        detector = CannyDetector()
        processed = detector(control_img)
    elif control_type == "depth":
        detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
        processed = detector(control_img)
    elif control_type == "pose":
        detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        processed = detector(control_img)
    elif control_type == "scribble":
        # Simple edge detection for scribble
        img_array = np.array(control_img.convert("L"))
        edges = cv2.Canny(img_array, 50, 150)
        processed = Image.fromarray(edges).convert("RGB")
    else:
        raise ValueError(f"Unknown control type: {control_type}")

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    if negative_prompt is None:
        negative_prompt = "blurry, low quality, distorted"

    print("Generating with ControlNet guidance...")
    start_time = time.time()

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=processed,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )

    elapsed = time.time() - start_time
    print(f"  Generated in {elapsed:.1f}s")

    return result.images[0], processed


def img2img(
    pipe: Any,
    prompt: str,
    init_image: Union[str, Path, Image.Image],
    strength: float = 0.75,
    negative_prompt: Optional[str] = None,
    num_inference_steps: int = 30,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Transform an existing image based on a prompt.

    Args:
        pipe: Image-to-image pipeline.
        prompt: Text description of desired result.
        init_image: Starting image.
        strength: How much to change the image (0-1).
        negative_prompt: What to avoid.
        num_inference_steps: Number of denoising steps.
        seed: Random seed.

    Returns:
        Transformed image.

    Example:
        >>> image = img2img(
        ...     pipe,
        ...     "Turn this photo into an oil painting",
        ...     "photo.jpg",
        ...     strength=0.7
        ... )
    """
    from diffusers import AutoPipelineForImage2Image

    # Load image
    if isinstance(init_image, (str, Path)):
        init_img = Image.open(init_image).convert("RGB")
    else:
        init_img = init_image.convert("RGB")

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    if negative_prompt is None:
        negative_prompt = "blurry, low quality"

    print(f"Transforming image with strength={strength}")
    start_time = time.time()

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_img,
        strength=strength,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )

    elapsed = time.time() - start_time
    print(f"  Transformed in {elapsed:.1f}s")

    return result.images[0]


def inpaint(
    pipe: Any,
    prompt: str,
    init_image: Union[str, Path, Image.Image],
    mask_image: Union[str, Path, Image.Image],
    negative_prompt: Optional[str] = None,
    num_inference_steps: int = 30,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Inpaint (fill in) masked regions of an image.

    Args:
        pipe: Inpainting pipeline.
        prompt: Description of what to fill in.
        init_image: Original image.
        mask_image: White = inpaint, Black = keep.
        negative_prompt: What to avoid.
        num_inference_steps: Number of steps.
        seed: Random seed.

    Returns:
        Inpainted image.

    Example:
        >>> result = inpaint(
        ...     pipe,
        ...     "A cute cat sitting",
        ...     "room.jpg",
        ...     "mask.png"  # White where the cat should appear
        ... )
    """
    # Load images
    if isinstance(init_image, (str, Path)):
        init_img = Image.open(init_image).convert("RGB")
    else:
        init_img = init_image.convert("RGB")

    if isinstance(mask_image, (str, Path)):
        mask_img = Image.open(mask_image).convert("L")
    else:
        mask_img = mask_image.convert("L")

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    if negative_prompt is None:
        negative_prompt = "blurry, low quality"

    print("Inpainting...")
    start_time = time.time()

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_img,
        mask_image=mask_img,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )

    elapsed = time.time() - start_time
    print(f"  Inpainted in {elapsed:.1f}s")

    return result.images[0]


def create_comparison_grid(
    images: List[Image.Image],
    labels: List[str],
    cols: int = 3,
    font_size: int = 20,
) -> Image.Image:
    """
    Create a comparison grid with labeled images.

    Args:
        images: List of images to compare.
        labels: Labels for each image.
        cols: Number of columns.
        font_size: Label font size.

    Returns:
        Grid image with labels.

    Example:
        >>> grid = create_comparison_grid(
        ...     [img1, img2, img3],
        ...     ["Steps=20", "Steps=30", "Steps=50"]
        ... )
    """
    from PIL import ImageDraw, ImageFont

    if len(images) != len(labels):
        raise ValueError("Number of images and labels must match")

    # Calculate grid dimensions
    rows = (len(images) + cols - 1) // cols
    padding = 10
    label_height = font_size + 10

    # Resize all images to same size
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    resized = []
    for img in images:
        if img.size != (max_width, max_height):
            img = img.resize((max_width, max_height), Image.Resampling.LANCZOS)
        resized.append(img)

    # Create canvas
    grid_width = cols * max_width + (cols + 1) * padding
    grid_height = rows * (max_height + label_height) + (rows + 1) * padding

    grid = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    # Place images with labels
    for idx, (img, label) in enumerate(zip(resized, labels)):
        row = idx // cols
        col = idx % cols

        x = padding + col * (max_width + padding)
        y = padding + row * (max_height + label_height + padding)

        grid.paste(img, (x, y))

        # Add label
        text_x = x + max_width // 2
        text_y = y + max_height + 5
        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font, anchor="mt")

    return grid


def save_generation_metadata(
    image: Image.Image,
    filepath: Union[str, Path],
    prompt: str,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    model_name: Optional[str] = None,
) -> None:
    """
    Save image with generation metadata in PNG info.

    Args:
        image: Generated image.
        filepath: Output path.
        prompt: Generation prompt.
        negative_prompt: Negative prompt used.
        seed: Random seed used.
        steps: Number of inference steps.
        guidance_scale: Guidance scale used.
        model_name: Model used for generation.

    Example:
        >>> save_generation_metadata(
        ...     image, "output.png",
        ...     prompt="A sunset over mountains",
        ...     seed=42, steps=30
        ... )
    """
    from PIL import PngImagePlugin

    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("prompt", prompt)

    if negative_prompt:
        metadata.add_text("negative_prompt", negative_prompt)
    if seed is not None:
        metadata.add_text("seed", str(seed))
    if steps is not None:
        metadata.add_text("steps", str(steps))
    if guidance_scale is not None:
        metadata.add_text("guidance_scale", str(guidance_scale))
    if model_name:
        metadata.add_text("model", model_name)

    image.save(filepath, pnginfo=metadata)
    print(f"Saved to {filepath} with metadata")


if __name__ == "__main__":
    print("Image Generation Utils - DGX Spark Optimized")
    print("=" * 50)
    print(f"Optimal dtype: {get_optimal_dtype()}")
    print("Available models:")
    print("  - SDXL (~8GB VRAM)")
    print("  - Flux (~24GB VRAM)")
    print("  - ControlNet (~12GB VRAM)")
