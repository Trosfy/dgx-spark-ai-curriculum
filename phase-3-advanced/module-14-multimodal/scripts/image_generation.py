"""
Image Generation Utilities

This module provides utilities for image generation using diffusion models
(SDXL, Flux, ControlNet) on DGX Spark.

Example usage:
    from image_generation import ImageGenerator

    # Initialize with SDXL
    gen = ImageGenerator(model_name="sdxl")
    gen.load()

    # Generate an image
    image = gen.generate("A sunset over mountains, photorealistic")
    image.save("sunset.png")

    # Clean up
    gen.cleanup()
"""

import torch
import gc
import time
import numpy as np
from typing import Optional, List, Tuple, Union
from PIL import Image


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_usage() -> str:
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        return f"Allocated: {allocated:.2f}GB"
    return "No GPU"


def create_image_grid(images: List[Image.Image], cols: int = 2, padding: int = 10) -> Image.Image:
    """
    Create a grid of images.

    Args:
        images: List of PIL Images
        cols: Number of columns
        padding: Padding between images in pixels

    Returns:
        Combined grid image

    Example:
        >>> images = [gen.generate(p) for p in prompts]
        >>> grid = create_image_grid(images, cols=2)
        >>> grid.save("grid.png")
    """
    if not images:
        raise ValueError("No images provided")

    rows = (len(images) + cols - 1) // cols

    # Get max dimensions
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create output image
    grid_width = cols * max_width + (cols + 1) * padding
    grid_height = rows * max_height + (rows + 1) * padding
    grid = Image.new('RGB', (grid_width, grid_height), color='white')

    # Place images
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = padding + col * (max_width + padding)
        y = padding + row * (max_height + padding)

        # Center smaller images
        x_offset = (max_width - img.width) // 2
        y_offset = (max_height - img.height) // 2

        grid.paste(img, (x + x_offset, y + y_offset))

    return grid


class ImageGenerator:
    """
    Image Generation Pipeline.

    Supports SDXL, Flux, and ControlNet for various image generation tasks.
    Optimized for DGX Spark's 128GB unified memory.

    Attributes:
        model_name: Name of the generation model
        pipe: Loaded pipeline instance

    Example:
        >>> gen = ImageGenerator(model_name="sdxl")
        >>> gen.load()
        >>> image = gen.generate("A cat sitting on a couch")
        >>> gen.cleanup()
    """

    SUPPORTED_MODELS = {
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        "flux-schnell": "black-forest-labs/FLUX.1-schnell",
        "flux-dev": "black-forest-labs/FLUX.1-dev",
    }

    DEFAULT_NEGATIVE = "ugly, blurry, low quality, distorted, disfigured, bad anatomy"

    def __init__(self, model_name: str = "sdxl"):
        """
        Initialize the Image Generator.

        Args:
            model_name: Model to use ('sdxl', 'flux-schnell', 'flux-dev')
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}")

        self.model_name = model_name
        self.model_id = self.SUPPORTED_MODELS[model_name]
        self.pipe = None
        self._loaded = False

    def load(self) -> None:
        """Load the generation pipeline."""
        if self._loaded:
            print(f"{self.model_name} already loaded!")
            return

        clear_gpu_memory()
        print(f"Loading {self.model_name}...")
        start_time = time.time()

        if self.model_name == "sdxl":
            self._load_sdxl()
        elif self.model_name.startswith("flux"):
            self._load_flux()

        load_time = time.time() - start_time
        self._loaded = True
        print(f"\nLoaded in {load_time:.1f} seconds!")
        print(f"Memory: {get_memory_usage()}")

    def _load_sdxl(self) -> None:
        """Load SDXL pipeline."""
        from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            variant="fp16"
        )

        # Use faster scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        self.pipe = self.pipe.to("cuda")

    def _load_flux(self) -> None:
        """Load Flux pipeline."""
        from diffusers import FluxPipeline

        self.pipe = FluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16
        )
        self.pipe = self.pipe.to("cuda")

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text description of desired image
            negative_prompt: What to avoid (SDXL only)
            num_steps: Number of denoising steps
            guidance_scale: How closely to follow prompt
            width: Output width in pixels
            height: Output height in pixels
            seed: Random seed for reproducibility

        Returns:
            Generated PIL Image

        Example:
            >>> image = gen.generate(
            ...     "A futuristic city at night",
            ...     negative_prompt="blurry, ugly",
            ...     seed=42
            ... )
        """
        if not self._loaded:
            self.load()

        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            generator = None
            seed = torch.randint(0, 2**32, (1,)).item()

        print(f"Generating (seed={seed})...")
        start_time = time.time()

        gen_kwargs = {
            "prompt": prompt,
            "num_inference_steps": num_steps,
            "generator": generator,
            "width": width,
            "height": height,
        }

        # SDXL-specific parameters
        if self.model_name == "sdxl":
            gen_kwargs["negative_prompt"] = negative_prompt or self.DEFAULT_NEGATIVE
            gen_kwargs["guidance_scale"] = guidance_scale

        # Flux uses fewer steps
        if self.model_name.startswith("flux"):
            if self.model_name == "flux-schnell":
                gen_kwargs["num_inference_steps"] = min(num_steps, 4)
            gen_kwargs["guidance_scale"] = 0.0  # Flux doesn't use guidance

        with torch.inference_mode():
            result = self.pipe(**gen_kwargs)

        gen_time = time.time() - start_time
        print(f"Generated in {gen_time:.1f}s")

        return result.images[0]

    def generate_variations(
        self,
        base_prompt: str,
        styles: List[str],
        seed: int = 42,
        **kwargs
    ) -> List[Image.Image]:
        """
        Generate the same scene in multiple styles.

        Args:
            base_prompt: Core scene description
            styles: List of style modifiers
            seed: Base seed (incremented for each style)
            **kwargs: Additional generation parameters

        Returns:
            List of generated images

        Example:
            >>> images = gen.generate_variations(
            ...     "A dragon flying over a castle",
            ...     ["oil painting", "anime style", "photorealistic"]
            ... )
        """
        images = []
        for i, style in enumerate(styles):
            full_prompt = f"{base_prompt}, {style}"
            image = self.generate(full_prompt, seed=seed + i, **kwargs)
            images.append(image)
        return images

    def cleanup(self) -> None:
        """Release pipeline from memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self._loaded = False
            clear_gpu_memory()
            print(f"{self.model_name} unloaded!")


class ControlNetGenerator:
    """
    ControlNet Image Generation.

    Generates images guided by control signals like edges, depth, or pose.

    Example:
        >>> gen = ControlNetGenerator(control_type="canny")
        >>> gen.load()
        >>> control_image = get_canny_edges(photo)
        >>> result = gen.generate("A beautiful house", control_image)
    """

    CONTROL_TYPES = {
        "canny": "diffusers/controlnet-canny-sdxl-1.0",
        "depth": "diffusers/controlnet-depth-sdxl-1.0",
    }

    def __init__(self, control_type: str = "canny"):
        """
        Initialize ControlNet Generator.

        Args:
            control_type: Type of control ('canny', 'depth')
        """
        if control_type not in self.CONTROL_TYPES:
            raise ValueError(f"Unsupported control type: {control_type}")

        self.control_type = control_type
        self.controlnet_id = self.CONTROL_TYPES[control_type]
        self.pipe = None
        self._loaded = False

    def load(self) -> None:
        """Load ControlNet pipeline."""
        if self._loaded:
            return

        clear_gpu_memory()
        print(f"Loading ControlNet ({self.control_type})...")

        from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_id,
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        )

        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            variant="fp16"
        )

        self.pipe = self.pipe.to("cuda")
        self._loaded = True
        print("Loaded!")

    def generate(
        self,
        prompt: str,
        control_image: Image.Image,
        negative_prompt: Optional[str] = None,
        controlnet_conditioning_scale: float = 0.7,
        num_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Generate an image guided by a control image.

        Args:
            prompt: Text description
            control_image: Control signal image (edges, depth, etc.)
            negative_prompt: What to avoid
            controlnet_conditioning_scale: How strictly to follow control (0-1)
            num_steps: Denoising steps
            guidance_scale: Prompt guidance
            seed: Random seed

        Returns:
            Generated PIL Image
        """
        if not self._loaded:
            self.load()

        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            generator = None

        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or "ugly, blurry, low quality",
                image=control_image,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator
            )

        return result.images[0]

    def cleanup(self) -> None:
        """Release pipeline from memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self._loaded = False
            clear_gpu_memory()


def get_canny_edges(
    image: Image.Image,
    low_threshold: int = 100,
    high_threshold: int = 200
) -> Image.Image:
    """
    Extract Canny edges from an image.

    Args:
        image: Input PIL Image
        low_threshold: Lower threshold for edge detection
        high_threshold: Upper threshold for edge detection

    Returns:
        Edge image as RGB PIL Image

    Example:
        >>> edges = get_canny_edges(photo)
        >>> result = controlnet_gen.generate("A house", edges)
    """
    import cv2

    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges_rgb = np.stack([edges, edges, edges], axis=-1)

    return Image.fromarray(edges_rgb)


if __name__ == "__main__":
    print("Image Generation Demo")
    print("=" * 50)

    # Simple SDXL demo
    gen = ImageGenerator(model_name="sdxl")
    gen.load()

    image = gen.generate(
        "A peaceful mountain landscape at sunset",
        seed=42,
        num_steps=25
    )
    image.save("demo_output.png")
    print("Saved demo_output.png")

    gen.cleanup()
