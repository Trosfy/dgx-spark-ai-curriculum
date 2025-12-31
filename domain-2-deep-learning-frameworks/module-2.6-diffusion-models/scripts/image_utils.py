"""
Image Utilities for Diffusion Models

Utilities for preprocessing, visualization, and image manipulation:
- Image loading and conversion
- Control image preparation (edges, depth, pose)
- Visualization grids
- Color space operations

Example Usage:
    >>> from scripts.image_utils import show_image_grid, get_canny_edges
    >>>
    >>> # Show a grid of generated images
    >>> show_image_grid(images, nrow=4, title="Generated Samples")
    >>>
    >>> # Prepare edge map for ControlNet
    >>> edges = get_canny_edges(image, low_threshold=100, high_threshold=200)

DGX Spark Optimization:
    - Batch processing for multiple images
    - GPU-accelerated preprocessing where possible
    - Memory-efficient image handling
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, List, Optional, Tuple
import warnings


def pil_to_tensor(
    image: Image.Image,
    normalize: bool = True,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert PIL Image to PyTorch tensor.

    Args:
        image: PIL Image (RGB or grayscale)
        normalize: If True, normalize to [-1, 1]; else [0, 1]
        dtype: Output tensor dtype

    Returns:
        Tensor of shape (C, H, W)

    Example:
        >>> img = Image.open("photo.jpg")
        >>> tensor = pil_to_tensor(img, normalize=True)
        >>> tensor.shape  # (3, H, W)
        >>> tensor.min(), tensor.max()  # (-1, 1)
    """
    # Convert to numpy
    np_image = np.array(image)

    # Handle grayscale
    if len(np_image.shape) == 2:
        np_image = np_image[:, :, np.newaxis]

    # Convert to tensor: (H, W, C) -> (C, H, W)
    tensor = torch.from_numpy(np_image).permute(2, 0, 1).to(dtype)

    # Normalize
    if normalize:
        # [0, 255] -> [-1, 1]
        tensor = (tensor / 127.5) - 1.0
    else:
        # [0, 255] -> [0, 1]
        tensor = tensor / 255.0

    return tensor


def tensor_to_pil(
    tensor: torch.Tensor,
    normalized: bool = True,
) -> Union[Image.Image, List[Image.Image]]:
    """
    Convert PyTorch tensor to PIL Image(s).

    Args:
        tensor: Tensor of shape (C, H, W) or (B, C, H, W)
        normalized: If True, tensor is in [-1, 1]; else [0, 1]

    Returns:
        Single PIL Image or list of PIL Images

    Example:
        >>> # Single image
        >>> img = tensor_to_pil(generated_tensor)
        >>> img.save("output.png")
        >>>
        >>> # Batch of images
        >>> images = tensor_to_pil(batch_tensor)
        >>> for i, img in enumerate(images):
        ...     img.save(f"output_{i}.png")
    """
    # Handle batch dimension
    if len(tensor.shape) == 4:
        return [tensor_to_pil(t, normalized) for t in tensor]

    # Move to CPU and convert to numpy
    tensor = tensor.detach().cpu()

    # Denormalize
    if normalized:
        # [-1, 1] -> [0, 255]
        tensor = ((tensor + 1.0) * 127.5).clamp(0, 255)
    else:
        # [0, 1] -> [0, 255]
        tensor = (tensor * 255).clamp(0, 255)

    # Convert to numpy: (C, H, W) -> (H, W, C)
    np_image = tensor.permute(1, 2, 0).numpy().astype(np.uint8)

    # Handle grayscale
    if np_image.shape[2] == 1:
        np_image = np_image.squeeze(-1)

    return Image.fromarray(np_image)


def load_and_preprocess(
    path: Union[str, Path],
    size: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Load an image and preprocess it for diffusion models.

    Args:
        path: Path to image file
        size: Target size (height, width) or None to keep original
        normalize: If True, normalize to [-1, 1]
        device: Target device
        dtype: Target dtype

    Returns:
        Preprocessed image tensor of shape (1, C, H, W)

    Example:
        >>> image = load_and_preprocess("input.jpg", size=(512, 512))
        >>> image.shape
        torch.Size([1, 3, 512, 512])
    """
    # Load image
    image = Image.open(path).convert("RGB")

    # Resize if needed
    if size is not None:
        image = image.resize((size[1], size[0]), Image.LANCZOS)

    # Convert to tensor
    tensor = pil_to_tensor(image, normalize=normalize)

    # Add batch dimension and move to device
    tensor = tensor.unsqueeze(0).to(device=device, dtype=dtype)

    return tensor


def show_image_grid(
    images: Union[torch.Tensor, List[Image.Image], List[torch.Tensor]],
    nrow: int = 4,
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    normalized: bool = True,
    show: bool = True,
) -> Optional[Image.Image]:
    """
    Display a grid of images.

    Args:
        images: Tensor (B, C, H, W) or list of images
        nrow: Number of images per row
        title: Optional title for the figure
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the figure
        normalized: Whether tensors are normalized to [-1, 1]
        show: Whether to display the figure

    Returns:
        Grid as PIL Image if save_path is None, else None

    Example:
        >>> # Display generated samples
        >>> show_image_grid(generated_images, nrow=4, title="DDPM Samples")
        >>>
        >>> # Save without displaying
        >>> show_image_grid(images, save_path="grid.png", show=False)
    """
    import matplotlib.pyplot as plt

    # Convert to list of PIL images
    if isinstance(images, torch.Tensor):
        pil_images = tensor_to_pil(images, normalized=normalized)
    elif isinstance(images[0], torch.Tensor):
        pil_images = [tensor_to_pil(img, normalized) for img in images]
    else:
        pil_images = images

    # Calculate grid dimensions
    n_images = len(pil_images)
    ncol = nrow
    nrow_actual = (n_images + ncol - 1) // ncol

    # Get image size from first image
    img_w, img_h = pil_images[0].size

    # Create grid image
    grid_w = ncol * img_w
    grid_h = nrow_actual * img_h
    grid = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))

    # Paste images
    for i, img in enumerate(pil_images):
        row = i // ncol
        col = i % ncol
        grid.paste(img, (col * img_w, row * img_h))

    # Display with matplotlib
    if show or title is not None:
        if figsize is None:
            figsize = (ncol * 2, nrow_actual * 2)

        plt.figure(figsize=figsize)
        plt.imshow(grid)
        plt.axis("off")
        if title:
            plt.title(title, fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"Saved grid to {save_path}")

        if show:
            plt.show()
        plt.close()

    elif save_path:
        grid.save(save_path)
        print(f"Saved grid to {save_path}")

    return grid


def get_canny_edges(
    image: Union[Image.Image, torch.Tensor, np.ndarray],
    low_threshold: int = 100,
    high_threshold: int = 200,
    return_pil: bool = True,
) -> Union[Image.Image, np.ndarray]:
    """
    Extract Canny edges from an image for ControlNet.

    ELI5: This finds the "outlines" in an image - like tracing a coloring book.
    The thresholds control how sensitive the edge detection is.

    Args:
        image: Input image (PIL, tensor, or numpy array)
        low_threshold: Lower threshold for edge detection (0-255)
        high_threshold: Upper threshold for edge detection (0-255)
        return_pil: If True, return PIL Image; else numpy array

    Returns:
        Edge map (white edges on black background)

    Example:
        >>> edges = get_canny_edges(photo, low_threshold=100, high_threshold=200)
        >>> edges.save("edges.png")
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV is required for edge detection. Install with: pip install opencv-python")

    # Convert to numpy array
    if isinstance(image, torch.Tensor):
        if len(image.shape) == 4:
            image = image[0]  # Remove batch dimension
        image = tensor_to_pil(image)

    if isinstance(image, Image.Image):
        np_image = np.array(image)
    else:
        np_image = image

    # Convert to grayscale if needed
    if len(np_image.shape) == 3:
        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = np_image

    # Apply Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    if return_pil:
        return Image.fromarray(edges)
    return edges


def get_depth_map(
    image: Union[Image.Image, torch.Tensor],
    model_type: str = "MiDaS_small",
    device: str = "cuda",
) -> Image.Image:
    """
    Estimate depth map from an image for ControlNet.

    ELI5: This guesses how far away each part of the picture is, like
    creating a "distance map" where darker = closer and lighter = farther.

    Args:
        image: Input image
        model_type: MiDaS model type ("MiDaS_small", "DPT_Hybrid", "DPT_Large")
        device: Device to run inference on

    Returns:
        Depth map as PIL Image (grayscale)

    Example:
        >>> depth = get_depth_map(photo)
        >>> depth.save("depth_map.png")

    Note:
        Requires torch hub. First run will download the model (~400MB for small).
    """
    # Convert to PIL if needed
    if isinstance(image, torch.Tensor):
        if len(image.shape) == 4:
            image = image[0]
        image = tensor_to_pil(image)

    # Load MiDaS model
    midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    midas = midas.to(device).eval()

    # Load transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    # Prepare input
    np_image = np.array(image)
    input_batch = transform(np_image).to(device)

    # Inference
    with torch.no_grad():
        prediction = midas(input_batch)

        # Resize to original size
        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=np_image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Normalize to [0, 255]
    depth = prediction.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
    depth = depth.astype(np.uint8)

    return Image.fromarray(depth)


def prepare_control_image(
    image: Union[Image.Image, torch.Tensor, str],
    control_type: str,
    size: Optional[Tuple[int, int]] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    **kwargs,
) -> torch.Tensor:
    """
    Prepare a control image for ControlNet.

    This is a convenience function that handles loading, preprocessing,
    and extracting control signals (edges, depth, etc.) in one call.

    Args:
        image: Input image (PIL, tensor, or path)
        control_type: Type of control ("canny", "depth", "none")
        size: Target size (height, width)
        device: Target device
        dtype: Target dtype
        **kwargs: Additional arguments for specific control types

    Returns:
        Control image tensor of shape (1, 3, H, W)

    Example:
        >>> # Prepare Canny edges
        >>> control = prepare_control_image(
        ...     "input.jpg",
        ...     control_type="canny",
        ...     size=(1024, 1024),
        ...     low_threshold=100,
        ...     high_threshold=200,
        ... )
        >>>
        >>> # Use with ControlNet pipeline
        >>> output = pipe(prompt="...", image=control)
    """
    # Load image if path
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, torch.Tensor):
        if len(image.shape) == 4:
            image = image[0]
        image = tensor_to_pil(image)

    # Resize if needed
    if size is not None:
        image = image.resize((size[1], size[0]), Image.LANCZOS)

    # Extract control signal
    if control_type == "canny":
        low = kwargs.get("low_threshold", 100)
        high = kwargs.get("high_threshold", 200)
        control = get_canny_edges(image, low, high)
        # Convert grayscale to RGB
        control = control.convert("RGB")
    elif control_type == "depth":
        model = kwargs.get("model_type", "MiDaS_small")
        control = get_depth_map(image, model, device)
        control = control.convert("RGB")
    elif control_type == "none" or control_type is None:
        control = image
    else:
        raise ValueError(f"Unknown control type: {control_type}")

    # Convert to tensor
    tensor = pil_to_tensor(control, normalize=True)
    tensor = tensor.unsqueeze(0).to(device=device, dtype=dtype)

    return tensor


def create_image_variations(
    image: Union[Image.Image, torch.Tensor],
    num_variations: int = 4,
    noise_strength: float = 0.3,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Create noisy variations of an image for img2img.

    This adds controlled amounts of noise to an image, which can then be
    denoised to create variations of the original.

    Args:
        image: Input image
        num_variations: Number of variations to create
        noise_strength: Amount of noise to add (0-1)
        device: Target device
        dtype: Target dtype

    Returns:
        Tensor of shape (num_variations, C, H, W)

    Example:
        >>> variations = create_image_variations(original, num_variations=4)
        >>> # Feed to img2img pipeline with appropriate denoising strength
    """
    if isinstance(image, Image.Image):
        tensor = pil_to_tensor(image, normalize=True)
        tensor = tensor.to(device=device, dtype=dtype)
    else:
        tensor = image

    # Expand to batch
    batch = tensor.unsqueeze(0).expand(num_variations, -1, -1, -1)

    # Add noise
    noise = torch.randn_like(batch)
    noisy = batch * (1 - noise_strength) + noise * noise_strength

    return noisy


def resize_for_diffusion(
    image: Image.Image,
    max_size: int = 1024,
    multiple_of: int = 64,
) -> Image.Image:
    """
    Resize image to be compatible with diffusion models.

    Diffusion models typically require dimensions to be multiples of 64
    (or 8 for some models). This function resizes while maintaining
    aspect ratio.

    Args:
        image: Input PIL Image
        max_size: Maximum dimension (width or height)
        multiple_of: Ensure dimensions are multiples of this

    Returns:
        Resized PIL Image

    Example:
        >>> img = Image.open("photo.jpg")  # 1920x1080
        >>> resized = resize_for_diffusion(img, max_size=1024)
        >>> resized.size  # (1024, 576) - maintains aspect ratio
    """
    w, h = image.size

    # Scale to max_size
    scale = min(max_size / w, max_size / h, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Round to multiple
    new_w = (new_w // multiple_of) * multiple_of
    new_h = (new_h // multiple_of) * multiple_of

    # Ensure minimum size
    new_w = max(new_w, multiple_of)
    new_h = max(new_h, multiple_of)

    if (new_w, new_h) != (w, h):
        image = image.resize((new_w, new_h), Image.LANCZOS)

    return image


def make_image_square(
    image: Image.Image,
    size: int = 1024,
    fill_color: Tuple[int, int, int] = (0, 0, 0),
    mode: str = "pad",
) -> Image.Image:
    """
    Make an image square by padding or cropping.

    Args:
        image: Input PIL Image
        size: Target size (will be size x size)
        fill_color: Color for padding (if mode="pad")
        mode: "pad" to add borders, "crop" to center crop

    Returns:
        Square PIL Image

    Example:
        >>> img = Image.open("landscape.jpg")  # 1920x1080
        >>> square = make_image_square(img, size=1024, mode="pad")
        >>> square.size  # (1024, 1024)
    """
    w, h = image.size

    if mode == "crop":
        # Center crop to square
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        image = image.crop((left, top, left + min_dim, top + min_dim))
        image = image.resize((size, size), Image.LANCZOS)
    else:
        # Resize maintaining aspect ratio, then pad
        scale = min(size / w, size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)

        # Create square canvas and paste
        canvas = Image.new("RGB", (size, size), fill_color)
        paste_x = (size - new_w) // 2
        paste_y = (size - new_h) // 2
        canvas.paste(image, (paste_x, paste_y))
        image = canvas

    return image


def blend_images(
    image1: Image.Image,
    image2: Image.Image,
    alpha: float = 0.5,
) -> Image.Image:
    """
    Blend two images together.

    Args:
        image1: First image
        image2: Second image
        alpha: Blend factor (0 = all image1, 1 = all image2)

    Returns:
        Blended PIL Image
    """
    # Ensure same size
    if image1.size != image2.size:
        image2 = image2.resize(image1.size, Image.LANCZOS)

    return Image.blend(image1, image2, alpha)


def create_mask_from_prompt(
    image: Image.Image,
    prompt: str,
    device: str = "cuda",
) -> Image.Image:
    """
    Create a segmentation mask from a text prompt using CLIPSeg.

    ELI5: Tell the AI what you want to select in the image with words,
    and it will highlight that part like a magic highlighter.

    Args:
        image: Input image
        prompt: Text description of what to segment (e.g., "the cat")
        device: Device for inference

    Returns:
        Mask as PIL Image (white = selected, black = not selected)

    Note:
        Requires transformers with CLIPSeg. Install if needed:
        pip install transformers
    """
    try:
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    except ImportError:
        raise ImportError(
            "CLIPSeg requires transformers. Install with: pip install transformers"
        )

    # Load model
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = model.to(device)

    # Process
    inputs = processor(
        text=[prompt],
        images=[image],
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        preds = outputs.logits

    # Resize and threshold
    mask = torch.sigmoid(preds)
    mask = F.interpolate(
        mask.unsqueeze(1),
        size=image.size[::-1],  # (H, W)
        mode="bilinear"
    ).squeeze()

    # Convert to PIL
    mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(mask_np)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Image utilities module loaded successfully!")
    print("\nAvailable functions:")
    print("  - pil_to_tensor: Convert PIL Image to tensor")
    print("  - tensor_to_pil: Convert tensor to PIL Image")
    print("  - load_and_preprocess: Load and prepare image for models")
    print("  - show_image_grid: Display/save grid of images")
    print("  - get_canny_edges: Extract edge map for ControlNet")
    print("  - get_depth_map: Estimate depth map for ControlNet")
    print("  - prepare_control_image: One-stop control image preparation")
    print("  - resize_for_diffusion: Resize to model-compatible dimensions")
    print("  - make_image_square: Pad or crop to square")

    # Demo with synthetic image
    print("\nDemo: Creating and processing a test image...")
    test_image = Image.new("RGB", (800, 600), color=(100, 150, 200))

    # Resize for diffusion
    resized = resize_for_diffusion(test_image, max_size=512)
    print(f"  Original: {test_image.size} -> Resized: {resized.size}")

    # Make square
    square = make_image_square(test_image, size=512, mode="pad")
    print(f"  Square (padded): {square.size}")

    print("\nImage utilities ready for use!")
