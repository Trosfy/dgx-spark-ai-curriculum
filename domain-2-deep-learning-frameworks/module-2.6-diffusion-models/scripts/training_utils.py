"""
Training Utilities for Diffusion Models

Utilities for training and fine-tuning diffusion models:
- Dataset preparation for style training
- LoRA configuration helpers
- Training loops and callbacks
- SNR-weighted loss computation

Example Usage:
    >>> from scripts.training_utils import prepare_dataset, DiffusionDataset
    >>>
    >>> # Prepare dataset for LoRA training
    >>> dataset = prepare_dataset(
    ...     image_dir="./my_style_images",
    ...     caption_ext=".txt",
    ...     resolution=1024,
    ... )
    >>>
    >>> # Create data loader
    >>> loader = DataLoader(dataset, batch_size=1, shuffle=True)

DGX Spark Optimization:
    - Efficient data loading for unified memory architecture
    - Gradient checkpointing for larger batch sizes
    - Mixed precision training (bfloat16)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Optional, List, Dict, Union, Callable, Tuple
import json
import random
from dataclasses import dataclass, field
import math
import warnings


@dataclass
class TrainingConfig:
    """
    Configuration for diffusion model training.

    This dataclass holds all training hyperparameters in one place.

    Example:
        >>> config = TrainingConfig(
        ...     output_dir="./lora_output",
        ...     learning_rate=1e-4,
        ...     num_epochs=100,
        ...     batch_size=4,
        ... )
    """
    # Output
    output_dir: str = "./output"
    logging_dir: str = "./logs"

    # Training
    num_epochs: int = 100
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # LoRA specific
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "to_q", "to_k", "to_v", "to_out.0"
    ])

    # Data
    resolution: int = 1024
    center_crop: bool = True
    random_flip: float = 0.5

    # Optimization
    use_8bit_adam: bool = False
    use_gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"  # bf16 for DGX Spark

    # Logging
    save_every: int = 500
    validate_every: int = 100
    log_every: int = 10

    # Seeds
    seed: int = 42

    def save(self, path: str):
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


class DiffusionDataset(Dataset):
    """
    Dataset for training diffusion models.

    Loads images and their corresponding captions for text-to-image training.
    Supports both .txt caption files and metadata.json format.

    Args:
        image_paths: List of paths to images
        captions: List of captions (one per image)
        resolution: Target image resolution
        center_crop: Whether to center crop (else random crop)
        random_flip: Probability of horizontal flip
        tokenizer: Optional tokenizer for pre-tokenizing captions

    Example:
        >>> dataset = DiffusionDataset(
        ...     image_paths=["img1.jpg", "img2.jpg"],
        ...     captions=["a photo of a cat", "a photo of a dog"],
        ...     resolution=1024,
        ... )
        >>> image, caption = dataset[0]
    """

    def __init__(
        self,
        image_paths: List[str],
        captions: List[str],
        resolution: int = 1024,
        center_crop: bool = True,
        random_flip: float = 0.5,
        tokenizer: Optional[Callable] = None,
    ):
        self.image_paths = image_paths
        self.captions = captions
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.tokenizer = tokenizer

        assert len(image_paths) == len(captions), \
            f"Mismatch: {len(image_paths)} images, {len(captions)} captions"

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")

        # Resize and crop
        image = self._resize_and_crop(image)

        # Random horizontal flip
        if random.random() < self.random_flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # Convert to tensor and normalize to [-1, 1]
        image = np.array(image).astype(np.float32) / 127.5 - 1.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        # Get caption
        caption = self.captions[idx]

        result = {
            "pixel_values": image,
            "caption": caption,
        }

        # Pre-tokenize if tokenizer provided
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            result["input_ids"] = tokens.input_ids.squeeze(0)

        return result

    def _resize_and_crop(self, image: Image.Image) -> Image.Image:
        """Resize image to resolution, then crop to square."""
        w, h = image.size

        # Resize shortest side to resolution
        if w < h:
            new_w = self.resolution
            new_h = int(h * self.resolution / w)
        else:
            new_h = self.resolution
            new_w = int(w * self.resolution / h)

        image = image.resize((new_w, new_h), Image.LANCZOS)

        # Crop to square
        if self.center_crop:
            left = (new_w - self.resolution) // 2
            top = (new_h - self.resolution) // 2
        else:
            left = random.randint(0, new_w - self.resolution)
            top = random.randint(0, new_h - self.resolution)

        image = image.crop((left, top, left + self.resolution, top + self.resolution))
        return image


def prepare_dataset(
    image_dir: Union[str, Path],
    caption_ext: str = ".txt",
    resolution: int = 1024,
    center_crop: bool = True,
    random_flip: float = 0.5,
    tokenizer: Optional[Callable] = None,
    recursive: bool = False,
) -> DiffusionDataset:
    """
    Prepare a dataset from a directory of images with caption files.

    Expected structure:
        image_dir/
            image1.jpg
            image1.txt  # Contains caption for image1
            image2.png
            image2.txt
            ...

    Args:
        image_dir: Directory containing images and captions
        caption_ext: Extension for caption files (.txt, .caption, etc.)
        resolution: Target resolution
        center_crop: Whether to center crop
        random_flip: Probability of horizontal flip
        tokenizer: Optional tokenizer for pre-processing
        recursive: Whether to search subdirectories

    Returns:
        DiffusionDataset instance

    Example:
        >>> dataset = prepare_dataset(
        ...     "./my_training_images",
        ...     caption_ext=".txt",
        ...     resolution=1024,
        ... )
        >>> print(f"Found {len(dataset)} training images")
    """
    image_dir = Path(image_dir)

    # Find all images
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if recursive:
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(image_dir.rglob(f"*{ext}"))
    else:
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(image_dir.glob(f"*{ext}"))

    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_dir}")

    # Find corresponding captions
    valid_paths = []
    captions = []

    for img_path in image_paths:
        caption_path = img_path.with_suffix(caption_ext)

        if caption_path.exists():
            with open(caption_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
            valid_paths.append(str(img_path))
            captions.append(caption)
        else:
            warnings.warn(f"No caption file found for {img_path}")

    print(f"Found {len(valid_paths)} images with captions in {image_dir}")

    return DiffusionDataset(
        image_paths=valid_paths,
        captions=captions,
        resolution=resolution,
        center_crop=center_crop,
        random_flip=random_flip,
        tokenizer=tokenizer,
    )


def compute_snr_weights(
    timesteps: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    snr_gamma: float = 5.0,
) -> torch.Tensor:
    """
    Compute SNR-based loss weights.

    From "Scalable Diffusion Models with Transformers" paper.
    This weighting helps the model focus more on important timesteps.

    ELI5: Not all noise levels are equally important for learning.
    This gives more importance to the "medium noise" steps where
    the model can actually learn useful patterns.

    Args:
        timesteps: Current timesteps, shape (B,)
        alphas_cumprod: Cumulative alpha schedule
        snr_gamma: Weighting exponent (higher = more emphasis on low SNR)

    Returns:
        Loss weights, shape (B,)

    Example:
        >>> weights = compute_snr_weights(timesteps, scheduler.alphas_cumprod)
        >>> loss = (loss * weights).mean()
    """
    snr = alphas_cumprod[timesteps] / (1 - alphas_cumprod[timesteps])

    # Min-SNR weighting
    weights = torch.clamp(snr, max=snr_gamma) / snr_gamma

    return weights


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create a cosine learning rate schedule with linear warmup.

    This is the standard LR schedule for training diffusion models.

    Args:
        optimizer: The optimizer
        num_warmup_steps: Steps for linear warmup
        num_training_steps: Total training steps
        num_cycles: Cosine cycles (0.5 = half cycle, ends at 0)

    Returns:
        LambdaLR scheduler

    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> scheduler = get_cosine_schedule_with_warmup(
        ...     optimizer,
        ...     num_warmup_steps=500,
        ...     num_training_steps=10000,
        ... )
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMAModel:
    """
    Exponential Moving Average of model parameters.

    Using EMA during training often produces better final results
    by smoothing out training noise.

    ELI5: Instead of using the model exactly as it is right now,
    we keep a "smoothed" version that averages over recent updates.
    Like how a moving average smooths out stock prices.

    Args:
        model: The model to track
        decay: EMA decay factor (higher = slower update, 0.9999 is typical)

    Example:
        >>> ema = EMAModel(unet, decay=0.9999)
        >>> # During training
        >>> loss.backward()
        >>> optimizer.step()
        >>> ema.update()
        >>> # For inference
        >>> ema.apply_to(unet)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        """Update EMA parameters."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name] = (
                        self.decay * self.shadow[name] +
                        (1 - self.decay) * param.data
                    )

    def apply_to(self, model: nn.Module):
        """Apply EMA parameters to model (backup originals)."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        """Restore original parameters from backup."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def get_lora_config(
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> Dict:
    """
    Get LoRA configuration for PEFT.

    Args:
        rank: LoRA rank (higher = more capacity, more memory)
        alpha: LoRA alpha (typically 2x rank)
        dropout: Dropout for LoRA layers
        target_modules: Modules to apply LoRA to

    Returns:
        Configuration dict for LoraConfig

    Example:
        >>> config = get_lora_config(rank=32, alpha=64)
        >>> from peft import LoraConfig
        >>> lora_config = LoraConfig(**config)
    """
    if target_modules is None:
        # Default for Stable Diffusion U-Net
        target_modules = [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "add_k_proj",
            "add_v_proj",
        ]

    return {
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "target_modules": target_modules,
        "bias": "none",
    }


class SimpleTrainer:
    """
    A simple trainer for diffusion models.

    This is a minimal training loop for educational purposes.
    For production, use HuggingFace accelerate or similar.

    Args:
        model: The model to train
        optimizer: Optimizer instance
        scheduler: Noise scheduler
        config: Training configuration
        device: Training device

    Example:
        >>> trainer = SimpleTrainer(
        ...     model=unet,
        ...     optimizer=optimizer,
        ...     scheduler=noise_scheduler,
        ...     config=TrainingConfig(),
        ... )
        >>> trainer.train(dataloader, num_epochs=100)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,  # NoiseScheduler
        config: TrainingConfig,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.noise_scheduler = scheduler
        self.config = config
        self.device = device
        self.global_step = 0

        # EMA
        self.ema = EMAModel(model, decay=0.9999)

        # LR scheduler
        self.lr_scheduler = None  # Set in train()

        # Scaler for mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if config.mixed_precision != "no" else None

    def train_step(self, batch: Dict) -> float:
        """Perform a single training step."""
        self.model.train()

        # Get data
        images = batch["pixel_values"].to(self.device)
        if "input_ids" in batch:
            encoder_hidden_states = batch["input_ids"].to(self.device)
        else:
            encoder_hidden_states = None

        # Sample noise
        noise = torch.randn_like(images)

        # Sample timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.num_timesteps,
            (images.shape[0],), device=self.device
        )

        # Add noise
        noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps)

        # Mixed precision forward pass
        dtype = torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float32
        with torch.amp.autocast('cuda', dtype=dtype):
            # Model prediction
            if encoder_hidden_states is not None:
                noise_pred = self.model(noisy_images, timesteps, encoder_hidden_states)
            else:
                noise_pred = self.model(noisy_images, timesteps)

            # Loss
            loss = F.mse_loss(noise_pred, noise)

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            loss.backward()
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Update EMA
        self.ema.update(self.model)

        # Update LR scheduler
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.global_step += 1
        return loss.item()

    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int,
        log_fn: Optional[Callable] = None,
    ):
        """
        Run the training loop.

        Args:
            dataloader: Training data loader
            num_epochs: Number of epochs
            log_fn: Optional logging function(step, loss, lr)
        """
        # Setup LR scheduler
        num_training_steps = num_epochs * len(dataloader) // self.config.gradient_accumulation_steps
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            self.config.lr_warmup_steps,
            num_training_steps,
        )

        print(f"Starting training for {num_epochs} epochs ({num_training_steps} steps)")

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                loss = self.train_step(batch)
                epoch_loss += loss

                # Logging
                if self.global_step % self.config.log_every == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    if log_fn:
                        log_fn(self.global_step, loss, lr)
                    else:
                        print(f"Step {self.global_step}: loss={loss:.4f}, lr={lr:.6f}")

                # Save checkpoint
                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")

            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs}: avg_loss={avg_loss:.4f}")

        # Final save
        self.save_checkpoint("final")

    def save_checkpoint(self, name: str):
        """Save a training checkpoint."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "ema_shadow": self.ema.shadow,
            "global_step": self.global_step,
            "config": self.config.__dict__,
        }

        path = output_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")


def caption_images_with_blip(
    image_paths: List[str],
    device: str = "cuda",
    batch_size: int = 8,
) -> List[str]:
    """
    Generate captions for images using BLIP.

    Useful for preparing training data when you have images but no captions.

    Args:
        image_paths: List of paths to images
        device: Device for inference
        batch_size: Batch size for processing

    Returns:
        List of generated captions

    Example:
        >>> captions = caption_images_with_blip(
        ...     ["img1.jpg", "img2.jpg"],
        ...     device="cuda",
        ... )
        >>> # Save captions
        >>> for path, caption in zip(image_paths, captions):
        ...     with open(path.replace(".jpg", ".txt"), "w") as f:
        ...         f.write(caption)
    """
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
    except ImportError:
        raise ImportError("BLIP requires transformers. Install with: pip install transformers")

    # Load model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large",
        torch_dtype=torch.bfloat16,
    ).to(device)

    captions = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]

        inputs = processor(images, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50)

        batch_captions = processor.batch_decode(outputs, skip_special_tokens=True)
        captions.extend(batch_captions)

        print(f"Captioned {min(i + batch_size, len(image_paths))}/{len(image_paths)} images")

    return captions


if __name__ == "__main__":
    print("Training utilities module loaded successfully!")
    print("\nAvailable classes and functions:")
    print("  - TrainingConfig: Configuration dataclass")
    print("  - DiffusionDataset: Dataset for training")
    print("  - prepare_dataset: Create dataset from directory")
    print("  - compute_snr_weights: SNR-based loss weighting")
    print("  - get_cosine_schedule_with_warmup: LR scheduler")
    print("  - EMAModel: Exponential moving average")
    print("  - get_lora_config: LoRA configuration helper")
    print("  - SimpleTrainer: Basic training loop")
    print("  - caption_images_with_blip: Auto-caption images")

    # Demo config
    print("\nDemo: Creating training config...")
    config = TrainingConfig(
        output_dir="./lora_output",
        learning_rate=1e-4,
        num_epochs=100,
        lora_rank=16,
    )
    print(f"  Output dir: {config.output_dir}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  LoRA rank: {config.lora_rank}")

    print("\nTraining utilities ready for use!")
