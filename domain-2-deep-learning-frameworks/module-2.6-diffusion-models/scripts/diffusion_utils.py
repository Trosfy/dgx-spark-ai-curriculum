"""
Diffusion Model Utilities

Core utilities for implementing and working with diffusion models:
- Noise schedulers (linear, cosine, scaled linear)
- Forward diffusion (adding noise)
- Reverse diffusion helpers
- Timestep embeddings

Example Usage:
    >>> from scripts.diffusion_utils import NoiseScheduler, add_noise
    >>>
    >>> # Create a noise scheduler
    >>> scheduler = NoiseScheduler(num_timesteps=1000, schedule_type="cosine")
    >>>
    >>> # Add noise to an image at timestep t=500
    >>> noisy_image, noise = add_noise(clean_image, t=500, scheduler=scheduler)
    >>>
    >>> # Get timestep embedding for the model
    >>> t_emb = get_timestep_embedding(timesteps=torch.tensor([500]), dim=256)

DGX Spark Optimization:
    - All operations use bfloat16 by default for Blackwell compatibility
    - Memory-efficient implementations for 128GB unified memory
    - GPU-accelerated noise generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Literal, Union
from dataclasses import dataclass


@dataclass
class SchedulerOutput:
    """Output from a scheduler step."""
    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class NoiseScheduler:
    """
    Noise scheduler for diffusion models.

    Implements various noise schedules used in diffusion models:
    - Linear: Original DDPM schedule
    - Cosine: Improved schedule from "Improved DDPM"
    - Scaled Linear: SDXL-style schedule

    The scheduler manages:
    - Beta schedule (variance of noise added at each step)
    - Alpha schedule (cumulative product of 1-beta)
    - SNR (Signal-to-Noise Ratio) for each timestep

    ELI5: Think of this like a recipe for gradually adding static to a TV image.
    The schedule tells us exactly how much static to add at each step, so we can
    later reverse it perfectly.

    Example:
        >>> scheduler = NoiseScheduler(num_timesteps=1000, schedule_type="cosine")
        >>>
        >>> # Get noise level at timestep 500
        >>> alpha = scheduler.alphas_cumprod[500]  # ~0.5
        >>>
        >>> # Forward diffusion: add noise
        >>> noisy = scheduler.add_noise(clean_image, noise, timestep=500)
        >>>
        >>> # Reverse diffusion: remove noise (given model prediction)
        >>> less_noisy = scheduler.step(model_output, timestep=500, sample=noisy)

    Args:
        num_timesteps: Number of diffusion steps (typically 1000)
        schedule_type: Type of beta schedule ("linear", "cosine", "scaled_linear")
        beta_start: Starting beta value (for linear/scaled_linear)
        beta_end: Ending beta value (for linear/scaled_linear)
        device: Device to place tensors on
        dtype: Data type for tensors (default: bfloat16 for DGX Spark)
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule_type: Literal["linear", "cosine", "scaled_linear"] = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        self.device = device
        self.dtype = dtype

        # Compute beta schedule
        if schedule_type == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, device=device, dtype=torch.float32
            )
        elif schedule_type == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps).to(device)
        elif schedule_type == "scaled_linear":
            # SDXL uses scaled linear (square root scaling)
            self.betas = torch.linspace(
                beta_start**0.5, beta_end**0.5, num_timesteps, device=device, dtype=torch.float32
            ) ** 2
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        # Compute derived quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # For q(x_{t-1} | x_t, x_0) - the reverse process
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # Posterior variance for DDPM sampling
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

        # SNR for loss weighting
        self.snr = self.alphas_cumprod / (1.0 - self.alphas_cumprod)

    def _cosine_beta_schedule(self, num_timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine schedule as proposed in "Improved DDPM" paper.

        This schedule provides more gradual noise addition at the start and end,
        which improves image quality.
        """
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to samples according to the forward diffusion process.

        This implements: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise

        Args:
            original_samples: Clean images x_0, shape (B, C, H, W)
            noise: Gaussian noise, same shape as original_samples
            timesteps: Timesteps for each sample, shape (B,)

        Returns:
            Noisy samples x_t
        """
        # Get the right shape for broadcasting
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]

        # Reshape for broadcasting over image dimensions
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = (
            sqrt_alpha_prod.to(original_samples.dtype) * original_samples
            + sqrt_one_minus_alpha_prod.to(original_samples.dtype) * noise
        )

        return noisy_samples

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        predict_epsilon: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> SchedulerOutput:
        """
        Perform one step of the reverse diffusion process.

        Given the model's noise prediction (or x_0 prediction), compute x_{t-1}.

        Args:
            model_output: Model prediction (noise or x_0 depending on predict_epsilon)
            timestep: Current timestep t
            sample: Current noisy sample x_t
            predict_epsilon: If True, model predicts noise; if False, predicts x_0
            generator: Optional random generator for reproducibility

        Returns:
            SchedulerOutput with prev_sample (x_{t-1}) and pred_original_sample (x_0)
        """
        t = timestep

        # 1. Predict x_0 from model output
        if predict_epsilon:
            # Model predicts epsilon (noise)
            pred_original_sample = (
                sample - self.sqrt_one_minus_alphas_cumprod[t] * model_output
            ) / self.sqrt_alphas_cumprod[t]
        else:
            # Model directly predicts x_0
            pred_original_sample = model_output

        # 2. Clip predicted x_0 for stability
        pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)

        # 3. Compute posterior mean
        posterior_mean = (
            self.posterior_mean_coef1[t] * pred_original_sample
            + self.posterior_mean_coef2[t] * sample
        )

        # 4. Add noise (except for t=0)
        if t > 0:
            noise = torch.randn(sample.shape, device=sample.device, generator=generator)
            variance = torch.exp(0.5 * self.posterior_log_variance_clipped[t])
            prev_sample = posterior_mean + variance * noise
        else:
            prev_sample = posterior_mean

        return SchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=pred_original_sample,
        )

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute velocity target for v-prediction models.

        v = sqrt(alpha_cumprod) * noise - sqrt(1 - alpha_cumprod) * sample

        This is used by some models (e.g., Imagen) that predict velocity instead of noise.
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]

        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity


def add_noise(
    image: torch.Tensor,
    timestep: int,
    scheduler: NoiseScheduler,
    noise: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function to add noise to an image.

    Args:
        image: Clean image tensor, shape (B, C, H, W) or (C, H, W)
        timestep: Diffusion timestep (0 = clean, num_timesteps-1 = pure noise)
        scheduler: NoiseScheduler instance
        noise: Optional pre-generated noise (generated if not provided)

    Returns:
        Tuple of (noisy_image, noise)

    Example:
        >>> scheduler = NoiseScheduler(num_timesteps=1000)
        >>> noisy, noise = add_noise(clean_image, timestep=500, scheduler=scheduler)
    """
    # Handle single image (add batch dimension)
    squeeze = False
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
        squeeze = True

    # Generate noise if not provided
    if noise is None:
        noise = torch.randn_like(image)

    # Create timestep tensor
    timesteps = torch.tensor([timestep], device=image.device).expand(image.shape[0])

    # Add noise
    noisy_image = scheduler.add_noise(image, noise, timesteps)

    if squeeze:
        noisy_image = noisy_image.squeeze(0)
        noise = noise.squeeze(0)

    return noisy_image, noise


def denoise_step(
    model: nn.Module,
    noisy_image: torch.Tensor,
    timestep: int,
    scheduler: NoiseScheduler,
    class_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Perform one denoising step using a trained model.

    Args:
        model: Trained denoising model (predicts noise given noisy image and timestep)
        noisy_image: Current noisy image x_t
        timestep: Current timestep
        scheduler: NoiseScheduler instance
        class_labels: Optional class conditioning (for class-conditional models)

    Returns:
        Denoised image x_{t-1}

    Example:
        >>> # Single denoising step
        >>> x_t_minus_1 = denoise_step(model, x_t, t, scheduler)
        >>>
        >>> # Full denoising loop
        >>> x = torch.randn(1, 3, 64, 64)  # Pure noise
        >>> for t in reversed(range(1000)):
        ...     x = denoise_step(model, x, t, scheduler)
    """
    # Ensure batch dimension
    squeeze = False
    if len(noisy_image.shape) == 3:
        noisy_image = noisy_image.unsqueeze(0)
        squeeze = True

    # Get timestep embedding
    t_tensor = torch.tensor([timestep], device=noisy_image.device).expand(noisy_image.shape[0])

    # Model prediction
    with torch.no_grad():
        if class_labels is not None:
            noise_pred = model(noisy_image, t_tensor, class_labels)
        else:
            noise_pred = model(noisy_image, t_tensor)

    # Scheduler step
    output = scheduler.step(noise_pred, timestep, noisy_image)
    denoised = output.prev_sample

    if squeeze:
        denoised = denoised.squeeze(0)

    return denoised


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: int = 10000,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    This is the standard positional embedding used in transformers, adapted for
    timesteps in diffusion models. The embedding allows the model to understand
    "how noisy" the current image is.

    ELI5: Think of this like giving each timestep a unique "fingerprint" that the
    model can recognize. Just like how you can identify a song by its melody,
    the model identifies timesteps by their embedding pattern.

    Args:
        timesteps: 1D tensor of timesteps, shape (B,)
        embedding_dim: Dimension of the output embedding
        max_period: Maximum period for sinusoidal functions
        dtype: Output data type

    Returns:
        Timestep embeddings, shape (B, embedding_dim)

    Example:
        >>> t = torch.tensor([0, 500, 999])
        >>> emb = get_timestep_embedding(t, embedding_dim=256)
        >>> emb.shape
        torch.Size([3, 256])
    """
    half_dim = embedding_dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half_dim, device=timesteps.device) / half_dim
    )

    # Outer product: (B,) x (half_dim,) -> (B, half_dim)
    args = timesteps[:, None].float() * freqs[None, :]

    # Concatenate sin and cos
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    # Handle odd embedding_dim
    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))

    return embedding.to(dtype)


class SimpleDiffusionPipeline:
    """
    A minimal diffusion pipeline for educational purposes.

    This demonstrates the core generation loop without the complexity of
    production pipelines like diffusers.

    Example:
        >>> # Create pipeline with trained model
        >>> pipeline = SimpleDiffusionPipeline(model, scheduler)
        >>>
        >>> # Generate images
        >>> images = pipeline.generate(
        ...     num_images=4,
        ...     image_size=64,
        ...     num_channels=3,
        ...     num_inference_steps=50,
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        scheduler: NoiseScheduler,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.scheduler = scheduler
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        num_images: int = 1,
        image_size: int = 64,
        num_channels: int = 3,
        num_inference_steps: int = 50,
        class_labels: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate images from pure noise.

        Args:
            num_images: Number of images to generate
            image_size: Size of generated images (square)
            num_channels: Number of color channels (1 for grayscale, 3 for RGB)
            num_inference_steps: Number of denoising steps
            class_labels: Optional class conditioning
            generator: Random generator for reproducibility
            show_progress: Whether to show progress bar

        Returns:
            Generated images, shape (num_images, num_channels, image_size, image_size)
        """
        # Start from pure noise
        shape = (num_images, num_channels, image_size, image_size)
        images = torch.randn(shape, device=self.device, generator=generator)

        # Compute timestep schedule (evenly spaced)
        step_ratio = self.scheduler.num_timesteps // num_inference_steps
        timesteps = list(range(0, self.scheduler.num_timesteps, step_ratio))[::-1]

        # Denoising loop
        iterator = timesteps
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(timesteps, desc="Generating")
            except ImportError:
                pass

        self.model.eval()
        for t in iterator:
            # Predict noise
            t_batch = torch.tensor([t], device=self.device).expand(num_images)

            if class_labels is not None:
                noise_pred = self.model(images, t_batch, class_labels)
            else:
                noise_pred = self.model(images, t_batch)

            # Take denoising step
            output = self.scheduler.step(noise_pred, t, images)
            images = output.prev_sample

        # Clamp to valid range
        images = torch.clamp(images, -1.0, 1.0)

        return images


def compute_loss(
    model: nn.Module,
    clean_images: torch.Tensor,
    scheduler: NoiseScheduler,
    loss_type: Literal["mse", "l1", "huber"] = "mse",
    class_labels: Optional[torch.Tensor] = None,
    snr_weighting: bool = False,
) -> torch.Tensor:
    """
    Compute diffusion training loss.

    This implements the standard DDPM training objective:
    L = E[||epsilon - model(x_t, t)||^2]

    Args:
        model: Denoising model
        clean_images: Clean training images x_0
        scheduler: NoiseScheduler instance
        loss_type: Type of loss ("mse", "l1", or "huber")
        class_labels: Optional class conditioning
        snr_weighting: Whether to weight loss by SNR (improves quality)

    Returns:
        Scalar loss value

    Example:
        >>> loss = compute_loss(model, batch_images, scheduler)
        >>> loss.backward()
    """
    batch_size = clean_images.shape[0]
    device = clean_images.device

    # Sample random timesteps
    timesteps = torch.randint(
        0, scheduler.num_timesteps, (batch_size,), device=device
    )

    # Sample noise
    noise = torch.randn_like(clean_images)

    # Add noise to images
    noisy_images = scheduler.add_noise(clean_images, noise, timesteps)

    # Predict noise
    if class_labels is not None:
        noise_pred = model(noisy_images, timesteps, class_labels)
    else:
        noise_pred = model(noisy_images, timesteps)

    # Compute loss
    if loss_type == "mse":
        loss = F.mse_loss(noise_pred, noise, reduction="none")
    elif loss_type == "l1":
        loss = F.l1_loss(noise_pred, noise, reduction="none")
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise_pred, noise, reduction="none")
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Average over spatial dimensions
    loss = loss.mean(dim=list(range(1, len(loss.shape))))

    # SNR weighting (from "Scalable Diffusion Models with Transformers")
    if snr_weighting:
        snr = scheduler.snr[timesteps]
        weights = snr / (snr + 1)
        loss = loss * weights

    return loss.mean()


if __name__ == "__main__":
    # Demo: Create scheduler and visualize noise levels
    print("Creating noise scheduler...")
    scheduler = NoiseScheduler(num_timesteps=1000, schedule_type="cosine")

    # Show alpha values at different timesteps
    print("\nAlpha (signal preserved) at different timesteps:")
    for t in [0, 100, 250, 500, 750, 900, 999]:
        alpha = scheduler.alphas_cumprod[t].item()
        snr = scheduler.snr[t].item()
        print(f"  t={t:4d}: alpha={alpha:.4f}, SNR={snr:.4f}")

    # Demo: Add noise to a dummy image
    print("\nDemo: Adding noise to an image...")
    dummy_image = torch.randn(1, 3, 64, 64, device="cuda")

    for t in [0, 250, 500, 750, 999]:
        noisy, noise = add_noise(dummy_image, t, scheduler)
        noise_level = torch.std(noisy - dummy_image).item()
        print(f"  t={t:4d}: noise_std={noise_level:.4f}")

    print("\nDiffusion utilities ready!")
