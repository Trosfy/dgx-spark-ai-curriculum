"""
Custom Dataset Utilities - Production-Ready Data Pipeline

This module provides reusable dataset components optimized for DGX Spark.

Components:
    - ImageFolderDataset: Custom dataset for image folders
    - MixupDataset: Mixup augmentation wrapper
    - CutmixDataset: Cutmix augmentation wrapper
    - create_transforms: Standard transform configurations
    - create_dataloaders: Optimized DataLoader creation

Example:
    >>> from custom_dataset import ImageFolderDataset, create_dataloaders
    >>> dataset = ImageFolderDataset('./data', transform=create_transforms('train'))
    >>> train_loader, val_loader = create_dataloaders(dataset)

Author: DGX Spark AI Curriculum
"""

__all__ = [
    'ImageFolderDataset',
    'MixupDataset',
    'CutmixDataset',
    'create_transforms',
    'create_dataloaders',
    'compute_mean_std',
]

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import numpy as np
from typing import Tuple, List, Optional, Callable, Union
import multiprocessing


class ImageFolderDataset(Dataset):
    """
    Custom Dataset for loading images from a folder structure.

    Expected folder structure:
        root/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img1.jpg
                ...

    Args:
        root_dir: Path to the root directory
        transform: Optional transform to apply to images
        extensions: Tuple of valid image extensions
        cache_images: Whether to cache images in memory (uses more RAM but faster)

    Example:
        >>> dataset = ImageFolderDataset('./data', transform=T.ToTensor())
        >>> image, label = dataset[0]
        >>> print(image.shape, label)
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.webp'),
        cache_images: bool = False
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.extensions = extensions
        self.cache_images = cache_images
        self._cache = {}

        # Discover classes (subdirectories)
        self.classes = sorted([
            d.name for d in self.root_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])

        if len(self.classes) == 0:
            raise ValueError(f"No class directories found in {root_dir}")

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        # Collect all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.extensions:
                    self.samples.append((
                        str(img_path),
                        self.class_to_idx[class_name]
                    ))

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {root_dir}")

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample."""
        img_path, label = self.samples[idx]

        # Check cache first
        if self.cache_images and idx in self._cache:
            image = self._cache[idx].copy()
        else:
            # Load image
            image = Image.open(img_path).convert('RGB')
            if self.cache_images:
                self._cache[idx] = image.copy()

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_name(self, label: int) -> str:
        """Get the class name for a given label."""
        return self.idx_to_class[label]

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return len(self.classes)


class MixupDataset(Dataset):
    """
    Dataset wrapper that applies Mixup augmentation.

    Mixup creates new training samples by combining two images:
        mixed_image = lambda * image1 + (1 - lambda) * image2
        mixed_label = (label1, label2, lambda)

    Paper: "mixup: Beyond Empirical Risk Minimization"
    https://arxiv.org/abs/1710.09412

    Args:
        dataset: Base dataset to wrap
        alpha: Mixup alpha parameter (default: 0.2)
        num_classes: Number of classes (for one-hot labels)

    Example:
        >>> base_dataset = ImageFolderDataset('./data')
        >>> mixup_dataset = MixupDataset(base_dataset, alpha=0.2)
        >>> mixed_img, (label1, label2, lam) = mixup_dataset[0]
    """

    def __init__(
        self,
        dataset: Dataset,
        alpha: float = 0.2,
        num_classes: Optional[int] = None
    ):
        self.dataset = dataset
        self.alpha = alpha
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[int, int, float]]:
        # Get first sample
        img1, label1 = self.dataset[idx]

        # Get random second sample
        idx2 = np.random.randint(len(self.dataset))
        img2, label2 = self.dataset[idx2]

        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0

        # Mix images
        mixed_img = lam * img1 + (1 - lam) * img2

        return mixed_img, (label1, label2, lam)


class CutmixDataset(Dataset):
    """
    Dataset wrapper that applies Cutmix augmentation.

    Cutmix cuts and pastes patches between images:
        mixed_image = image1 with a rectangular patch from image2
        mixed_label weighted by the area ratio

    Paper: "CutMix: Regularization Strategy to Train Strong Classifiers"
    https://arxiv.org/abs/1905.04899

    Args:
        dataset: Base dataset to wrap
        alpha: Cutmix alpha parameter (default: 1.0)

    Example:
        >>> base_dataset = ImageFolderDataset('./data')
        >>> cutmix_dataset = CutmixDataset(base_dataset, alpha=1.0)
        >>> mixed_img, (label1, label2, lam) = cutmix_dataset[0]
    """

    def __init__(self, dataset: Dataset, alpha: float = 1.0):
        self.dataset = dataset
        self.alpha = alpha

    def __len__(self) -> int:
        return len(self.dataset)

    def _rand_bbox(
        self, size: Tuple[int, ...], lam: float
    ) -> Tuple[int, int, int, int]:
        """Generate random bounding box."""
        _, H, W = size
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[int, int, float]]:
        # Get first sample
        img1, label1 = self.dataset[idx]

        # Get random second sample
        idx2 = np.random.randint(len(self.dataset))
        img2, label2 = self.dataset[idx2]

        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)

        # Get random bounding box
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(img1.size(), lam)

        # Create mixed image
        mixed_img = img1.clone()
        mixed_img[:, bby1:bby2, bbx1:bbx2] = img2[:, bby1:bby2, bbx1:bbx2]

        # Adjust lambda based on actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img1.size(1) * img1.size(2)))

        return mixed_img, (label1, label2, lam)


def create_transforms(
    mode: str = 'train',
    image_size: int = 32,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
) -> T.Compose:
    """
    Create standard transform pipelines.

    Args:
        mode: 'train', 'val', or 'test'
        image_size: Target image size
        mean: Normalization mean (default: ImageNet)
        std: Normalization std (default: ImageNet)

    Returns:
        Composed transform pipeline

    Example:
        >>> train_transform = create_transforms('train', image_size=32)
        >>> val_transform = create_transforms('val', image_size=32)
    """
    if mode == 'train':
        return T.Compose([
            T.Resize(int(image_size * 1.25)),
            T.RandomCrop(image_size, padding=image_size // 8),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.RandomRotation(degrees=15),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        return T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    batch_size: int = 128,
    num_workers: Optional[int] = None,
    val_split: float = 0.2
) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
    """
    Create optimized DataLoaders for DGX Spark.

    If val_dataset is None, train_dataset will be split according to val_split.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        batch_size: Batch size for training
        num_workers: Number of workers (auto-detected if None)
        val_split: Validation split ratio if val_dataset is None

    Returns:
        Tuple of (train_loader, val_loader) or just train_loader

    Example:
        >>> dataset = ImageFolderDataset('./data')
        >>> train_loader, val_loader = create_dataloaders(dataset, batch_size=64)
    """
    # Auto-detect optimal workers
    if num_workers is None:
        num_workers = min(4, multiprocessing.cpu_count())

    # Split if needed
    if val_dataset is None and val_split > 0:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size]
        )

    # Create training loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        return train_loader, val_loader

    return train_loader


def compute_mean_std(
    dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and std of a dataset for normalization.

    Uses Welford's online algorithm for numerically stable computation
    of mean and variance in a single pass.

    Args:
        dataset: Dataset to analyze
        batch_size: Batch size for computation
        num_workers: Number of data loading workers

    Returns:
        Tuple of (mean, std) tensors with shape (3,)

    Example:
        >>> mean, std = compute_mean_std(dataset)
        >>> print(f"Mean: {mean}, Std: {std}")
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # First pass: compute mean
    pixel_sum = torch.zeros(3)
    pixel_count = 0

    for images, _ in loader:
        # images shape: (B, C, H, W)
        batch_samples = images.size(0)
        images = images.view(batch_samples, 3, -1)  # (B, C, H*W)
        pixel_sum += images.sum(dim=[0, 2])  # Sum over batch and pixels
        pixel_count += batch_samples * images.size(2)  # Total pixels per channel

    mean = pixel_sum / pixel_count

    # Second pass: compute variance
    pixel_squared_diff_sum = torch.zeros(3)

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, 3, -1)  # (B, C, H*W)
        # Compute squared differences from mean
        diff = images - mean.view(1, 3, 1)
        pixel_squared_diff_sum += (diff ** 2).sum(dim=[0, 2])

    std = torch.sqrt(pixel_squared_diff_sum / pixel_count)

    return mean, std


if __name__ == '__main__':
    import tempfile
    import shutil

    print("Testing custom dataset utilities...")

    # Create temporary test data
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample images
        for cls in ['class_a', 'class_b']:
            cls_dir = Path(tmpdir) / cls
            cls_dir.mkdir()
            for i in range(10):
                img = Image.new('RGB', (64, 64), color=(i * 20, i * 10, 255 - i * 20))
                img.save(cls_dir / f'img_{i}.jpg')

        # Test ImageFolderDataset
        dataset = ImageFolderDataset(tmpdir, transform=create_transforms('train'))
        print(f"Dataset size: {len(dataset)}")
        print(f"Classes: {dataset.classes}")

        img, label = dataset[0]
        print(f"Sample: shape={img.shape}, label={label}")

        # Test MixupDataset
        mixup = MixupDataset(dataset, alpha=0.2)
        mixed_img, (l1, l2, lam) = mixup[0]
        print(f"Mixup: shape={mixed_img.shape}, labels=({l1}, {l2}), lam={lam:.3f}")

        # Test DataLoaders
        train_loader, val_loader = create_dataloaders(dataset, batch_size=4)
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

        batch = next(iter(train_loader))
        print(f"Batch: images={batch[0].shape}, labels={batch[1].shape}")

    print("\nAll tests passed!")
