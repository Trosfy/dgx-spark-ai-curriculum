"""
Visualization Utilities for Computer Vision Module

This module provides visualization helpers for images, training metrics,
and model outputs optimized for DGX Spark workflows.

Functions included:
- plot_training_history: Loss/accuracy curves
- visualize_predictions: Model predictions with labels
- plot_confusion_matrix: Classification analysis
- visualize_feature_maps: CNN layer activations
- visualize_segmentation: Semantic segmentation results
- visualize_detections: Object detection boxes

Example usage:
    from visualization_utils import plot_training_history, visualize_predictions

    plot_training_history(history)
    visualize_predictions(model, test_loader, class_names)
"""

__all__ = [
    # Constants
    'CIFAR10_CLASSES',
    'VOC_CLASSES',
    'VOC_COLORMAP',
    'IMAGENET_MEAN',
    'IMAGENET_STD',
    'CIFAR10_MEAN',
    'CIFAR10_STD',
    # Functions
    'denormalize',
    'plot_training_history',
    'visualize_predictions',
    'plot_confusion_matrix',
    'visualize_feature_maps',
    'visualize_segmentation',
    'visualize_detections',
    'compare_models',
]

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path


# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# VOC segmentation class names
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# VOC color palette
VOC_COLORMAP = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
    [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128]
], dtype=np.uint8)

# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# CIFAR-10 normalization (computed from the training set)
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD = np.array([0.2023, 0.1994, 0.2010])


def denormalize(
    tensor: torch.Tensor,
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD
) -> np.ndarray:
    """
    Denormalize a tensor for visualization.

    Args:
        tensor: Normalized image tensor [C, H, W] or [B, C, H, W]
        mean: Normalization mean
        std: Normalization std

    Returns:
        Denormalized numpy array
    """
    if tensor.dim() == 4:
        # Batch dimension
        tensor = tensor[0]

    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)

    tensor = tensor.cpu() * std + mean
    tensor = tensor.permute(1, 2, 0).numpy()
    return tensor.clip(0, 1)


def plot_training_history(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> None:
    """
    Plot training history (loss and accuracy).

    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        figsize: Figure size
        save_path: Optional path to save the figure

    Example:
        >>> history = trainer.fit(epochs=10)
        >>> plot_training_history(history)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train', linewidth=2)
    if 'val_acc' in history:
        axes[1].plot(history['val_acc'], label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def visualize_predictions(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    class_names: List[str] = CIFAR10_CLASSES,
    num_images: int = 16,
    device: str = 'cuda',
    figsize: Tuple[int, int] = (15, 12)
) -> None:
    """
    Visualize model predictions on a batch of images.

    Args:
        model: Trained PyTorch model
        data_loader: Data loader with test images
        class_names: List of class names
        num_images: Number of images to show
        device: Device to run inference on
        figsize: Figure size
    """
    model.eval()
    model = model.to(device)

    images, labels = next(iter(data_loader))
    images = images[:num_images]
    labels = labels[:num_images]

    with torch.no_grad():
        outputs = model(images.to(device))
        _, predictions = outputs.max(1)
        predictions = predictions.cpu()

    rows = int(np.ceil(np.sqrt(num_images)))
    cols = int(np.ceil(num_images / rows))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx >= num_images:
            ax.axis('off')
            continue

        img = denormalize(images[idx])
        pred = predictions[idx].item()
        true = labels[idx].item()

        ax.imshow(img)
        color = 'green' if pred == true else 'red'
        ax.set_title(
            f'Pred: {class_names[pred]}\nTrue: {class_names[true]}',
            color=color,
            fontsize=10
        )
        ax.axis('off')

    plt.suptitle('Model Predictions', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    class_names: List[str] = CIFAR10_CLASSES,
    device: str = 'cuda',
    figsize: Tuple[int, int] = (10, 8),
    normalize: bool = True
) -> None:
    """
    Plot confusion matrix for model predictions.

    Args:
        model: Trained PyTorch model
        data_loader: Data loader with test images
        class_names: List of class names
        device: Device to run inference on
        figsize: Figure size
        normalize: Normalize confusion matrix rows
    """
    model.eval()
    model = model.to(device)

    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images.to(device))
            _, predictions = outputs.max(1)

            for pred, true in zip(predictions.cpu(), labels):
                confusion[true][pred] += 1

    if normalize:
        confusion = confusion.astype(np.float32)
        confusion = confusion / confusion.sum(axis=1, keepdims=True)

    plt.figure(figsize=figsize)
    plt.imshow(confusion, cmap='Blues')
    plt.colorbar()

    plt.xticks(range(num_classes), class_names, rotation=45, ha='right')
    plt.yticks(range(num_classes), class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Add text annotations
    thresh = confusion.max() / 2
    for i in range(num_classes):
        for j in range(num_classes):
            val = confusion[i, j]
            text = f'{val:.2f}' if normalize else f'{val}'
            plt.text(j, i, text, ha='center', va='center',
                    color='white' if val > thresh else 'black', fontsize=8)

    plt.tight_layout()
    plt.show()


def visualize_feature_maps(
    model: nn.Module,
    image: torch.Tensor,
    layer_name: str,
    num_features: int = 16,
    figsize: Tuple[int, int] = (15, 12)
) -> None:
    """
    Visualize feature maps from a specific layer.

    Args:
        model: PyTorch model
        image: Input image tensor [1, C, H, W]
        layer_name: Name of the layer to visualize
        num_features: Number of feature maps to show
        figsize: Figure size

    Example:
        >>> visualize_feature_maps(model, image, 'conv1')
    """
    activation = {}

    def hook_fn(name):
        def hook(module, input, output):
            activation[name] = output.detach()
        return hook

    # Register hook
    for name, module in model.named_modules():
        if name == layer_name:
            module.register_forward_hook(hook_fn(name))
            break

    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(image)

    if layer_name not in activation:
        print(f"Layer '{layer_name}' not found. Available layers:")
        for name, _ in model.named_modules():
            print(f"  - {name}")
        return

    features = activation[layer_name][0]  # [C, H, W]
    num_features = min(num_features, features.shape[0])

    rows = int(np.ceil(np.sqrt(num_features)))
    cols = int(np.ceil(num_features / rows))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx >= num_features:
            ax.axis('off')
            continue

        feature_map = features[idx].cpu().numpy()
        ax.imshow(feature_map, cmap='viridis')
        ax.set_title(f'Filter {idx}', fontsize=9)
        ax.axis('off')

    plt.suptitle(f'Feature Maps from "{layer_name}"', fontsize=14)
    plt.tight_layout()
    plt.show()


def visualize_segmentation(
    image: torch.Tensor,
    mask: torch.Tensor,
    prediction: Optional[torch.Tensor] = None,
    colormap: np.ndarray = VOC_COLORMAP,
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """
    Visualize segmentation results.

    Args:
        image: Input image tensor [C, H, W]
        mask: Ground truth mask [H, W]
        prediction: Optional predicted mask [H, W]
        colormap: Color map for classes
        figsize: Figure size
    """
    # Denormalize image
    img = denormalize(image)

    # Convert masks to colored images
    mask_np = mask.cpu().numpy()
    mask_colored = colormap[mask_np.clip(0, len(colormap)-1)] / 255.0

    if prediction is not None:
        fig, axes = plt.subplots(1, 4, figsize=figsize)

        pred_np = prediction.cpu().numpy()
        pred_colored = colormap[pred_np.clip(0, len(colormap)-1)] / 255.0

        axes[0].imshow(img)
        axes[0].set_title('Input')
        axes[0].axis('off')

        axes[1].imshow(mask_colored)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        axes[2].imshow(pred_colored)
        axes[2].set_title('Prediction')
        axes[2].axis('off')

        # Overlay
        axes[3].imshow(img)
        axes[3].imshow(pred_colored, alpha=0.5)
        axes[3].set_title('Overlay')
        axes[3].axis('off')
    else:
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        axes[0].imshow(img)
        axes[0].set_title('Input')
        axes[0].axis('off')

        axes[1].imshow(mask_colored)
        axes[1].set_title('Mask')
        axes[1].axis('off')

        axes[2].imshow(img)
        axes[2].imshow(mask_colored, alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    class_names: List[str],
    score_threshold: float = 0.5,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Visualize object detection results.

    Args:
        image: Input image [H, W, C] as numpy array
        boxes: Detection boxes [N, 4] as [x1, y1, x2, y2]
        labels: Class labels [N]
        scores: Confidence scores [N]
        class_names: List of class names
        score_threshold: Minimum score to display
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.imshow(image)
    ax = plt.gca()

    colors = plt.cm.hsv(np.linspace(0, 1, len(class_names)))

    for box, label, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue

        x1, y1, x2, y2 = box
        color = colors[label]

        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor=color, linewidth=2
        )
        ax.add_patch(rect)

        text = f'{class_names[label]}: {score:.2f}'
        ax.text(x1, y1 - 5, text, fontsize=10,
               color='white', backgroundcolor=color[:3])

    plt.axis('off')
    plt.title(f'Detections (threshold={score_threshold})')
    plt.tight_layout()
    plt.show()


def compare_models(
    histories: Dict[str, Dict[str, List[float]]],
    metric: str = 'val_acc',
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Compare training curves from multiple models.

    Args:
        histories: Dictionary mapping model names to their histories
        metric: Metric to compare ('val_acc', 'val_loss', 'train_acc', 'train_loss')
        figsize: Figure size

    Example:
        >>> histories = {
        ...     'ResNet': resnet_history,
        ...     'VGG': vgg_history,
        ... }
        >>> compare_models(histories, metric='val_acc')
    """
    plt.figure(figsize=figsize)

    for name, history in histories.items():
        if metric in history:
            plt.plot(history[metric], label=name, linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'Model Comparison: {metric}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Visualization utilities loaded successfully!")
    print(f"Available functions: plot_training_history, visualize_predictions, ")
    print(f"                     plot_confusion_matrix, visualize_feature_maps, ")
    print(f"                     visualize_segmentation, visualize_detections")
