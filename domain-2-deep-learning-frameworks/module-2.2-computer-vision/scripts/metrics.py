"""
Evaluation Metrics for Computer Vision Module

This module provides evaluation metrics for computer vision tasks on DGX Spark.

Metrics included:
- Classification: accuracy, precision, recall, F1-score
- Segmentation: IoU, mIoU, Dice coefficient, pixel accuracy
- Detection: mAP, precision-recall curves

Example usage:
    from metrics import compute_iou, ClassificationMetrics

    # Segmentation evaluation
    iou = compute_iou(predictions, targets, num_classes=21)
    print(f"mIoU: {iou['miou']:.2%}")

    # Classification evaluation
    metrics = ClassificationMetrics(num_classes=10)
    metrics.update(predictions, targets)
    results = metrics.compute()
"""

__all__ = [
    'ClassificationMetrics',
    'SegmentationMetrics',
    'compute_iou',
    'compute_dice',
    'compute_pixel_accuracy',
    'compute_detection_metrics',
    'box_iou',
]

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class ClassificationMetrics:
    """
    Compute classification metrics.

    Tracks predictions and computes accuracy, precision, recall, F1.

    Example:
        >>> metrics = ClassificationMetrics(num_classes=10)
        >>> for batch in loader:
        ...     preds = model(batch)
        ...     metrics.update(preds, labels)
        >>> results = metrics.compute()
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset all counters."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update confusion matrix with new predictions.

        Args:
            predictions: Predicted class indices [B] or logits [B, C]
            targets: Ground truth class indices [B]
        """
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=1)

        preds = predictions.cpu().numpy()
        targs = targets.cpu().numpy()

        for p, t in zip(preds, targs):
            self.confusion_matrix[t, p] += 1

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.

        Returns:
            Dictionary with accuracy, per-class precision/recall/f1, and macro averages
        """
        # Accuracy
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        accuracy = correct / total if total > 0 else 0

        # Per-class metrics
        precision = []
        recall = []
        f1 = []

        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp

            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0

            precision.append(p)
            recall.append(r)
            f1.append(f)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'macro_precision': np.mean(precision),
            'macro_recall': np.mean(recall),
            'macro_f1': np.mean(f1),
            'confusion_matrix': self.confusion_matrix.copy()
        }


def compute_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 21,
    ignore_index: int = 255
) -> Dict[str, float]:
    """
    Compute Intersection over Union (IoU) for segmentation.

    IoU = TP / (TP + FP + FN)

    Args:
        predictions: Predicted class indices [B, H, W] or [H, W]
        targets: Ground truth class indices [B, H, W] or [H, W]
        num_classes: Number of classes
        ignore_index: Index to ignore (e.g., boundary pixels)

    Returns:
        Dictionary with per-class IoU and mean IoU

    Example:
        >>> preds = model(images).argmax(dim=1)
        >>> metrics = compute_iou(preds, masks, num_classes=21)
        >>> print(f"mIoU: {metrics['miou']:.2%}")
    """
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)
        targets = targets.unsqueeze(0)

    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    # Ignore specified index
    valid_mask = targets != ignore_index

    ious = []
    class_ious = []

    for cls in range(num_classes):
        pred_cls = (predictions == cls) & valid_mask
        target_cls = (targets == cls) & valid_mask

        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()

        if union > 0:
            iou = intersection / union
            ious.append(iou)
            class_ious.append(iou)
        else:
            class_ious.append(float('nan'))

    miou = np.mean(ious) if ious else 0.0

    return {
        'per_class_iou': class_ious,
        'miou': miou,
        'num_valid_classes': len(ious)
    }


def compute_dice(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 21,
    smooth: float = 1.0
) -> Dict[str, float]:
    """
    Compute Dice coefficient for segmentation.

    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Args:
        predictions: Predicted class indices [B, H, W]
        targets: Ground truth class indices [B, H, W]
        num_classes: Number of classes
        smooth: Smoothing factor for numerical stability

    Returns:
        Dictionary with per-class Dice and mean Dice
    """
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)
        targets = targets.unsqueeze(0)

    # One-hot encode
    pred_one_hot = torch.nn.functional.one_hot(predictions.long(), num_classes)
    target_one_hot = torch.nn.functional.one_hot(targets.long(), num_classes)

    # Reshape to [B, C, H*W]
    pred_flat = pred_one_hot.permute(0, 3, 1, 2).flatten(2).float()
    target_flat = target_one_hot.permute(0, 3, 1, 2).flatten(2).float()

    # Compute Dice per class
    intersection = (pred_flat * target_flat).sum(dim=2).sum(dim=0)
    pred_sum = pred_flat.sum(dim=2).sum(dim=0)
    target_sum = target_flat.sum(dim=2).sum(dim=0)

    dice_per_class = (2 * intersection + smooth) / (pred_sum + target_sum + smooth)

    return {
        'per_class_dice': dice_per_class.tolist(),
        'mean_dice': dice_per_class.mean().item()
    }


def compute_pixel_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 255
) -> float:
    """
    Compute pixel accuracy for segmentation.

    Args:
        predictions: Predicted class indices [B, H, W]
        targets: Ground truth class indices [B, H, W]
        ignore_index: Index to ignore

    Returns:
        Pixel accuracy as float
    """
    valid_mask = targets != ignore_index
    correct = (predictions == targets) & valid_mask
    return correct.sum().item() / valid_mask.sum().item()


def compute_detection_metrics(
    pred_boxes: List[np.ndarray],
    pred_labels: List[np.ndarray],
    pred_scores: List[np.ndarray],
    gt_boxes: List[np.ndarray],
    gt_labels: List[np.ndarray],
    num_classes: int,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute detection metrics (precision, recall, AP).

    Args:
        pred_boxes: List of predicted boxes per image [N, 4]
        pred_labels: List of predicted labels per image [N]
        pred_scores: List of confidence scores per image [N]
        gt_boxes: List of ground truth boxes per image [M, 4]
        gt_labels: List of ground truth labels per image [M]
        num_classes: Number of classes
        iou_threshold: IoU threshold for matching

    Returns:
        Dictionary with per-class AP and mAP
    """
    # Collect all detections and ground truths by class
    all_detections = defaultdict(list)
    all_ground_truths = defaultdict(list)

    for img_idx, (pboxes, plabels, pscores, gboxes, glabels) in enumerate(
        zip(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels)
    ):
        # Add detections
        for box, label, score in zip(pboxes, plabels, pscores):
            all_detections[label].append({
                'image_id': img_idx,
                'box': box,
                'score': score
            })

        # Add ground truths
        for box, label in zip(gboxes, glabels):
            all_ground_truths[label].append({
                'image_id': img_idx,
                'box': box,
                'matched': False
            })

    # Compute AP per class
    aps = []
    class_aps = {}

    for cls in range(num_classes):
        dets = all_detections[cls]
        gts = all_ground_truths[cls]

        if len(gts) == 0:
            continue

        # Sort detections by score
        dets = sorted(dets, key=lambda x: x['score'], reverse=True)

        tp = np.zeros(len(dets))
        fp = np.zeros(len(dets))

        # Reset matched flags
        for gt in gts:
            gt['matched'] = False

        for det_idx, det in enumerate(dets):
            # Find best matching ground truth
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gts):
                if gt['image_id'] != det['image_id'] or gt['matched']:
                    continue

                iou = box_iou(det['box'], gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                tp[det_idx] = 1
                gts[best_gt_idx]['matched'] = True
            else:
                fp[det_idx] = 1

        # Compute precision-recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / len(gts)

        # Compute AP using 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            p = precision[recall >= t].max() if (recall >= t).any() else 0
            ap += p / 11

        aps.append(ap)
        class_aps[cls] = ap

    return {
        'per_class_ap': class_aps,
        'mAP': np.mean(aps) if aps else 0.0
    }


def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes.

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


class SegmentationMetrics:
    """
    Accumulate and compute segmentation metrics over batches.

    Example:
        >>> metrics = SegmentationMetrics(num_classes=21)
        >>> for batch in loader:
        ...     preds = model(batch).argmax(dim=1)
        ...     metrics.update(preds, labels)
        >>> results = metrics.compute()
    """

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """Reset accumulators."""
        self.intersection = np.zeros(self.num_classes)
        self.union = np.zeros(self.num_classes)
        self.correct = 0
        self.total = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with new batch.

        Args:
            predictions: Predicted class indices [B, H, W]
            targets: Ground truth class indices [B, H, W]
        """
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()

        valid_mask = targets != self.ignore_index

        for cls in range(self.num_classes):
            pred_cls = (predictions == cls) & valid_mask
            target_cls = (targets == cls) & valid_mask

            self.intersection[cls] += (pred_cls & target_cls).sum()
            self.union[cls] += (pred_cls | target_cls).sum()

        self.correct += ((predictions == targets) & valid_mask).sum()
        self.total += valid_mask.sum()

    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.

        Returns:
            Dictionary with mIoU, per-class IoU, pixel accuracy
        """
        ious = []
        per_class_iou = []

        for cls in range(self.num_classes):
            if self.union[cls] > 0:
                iou = self.intersection[cls] / self.union[cls]
                ious.append(iou)
                per_class_iou.append(iou)
            else:
                per_class_iou.append(float('nan'))

        return {
            'miou': np.mean(ious) if ious else 0.0,
            'per_class_iou': per_class_iou,
            'pixel_accuracy': self.correct / self.total if self.total > 0 else 0.0
        }


if __name__ == "__main__":
    print("Metrics module loaded successfully!")

    # Test classification metrics
    metrics = ClassificationMetrics(num_classes=10)
    preds = torch.randint(0, 10, (100,))
    targets = torch.randint(0, 10, (100,))
    metrics.update(preds, targets)
    results = metrics.compute()
    print(f"Classification accuracy: {results['accuracy']:.2%}")

    # Test segmentation metrics
    preds = torch.randint(0, 21, (2, 256, 256))
    targets = torch.randint(0, 21, (2, 256, 256))
    iou_results = compute_iou(preds, targets, num_classes=21)
    print(f"Segmentation mIoU: {iou_results['miou']:.2%}")
