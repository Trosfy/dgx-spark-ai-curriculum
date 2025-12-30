"""
CNN Architectures for Computer Vision Module

This module contains implementations of classic and modern CNN architectures
optimized for DGX Spark.

Architectures included:
- LeNet-5 (1998)
- AlexNet (2012)
- VGG-11 (2014)
- ResNet-18 (2015)
- U-Net for Segmentation

Example usage:
    from cnn_architectures import LeNet5, AlexNet, VGG11, ResNet18, UNet

    model = ResNet18(num_classes=10)
    output = model(torch.randn(1, 3, 32, 32))
"""

__all__ = [
    'LeNet5',
    'AlexNet',
    'VGG11',
    'BasicBlock',
    'ResNet18',
    'DoubleConv',
    'Down',
    'Up',
    'UNet',
    'get_model',
    'count_parameters',
]

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class LeNet5(nn.Module):
    """
    LeNet-5 implementation for CIFAR-10 (adapted for 3-channel input).

    Original paper: "Gradient-Based Learning Applied to Document Recognition"
    by Yann LeCun et al., 1998

    Args:
        num_classes: Number of output classes (default: 10)

    Example:
        >>> model = LeNet5(num_classes=10)
        >>> x = torch.randn(1, 3, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([1, 10])
    """

    def __init__(self, num_classes: int = 10):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AlexNet(nn.Module):
    """
    AlexNet implementation adapted for 32x32 CIFAR-10 images.

    Original paper: "ImageNet Classification with Deep Convolutional Neural Networks"
    by Alex Krizhevsky et al., 2012

    Key features:
    - ReLU activation
    - Dropout regularization
    - Batch normalization (modern addition)

    Args:
        num_classes: Number of output classes (default: 10)
    """

    def __init__(self, num_classes: int = 10):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG11(nn.Module):
    """
    VGG-11 implementation (Configuration A).

    Original paper: "Very Deep Convolutional Networks for Large-Scale Image Recognition"
    by Karen Simonyan and Andrew Zisserman, 2014

    Key features:
    - Uniform 3x3 convolutions
    - Doubling filters after pooling

    Args:
        num_classes: Number of output classes (default: 10)
    """

    def __init__(self, num_classes: int = 10):
        super(VGG11, self).__init__()

        def vgg_block(in_ch: int, out_ch: int, num_convs: int) -> nn.Sequential:
            layers = []
            for i in range(num_convs):
                layers.extend([
                    nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ])
            layers.append(nn.MaxPool2d(2, 2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            vgg_block(3, 64, 1),
            vgg_block(64, 128, 1),
            vgg_block(128, 256, 2),
            vgg_block(256, 512, 2),
            vgg_block(512, 512, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    """
    ResNet-18 implementation.

    Original paper: "Deep Residual Learning for Image Recognition"
    by Kaiming He et al., 2015

    Key features:
    - Skip connections (identity shortcuts)
    - Batch normalization
    - Global average pooling

    Args:
        num_classes: Number of output classes (default: 10)
    """

    def __init__(self, num_classes: int = 10):
        super(ResNet18, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ==================== U-Net for Segmentation ====================

class DoubleConv(nn.Module):
    """Double convolution block for U-Net."""

    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling block for U-Net."""

    def __init__(self, in_channels: int, out_channels: int):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling block for U-Net."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, 2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for semantic segmentation.

    Original paper: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    by Olaf Ronneberger et al., 2015

    Args:
        n_channels: Number of input channels (default: 3)
        n_classes: Number of output classes (default: 21 for VOC)
        bilinear: Use bilinear upsampling (default: True)

    Example:
        >>> model = UNet(n_channels=3, n_classes=21)
        >>> x = torch.randn(1, 3, 256, 256)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([1, 21, 256, 256])
    """

    def __init__(self, n_channels: int = 3, n_classes: int = 21, bilinear: bool = True):
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)


def get_model(name: str, num_classes: int = 10, **kwargs) -> nn.Module:
    """
    Factory function to get a model by name.

    Args:
        name: Model name ('lenet', 'alexnet', 'vgg11', 'resnet18', 'unet')
        num_classes: Number of output classes
        **kwargs: Additional arguments for the model

    Returns:
        Instantiated model

    Example:
        >>> model = get_model('resnet18', num_classes=10)
    """
    models = {
        'lenet': LeNet5,
        'alexnet': AlexNet,
        'vgg11': VGG11,
        'resnet18': ResNet18,
        'unet': UNet,
    }

    name = name.lower()
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")

    if name == 'unet':
        return models[name](n_classes=num_classes, **kwargs)
    return models[name](num_classes=num_classes, **kwargs)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test all architectures
    print("Testing CNN Architectures...")

    x = torch.randn(2, 3, 32, 32)

    for name in ['lenet', 'alexnet', 'vgg11', 'resnet18']:
        model = get_model(name, num_classes=10)
        out = model(x)
        params = count_parameters(model)
        print(f"{name.upper():>10}: {out.shape} | {params:,} params")

    # Test U-Net
    x_seg = torch.randn(2, 3, 256, 256)
    unet = get_model('unet', num_classes=21)
    out_seg = unet(x_seg)
    print(f"{'UNET':>10}: {out_seg.shape} | {count_parameters(unet):,} params")
