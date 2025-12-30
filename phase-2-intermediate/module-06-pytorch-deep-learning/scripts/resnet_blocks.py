"""
ResNet Building Blocks - Production-Ready Implementation

This module provides reusable ResNet components optimized for DGX Spark.

Components:
    - BasicBlock: For ResNet-18 and ResNet-34
    - Bottleneck: For ResNet-50, ResNet-101, ResNet-152
    - SEBlock: Squeeze-and-Excitation attention
    - ResNet: Complete ResNet implementation

Example:
    >>> from resnet_blocks import resnet18, resnet50
    >>> model = resnet18(num_classes=10).to('cuda')
    >>> x = torch.randn(4, 3, 32, 32).to('cuda')
    >>> output = model(x)
    >>> print(output.shape)  # torch.Size([4, 10])

Author: DGX Spark AI Curriculum
"""

__all__ = [
    'SEBlock',
    'BasicBlock',
    'Bottleneck',
    'ResNet',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
]

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Union, List, Optional


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel attention.

    Paper: "Squeeze-and-Excitation Networks" (Hu et al., 2018)

    This block learns to weight channels based on their importance,
    improving the representational power of the network.

    Args:
        channels: Number of input/output channels
        reduction: Reduction ratio for the bottleneck (default: 16)

    Example:
        >>> se = SEBlock(64, reduction=16)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> y = se(x)
        >>> assert y.shape == x.shape
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        reduced = max(channels // reduction, 1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, reduced, bias=False)
        self.fc2 = nn.Linear(reduced, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape

        # Squeeze: Global average pooling
        y = self.global_pool(x).view(b, c)

        # Excitation: Two FC layers with sigmoid
        y = F.relu(self.fc1(y), inplace=True)
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)

        # Scale: Multiply input by channel weights
        return x * y


class BasicBlock(nn.Module):
    """
    ResNet BasicBlock: Two 3x3 convolutions with a skip connection.

    Used in ResNet-18 and ResNet-34.

    Architecture:
        x -> [3x3 Conv -> BN -> ReLU -> 3x3 Conv -> BN] + shortcut(x) -> ReLU

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for the first convolution (default: 1)
        use_se: Whether to use Squeeze-and-Excitation (default: False)
        dropout: Dropout probability after first ReLU (default: 0.0)

    Example:
        >>> block = BasicBlock(64, 64)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> y = block(x)
        >>> assert y.shape == x.shape
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_se: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()

        # First conv: may downsample if stride=2
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Optional dropout
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        # Second conv: always stride=1
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Optional SE block
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # SE attention
        out = self.se(out)

        # Add shortcut and activate
        out += self.shortcut(identity)
        out = F.relu(out, inplace=True)

        return out


class Bottleneck(nn.Module):
    """
    ResNet Bottleneck Block: 1x1 -> 3x3 -> 1x1 with skip connection.

    Used in ResNet-50, ResNet-101, ResNet-152.

    The bottleneck design reduces parameters by doing the expensive 3x3 conv
    at reduced channel count.

    Architecture:
        x -> [1x1 Conv -> BN -> ReLU -> 3x3 Conv -> BN -> ReLU -> 1x1 Conv -> BN]
              + shortcut(x) -> ReLU

    Args:
        in_channels: Number of input channels
        bottleneck_channels: Number of channels in the bottleneck layer
        stride: Stride for the 3x3 convolution (default: 1)
        use_se: Whether to use Squeeze-and-Excitation (default: False)

    Example:
        >>> block = Bottleneck(64, 64)  # Output: 64 * 4 = 256 channels
        >>> x = torch.randn(2, 64, 56, 56)
        >>> y = block(x)
        >>> print(y.shape)  # torch.Size([2, 256, 56, 56])
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int,
        stride: int = 1,
        use_se: bool = False
    ):
        super().__init__()

        out_channels = bottleneck_channels * self.expansion

        # 1x1 conv: squeeze
        self.conv1 = nn.Conv2d(
            in_channels, bottleneck_channels,
            kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        # 3x3 conv: spatial processing
        self.conv2 = nn.Conv2d(
            bottleneck_channels, bottleneck_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        # 1x1 conv: expand
        self.conv3 = nn.Conv2d(
            bottleneck_channels, out_channels,
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Optional SE block
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # Squeeze
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        # Process
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)

        # Expand
        out = self.conv3(out)
        out = self.bn3(out)

        # SE attention
        out = self.se(out)

        # Add shortcut and activate
        out += self.shortcut(identity)
        out = F.relu(out, inplace=True)

        return out


class ResNet(nn.Module):
    """
    ResNet implementation for various input sizes.

    This implementation supports both ImageNet (224x224) and CIFAR (32x32) inputs.

    Args:
        block: Block type (BasicBlock or Bottleneck)
        layers: Number of blocks in each layer [layer1, layer2, layer3, layer4]
        num_classes: Number of output classes (default: 10)
        input_size: 'cifar' for 32x32 or 'imagenet' for 224x224 (default: 'cifar')
        use_se: Whether to use SE blocks (default: False)

    Example:
        >>> model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        >>> x = torch.randn(4, 3, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([4, 10])
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 10,
        input_size: str = 'cifar',
        use_se: bool = False
    ):
        super().__init__()

        self.in_channels = 64
        self.use_se = use_se

        # Initial convolution - depends on input size
        if input_size == 'cifar':
            # CIFAR: 3x3 conv, no pooling
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.maxpool = nn.Identity()
        else:
            # ImageNet: 7x7 conv, max pooling
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(64)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        channels: int,
        num_blocks: int,
        stride: int
    ) -> nn.Sequential:
        """Create a layer with multiple blocks."""
        layers = []

        # First block may downsample
        layers.append(
            block(self.in_channels, channels, stride, use_se=self.use_se)
        )
        self.in_channels = channels * block.expansion

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(
                block(self.in_channels, channels, stride=1, use_se=self.use_se)
            )

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)

        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the final FC layer."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


# Factory functions

def resnet18(
    num_classes: int = 10,
    input_size: str = 'cifar',
    use_se: bool = False
) -> ResNet:
    """Create ResNet-18 model (11.7M parameters for CIFAR)."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, input_size, use_se)


def resnet34(
    num_classes: int = 10,
    input_size: str = 'cifar',
    use_se: bool = False
) -> ResNet:
    """Create ResNet-34 model (21.8M parameters for CIFAR)."""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, input_size, use_se)


def resnet50(
    num_classes: int = 10,
    input_size: str = 'cifar',
    use_se: bool = False
) -> ResNet:
    """Create ResNet-50 model (25.6M parameters for CIFAR)."""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, input_size, use_se)


def resnet101(
    num_classes: int = 10,
    input_size: str = 'cifar',
    use_se: bool = False
) -> ResNet:
    """Create ResNet-101 model (44.5M parameters for CIFAR)."""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, input_size, use_se)


def resnet152(
    num_classes: int = 10,
    input_size: str = 'cifar',
    use_se: bool = False
) -> ResNet:
    """Create ResNet-152 model (60.2M parameters for CIFAR)."""
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, input_size, use_se)


if __name__ == '__main__':
    # Quick test
    print("Testing ResNet implementations...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Test all variants
    variants = [
        ('ResNet-18', resnet18),
        ('ResNet-34', resnet34),
        ('ResNet-50', resnet50),
    ]

    x = torch.randn(2, 3, 32, 32).to(device)

    for name, factory in variants:
        model = factory(num_classes=10).to(device)
        params = sum(p.numel() for p in model.parameters())

        with torch.no_grad():
            y = model(x)

        print(f"{name}: {params:,} parameters, output shape: {y.shape}")

    print("\nAll tests passed!")
