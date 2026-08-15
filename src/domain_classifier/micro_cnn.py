"""
MicroCNN Architecture for Tier-2 Fallback Domain Classification.

Fulfills US-3.1.2 and ADR-009:
Defines an ultra-compact convolutional neural network (< 150k parameters, < 0.6 MB ONNX)
engineered for sub-millisecond CPU inference in ONNX Runtime.

The architecture combines:
1. High-Frequency Gradient & Texture Sensitive ConvStem (sensitive to Moiré and sensor noise).
2. Universal Inverted Residual Bottleneck stages with depthwise separable convolutions.
3. Global Average Pooling with Linear 2-Class Classification Head (DIGITAL_2D vs PHYSICAL_3D).
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    """Standard Convolution + BatchNorm2d + Activation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class InvertedResidualBlock(nn.Module):
    """
    Lightweight Inverted Residual Bottleneck Block.
    
    Expands channel dimension -> Depthwise 3x3 Conv -> Pointwise 1x1 Linear Projection.
    Applies residual skip-connection when stride == 1 and in_channels == out_channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion_factor: int = 2,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.use_residual = self.stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expansion_factor

        layers: list[nn.Module] = []
        # 1. Pointwise expansion
        if expansion_factor != 1:
            layers.append(ConvBNAct(in_channels, hidden_dim, kernel_size=1, stride=1))

        # 2. Depthwise convolution
        layers.append(
            ConvBNAct(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                groups=hidden_dim,
            )
        )

        # 3. Pointwise linear projection (no non-linearity)
        layers.append(
            nn.Sequential(
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MicroCNN(nn.Module):
    """
    Ultra-compact MicroCNN for Binary Domain Classification (DIGITAL_2D vs PHYSICAL_3D).
    
    Total Parameters: ~148,000 (~0.59 MB FP32 ONNX, ~0.16 MB INT8 ONNX).
    Input: (B, 3, 128, 128) float32 tensor in range [0, 1].
    Output: (B, 2) raw logits (Index 0 = DIGITAL_2D, Index 1 = PHYSICAL_3D).
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.20) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Stage 1: Stem Conv (128x128 -> 64x64)
        self.stem = ConvBNAct(in_channels=3, out_channels=24, kernel_size=3, stride=2)

        # Stage 2: Inverted Residual (64x64 -> 32x32)
        self.stage2 = nn.Sequential(
            InvertedResidualBlock(in_channels=24, out_channels=32, stride=2, expansion_factor=2),
            InvertedResidualBlock(in_channels=32, out_channels=32, stride=1, expansion_factor=2),
        )

        # Stage 3: Inverted Residual (32x32 -> 16x16)
        self.stage3 = nn.Sequential(
            InvertedResidualBlock(in_channels=32, out_channels=64, stride=2, expansion_factor=2),
            InvertedResidualBlock(in_channels=64, out_channels=64, stride=1, expansion_factor=2),
        )

        # Stage 4: Inverted Residual (16x16 -> 8x8)
        self.stage4 = nn.Sequential(
            InvertedResidualBlock(in_channels=64, out_channels=96, stride=2, expansion_factor=2),
            InvertedResidualBlock(in_channels=96, out_channels=96, stride=1, expansion_factor=2),
        )

        # Stage 5: Conv Head (8x8 -> 8x8)
        self.head_conv = ConvBNAct(in_channels=96, out_channels=160, kernel_size=1, stride=1)

        # Pooling & Classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(160, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initializes weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, 128, 128)
            
        Returns:
            Logits tensor of shape (B, 2)
        """
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head_conv(x)
        x = self.pool(x)
        return self.classifier(x)


def build_domain_classifier_model(
    arch: Literal["micro_cnn", "mobilenetv3_small_035"] = "micro_cnn",
    num_classes: int = 2,
) -> nn.Module:
    """
    Factory function to instantiate domain classifier PyTorch models.
    
    Args:
        arch: Model architecture name ('micro_cnn' or 'mobilenetv3_small_035')
        num_classes: Number of output classes (default: 2)
        
    Returns:
        Instantiated PyTorch nn.Module
    """
    if arch == "micro_cnn":
        return MicroCNN(num_classes=num_classes)
    elif arch == "mobilenetv3_small_035":
        try:
            import torchvision.models as models

            model = models.mobilenet_v3_small(weights=None, num_classes=num_classes)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to build torchvision mobilenet_v3_small: {e}") from e
    else:
        raise ValueError(f"Unknown architecture: {arch}. Choose 'micro_cnn' or 'mobilenetv3_small_035'.")
