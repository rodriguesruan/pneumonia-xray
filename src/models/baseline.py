"""Compact CNN baseline architecture."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """Three-block convolutional baseline for comparison with transfer learning."""

    def __init__(self, num_classes: int, channels: Sequence[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_channels = 3
        for output_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            input_channels = output_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(p=dropout), nn.Linear(channels[-1], num_classes)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute raw class logits."""
        return self.classifier(self.pool(self.features(inputs)))
