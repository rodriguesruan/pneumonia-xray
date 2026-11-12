"""Construct configured baseline and ImageNet transfer-learning models."""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from torchvision import models

from src.models.baseline import SmallCNN


def build_model(config: dict[str, Any]) -> nn.Module:
    """Build a model and replace its classification head."""
    model_config = config["model"]
    model_name = model_config["name"].lower()
    num_classes = model_config["num_classes"]
    dropout = model_config["dropout"]

    if model_name == "small_cnn":
        return SmallCNN(
            num_classes=num_classes,
            channels=model_config["baseline_channels"],
            dropout=dropout,
        )
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if model_config["pretrained"] else None
        model = models.resnet50(weights=weights)
        if model_config["freeze_backbone"]:
            for parameter in model.parameters():
                parameter.requires_grad = False
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(model.fc.in_features, num_classes)
        )
        return model
    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if model_config["pretrained"] else None
        model = models.mobilenet_v3_small(weights=weights)
        if model_config["freeze_backbone"]:
            for parameter in model.parameters():
                parameter.requires_grad = False
        input_features = model.classifier[0].in_features
        hidden_features = model.classifier[0].out_features
        model.classifier = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_features, num_classes),
        )
        return model
    if model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if model_config["pretrained"] else None
        model = models.densenet121(weights=weights)
        if model_config["freeze_backbone"]:
            for parameter in model.parameters():
                parameter.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(model.classifier.in_features, num_classes)
        )
        return model
    raise ValueError(f"Unsupported model name: {model_name}")
