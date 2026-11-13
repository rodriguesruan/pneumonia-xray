"""Graceful behavior when Grad-CAM's OpenCV dependency cannot load."""

from __future__ import annotations

import builtins

import numpy as np
import pytest


def test_create_gradcam_returns_none_when_gradcam_import_fails(monkeypatch):
    torch = pytest.importorskip("torch")
    from src.explain.gradcam import create_gradcam

    original_import = builtins.__import__

    def fail_gradcam_import(name, *args, **kwargs):
        if name == "pytorch_grad_cam":
            raise ImportError("simulated OpenCV runtime load failure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_gradcam_import)
    image = torch.zeros((3, 8, 8))

    result = create_gradcam(
        model=torch.nn.Identity(),
        model_name="mobilenet_v3_small",
        image_tensor=image,
        original_rgb=np.zeros((8, 8, 3), dtype=np.float32),
        positive_class_index=1,
    )

    assert result is None


def test_create_gradcam_computes_with_frozen_feature_backbone():
    torch = pytest.importorskip("torch")
    pytest.importorskip("cv2")
    from src.explain.gradcam import create_gradcam

    class TinyClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 4, kernel_size=3, padding=1), torch.nn.ReLU()
            )
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = torch.nn.Linear(4, 2)
            for parameter in self.features.parameters():
                parameter.requires_grad_(False)

        def forward(self, inputs):
            return self.classifier(self.pool(self.features(inputs)).flatten(1))

    model = TinyClassifier().eval()
    image = torch.rand((3, 16, 16))
    overlay = create_gradcam(
        model=model,
        model_name="mobilenet_v3_small",
        image_tensor=image,
        original_rgb=image.permute(1, 2, 0).numpy(),
        positive_class_index=1,
    )

    assert overlay is not None
    assert overlay.shape == (16, 16, 3)
