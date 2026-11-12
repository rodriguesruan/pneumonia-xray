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
