"""Grad-CAM helpers for supported model architectures."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def target_layer_for_model(model: nn.Module, model_name: str) -> nn.Module:
    """Return the final convolutional feature layer for Grad-CAM."""
    if model_name == "resnet50":
        return model.layer4[-1]
    if model_name == "mobilenet_v3_small":
        return model.features[-1]
    if model_name == "densenet121":
        return model.features.denseblock4
    if model_name == "small_cnn":
        convolution_layers = [
            module for module in model.features.modules() if isinstance(module, nn.Conv2d)
        ]
        if not convolution_layers:
            raise ValueError("SmallCNN has no convolutional layer for Grad-CAM")
        return convolution_layers[-1]
    raise ValueError(f"Grad-CAM is not configured for {model_name}")


def create_gradcam(
    model: nn.Module,
    model_name: str,
    image_tensor: torch.Tensor,
    original_rgb: np.ndarray,
    positive_class_index: int,
) -> np.ndarray | None:
    """Create an RGB Grad-CAM overlay, or return None if OpenCV is unavailable."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except (ImportError, OSError):
        # OpenCV may be installed but fail to load when a system library is absent.
        return None

    target_layers = [target_layer_for_model(model, model_name)]
    # The production classifier freezes its feature backbone. Requiring gradients
    # on the input keeps gradients flowing through that frozen backbone to the CAM layer.
    gradcam_input = image_tensor.unsqueeze(0).detach().requires_grad_(True)
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(
            input_tensor=gradcam_input,
            targets=[ClassifierOutputTarget(positive_class_index)],
        )[0]
    return show_cam_on_image(original_rgb, grayscale_cam, use_rgb=True)
