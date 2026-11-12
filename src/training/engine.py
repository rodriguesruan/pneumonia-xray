"""Training and validation epoch functions."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    threshold: float,
    positive_class_index: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Run one training or validation epoch and return aggregate metrics."""
    training = optimizer is not None
    model.train(training)
    if training:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(
                parameter.requires_grad for parameter in module.parameters()
            ):
                module.eval()
    loss_total = 0.0
    sample_count = 0
    all_labels: list[int] = []
    all_probabilities: list[float] = []
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled and training)

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = loss_function(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        batch_size = int(labels.shape[0])
        loss_total += float(loss.detach().item()) * batch_size
        sample_count += batch_size
        probabilities = torch.softmax(logits.detach().float(), dim=1)[:, positive_class_index]
        all_probabilities.extend(probabilities.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    if sample_count == 0:
        raise ValueError("Cannot calculate metrics for an empty data loader")
    labels_array = np.asarray(all_labels)
    probabilities_array = np.asarray(all_probabilities)
    predictions = probabilities_array >= threshold
    positives = labels_array == positive_class_index
    negatives = ~positives
    true_positive = int(np.logical_and(predictions, positives).sum())
    true_negative = int(np.logical_and(~predictions, negatives).sum())
    false_negative = int(np.logical_and(~predictions, positives).sum())
    false_positive = int(np.logical_and(predictions, negatives).sum())
    auc = (
        float(roc_auc_score(labels_array, probabilities_array))
        if len(np.unique(labels_array)) > 1
        else float("nan")
    )
    return {
        "loss": loss_total / sample_count,
        "roc_auc": auc,
        "monitor_threshold": threshold,
        "sensitivity_at_threshold": true_positive / max(true_positive + false_negative, 1),
        "specificity_at_threshold": true_negative / max(true_negative + false_positive, 1),
        "sample_count": sample_count,
        "labels": all_labels,
        "probabilities": all_probabilities,
    }
