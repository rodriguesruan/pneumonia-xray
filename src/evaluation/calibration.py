"""Temperature scaling fitted exclusively on validation logits."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.optim import LBFGS


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iterations: int) -> float:
    """Fit one positive temperature by minimizing validation cross-entropy."""
    logits_tensor = torch.as_tensor(logits, dtype=torch.float64)
    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    if logits_tensor.ndim != 2 or logits_tensor.shape[0] != len(labels_tensor):
        raise ValueError("logits must have shape [samples, classes] and align with labels")
    if len(torch.unique(labels_tensor)) < 2:
        raise ValueError("Temperature scaling requires both classes in validation")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    log_temperature = nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = LBFGS([log_temperature], max_iter=max_iterations, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp()
        loss = nn.functional.cross_entropy(logits_tensor / temperature, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(min=1e-3, max=1e3))


def probabilities_from_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Convert logits into calibrated softmax probabilities."""
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2 or not np.isfinite(logits).all():
        raise ValueError("logits must be a finite two-dimensional array")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    scaled = logits / temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)
