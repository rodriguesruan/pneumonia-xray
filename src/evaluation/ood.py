"""Energy-score OOD alert calibrated on validation data."""

from __future__ import annotations

import numpy as np


def fit_energy_threshold(validation_energies: np.ndarray, quantile: float) -> float:
    """Set a one-sided OOD alert threshold from validation energy scores."""
    energies = np.asarray(validation_energies, dtype=float)
    if not len(energies) or not np.isfinite(energies).all():
        raise ValueError("Validation energies must be non-empty and finite")
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1)")
    return float(np.quantile(energies, quantile))


def energy_score(logits) -> np.ndarray:
    """Return energy scores where larger values indicate lower model confidence."""
    import torch

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    return (-torch.logsumexp(logits.float(), dim=1)).detach().cpu().numpy()
