"""Classification metrics, validation threshold selection, and bootstrap intervals."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate discrimination, calibration, and operating-point metrics."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must be aligned one-dimensional arrays")
    if not len(labels) or not np.isfinite(probabilities).all():
        raise ValueError("Evaluation data must be non-empty with finite probabilities")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be between zero and one")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must contain only binary values 0 and 1")
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    has_both_classes = len(np.unique(labels)) == 2
    return {
        "threshold": float(threshold),
        "n": int(len(labels)),
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(labels, probabilities)) if has_both_classes else float("nan")
        ),
        "pr_auc": (
            float(average_precision_score(labels, probabilities))
            if has_both_classes
            else float("nan")
        ),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }


def select_threshold_for_sensitivity(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_sensitivity: float,
) -> dict[str, float]:
    """Choose the highest validation threshold meeting a sensitivity target."""
    if not 0 < target_sensitivity <= 1:
        raise ValueError("target_sensitivity must be in (0, 1]")
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(labels) != len(probabilities) or not len(labels):
        raise ValueError("labels and probabilities must be non-empty and aligned")
    if not np.any(labels == 1):
        raise ValueError("Validation set must contain positive examples")
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    eligible: list[tuple[float, float]] = []
    for threshold in candidates:
        metrics = classification_metrics(labels, probabilities, float(threshold))
        if metrics["sensitivity"] >= target_sensitivity:
            eligible.append((float(threshold), float(metrics["specificity"])))
    if not eligible:
        raise RuntimeError("No threshold met the requested sensitivity on validation data")
    threshold, specificity = max(eligible, key=lambda item: item[0])
    selected = classification_metrics(labels, probabilities, threshold)
    return {
        "threshold": threshold,
        "target_sensitivity": float(target_sensitivity),
        "validation_sensitivity": float(selected["sensitivity"]),
        "validation_specificity": specificity,
    }


def bootstrap_confidence_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    replicates: int,
    confidence_level: float,
    seed: int,
    groups: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """Estimate percentile confidence intervals using image or cluster bootstrap."""
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(labels) != len(probabilities) or not len(labels):
        raise ValueError("labels and probabilities must be non-empty and aligned")
    rng = np.random.default_rng(seed)
    if groups is None:
        units = np.arange(len(labels))
        members = {int(index): np.array([index]) for index in units}
    else:
        groups = np.asarray(groups)
        if len(groups) != len(labels):
            raise ValueError("groups must align with labels")
        units = np.unique(groups)
        members = {unit: np.flatnonzero(groups == unit) for unit in units}
    metric_values: dict[str, list[float]] = {}
    for _ in range(replicates):
        sampled_units = rng.choice(units, size=len(units), replace=True)
        indices = np.concatenate([members[unit] for unit in sampled_units])
        metrics = classification_metrics(labels[indices], probabilities[indices], threshold)
        for name, value in metrics.items():
            if name != "threshold" and np.isfinite(value):
                metric_values.setdefault(name, []).append(float(value))
    alpha = (1 - confidence_level) / 2
    intervals: dict[str, dict[str, float]] = {}
    for name, values in metric_values.items():
        if values:
            lower, upper = np.quantile(values, [alpha, 1 - alpha])
            intervals[name] = {
                "lower": float(lower),
                "upper": float(upper),
                "valid_replicates": float(len(values)),
            }
    return intervals


def decision_curve(
    labels: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray
) -> dict[str, np.ndarray]:
    """Calculate model, treat-all, and treat-none net benefit across thresholds."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if np.any((thresholds <= 0) | (thresholds >= 1)):
        raise ValueError("Decision-curve thresholds must lie strictly between zero and one")
    prevalence = float(labels.mean())
    model_benefits = []
    for threshold in thresholds:
        predictions = probabilities >= threshold
        true_positive = np.logical_and(predictions, labels == 1).sum()
        false_positive = np.logical_and(predictions, labels == 0).sum()
        model_benefits.append(
            true_positive / len(labels) - false_positive / len(labels) * threshold / (1 - threshold)
        )
    treat_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
    return {
        "thresholds": thresholds,
        "model": np.asarray(model_benefits),
        "treat_all": treat_all,
        "treat_none": np.zeros_like(thresholds),
    }
