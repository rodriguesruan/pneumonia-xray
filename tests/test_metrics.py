import numpy as np
import pytest

from src.evaluation.metrics import (
    bootstrap_confidence_intervals,
    classification_metrics,
    select_threshold_for_sensitivity,
)


def test_classification_metrics_compute_operating_point() -> None:
    metrics = classification_metrics(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.7, 0.8, 0.9]), threshold=0.5
    )
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 0.5
    assert metrics["false_positive"] == 1
    assert metrics["roc_auc"] == 1.0


def test_threshold_meets_validation_sensitivity_target() -> None:
    selected = select_threshold_for_sensitivity(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.7, 0.8, 0.9]), 1.0
    )
    assert selected["validation_sensitivity"] >= 1.0
    assert selected["validation_specificity"] == 1.0


def test_group_bootstrap_is_reproducible() -> None:
    labels = np.array([0, 1, 0, 1, 0, 1])
    probabilities = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    first = bootstrap_confidence_intervals(labels, probabilities, 0.5, 40, 0.95, 23, groups=groups)
    second = bootstrap_confidence_intervals(labels, probabilities, 0.5, 40, 0.95, 23, groups=groups)
    assert first == second
    assert first["sensitivity"]["valid_replicates"] == 40.0


def test_metrics_reject_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        classification_metrics(np.array([0, 1]), np.array([0.1, 1.2]), 0.5)
