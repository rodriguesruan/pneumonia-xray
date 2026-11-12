import numpy as np
import pytest


def test_temperature_scaling_returns_valid_probabilities() -> None:
    pytest.importorskip("torch")
    from src.evaluation.calibration import fit_temperature, probabilities_from_logits

    logits = np.array([[3.0, -1.0], [-2.0, 2.0], [2.5, -0.5], [-1.5, 1.5]])
    labels = np.array([0, 1, 0, 1])
    temperature = fit_temperature(logits, labels, max_iterations=20)
    probabilities = probabilities_from_logits(logits, temperature)
    assert temperature > 0
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_temperature_scaling_requires_both_classes() -> None:
    pytest.importorskip("torch")
    from src.evaluation.calibration import fit_temperature

    with pytest.raises(ValueError, match="both classes"):
        fit_temperature(np.array([[1.0, 0.0]]), np.array([0]), max_iterations=10)
