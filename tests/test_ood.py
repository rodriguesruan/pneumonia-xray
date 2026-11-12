import numpy as np
import pytest

from src.evaluation.ood import fit_energy_threshold


def test_energy_threshold_uses_validation_quantile() -> None:
    threshold = fit_energy_threshold(np.array([-3.0, -2.0, -1.0, 0.0]), 0.75)
    assert threshold == -0.75


def test_energy_threshold_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty and finite"):
        fit_energy_threshold(np.array([np.nan]), 0.95)
