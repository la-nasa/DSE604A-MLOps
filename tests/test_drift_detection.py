import numpy as np

from src.drift_detection import diagnose_accuracy_drop, population_stability_index


def test_psi_is_near_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 5000)
    psi = population_stability_index(reference, reference.copy())
    assert psi < 0.01


def test_psi_detects_shifted_distribution():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 5000)
    shifted = rng.normal(3, 1, 5000)
    psi = population_stability_index(reference, shifted)
    assert psi > 0.25


def test_diagnose_no_drop_is_healthy():
    report = diagnose_accuracy_drop(training_accuracy=0.95, production_accuracy=0.94)
    assert report.diagnosis == "healthy"
    assert report.alert is False


def test_diagnose_manual_scenario_with_data_drift():
    # The exact numbers from the manual: 95% -> 87%, with a feature that
    # clearly shifted (PSI above the drift threshold).
    report = diagnose_accuracy_drop(
        training_accuracy=0.95,
        production_accuracy=0.87,
        feature_psi={"temperature": 0.4, "vibration": 0.05},
    )
    assert report.alert is True
    assert report.diagnosis == "data_drift"
    assert "temperature" in report.recommended_action


def test_diagnose_manual_scenario_without_feature_shift_is_concept_drift():
    report = diagnose_accuracy_drop(
        training_accuracy=0.95,
        production_accuracy=0.87,
        feature_psi={"temperature": 0.05, "vibration": 0.02},
    )
    assert report.alert is True
    assert report.diagnosis == "concept_drift"
