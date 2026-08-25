"""
Lab 14 -- Monitoring and Model Drift.

Detects changes in production data and model performance, and helps decide
whether an accuracy drop points to data drift, concept drift, or an
implementation problem -- the exact investigation the manual poses:

    Training Accuracy = 95%
    Production Accuracy = 87%
"""
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import mlflow
import numpy as np


def population_stability_index(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index (PSI) between a reference (training) and a
    current (production) numeric distribution.

    PSI < 0.1  -> no significant shift
    0.1 - 0.25 -> moderate shift, worth investigating
    > 0.25     -> significant distribution shift (data drift)
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    breakpoints = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_counts = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Avoid divide-by-zero / log(0) on empty buckets.
    expected_counts = np.clip(expected_counts, 1e-6, None)
    actual_counts = np.clip(actual_counts, 1e-6, None)

    psi = np.sum((actual_counts - expected_counts) * np.log(actual_counts / expected_counts))
    return float(psi)


@dataclass
class DriftReport:
    accuracy_drop: float
    feature_psi: Dict[str, float] = field(default_factory=dict)
    diagnosis: str = ""
    alert: bool = False
    recommended_action: str = ""


def diagnose_accuracy_drop(
    training_accuracy: float,
    production_accuracy: float,
    feature_psi: Optional[Dict[str, float]] = None,
    accuracy_drop_threshold: float = 0.05,
    psi_drift_threshold: float = 0.25,
) -> DriftReport:
    """Classify an accuracy drop as data drift, concept drift, or an
    implementation problem, following the decision procedure taught in Lab 14.

    Heuristic:
      - No meaningful drop -> healthy, no action.
      - Drop exceeds threshold AND input feature distributions shifted
        significantly (PSI) -> data drift (retrain on recent data).
      - Drop exceeds threshold AND feature distributions are stable ->
        concept drift (the input/output relationship changed; retrain and
        review labeling/feature engineering) or an implementation bug if the
        drop is extreme and sudden (checked by the caller/operator).
    """
    feature_psi = feature_psi or {}
    accuracy_drop = training_accuracy - production_accuracy

    if accuracy_drop < accuracy_drop_threshold:
        return DriftReport(
            accuracy_drop=accuracy_drop,
            feature_psi=feature_psi,
            diagnosis="healthy",
            alert=False,
            recommended_action="No action required; continue routine monitoring.",
        )

    drifted_features = {name: psi for name, psi in feature_psi.items() if psi >= psi_drift_threshold}

    if drifted_features:
        diagnosis = "data_drift"
        recommended_action = (
            f"Input distribution shifted for {list(drifted_features.keys())}. "
            "Retrain on recent production data and refresh the reference dataset."
        )
    else:
        diagnosis = "concept_drift"
        recommended_action = (
            "Input distributions are stable but accuracy dropped: the relationship "
            "between features and the target likely changed. Investigate recent "
            "labels/business changes, then retrain; also double-check the serving "
            "pipeline for implementation regressions (feature order, encoding, "
            "stale preprocessing) before ruling out a bug."
        )

    return DriftReport(
        accuracy_drop=accuracy_drop,
        feature_psi=feature_psi,
        diagnosis=diagnosis,
        alert=True,
        recommended_action=recommended_action,
    )


def log_drift_report(report: DriftReport, run_name: str = "drift_check") -> str:
    """Log a drift report to MLflow so it shows up alongside training runs."""
    mlflow.set_experiment("Model_Monitoring")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_metric("accuracy_drop", report.accuracy_drop)
        for feature, psi in report.feature_psi.items():
            mlflow.log_metric(f"psi_{feature}", psi)
        mlflow.log_param("diagnosis", report.diagnosis)
        mlflow.log_param("alert", report.alert)
        mlflow.set_tag("recommended_action", report.recommended_action)
        return run.info.run_id


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose a production accuracy drop (Lab 14).")
    parser.add_argument("--training_accuracy", type=float, default=0.95)
    parser.add_argument("--production_accuracy", type=float, default=0.87)
    parser.add_argument("--log_to_mlflow", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    report = diagnose_accuracy_drop(args.training_accuracy, args.production_accuracy)
    print(f"Accuracy drop: {report.accuracy_drop:.4f}")
    print(f"Diagnosis: {report.diagnosis}")
    print(f"Alert: {report.alert}")
    print(f"Recommended action: {report.recommended_action}")

    if args.log_to_mlflow:
        run_id = log_drift_report(report)
        print(f"Logged drift report to MLflow run {run_id}")


if __name__ == "__main__":
    main()
