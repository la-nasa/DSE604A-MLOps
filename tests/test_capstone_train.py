from src.capstone_train import train_and_track


def test_train_and_track_runs_both_configs_and_reports_metrics():
    summary = train_and_track(
        experiment_name="CI_Capstone_Test",
        registered_model_name="CI_MachineFailurePredictor",
        promotion_threshold=1.1,  # deliberately unreachable to skip registry in CI
    )

    assert set(summary["results"].keys()) == {"random_forest", "gradient_boosting"}
    for config_result in summary["results"].values():
        assert "run_id" in config_result
        metrics = config_result["metrics"]
        for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0

    assert summary["best_config"] in {"random_forest", "gradient_boosting"}
    assert summary["registration"]["registered"] is False
