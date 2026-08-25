"""
Lab 15 -- Complete MLOps Capstone.

Industrial scenario: predict machine failures from temperature, vibration,
pressure and operating-hours sensor data. This script implements the
required workflow end to end:

    Sensor Data -> Preprocessing -> Training (>= 2 configurations) ->
    MLflow Tracking -> Model Evaluation -> Model Registry

Docker packaging (src/app.py) and Kubernetes deployment reuse the model
registered here via the MLFLOW_MODEL_URI environment variable.
"""
import argparse

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_processing import SENSOR_FEATURE_COLUMNS, get_sensor_train_test_split

MODEL_CONFIGS = {
    "random_forest": {
        "estimator": RandomForestClassifier,
        "params": {"n_estimators": 200, "max_depth": 8, "random_state": 42, "class_weight": "balanced"},
    },
    "gradient_boosting": {
        "estimator": GradientBoostingClassifier,
        "params": {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.1, "random_state": 42},
    },
}


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def train_and_track(
    experiment_name: str = "Machine_Failure_Prediction",
    registered_model_name: str = "MachineFailurePredictor",
    promotion_metric: str = "f1_score",
    promotion_threshold: float = 0.6,
):
    mlflow.set_experiment(experiment_name)

    X_train, X_test, y_train, y_test = get_sensor_train_test_split()

    results = {}
    for config_name, config in MODEL_CONFIGS.items():
        with mlflow.start_run(run_name=config_name) as run:
            pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("classifier", config["estimator"](**config["params"])),
                ]
            )
            pipeline.fit(X_train[SENSOR_FEATURE_COLUMNS], y_train)

            metrics = evaluate(pipeline, X_test[SENSOR_FEATURE_COLUMNS], y_test)

            mlflow.log_param("model_type", config_name)
            for param_name, param_value in config["params"].items():
                mlflow.log_param(param_name, param_value)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, "model")

            results[config_name] = {"run_id": run.info.run_id, "metrics": metrics}
            print(f"[{config_name}] metrics: {metrics}")

    best_config = max(results, key=lambda name: results[name]["metrics"][promotion_metric])
    best_run_id = results[best_config]["run_id"]
    best_metrics = results[best_config]["metrics"]
    print(f"Best configuration: {best_config} ({promotion_metric}={best_metrics[promotion_metric]:.4f})")

    registration = {"registered": False, "version": None, "stage": None}
    if best_metrics[promotion_metric] >= promotion_threshold:
        client = MlflowClient()
        try:
            client.get_registered_model(registered_model_name)
        except Exception:
            client.create_registered_model(registered_model_name)

        model_uri = f"runs:/{best_run_id}/model"
        mv = client.create_model_version(name=registered_model_name, source=model_uri, run_id=best_run_id)
        client.transition_model_version_stage(
            name=registered_model_name, version=mv.version, stage="Staging", archive_existing_versions=False
        )
        registration = {"registered": True, "version": mv.version, "stage": "Staging"}
        print(f"Registered '{registered_model_name}' v{mv.version} and promoted to Staging.")
    else:
        print(
            f"Best model {promotion_metric}={best_metrics[promotion_metric]:.4f} "
            f"below threshold {promotion_threshold}; not registered."
        )

    return {"results": results, "best_config": best_config, "registration": registration}


def parse_args():
    parser = argparse.ArgumentParser(description="Train and register the Lab 15 machine-failure model.")
    parser.add_argument("--experiment_name", type=str, default="Machine_Failure_Prediction")
    parser.add_argument("--registered_model_name", type=str, default="MachineFailurePredictor")
    parser.add_argument("--promotion_metric", type=str, default="f1_score")
    parser.add_argument("--promotion_threshold", type=float, default=0.6)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = train_and_track(
        experiment_name=args.experiment_name,
        registered_model_name=args.registered_model_name,
        promotion_metric=args.promotion_metric,
        promotion_threshold=args.promotion_threshold,
    )
    print("Summary:", summary)


if __name__ == "__main__":
    main()
