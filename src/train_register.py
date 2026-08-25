import argparse
import time
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_and_register(n_estimators: int, max_depth: int, registered_model_name: str, stage_threshold: float, experiment_name: str):
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

        mlflow.log_param("algorithm", "RandomForest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", 42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "model")

        run_id = run.info.run_id
        print(f"Run completed: {run_id}")

    model_uri = f"runs:/{run_id}/model"
    result = {
        "run_id": run_id,
        "registered_model_name": None,
        "version": None,
        "status": None,
        "metrics": {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1},
    }

    try:
        client = MlflowClient()
        try:
            client.get_registered_model(registered_model_name)
        except Exception:
            client.create_registered_model(registered_model_name)

        mv = client.create_model_version(name=registered_model_name, source=model_uri, run_id=run_id)
        version = mv.version

        for _ in range(30):
            mv = client.get_model_version(registered_model_name, version)
            if mv.status == "READY":
                break
            time.sleep(1)

        print(f"Registered model '{registered_model_name}' version {version} status={mv.status}")

        if accuracy >= stage_threshold:
            client.transition_model_version_stage(registered_model_name, version, "Staging", archive_existing_versions=False)
            print(f"Model version {version} transitioned to Staging (accuracy={accuracy:.4f} >= {stage_threshold}).")
        else:
            print(f"Model accuracy {accuracy:.4f} below threshold {stage_threshold}; not promoted.")

        result.update({
            "registered_model_name": registered_model_name,
            "version": version,
            "status": mv.status,
        })
    except Exception as e:
        # Model Registry may be unavailable for local file-store tracking servers.
        print(f"Model Registry unavailable or failed: {e}")
        print(f"Model is logged to run: {run_id} with uri {model_uri}")

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Train RandomForest, log to MLflow, register and optionally stage.")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--registered_model_name", type=str, default="RetailForecast_RF")
    parser.add_argument("--stage_threshold", type=float, default=0.90, help="Accuracy threshold to promote model to Staging")
    parser.add_argument("--experiment_name", type=str, default="Retail_Demand_Experiment")
    return parser.parse_args()


def main():
    args = parse_args()
    result = train_and_register(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        registered_model_name=args.registered_model_name,
        stage_threshold=args.stage_threshold,
        experiment_name=args.experiment_name,
    )

    print("Result:", result)


if __name__ == "__main__":
    main()
