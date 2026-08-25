import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


def evaluate_model(model_uri: str = None):
    # Use iris test set by default
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_uri:
        try:
            model = mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            print(f"Failed to load model from {model_uri}: {e}")
            return None
    else:
        print("No model_uri provided; cannot evaluate.")
        return None

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    print("Evaluation metrics:", metrics)
    return metrics


if __name__ == "__main__":
    # Example: evaluate a model from a run
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_uri", type=str, help="MLflow model uri, e.g. runs:/<run_id>/model")
    args = parser.parse_args()
    evaluate_model(args.model_uri)
