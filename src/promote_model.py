import mlflow
from mlflow.tracking import MlflowClient

def promote_model(model_name, version, accuracy_threshold=0.90):
    client = MlflowClient()
    
    # Récupérer les métriques du modèle
    model_version = client.get_model_version(model_name, version)
    run_id = model_version.run_id
    run = client.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy", 0)
    
    if accuracy >= accuracy_threshold:
        # Promouvoir en production
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        print(f"Model {model_name} v{version} approved for production")
        print(f"Accuracy: {accuracy:.3f}")
    else:
        print(f"Model rejected. Accuracy {accuracy:.3f} below threshold {accuracy_threshold}")

if __name__ == "__main__":
    promote_model("IrisClassifier", version=2, accuracy_threshold=0.90)