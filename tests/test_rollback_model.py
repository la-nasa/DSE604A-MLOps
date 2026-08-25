import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyClassifier

from src.rollback_model import rollback_model


def _register_version(client: MlflowClient, model_name: str, experiment_name: str):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        model = DummyClassifier(strategy="most_frequent").fit([[0], [1]], [0, 1])
        mlflow.sklearn.log_model(model, "model")
        run_id = run.info.run_id

    mv = client.create_model_version(name=model_name, source=f"runs:/{run_id}/model", run_id=run_id)
    for _ in range(30):
        mv = client.get_model_version(model_name, mv.version)
        if mv.status == "READY":
            break
    return mv.version


def test_rollback_archives_the_actual_faulty_production_version():
    client = MlflowClient()
    model_name = "CI_RollbackTest_Model"
    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(model_name)

    good_version = _register_version(client, model_name, "CI_Rollback_Test")
    faulty_version = _register_version(client, model_name, "CI_Rollback_Test")

    # Promote the "good" version to Production first, then simulate a bad
    # deploy by promoting the faulty version over it.
    client.transition_model_version_stage(model_name, good_version, "Production", archive_existing_versions=False)
    client.transition_model_version_stage(model_name, faulty_version, "Production", archive_existing_versions=False)

    rollback_model(model_name, target_version=good_version)

    good_mv = client.get_model_version(model_name, good_version)
    faulty_mv = client.get_model_version(model_name, faulty_version)

    assert good_mv.current_stage == "Production"
    assert faulty_mv.current_stage == "Archived"
