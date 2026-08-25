from mlflow.tracking import MlflowClient
import datetime


def rollback_model(model_name, target_version):
    """Roll back `model_name` to `target_version` in Production.

    The version currently serving Production (whichever it is) is archived
    instead of a hardcoded version number, so this works regardless of how
    many promotions/rollbacks have already happened.
    """
    client = MlflowClient()

    current_production = client.get_latest_versions(model_name, stages=["Production"])
    faulty_versions = [mv.version for mv in current_production if mv.version != str(target_version)]

    incident_log = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": model_name,
        "action": "rollback",
        "target_version": target_version,
        "archived_versions": faulty_versions,
        "reason": "Accuracy decrease and latency increase in production",
    }

    # Promote the known-good version back to Production
    client.transition_model_version_stage(
        name=model_name,
        version=target_version,
        stage="Production",
        archive_existing_versions=False,
    )

    # Archive whatever was previously serving Production (the faulty version)
    for version in faulty_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Archived",
        )

    print(f"Rollback to version {target_version} completed")
    print(f"Incident documented: {incident_log}")

    return incident_log


if __name__ == "__main__":
    rollback_model("IrisClassifier", target_version=2)