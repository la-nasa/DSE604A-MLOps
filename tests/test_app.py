import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.app import app  # noqa: E402


def test_health_endpoint_ok_without_a_loaded_model():
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_home_endpoint():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200


def test_predict_without_features_returns_400():
    client = app.test_client()
    response = client.post("/predict", json={})
    assert response.status_code == 400


def test_predict_without_model_returns_503(monkeypatch):
    # No MLFLOW_MODEL_URI is configured and no mlruns/ artifacts are mounted
    # in the test environment, so get_model() should fail gracefully (503),
    # never as an unhandled 500.
    import src.app as app_module

    monkeypatch.setattr(app_module, "get_model", lambda: (_ for _ in ()).throw(RuntimeError("no model")))
    client = app.test_client()
    response = client.post("/predict", json={"features": [1, 2, 3, 4]})
    assert response.status_code == 503
