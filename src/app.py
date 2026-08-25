from flask import Flask, request, jsonify
import os
import mlflow
import mlflow.sklearn

app = Flask(__name__)

# Load model at startup if MLFLOW_MODEL_URI is provided; else lazy-load on first predict
MODEL_URI = os.environ.get("MLFLOW_MODEL_URI")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
_MODEL = None


def _load_from_registered_model(name_or_uri: str):
    # support model URI formats like "models:/ModelName/1" or registered model name only
    try:
        if name_or_uri.startswith("models:") or name_or_uri.startswith("mlflow://"):
            uri = name_or_uri.replace("mlflow://", "models:")
            return mlflow.sklearn.load_model(uri)
        # try loading by registered model name (latest production/staging)
        return mlflow.sklearn.load_model(f"models:/{name_or_uri}/Staging")
    except Exception:
        return None


def _scan_local_mlruns():
    # Look for latest model artifact under ./mlruns (useful when running with mounted mlruns)
    import glob
    runs_dir = os.path.abspath(os.environ.get("MLRUNS_DIR", "mlruns"))
    pattern = os.path.join(runs_dir, "**", "artifacts", "*")
    candidates = glob.glob(pattern, recursive=True)
    # prefer skops/pickled artifacts
    for c in sorted(candidates, reverse=True):
        try:
            # mlflow expects a model directory; attempt to load common file names
            for name in ("model", "model.skops", "model.pkl"):
                path = os.path.join(c, name)
                if os.path.exists(path):
                    return mlflow.sklearn.load_model(path)
        except Exception:
            continue
    return None


def get_model():
    global _MODEL
    if _MODEL is None:
        # prefer explicit uri
        if MODEL_URI:
            m = _load_from_registered_model(MODEL_URI)
            if m:
                _MODEL = m
        # try using tracking server + registered model name
        if _MODEL is None and MLFLOW_TRACKING_URI and MODEL_URI:
            try:
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
                m = _load_from_registered_model(MODEL_URI)
                if m:
                    _MODEL = m
            except Exception:
                pass

        # fallback to scanning local mlruns
        if _MODEL is None:
            _MODEL = _scan_local_mlruns()

        if _MODEL is None:
            raise RuntimeError("No model found. Set MLFLOW_MODEL_URI or mount mlruns/ into the container.")
    return _MODEL


@app.route("/")
def home():
    return "MLOps Model API"


@app.route("/health")
def health():
    """Lightweight liveness/readiness probe endpoint (does not force a model load)."""
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    if not payload or "features" not in payload:
        return jsonify({"error": "request must contain 'features' array"}), 400

    features = payload["features"]
    try:
        model = get_model()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503

    try:
        pred = model.predict([features])
        return jsonify({"prediction": int(pred[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
