# DSE604A -- MLOps & Industrialization

Complete practical lab project: taking a machine learning experiment from
notebook to a reproducible, versioned, containerized, continuously tested,
scalable and monitored production system.

**Program:** Master of Engineering in Data Science and Artificial Intelligence (MEng DSAI)
**Theme:** From Machine Learning Experimentation to Production MLOps

This repository implements all 15 labs from the course manual
(`DSE604A_MLOps_Industrialization_Complete_Laboratory_Manual-1.docx`), end to
end: MLflow experiment tracking and model registry, Git/GitHub workflow, CI
with GitHub Actions, Docker packaging, Kubernetes deployment and scaling,
monitoring, drift detection, and a capstone predictive-maintenance project.

## Project layout

```
DSE604A_MLOps/
|-- data/                    sensor_data.csv (generated on first run, Lab 15)
|-- notebooks/                exploratory analysis
|-- src/
|   |-- experiment.py         Lab 1  - first MLflow experiment
|   |-- train.py               Lab 2  - real experiment (Iris, RandomForest)
|   |-- train_register.py      Lab 4  - train, register, auto-stage in MLflow Model Registry
|   |-- promote_model.py       Lab 5  - promotion rule (accuracy threshold)
|   |-- rollback_model.py      Lab 5  - rollback + incident documentation
|   |-- model_registry.py      Lab 4/5 - reusable registry manager (create/promote/compare/rollback)
|   |-- model_promotion.py     Lab 3  - multi-model comparison + fairness/latency/size gating (Telco churn)
|   |-- evaluate.py            shared evaluation helper
|   |-- data_processing.py     Lab 15 - sensor dataset loader/generator
|   |-- capstone_train.py      Lab 15 - full capstone training/registration pipeline
|   |-- drift_detection.py     Lab 14 - PSI-based drift detection + diagnosis
|   |-- system_metrics.py      Lab 14 - Prometheus metrics + system/model monitoring
|   |-- tracing.py             observability / OpenTelemetry tracing helper
|   |-- app.py                 Lab 9  - Flask prediction API
|   `-- config.py              central configuration
|-- tests/                    pytest suite covering every module above
|-- docker/                   Dockerfile, mlflow.Dockerfile, pinned requirements
|-- Dockerfile                Lab 8/9 - prediction API image (root, matches manual naming)
|-- docker-compose.yml        mlflow server + prediction API, wired together
|-- kubernetes/               Lab 10/11/12 - Deployment + Service manifests
|-- .github/workflows/        Lab 7/13 - CI (test + docker build) and CD (build/push/deploy)
`-- requirements.txt          pinned dependencies (kept in sync with docker/requirements.txt)
```

## 1. Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> **Note (Windows / Python 3.14):** this project pins `protobuf==6.33.6` and
> matching `opentelemetry-*==1.44.0` packages. Older protobuf releases ship a
> C extension that is incompatible with Python 3.14
> (`TypeError: Metaclasses with custom tp_new are not supported`), which
> breaks every `import mlflow`. If you ever see that error, reinstall protobuf:
> `pip install --upgrade protobuf==6.33.6 opentelemetry-api==1.44.0 opentelemetry-sdk==1.44.0 opentelemetry-proto==1.44.0`

## 2. Run the tests

```bash
pytest -q
```

The suite (`tests/conftest.py`) points MLflow at a throwaway SQLite store for
the duration of the run, so it never pollutes your real `mlruns/` /
`mlflow.db`.

## 3. MLflow tracking server (Labs 1-5)

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open http://localhost:5000, then run any lab script:

```bash
python src/experiment.py           # Lab 1
python src/train.py                # Lab 2
python src/train_register.py       # Lab 4 (train + register + auto-stage)
python src/promote_model.py        # Lab 5 (promote by accuracy threshold)
python src/rollback_model.py       # Lab 5 (rollback + incident log)
```

## 4. Capstone pipeline (Lab 15)

```bash
python -m src.data_processing   # generates data/sensor_data.csv on first run
python -m src.capstone_train    # trains RandomForest + GradientBoosting, tracks both,
                                 # registers the best (by F1) as "MachineFailurePredictor"
```

## 5. Monitoring & drift detection (Lab 14)

```bash
python -m src.drift_detection --training_accuracy 0.95 --production_accuracy 0.87
```

Classifies the drop as `healthy`, `data_drift` or `concept_drift` based on
per-feature Population Stability Index (PSI), and can log the report back to
MLflow with `--log_to_mlflow`. `src/system_metrics.py` exposes the
Prometheus counters/gauges (`model_predictions_total`,
`model_prediction_latency_seconds`, `model_accuracy`, `system_cpu_usage_percent`,
...) used to feed a Grafana/Prometheus stack in production.

## 6. Docker (Lab 8/9)

```bash
docker compose up --build
```

This starts:
- `mlflow` on http://localhost:5000 (backed by the shared `./mlruns` volume)
- `app` (the prediction API) on http://localhost:8081

Or build/run the prediction API directly:

```bash
docker build -t dse604a-mlflow-app:1.0 .
docker run -p 8080:8080 -e MLFLOW_MODEL_URI=IrisClassifier -v "${PWD}/mlruns:/app/mlruns" dse604a-mlflow-app:1.0

curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" \
     -d '{"features":[5.1,3.5,1.4,0.2]}'
```

## 7. Kubernetes (Lab 10/11/12)

```bash
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl get pods
kubectl scale deployment ml-model --replicas=5
minikube service ml-model-service
```

The Deployment ships with liveness/readiness probes against `/health`,
resource requests/limits, and `MLFLOW_MODEL_URI`/`MLFLOW_TRACKING_URI`
environment variables so it can actually serve a registered model rather
than failing every prediction with no model loaded.

## 8. CI/CD (Lab 7/13)

- `.github/workflows/ci.yml` -- runs `pytest` on every push/PR, then builds
  both Docker images (prediction API + MLflow server) to catch packaging
  regressions.
- `.github/workflows/cd.yml` -- on version tags (`v*`) or manual dispatch,
  builds and pushes the image to Docker Hub (`DOCKER_USERNAME`/
  `DOCKER_PASSWORD` repo secrets), then deploys to staging/production.

## Known, intentional simplifications

- `data/sensor_data.csv` is synthetically generated (`src/data_processing.py`)
  since the manual's capstone describes a fictional manufacturer; the
  generator encodes realistic failure drivers (temperature, vibration,
  pressure deviation, operating hours) so the pipeline behaves like a real
  predictive-maintenance problem.
- `deploy-staging`/`deploy-production` steps in `cd.yml` are illustrative
  (`kubectl set image ...` left commented) -- wire them to your actual
  cluster/namespace before relying on them.
- MLflow's stage-based registry API (`transition_model_version_stage`,
  `get_latest_versions`) is used throughout to match the manual; MLflow has
  deprecated it in favor of aliases since 2.9 (still functional in 3.15, with
  a `FutureWarning`). See `REPORT.md` for the migration note.
