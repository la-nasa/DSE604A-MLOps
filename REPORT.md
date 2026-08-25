# DSE604A -- MLOps & Industrialization
## Report

**Programme:** Master of Engineering in Data Science and Artificial Intelligence (MEng DSAI)
**Theme:** From Machine Learning Experimentation to Production MLOps



### 1. Introduction

Machine learning models rarely fail because the modeling code is wrong --
they fail because nothing around the model is engineered: no experiment
history, no versioned artifacts, no reproducible packaging, no deployment
pipeline, no way to detect that production has quietly drifted from what was
validated. This project walks a single problem -- predicting machine
failures from sensor telemetry -- through the full MLOps lifecycle: MLflow
experiment tracking and model registry, Git/GitHub-based collaboration, CI
with GitHub Actions, Docker containerization, Kubernetes deployment and
scaling, and production monitoring with drift detection. Labs 1-14 build the
tooling; Lab 15 (the capstone, `src/capstone_train.py`) assembles it into one
coherent industrial pipeline.

### 2. Dataset description

The capstone scenario (Lab 15) is a manufacturing company predicting machine
failures from four sensor readings: **temperature** (°C), **vibration**
(accelerometer amplitude), **pressure** (a nominal-band signal) and
**cumulative operating hours**. No public dataset matches this exact
fictional scenario, so `src/data_processing.py::generate_sensor_data()`
synthesizes one with physically-motivated relationships: failure risk rises
with higher temperature, higher vibration, pressure deviating from its
nominal band, and more accumulated operating hours, combined through a
logistic model with noise. The generator is deterministic (fixed random
seed) and produces ~5,000 rows with a realistic, imbalanced failure rate
(roughly 5-20%), written to `data/sensor_data.csv` on first use.

Earlier labs use two additional datasets to keep each concept isolated:
scikit-learn's built-in **Iris** dataset (Labs 1-2, 4-5 -- fast, dependency-free,
ideal for teaching registry mechanics) and the public **IBM Telco Customer
Churn** dataset (`src/model_promotion.py`, Lab 3 extension -- exercises
categorical preprocessing, multiple model families and fairness metrics).

### 3. Data preprocessing

- **Capstone (sensor data):** `get_sensor_train_test_split()` performs a
  stratified 80/20 split on the four numeric sensor columns; each model
  pipeline (`src/capstone_train.py`) applies `StandardScaler` before the
  classifier, since vibration and operating hours live on very different
  scales.
- **Telco churn (`model_promotion.py`):** categorical columns are one-hot
  encoded and numeric columns standardized via a `ColumnTransformer`;
  `TotalCharges` (shipped as a string with stray spaces) is coerced to
  numeric and rows that fail to parse are dropped; the split is stratified
  on the target to preserve class balance.
- **Iris:** used as-is (already numeric, balanced, no missing values) so the
  early labs can focus on MLflow mechanics rather than data cleaning.

### 4. Model development

Three model families are used across the labs: `LogisticRegression`,
`RandomForestClassifier`, `GradientBoostingClassifier` and `XGBClassifier`.
The capstone (`src/capstone_train.py::MODEL_CONFIGS`) trains two
configurations -- a 200-tree balanced-class-weight Random Forest and a
150-estimator Gradient Boosting model -- inside a `Pipeline` (scaler +
classifier) so preprocessing travels with the model artifact and can't drift
out of sync between training and serving.

### 5. MLflow experiment tracking

Every training run logs parameters, metrics and the model artifact via
`mlflow.log_param` / `mlflow.log_metric` / `mlflow.sklearn.log_model`
(`src/experiment.py`, `src/train.py`, `src/train_register.py`,
`src/capstone_train.py`). Runs are grouped per experiment
(`Iris_Classification`, `Telco_Churn_Promotion`, `Machine_Failure_Prediction`,
`Model_Monitoring`) so comparisons stay meaningful. Start the tracking UI
with:


mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000


### 6. Model comparison

`src/capstone_train.py` trains both configurations in the same experiment
and selects the best one by F1-score (`max(results, key=...)`), printing a
full metrics table (accuracy, precision, recall, F1, ROC-AUC) for both. The
same pattern appears in `src/model_promotion.py`, which additionally gates
promotion on **latency**, **model size** and **fairness disparity** (recall
gap between groups) -- not accuracy alone -- directly answering the Lab 3
discussion question "why model selection should not rely only on accuracy."

### 7. Model Registry

`src/model_registry.py::ModelRegistryManager` wraps the full registry
lifecycle: create, register, transition stage, compare two versions'
metrics, export registry state as JSON, and roll back. `src/train_register.py`
demonstrates the minimal path (register + auto-promote to Staging above an
accuracy threshold); `src/capstone_train.py` reuses the same pattern for the
capstone model, registering it as `MachineFailurePredictor`.

> **Migration note:** stage-based APIs (`transition_model_version_stage`,
> `get_latest_versions(stages=...)`) are deprecated since MLflow 2.9 in favor
> of model aliases/tags, and still work (with a `FutureWarning`) on the
> MLflow 3.15 pinned here. They're used throughout to match the manual's
> terminology (Staging/Production/Archived); a production migration would
> move to `set_registered_model_alias`.

### 8. Docker implementation

`Dockerfile` (root, matches the `docker build -t dse604a-mlflow-app:1.0 .`
command from Lab 8) packages the Flask prediction API on Python 3.11-slim,
running as a non-root user, with a `HEALTHCHECK` against the app's own
`/health` endpoint. `docker/mlflow.Dockerfile` packages a standalone MLflow
tracking server. `docker-compose.yml` wires both together with a shared
`./mlruns` volume so the API can load whatever model the tracking server has
registered. Dependency versions in `docker/requirements.txt` are pinned to
match the training environment (`scikit-learn==1.9.0`, `numpy==2.5.2`) --
serving a model with a different scikit-learn version than the one it was
pickled with is a common, silent source of production breakage.

### 9. Kubernetes deployment

`kubernetes/deployment.yaml` runs 2 replicas of the prediction API with
`MLFLOW_TRACKING_URI`/`MLFLOW_MODEL_URI` environment variables, CPU/memory
requests and limits, and readiness/liveness probes against `/health` (so a
pod that can't yet serve traffic is never sent requests, and a wedged pod
gets restarted automatically). `kubernetes/service.yaml` exposes it as a
`NodePort` Service -- routing through a stable virtual IP and label selector
instead of a Pod's ephemeral IP, so the app keeps working across pod
restarts, rescheduling and rolling updates.

### 10. CI/CD pipeline

`.github/workflows/ci.yml` runs on every push/PR: install dependencies, run
`pytest`, then build both Docker images to catch packaging regressions
before merge. `.github/workflows/cd.yml` triggers on version tags or manual
dispatch: builds and pushes the image to Docker Hub, then deploys to
staging/production with an automatic rollback job if the production deploy
fails. (An earlier duplicate pair of workflow files lived in
`.github/workflow/` -- singular, not the `workflows/` directory GitHub
Actions actually reads -- and had silently never run; it has been merged
into the real `workflows/` directory as part of this project.)

### 11. Monitoring strategy

`src/system_metrics.py::SystemMetricsCollector` exposes Prometheus counters
and gauges: `model_predictions_total`, `model_prediction_errors_total`,
`model_prediction_latency_seconds` (histogram), `model_accuracy`,
`model_f1_score`, plus system-level `system_cpu_usage_percent` /
`system_memory_usage_percent` / `system_disk_usage_percent`. `track_prediction()`
is a context manager for wrapping serving code; `monitor_system_health()`
polls system metrics on an interval and logs warnings past 90% utilization.
`src/tracing.py` adds OpenTelemetry-based distributed tracing and logs
prediction/training/registration events as MLflow artifacts for lineage.

### 12. Drift detection

`src/drift_detection.py` implements the Lab 14 investigation directly. Given
a training vs. production accuracy gap, `diagnose_accuracy_drop()`:

1. Returns `healthy` if the gap is below threshold (default 5 points).
2. Otherwise computes/accepts per-feature **Population Stability Index**
   (`population_stability_index()`) between the training and production
   feature distributions.
3. If any feature's PSI exceeds 0.25, diagnoses **data drift** (input
   distribution shifted -- retrain on recent data).
4. Otherwise diagnoses **concept drift** (the feature/target relationship
   changed even though inputs look stable) and flags that an implementation
   bug should also be ruled out before retraining.

Reproduce the manual's exact scenario (95% -> 87%):

```bash
python -m src.drift_detection --training_accuracy 0.95 --production_accuracy 0.87 --log_to_mlflow
```

### 13. Rollback strategy

`src/rollback_model.py::rollback_model()` implements Lab 5's rollback
scenario: it looks up whichever version is *actually* serving Production via
`get_latest_versions(stages=["Production"])`, promotes the known-good target
version back to Production, and archives the version(s) it replaces --
rather than assuming a fixed version number. It also returns a structured
incident log (timestamp, model, action, archived versions, reason) suitable
for an incident tracker. `src/model_registry.py::ModelRegistryManager.rollback()`
provides the equivalent generic version for any registered model.

### 14. Execution screenshots

Not included here -- capture these on your own machine while running the
commands in `README.md` (MLflow UI showing the runs/registry, `docker ps`
output, `kubectl get pods`/`kubectl get deployment` after scaling, the
GitHub Actions run, `curl` against `/predict`). Every command referenced in
this report and in `README.md` has been executed and verified during
development (see Section 15).

### 15. Conclusion

Every script in this repository was audited and exercised end to end while
preparing this project, and two categories of problems were found and fixed:

- **Environment:** the virtual environment had a corrupted `protobuf`
  install (a botched 4.25.3 -> 6.33.6 upgrade left a broken
  `~rotobuf-6.33.6.dist-info`), and its C extension is incompatible with
  Python 3.14, so `import mlflow` failed outright. Fixed by cleanly
  reinstalling `protobuf==6.33.6` and the matching `opentelemetry-*==1.44.0`
  packages.
- **Code:** `rollback_model.py` archived a hardcoded version number instead
  of the actual current Production version; `requirements.txt` was missing
  packages the code imports (`psutil`, `prometheus_client`, `schedule`,
  `opentelemetry-exporter-otlp-proto-grpc`, `pytest`); `docker/requirements.txt`
  pinned `scikit-learn==1.2.2` against a `1.9.0` training environment (a
  silent model-loading break waiting to happen); the root `Dockerfile`
  referenced a non-existent `src.main` module and a health check on the
  wrong port; `kubernetes/deployment.yaml` had no probes, resources or model
  configuration; a duplicate CI/CD workflow pair sat in `.github/workflow/`
  (singular) where GitHub Actions never reads it; and `.gitignore` was
  silently excluding every notebook (`*.ipynb`) from version control.

With those fixed, `pytest -q` passes 15/15 (`tests/`), both Docker images
build successfully, and the Kubernetes manifests validate. The project now
demonstrates, with working code rather than slideware, the full path from a
single `mlflow.start_run()` call to a monitored, horizontally scalable,
continuously tested production deployment -- and, just as importantly, a
concrete process for deciding *when* that deployment needs to be rolled back
or retrained.
