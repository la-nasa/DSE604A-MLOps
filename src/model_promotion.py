import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
import time
import numpy as np
import os
import tempfile
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load data
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
print(df.shape)
df.head()

# Drop customer ID
df = df.drop('customerID', axis=1)

# Convert TotalCharges to numeric (handling spaces)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])

# Target: Churn -> 1 if Yes, 0 if No
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Split data (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# We will also keep protected groups for fairness evaluation.
# For simplicity, we use 'gender' as protected attribute.
groups_train = X_train['gender'].map({'Female': 0, 'Male': 1})
groups_test = X_test['gender'].map({'Female': 0, 'Male': 1})

# Promotion criteria
promotion_criteria = {
    "f1_score_min": 0.65,
    "latency_max_ms": 10,
    "model_size_max_mb": 100,
    "fairness_max_disparity": 0.10,
}

# MLflow setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")  # local SQLite for simplicity
mlflow.set_experiment("Telco_Churn_Promotion")
client = MlflowClient()

# Fixed evaluate_model function
def evaluate_model(model, X_test, y_test, groups_test, criteria):
    # Predict
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    
    # Latency (average over 100 predictions on first sample)
    start = time.time()
    for _ in range(100):
        model.predict(X_test[:1])
    latency_ms = (time.time() - start) * 1000 / 100  # ms per prediction
    
    # Fairness: recall for each group
    recall_group0 = recall_score(y_test[groups_test == 0], y_pred[groups_test == 0])
    recall_group1 = recall_score(y_test[groups_test == 1], y_pred[groups_test == 1])
    fairness_disparity = abs(recall_group0 - recall_group1)
    
    # Model size - Fixed file handling
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = tmp.name
        # Close the file before joblib writes to it
        tmp.close()
        
        # Save model to the temporary file
        joblib.dump(model, tmp_path)
        
        # Get file size
        model_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        
        # Safely delete the file
        try:
            os.unlink(tmp_path)
        except PermissionError:
            # If can't delete immediately, wait and try again
            time.sleep(0.1)
            try:
                os.unlink(tmp_path)
            except PermissionError:
                print(f"Warning: Could not delete temporary file {tmp_path}")
    
    return {
        "f1_score": f1,
        "latency_ms": latency_ms,
        "model_size_mb": model_size_mb,
        "fairness_disparity": fairness_disparity,
    }

# Logistic Regression
with mlflow.start_run(run_name="logistic_regression") as run:
    # Preprocess and train
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(pipeline, X_test, y_test, groups_test, promotion_criteria)
    
    # Log parameters and metrics
    mlflow.log_param("model_type", "logistic_regression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("f1_score", metrics["f1_score"])
    mlflow.log_metric("latency_ms", metrics["latency_ms"])
    mlflow.log_metric("model_size_mb", metrics["model_size_mb"])
    mlflow.log_metric("fairness_disparity", metrics["fairness_disparity"])
    
    # Log model
    mlflow.sklearn.log_model(pipeline, "model")
    run_id_lr = run.info.run_id

# Random Forest
with mlflow.start_run(run_name="random_forest") as run:
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    metrics = evaluate_model(pipeline, X_test, y_test, groups_test, promotion_criteria)
    
    mlflow.log_param("model_type", "random_forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("f1_score", metrics["f1_score"])
    mlflow.log_metric("latency_ms", metrics["latency_ms"])
    mlflow.log_metric("model_size_mb", metrics["model_size_mb"])
    mlflow.log_metric("fairness_disparity", metrics["fairness_disparity"])
    
    mlflow.sklearn.log_model(pipeline, "model")
    run_id_rf = run.info.run_id

# XGBoost - Fixed logging approach
with mlflow.start_run(run_name="xgboost") as run:
    # Create and train the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    metrics = evaluate_model(pipeline, X_test, y_test, groups_test, promotion_criteria)
    
    mlflow.log_param("model_type", "xgboost")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 6)
    mlflow.log_metric("f1_score", metrics["f1_score"])
    mlflow.log_metric("latency_ms", metrics["latency_ms"])
    mlflow.log_metric("model_size_mb", metrics["model_size_mb"])
    mlflow.log_metric("fairness_disparity", metrics["fairness_disparity"])
    
    # Option 1: Log using pyfunc flavor with custom wrapper
    # Create a custom model class that includes preprocessing
    from mlflow.pyfunc import PythonModel
    import pandas as pd
    
    class XGBoostPipelineWrapper(PythonModel):
        def __init__(self, pipeline):
            self.pipeline = pipeline
        
        def predict(self, context, model_input):
            return self.pipeline.predict(model_input)
    
    # Log the pipeline as a pyfunc model
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=XGBoostPipelineWrapper(pipeline),
        registered_model_name=None
    )
    
    # Option 2 (Alternative): Log just the XGBoost model
    # mlflow.xgboost.log_model(pipeline.named_steps['classifier'], "model")
    
    # Option 3 (Alternative): Log with trusted types
    # from xgboost import XGBClassifier
    # from xgboost.core import Booster
    # mlflow.sklearn.log_model(
    #     pipeline, 
    #     "model",
    #     skops_trusted_types=[XGBClassifier, Booster]
    # )
    
    run_id_xgb = run.info.run_id

# Query all runs from the experiment
experiment = mlflow.get_experiment_by_name("Telco_Churn_Promotion")
runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Display runs
print(runs_df[["run_id", "metrics.f1_score", "metrics.latency_ms", 
               "metrics.model_size_mb", "metrics.fairness_disparity"]])

# Find run with highest F1-score
best_run = runs_df.loc[runs_df["metrics.f1_score"].idxmax()]
best_run_id = best_run["run_id"]
print(f"Best run: {best_run_id}")

# Check if best run passes all criteria
criteria_checks = {
    "f1_score": best_run["metrics.f1_score"] >= promotion_criteria["f1_score_min"],
    "latency_ms": best_run["metrics.latency_ms"] <= promotion_criteria["latency_max_ms"],
    "model_size_mb": best_run["metrics.model_size_mb"] <= promotion_criteria["model_size_max_mb"],
    "fairness_disparity": best_run["metrics.fairness_disparity"] <= promotion_criteria["fairness_max_disparity"],
}

print("Criteria checks:", criteria_checks)

if all(criteria_checks.values()):
    print("Promoting model to Staging and Production")
    
    # Register model if not already registered
    try:
        client.create_registered_model("Telco_Churn_Model")
    except Exception as e:
        print(f"Model already exists or error: {e}")
    
    # Create model version from the run
    model_uri = f"runs:/{best_run_id}/model"
    mv = client.create_model_version(
        name="Telco_Churn_Model",
        source=model_uri,
        run_id=best_run_id
    )
    print(f"Created model version {mv.version}")
    
    # Transition to Staging
    client.transition_model_version_stage(
        name="Telco_Churn_Model",
        version=mv.version,
        stage="Staging"
    )
    
    # Transition to Production (after manual approval, here we do it directly)
    client.transition_model_version_stage(
        name="Telco_Churn_Model",
        version=mv.version,
        stage="Production"
    )
    
    print("Model promoted to Production.")
else:
    print("Best model does not meet all promotion criteria.")