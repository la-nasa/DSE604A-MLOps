"""
Configuration module for MLOps pipeline
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MLflowConfig:
    """MLflow configuration"""
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "Telco_Churn_Promotion"
    registry_uri: str = "sqlite:///mlflow.db"
    model_name: str = "Telco_Churn_Model"
    
@dataclass
class PrometheusConfig:
    """Prometheus configuration"""
    host: str = "localhost"
    port: int = 8000
    metrics_path: str = "/metrics"
    collect_interval: int = 15

@dataclass
class DataConfig:
    """Data configuration"""
    url: str = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    test_size: float = 0.2
    random_state: int = 42
    target_column: str = "Churn"
    protected_attribute: str = "gender"

@dataclass
class ModelConfig:
    """Model configuration"""
    promotion_criteria: Dict[str, float] = field(default_factory=lambda: {
        "f1_score_min": 0.65,
        "latency_max_ms": 10,
        "model_size_max_mb": 100,
        "fairness_max_disparity": 0.10,
    })
    models: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "logistic_regression": {
            "max_iter": 1000,
            "random_state": 42
        },
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        },
        "xgboost": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42
        }
    })

@dataclass
class AppConfig:
    """Main application configuration"""
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    prometheus: PrometheusConfig = field(default_factory=PrometheusConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    log_level: str = "INFO"
    environment: str = "development"
    
    def setup(self):
        """Setup configuration"""
        os.environ['MLFLOW_TRACKING_URI'] = self.mlflow.tracking_uri
        logger.info(f"Configuration loaded for {self.environment} environment")

# Singleton config
config = AppConfig()