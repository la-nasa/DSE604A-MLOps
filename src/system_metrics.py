"""
System metrics collection and monitoring module
"""
import time
import psutil
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
import mlflow
from mlflow.tracking import MlflowClient
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Prometheus metrics
MODEL_PREDICTIONS = Counter(
    'model_predictions_total',
    'Total number of model predictions',
    ['model_name', 'model_version']
)

MODEL_PREDICTION_ERRORS = Counter(
    'model_prediction_errors_total',
    'Total number of model prediction errors',
    ['model_name', 'model_version']
)

MODEL_LATENCY = Histogram(
    'model_prediction_latency_seconds',
    'Model prediction latency in seconds',
    ['model_name', 'model_version'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

MODEL_ACCURACY = Gauge(
    'model_accuracy',
    'Model accuracy metric',
    ['model_name', 'model_version']
)

MODEL_F1_SCORE = Gauge(
    'model_f1_score',
    'Model F1 score metric',
    ['model_name', 'model_version']
)

SYSTEM_CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage'
)

SYSTEM_MEMORY_USAGE = Gauge(
    'system_memory_usage_percent',
    'System memory usage percentage'
)

SYSTEM_DISK_USAGE = Gauge(
    'system_disk_usage_percent',
    'System disk usage percentage'
)

MODEL_TRAINING_DURATION = Summary(
    'model_training_duration_seconds',
    'Model training duration in seconds',
    ['model_name']
)

class SystemMetricsCollector:
    """Collect system and model metrics"""
    
    def __init__(self, model_name: str = "Telco_Churn_Model"):
        self.model_name = model_name
        self.mlflow_client = MlflowClient()
        
    def start_prometheus_server(self, port: int = 8000):
        """Start Prometheus metrics server"""
        try:
            start_http_server(port)
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
    
    def collect_system_metrics(self):
        """Collect system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            SYSTEM_CPU_USAGE.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            SYSTEM_MEMORY_USAGE.set(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            SYSTEM_DISK_USAGE.set(disk.percent)
            
            metrics = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent
            }
            
            # Log to MLflow
            mlflow.log_metrics({
                'system_cpu_percent': cpu_percent,
                'system_memory_percent': memory.percent,
                'system_disk_percent': disk.percent
            })
            
            return metrics
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    @contextmanager
    def track_prediction(self, model_version: str = "latest"):
        """Context manager to track prediction metrics"""
        start_time = time.time()
        try:
            yield
            MODEL_PREDICTIONS.labels(
                model_name=self.model_name,
                model_version=model_version
            ).inc()
            
            latency = time.time() - start_time
            MODEL_LATENCY.labels(
                model_name=self.model_name,
                model_version=model_version
            ).observe(latency)
            
        except Exception as e:
            MODEL_PREDICTION_ERRORS.labels(
                model_name=self.model_name,
                model_version=model_version
            ).inc()
            logger.error(f"Prediction error: {e}")
            raise
    
    def update_model_metrics(self, metrics: Dict[str, float], model_version: str = "latest"):
        """Update model performance metrics"""
        if 'accuracy' in metrics:
            MODEL_ACCURACY.labels(
                model_name=self.model_name,
                model_version=model_version
            ).set(metrics['accuracy'])
            
        if 'f1_score' in metrics:
            MODEL_F1_SCORE.labels(
                model_name=self.model_name,
                model_version=model_version
            ).set(metrics['f1_score'])
    
    def track_training(self, model_name: str):
        """Decorator to track training duration"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    MODEL_TRAINING_DURATION.labels(
                        model_name=model_name
                    ).observe(duration)
                    logger.info(f"Training completed in {duration:.2f} seconds")
                    return result
                except Exception as e:
                    logger.error(f"Training failed: {e}")
                    raise
            return wrapper
        return decorator
    
    def monitor_system_health(self, interval: int = 60):
        """Continuously monitor system health"""
        import threading
        import schedule
        
        def job():
            metrics = self.collect_system_metrics()
            logger.info(f"System metrics: {metrics}")
            
            # Check thresholds
            if metrics.get('cpu_percent', 0) > 90:
                logger.warning("High CPU usage detected!")
            if metrics.get('memory_percent', 0) > 90:
                logger.warning("High memory usage detected!")
            if metrics.get('disk_percent', 0) > 90:
                logger.warning("High disk usage detected!")
        
        schedule.every(interval).seconds.do(job)
        
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(1)
        
        thread = threading.Thread(target=run_scheduler, daemon=True)
        thread.start()
        logger.info(f"System monitoring started (interval: {interval}s)")
        
        return thread

class PerformanceProfiler:
    """Profile model performance"""
    
    def __init__(self):
        self.measurements = {}
    
    def measure_prediction_time(self, model, X_sample, n_iterations: int = 100) -> float:
        """Measure average prediction time"""
        # Warm-up
        model.predict(X_sample)
        
        start_time = time.time()
        for _ in range(n_iterations):
            model.predict(X_sample)
        total_time = time.time() - start_time
        
        avg_time_ms = (total_time * 1000) / n_iterations
        self.measurements['avg_prediction_time_ms'] = avg_time_ms
        
        return avg_time_ms
    
    def measure_memory_usage(self) -> Dict[str, float]:
        """Measure memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        measurements = {
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024),
            'percent': process.memory_percent()
        }
        
        self.measurements.update(measurements)
        return measurements
    
    def profile_model(self, model, X_sample) -> Dict[str, Any]:
        """Complete model profiling"""
        profile = {
            'prediction_time': self.measure_prediction_time(model, X_sample),
            'memory_usage': self.measure_memory_usage()
        }
        
        logger.info(f"Model profiling: {profile}")
        return profile