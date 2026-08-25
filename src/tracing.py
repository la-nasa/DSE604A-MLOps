"""
Distributed tracing and observability module
"""
import logging
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import mlflow
from mlflow.tracking import MlflowClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)

# Setup OpenTelemetry
resource = Resource.create({"service.name": "telco-churn-model"})
provider = TracerProvider(resource=resource)
trace.set_tracer_provider(provider)

# Configure OTLP exporter (can be disabled if no collector is available)
try:
    otlp_exporter = OTLPSpanExporter(endpoint="localhost:4317", insecure=True)
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
except Exception as e:
    logger.warning(f"OTLP exporter not configured: {e}")

tracer = trace.get_tracer(__name__)

@dataclass
class TraceEvent:
    """Trace event data"""
    timestamp: datetime
    event_type: str
    data: Dict[str, Any]
    trace_id: str
    span_id: str

class ModelTracer:
    """Model tracing and observability"""
    
    def __init__(self, model_name: str = "Telco_Churn_Model"):
        self.model_name = model_name
        self.mlflow_client = MlflowClient()
        self.events: List[TraceEvent] = []
    
    def start_span(self, span_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Start a new trace span"""
        span = tracer.start_span(span_name)
        if attributes:
            span.set_attributes(attributes)
        return span
    
    def log_model_prediction(self, input_data: Dict[str, Any], 
                            prediction: Any, 
                            confidence: Optional[float] = None,
                            trace_id: Optional[str] = None):
        """Log model prediction for tracing"""
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        
        event_data = {
            'trace_id': trace_id,
            'model_name': self.model_name,
            'input': input_data,
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Log to MLflow
        try:
            mlflow.log_dict(event_data, f"traces/prediction_{trace_id}.json")
        except Exception as e:
            logger.error(f"Failed to log trace to MLflow: {e}")
        
        # Store in memory
        self.events.append(TraceEvent(
            timestamp=datetime.utcnow(),
            event_type='prediction',
            data=event_data,
            trace_id=trace_id,
            span_id=str(uuid.uuid4())
        ))
        
        return trace_id
    
    def log_training_run(self, model_type: str, 
                        parameters: Dict[str, Any],
                        metrics: Dict[str, float],
                        trace_id: Optional[str] = None):
        """Log training run for tracing"""
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        
        event_data = {
            'trace_id': trace_id,
            'model_type': model_type,
            'parameters': parameters,
            'metrics': metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Log to MLflow
        try:
            mlflow.log_dict(event_data, f"traces/training_{trace_id}.json")
        except Exception as e:
            logger.error(f"Failed to log training trace: {e}")
        
        self.events.append(TraceEvent(
            timestamp=datetime.utcnow(),
            event_type='training',
            data=event_data,
            trace_id=trace_id,
            span_id=str(uuid.uuid4())
        ))
        
        return trace_id
    
    def log_model_registration(self, model_version: str, 
                              stage: str,
                              trace_id: Optional[str] = None):
        """Log model registration events"""
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        
        event_data = {
            'trace_id': trace_id,
            'model_name': self.model_name,
            'model_version': model_version,
            'stage': stage,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            mlflow.log_dict(event_data, f"traces/registration_{trace_id}.json")
        except Exception as e:
            logger.error(f"Failed to log registration trace: {e}")
        
        self.events.append(TraceEvent(
            timestamp=datetime.utcnow(),
            event_type='registration',
            data=event_data,
            trace_id=trace_id,
            span_id=str(uuid.uuid4())
        ))
        
        return trace_id
    
    def export_traces(self, format: str = 'json') -> str:
        """Export traces to file"""
        if format == 'json':
            trace_data = [event.data for event in self.events]
            return json.dumps(trace_data, indent=2)
        elif format == 'csv':
            import csv
            import io
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=['timestamp', 'event_type', 'trace_id', 'data'])
            writer.writeheader()
            for event in self.events:
                writer.writerow({
                    'timestamp': event.timestamp,
                    'event_type': event.event_type,
                    'trace_id': event.trace_id,
                    'data': json.dumps(event.data)
                })
            return output.getvalue()
    
    def get_recent_events(self, minutes: int = 60) -> List[TraceEvent]:
        """Get recent trace events"""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        return [event for event in self.events if event.timestamp > cutoff]

class MLflowTracer:
    """MLflow-specific tracing integration"""
    
    def __init__(self):
        self.client = MlflowClient()
    
    def log_artifact_trace(self, run_id: str, artifact_name: str, data: Any):
        """Log artifact for tracing"""
        with mlflow.start_run(run_id=run_id):
            mlflow.log_dict(data, f"traces/{artifact_name}.json")
    
    def get_run_trace(self, run_id: str) -> Dict[str, Any]:
        """Get complete trace for a run"""
        run = self.client.get_run(run_id)
        trace_data = {
            'run_id': run_id,
            'run_name': run.data.tags.get('mlflow.runName', ''),
            'status': run.info.status,
            'start_time': run.info.start_time,
            'end_time': run.info.end_time,
            'parameters': run.data.params,
            'metrics': run.data.metrics,
            'tags': run.data.tags
        }
        return trace_data
    
    def trace_model_lineage(self, model_name: str) -> List[Dict[str, Any]]:
        """Trace model lineage"""
        lineage = []
        versions = self.client.search_model_versions(f"name='{model_name}'")
        
        for version in versions:
            version_info = {
                'version': version.version,
                'stage': version.current_stage,
                'status': version.status,
                'run_id': version.run_id,
                'created_at': version.creation_timestamp,
                'last_updated': version.last_updated_timestamp
            }
            lineage.append(version_info)
        
        return sorted(lineage, key=lambda x: x['version'])