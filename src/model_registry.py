import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
import json

logger = logging.getLogger(__name__)

class ModelRegistryManager:
    """Manage model registry operations"""
    
    def __init__(self, model_name: str = "Telco_Churn_Model", 
                 registry_uri: Optional[str] = None):
        self.model_name = model_name
        self.client = MlflowClient(registry_uri=registry_uri)
        
    def create_model(self, description: str = None, tags: Dict[str, str] = None) -> bool:
        """Create a new registered model"""
        try:
            self.client.create_registered_model(
                name=self.model_name,
                description=description,
                tags=tags
            )
            logger.info(f"Created model: {self.model_name}")
            return True
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"Model {self.model_name} already exists")
                return False
            logger.error(f"Failed to create model: {e}")
            return False
    
    def register_model(self, run_id: str, model_uri: str, 
                      description: str = None) -> Optional[ModelVersion]:
        """Register a new model version"""
        try:
            model_version = self.client.create_model_version(
                name=self.model_name,
                source=model_uri,
                run_id=run_id,
                description=description
            )
            logger.info(f"Registered model version {model_version.version}")
            return model_version
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None
    
    def transition_stage(self, version: str, stage: str, 
                        archive_existing: bool = True) -> bool:
        """Transition model version to a new stage"""
        try:
            if archive_existing and stage in ["Production", "Staging"]:
                # Archive existing versions in the target stage
                existing_versions = self.get_versions_by_stage(stage)
                for existing in existing_versions:
                    if existing.version != version:
                        self.client.transition_model_version_stage(
                            name=self.model_name,
                            version=existing.version,
                            stage="Archived"
                        )
            
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage=stage
            )
            logger.info(f"Transitioned model version {version} to {stage}")
            return True
        except Exception as e:
            logger.error(f"Failed to transition model: {e}")
            return False
    
    def get_production_model(self) -> Optional[ModelVersion]:
        """Get current production model"""
        try:
            versions = self.client.get_latest_versions(self.model_name, stages=["Production"])
            return versions[0] if versions else None
        except Exception as e:
            logger.error(f"Failed to get production model: {e}")
            return None
    
    def get_model_version(self, version: str) -> Optional[ModelVersion]:
        """Get specific model version"""
        try:
            return self.client.get_model_version(self.model_name, version)
        except Exception as e:
            logger.error(f"Failed to get model version: {e}")
            return None
    
    def get_versions_by_stage(self, stage: str) -> List[ModelVersion]:
        """Get all model versions in a stage"""
        try:
            return self.client.get_latest_versions(self.model_name, stages=[stage])
        except Exception as e:
            logger.error(f"Failed to get versions by stage: {e}")
            return []
    
    def list_versions(self) -> List[ModelVersion]:
        """List all model versions"""
        try:
            return self.client.search_model_versions(f"name='{self.model_name}'")
        except Exception as e:
            logger.error(f"Failed to list versions: {e}")
            return []
    
    def delete_version(self, version: str) -> bool:
        """Delete a model version"""
        try:
            self.client.delete_model_version(self.model_name, version)
            logger.info(f"Deleted model version {version}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete version: {e}")
            return False
    
    def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two model versions"""
        v1 = self.get_model_version(version1)
        v2 = self.get_model_version(version2)
        
        if not v1 or not v2:
            return {}
        
        # Get metrics from associated runs
        run1 = self.client.get_run(v1.run_id)
        run2 = self.client.get_run(v2.run_id)
        
        comparison = {
            'version1': {
                'version': version1,
                'stage': v1.current_stage,
                'metrics': run1.data.metrics,
                'parameters': run1.data.params
            },
            'version2': {
                'version': version2,
                'stage': v2.current_stage,
                'metrics': run2.data.metrics,
                'parameters': run2.data.params
            },
            'metrics_diff': {}
        }
        
        # Calculate metric differences
        all_metrics = set(list(run1.data.metrics.keys()) + list(run2.data.metrics.keys()))
        for metric in all_metrics:
            m1 = run1.data.metrics.get(metric, 0)
            m2 = run2.data.metrics.get(metric, 0)
            comparison['metrics_diff'][metric] = m2 - m1
        
        return comparison
    
    def export_registry_info(self) -> str:
        """Export registry information as JSON"""
        versions = self.list_versions()
        registry_info = []
        
        for version in versions:
            run = self.client.get_run(version.run_id)
            registry_info.append({
                'version': version.version,
                'stage': version.current_stage,
                'status': version.status,
                'run_id': version.run_id,
                'metrics': run.data.metrics,
                'parameters': run.data.params,
                'created_at': version.creation_timestamp,
                'last_updated': version.last_updated_timestamp
            })
        
        return json.dumps(registry_info, indent=2)
    
    def rollback(self, from_stage: str = "Production") -> bool:
        """Rollback to previous version in stage"""
        try:
            # Get versions in the stage
            versions = self.get_versions_by_stage(from_stage)
            if not versions:
                logger.warning(f"No versions found in {from_stage}")
                return False
            
            # Get all versions sorted by version number
            all_versions = sorted(self.list_versions(), 
                                key=lambda x: int(x.version), 
                                reverse=True)
            
            # Find the previous version not in the current stage
            current_version = versions[0].version
            for version in all_versions:
                if int(version.version) < int(current_version):
                    # Rollback to this version
                    logger.info(f"Rolling back from version {current_version} to {version.version}")
                    return self.transition_stage(version.version, from_stage)
            
            logger.warning("No previous version found for rollback")
            return False
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False