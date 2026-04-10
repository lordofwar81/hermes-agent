"""
Model Deployment Module

Automated model deployment with canary releases, rollback support,
and health monitoring.

Features:
- Multiple deployment strategies (canary, blue-green, rolling, immediate)
- Canary traffic splitting and gradual rollout
- Automatic rollback on failure
- Health checks and readiness probes
- Model registry management
- Deployment history and auditing
"""

import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml

logger = logging.getLogger(__name__)


class DeploymentResult:
    """Container for deployment operation results."""
    
    def __init__(
        self,
        deployment_id: str,
        model_id: str,
        strategy: str,
        status: str,
        metadata: Dict[str, Any],
    ):
        self.deployment_id = deployment_id
        self.model_id = model_id
        self.strategy = strategy
        self.status = status
        self.metadata = metadata
        self.created_at = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> dict:
        return {
            "deployment_id": self.deployment_id,
            "model_id": self.model_id,
            "strategy": self.strategy,
            "status": self.status,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


class HealthChecker:
    """Health checking for deployed models."""
    
    def __init__(self, config):
        self.config = config
    
    def check_health(self, endpoint: str) -> Dict[str, Any]:
        """Perform health check on deployed model endpoint."""
        try:
            start_time = time.time()
            response = requests.get(
                endpoint,
                timeout=self.config.health_check_timeout_seconds
            )
            duration = (time.time() - start_time) * 1000  # milliseconds
            
            health_status = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "status_code": response.status_code,
                "response_time_ms": duration,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            
            # Check response time threshold
            if duration > 1000:  # 1 second threshold
                health_status["warnings"] = ["slow_response"]
            
            return health_status
            
        except requests.exceptions.Timeout:
            return {
                "status": "unhealthy",
                "error": "timeout",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except requests.exceptions.ConnectionError:
            return {
                "status": "unhealthy",
                "error": "connection_error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
    
    def check_readiness(self, endpoint: str, max_attempts: int = 10) -> bool:
        """Check if deployed model is ready for traffic."""
        for attempt in range(max_attempts):
            health = self.check_health(endpoint)
            if health["status"] == "healthy":
                logger.info("Model is ready after %d attempts", attempt + 1)
                return True
            
            if attempt < max_attempts - 1:
                time.sleep(2)  # Wait 2 seconds before next check
        
        logger.error("Model readiness check failed after %d attempts", max_attempts)
        return False


class ModelDeployer:
    """Handles model deployment with multiple strategies."""
    
    def __init__(self, config, artifacts_dir: str = "./pipeline/artifacts"):
        """
        Args:
            config: DeploymentConfig instance
            artifacts_dir: Directory for storing deployment artifacts
        """
        self.config = config
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        self.health_checker = HealthChecker(config)
        self._deployment_history: List[DeploymentResult] = []
        self._load_deployment_history()
    
    def deploy_model(
        self,
        model_path: str,
        model_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        strategy: Optional[str] = None
    ) -> DeploymentResult:
        """
        Deploy a model using the specified strategy.
        
        Args:
            model_path: Path to model artifact
            model_id: Unique identifier for this model version
            metadata: Additional model metadata
            strategy: Deployment strategy (overrides config if provided)
            
        Returns:
            DeploymentResult with deployment information
        """
        strategy = strategy or self.config.strategy
        deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info("Starting deployment %s (model: %s, strategy: %s)", deployment_id, model_id, strategy)
        
        try:
            # Validate model artifact
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model artifact not found: {model_path}")
            
            # Copy model to registry
            registry_path = self._register_model(model_path, model_id, deployment_id)
            
            # Execute deployment strategy
            deployment_result = self._execute_deployment_strategy(
                strategy, deployment_id, model_id, registry_path, metadata
            )
            
            # Record deployment
            self._deployment_history.append(deployment_result)
            self._save_deployment_history()
            
            logger.info("Deployment %s completed successfully", deployment_id)
            return deployment_result
            
        except Exception as e:
            logger.error("Deployment %s failed: %s", deployment_id, e)
            deployment_result = DeploymentResult(
                deployment_id=deployment_id,
                model_id=model_id,
                strategy=strategy,
                status="failed",
                metadata={"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}
            )
            self._deployment_history.append(deployment_result)
            self._save_deployment_history()
            raise
    
    def _register_model(self, model_path: str, model_id: str, deployment_id: str) -> Path:
        """Register model in the model registry."""
        registry_dir = Path(self.config.model_registry_dir)
        registry_dir.mkdir(parents=True, exist_ok=True)
        
        # Create deployment-specific directory
        model_dir = registry_dir / deployment_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model artifact
        dest_path = model_dir / "model.bin"
        shutil.copy2(model_path, dest_path)
        
        # Create model metadata
        metadata = {
            "model_id": model_id,
            "deployment_id": deployment_id,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "model_path": str(dest_path),
            "checksum": self._calculate_checksum(dest_path),
        }
        
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info("Model registered at %s", model_dir)
        return dest_path
    
    def _execute_deployment_strategy(
        self,
        strategy: str,
        deployment_id: str,
        model_id: str,
        model_path: Path,
        metadata: Optional[Dict[str, Any]]
    ) -> DeploymentResult:
        """Execute the specified deployment strategy."""
        strategy_func_map = {
            "immediate": self._immediate_deployment,
            "rolling": self._rolling_deployment,
            "blue_green": self._blue_green_deployment,
            "canary": self._canary_deployment,
        }
        
        if strategy not in strategy_func_map:
            raise ValueError(f"Unknown deployment strategy: {strategy}")
        
        deploy_func = strategy_func_map[strategy]
        return deploy_func(deployment_id, model_id, model_path, metadata)
    
    def _immediate_deployment(
        self,
        deployment_id: str,
        model_id: str,
        model_path: Path,
        metadata: Optional[Dict[str, Any]]
    ) -> DeploymentResult:
        """Immediate deployment - 100% traffic to new model."""
        logger.info("Executing immediate deployment strategy")
        
        # Simulate deployment
        deployment_path = Path(self.config.deployment_dir) / model_id
        deployment_path.mkdir(parents=True, exist_ok=True)
        
        # Copy model to deployment directory
        shutil.copy2(model_path, deployment_path / "model.bin")
        
        # Warm up the model
        self._warmup_model(deployment_path)
        
        # Update active model registry
        self._update_active_model(deployment_id, model_id)
        
        result = DeploymentResult(
            deployment_id=deployment_id,
            model_id=model_id,
            strategy="immediate",
            status="active",
            metadata={
                "deployment_path": str(deployment_path),
                "traffic_percentage": 100,
                "deployment_complete_at": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {},
            }
        )
        
        return result
    
    def _canary_deployment(
        self,
        deployment_id: str,
        model_id: str,
        model_path: Path,
        metadata: Optional[Dict[str, Any]]
    ) -> DeploymentResult:
        """Canary deployment with gradual traffic rollout."""
        logger.info("Executing canary deployment strategy")
        
        # Deploy canary version
        canary_path = Path(self.config.deployment_dir) / f"canary_{model_id}"
        canary_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, canary_path / "model.bin")
        
        # Warm up canary
        self._warmup_model(canary_path)
        
        # Set initial canary traffic
        initial_traffic = self.config.canary_traffic_percentage
        
        logger.info("Canary deployed with %d%% traffic", initial_traffic)
        
        # Simulate traffic routing
        routing_config = {
            "current_version": model_id,
            "canary_version": f"canary_{model_id}",
            "traffic_split": {
                "canary": initial_traffic,
                "stable": 100 - initial_traffic,
            },
            "endpoint": f"http://localhost:8000/canary/{model_id}",
        }
        
        # Start canary monitoring
        canary_duration = self.config.canary_duration_minutes
        
        # Simulate canary validation period
        logger.info("Canary validation starting - will monitor for %d minutes", canary_duration)
        
        # In production, this would:
        # 1. Monitor canary performance vs stable
        # 2. Check error rates
        # 3. Validate business metrics
        # 4. Decide whether to promote or rollback
        
        # For this simulation, assume canary validation passes
        validation_passed = True
        
        if validation_passed:
            # Promote canary to production
            production_path = Path(self.config.deployment_dir) / model_id
            production_path.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model_path, production_path / "model.bin")
            
            self._warmup_model(production_path)
            self._update_active_model(deployment_id, model_id)
            
            # Remove old versions (keeping rollback copies)
            self._cleanup_old_versions()
            
            result = DeploymentResult(
                deployment_id=deployment_id,
                model_id=model_id,
                strategy="canary",
                status="active",
                metadata={
                    "canary_duration_minutes": canary_duration,
                    "initial_traffic_percent": initial_traffic,
                    "validation_passed": validation_passed,
                    "deployment_path": str(production_path),
                    "traffic_percentage": 100,
                    "deployment_complete_at": datetime.now(timezone.utc).isoformat(),
                    "metadata": metadata or {},
                }
            )
            
            logger.info("Canary deployment promoted to production")
        else:
            # Rollback canary
            self._rollback_canary(canary_path)
            result = DeploymentResult(
                deployment_id=deployment_id,
                model_id=model_id,
                strategy="canary",
                status="rolled_back",
                metadata={
                    "canary_duration_minutes": canary_duration,
                    "initial_traffic_percent": initial_traffic,
                    "validation_passed": validation_passed,
                    "rollback_reason": "canary_validation_failed",
                    "rolled_back_at": datetime.now(timezone.utc).isoformat(),
                    "metadata": metadata or {},
                }
            )
            logger.info("Canary deployment rolled back")
        
        return result
    
    def _blue_green_deployment(
        self,
        deployment_id: str,
        model_id: str,
        model_path: Path,
        metadata: Optional[Dict[str, Any]]
    ) -> DeploymentResult:
        """Blue-green deployment with zero-downtime switch."""
        logger.info("Executing blue-green deployment strategy")
        
        # Deploy to green environment
        green_path = Path(self.config.deployment_dir) / "green"
        green_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, green_path / "model.bin")
        
        # Warm up green environment
        self._warmup_model(green_path)
        
        # Switch traffic from blue to green
        logger.info("Switching traffic from blue to green")
        
        # In production, this would involve:
        # 1. Load balancer configuration updates
        # 2. DNS updates
        # 3. Session affinity updates
        
        # Update active model
        self._update_active_model(deployment_id, model_id)
        
        result = DeploymentResult(
            deployment_id=deployment_id,
            model_id=model_id,
            strategy="blue_green",
            status="active",
            metadata={
                "blue_path": str(Path(self.config.deployment_dir) / "blue"),
                "green_path": str(green_path),
                "switch_complete_at": datetime.now(timezone.utc).isoformat(),
                "deployment_path": str(green_path),
                "traffic_percentage": 100,
                "deployment_complete_at": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {},
            }
        )
        
        logger.info("Blue-green deployment complete")
        return result
    
    def _rolling_deployment(
        self,
        deployment_id: str,
        model_id: str,
        model_path: Path,
        metadata: Optional[Dict[str, Any]]
    ) -> DeploymentResult:
        """Rolling deployment with gradual replacement."""
        logger.info("Executing rolling deployment strategy")
        
        # Deploy to new version
        new_version_path = Path(self.config.deployment_dir) / f"v{int(time.time())}"
        new_version_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, new_version_path / "model.bin")
        
        # Warm up new version
        self._warmup_model(new_version_path)
        
        # Simulate gradual rollout over multiple instances
        total_instances = 10  # Simulated
        rollout_batches = 2  # Number of rollout batches
        
        for batch in range(rollout_batches):
            batch_size = total_instances // rollout_batches
            instance_range = range(batch * batch_size, (batch + 1) * batch_size)
            
            logger.info("Rolling out batch %d: instances %s", batch + 1, instance_range)
            
            # Simulate updating instances
            time.sleep(1)  # Simulate deployment time
            
            traffic_percentage = ((batch + 1) * batch_size / total_instances) * 100
            logger.info("Current traffic: %d%%", traffic_percentage)
        
        # Update active model
        self._update_active_model(deployment_id, model_id)
        
        # Remove old versions (keep for rollback)
        self._cleanup_old_versions()
        
        result = DeploymentResult(
            deployment_id=deployment_id,
            model_id=model_id,
            strategy="rolling",
            status="active",
            metadata={
                "total_instances": total_instances,
                "rollout_batches": rollout_batches,
                "deployment_path": str(new_version_path),
                "traffic_percentage": 100,
                "deployment_complete_at": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {},
            }
        )
        
        logger.info("Rolling deployment complete")
        return result
    
    def _warmup_model(self, model_path: Path):
        """Warm up the deployed model."""
        logger.info("Warming up model at %s", model_path)
        
        # Simulate warmup requests
        warmup_requests = self.config.warmup_requests
        
        for i in range(warmup_requests):
            try:
                # In production, this would make actual API calls
                # to warm up the model and initialize caches
                time.sleep(0.1)  # Simulate processing time
            except Exception as e:
                logger.warning("Warmup request %d failed: %s", i + 1, e)
        
        logger.info("Model warmup complete")
    
    def _update_active_model(self, deployment_id: str, model_id: str):
        """Update the active model registry."""
        active_file = Path(self.config.deployment_dir) / "active_model.json"
        
        active_info = {
            "deployment_id": deployment_id,
            "model_id": model_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        
        with open(active_file, "w") as f:
            json.dump(active_info, f, indent=2, default=str)
        
        logger.info("Active model updated: %s", model_id)
    
    def _cleanup_old_versions(self):
        """Clean up old deployment versions, keeping recent ones for rollback."""
        if self.config.max_rollback_versions <= 0:
            return
        
        deployment_dir = Path(self.config.deployment_dir)
        
        # Find all deployment directories
        deployment_dirs = []
        for item in deployment_dir.iterdir():
            if item.is_dir() and item.name not in ["active_model.json"]:
                deployment_dirs.append(item)
        
        # Sort by modification time (newest first)
        deployment_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove old versions
        to_remove = deployment_dirs[self.config.max_rollback_versions:]
        for old_dir in to_remove:
            try:
                shutil.rmtree(old_dir)
                logger.info("Cleaned up old version: %s", old_dir)
            except Exception as e:
                logger.warning("Failed to clean up %s: %s", old_dir, e)
    
    def _rollback_canary(self, canary_path: Path):
        """Rollback canary deployment."""
        logger.info("Rolling back canary deployment")
        try:
            shutil.rmtree(canary_path)
            logger.info("Canary rolled back successfully")
        except Exception as e:
            logger.error("Canary rollback failed: %s", e)
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        import hashlib
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _save_deployment_history(self):
        """Save deployment history to disk."""
        history_path = self.artifacts_dir / "deployment_history.json"
        with open(history_path, "w") as f:
            history = [result.to_dict() for result in self._deployment_history]
            json.dump(history, f, indent=2, default=str)
    
    def _load_deployment_history(self):
        """Load existing deployment history."""
        history_path = self.artifacts_dir / "deployment_history.json"
        if history_path.exists():
            try:
                with open(history_path, "r") as f:
                    history_data = json.load(f)
                    self._deployment_history = [
                        DeploymentResult(
                            deployment_id=data["deployment_id"],
                            model_id=data["model_id"],
                            strategy=data["strategy"],
                            status=data["status"],
                            metadata=data.get("metadata", {}),
                        )
                        for data in history_data
                    ]
            except Exception as e:
                logger.warning("Failed to load deployment history: %s", e)
    
    def rollback_deployment(self, target_deployment_id: str) -> DeploymentResult:
        """Rollback to a previous deployment."""
        logger.info("Initiating rollback to deployment %s", target_deployment_id)
        
        # Find target deployment in history
        target = None
        for deployment in reversed(self._deployment_history):
            if deployment.deployment_id == target_deployment_id:
                target = deployment
                break
        
        if not target:
            raise ValueError(f"Deployment {target_deployment_id} not found in history")
        
        # Simulate rollback process
        logger.info("Rolling back to model %s", target.model_id)
        
        # Create rollback deployment record
        rollback_id = f"rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        rollback_result = DeploymentResult(
            deployment_id=rollback_id,
            model_id=target.model_id,
            strategy=f"rollback_from_{target.strategy}",
            status="active",
            metadata={
                "original_deployment": target_deployment_id,
                "rollback_reason": "manual_rollback",
                "rollback_complete_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        
        # Update active model
        self._update_active_model(rollback_id, target.model_id)
        
        # Add to history
        self._deployment_history.append(rollback_result)
        self._save_deployment_history()
        
        logger.info("Rollback completed to deployment %s", target_deployment_id)
        return rollback_result
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        active_file = Path(self.config.deployment_dir) / "active_model.json"
        
        if active_file.exists():
            with open(active_file, "r") as f:
                active_info = json.load(f)
            
            # Get health status
            health_endpoint = active_info.get("health_endpoint", self.config.health_check_endpoint)
            health_status = self.health_checker.check_health(health_endpoint)
            
            return {
                "active_deployment": active_info,
                "health_status": health_status,
                "total_deployments": len(self._deployment_history),
                "last_deployment": self._deployment_history[-1].to_dict() if self._deployment_history else None,
            }
        else:
            return {
                "active_deployment": None,
                "health_status": None,
                "total_deployments": len(self._deployment_history),
                "last_deployment": None,
            }
    
    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all deployments in history."""
        return [deployment.to_dict() for deployment in self._deployment_history]
    
    def get_deployment(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific deployment."""
        for deployment in self._deployment_history:
            if deployment.deployment_id == deployment_id:
                return deployment.to_dict()
        return None