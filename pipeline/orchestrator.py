"""
Pipeline Orchestrator

Coordinates the end-to-end automated retraining pipeline including
data collection, training, evaluation, monitoring, and deployment.

Features:
- Multi-stage workflow orchestration
- Conditional execution based on data freshness and performance
- Comprehensive logging and error handling
- Integration with external monitoring and deployment systems
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.config import PipelineConfig
from pipeline.data import DataCollector
from pipeline.training import ModelTrainer
from pipeline.evaluation import ModelEvaluator
from pipeline.monitoring import PerformanceMonitor
from pipeline.deployment import ModelDeployer
from pipeline.dashboards import DashboardGenerator

logger = logging.getLogger(__name__)


class PipelineStatus:
    """Tracks pipeline execution status and metadata."""
    
    def __init__(self, pipeline_name: str, run_id: str):
        self.pipeline_name = pipeline_name
        self.run_id = run_id
        self.status = "initialized"
        self.start_time = None
        self.end_time = None
        self.stage_results: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.total_duration_seconds = 0
        
    def mark_stage_complete(self, stage_name: str, result: Dict[str, Any]):
        """Mark a stage as complete and store its results."""
        self.stage_results[stage_name] = {
            "status": "completed",
            "result": result,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("Stage %s completed successfully", stage_name)
        
    def mark_stage_failed(self, stage_name: str, error: str):
        """Mark a stage as failed and record the error."""
        self.stage_results[stage_name] = {
            "status": "failed",
            "error": error,
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.errors.append(f"{stage_name}: {error}")
        logger.error("Stage %s failed: %s", stage_name, error)
        
    def start(self):
        """Start pipeline execution."""
        self.status = "running"
        self.start_time = datetime.now(timezone.utc)
        logger.info("Pipeline %s starting (run: %s)", self.pipeline_name, self.run_id)
        
    def complete(self):
        """Mark pipeline as completed."""
        self.status = "completed"
        self.end_time = datetime.now(timezone.utc)
        if self.start_time:
            self.total_duration_seconds = (self.end_time - self.start_time).total_seconds()
        logger.info("Pipeline %s completed in %.1f seconds", self.pipeline_name, self.total_duration_seconds)
        
    def abort(self, reason: str):
        """Abort pipeline execution."""
        self.status = "aborted"
        self.errors.append(f"ABORTED: {reason}")
        if not self.end_time and self.start_time:
            self.end_time = datetime.now(timezone.utc)
            self.total_duration_seconds = (self.end_time - self.start_time).total_seconds()
        logger.warning("Pipeline %s aborted: %s", self.pipeline_name, reason)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert status to dictionary for serialization."""
        return {
            "pipeline_name": self.pipeline_name,
            "run_id": self.run_id,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_seconds": self.total_duration_seconds,
            "stage_results": self.stage_results,
            "errors": self.errors,
        }


class RetrainingPipeline:
    """Main pipeline orchestrator for automated model retraining."""
    
    def __init__(self, config: PipelineConfig, artifacts_dir: str = "./pipeline/artifacts"):
        """
        Args:
            config: PipelineConfig instance with all settings
            artifacts_dir: Directory for storing pipeline artifacts
        """
        self.config = config
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_collector = DataCollector(config.data)
        self.trainer = ModelTrainer(config.training, artifacts_dir)
        self.evaluator = ModelEvaluator(config.evaluation, artifacts_dir)
        self.monitor = PerformanceMonitor(config.monitoring, artifacts_dir)
        self.deployer = ModelDeployer(config.deployment, artifacts_dir)
        self.dashboard_gen = DashboardGenerator(config, artifacts_dir)
        
        # Pipeline status tracking
        self.status = None
        self._current_stage = None
        
    def run(self, force_data_collection: bool = False) -> Dict[str, Any]:
        """
        Execute the complete pipeline workflow.
        
        Args:
            force_data_collection: Skip data freshness check and collect new data
            
        Returns:
            dict with pipeline execution results and status
        """
        run_id = f"{self.config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.status = PipelineStatus(self.config.name, run_id)
        
        try:
            self.status.start()
            
            # Stage 1: Data Collection
            if force_data_collection or self._should_collect_data():
                self._run_data_collection()
            else:
                logger.info("Skipping data collection - data is fresh")
                self.status.mark_stage_complete("data_collection", {
                    "skipped": True,
                    "reason": "data_fresh"
                })
            
            # Stage 2: Data Preprocessing
            if "data_collection" in self.status.stage_results and self.status.stage_results["data_collection"].get("result"):
                self._run_data_preprocessing()
            
            # Stage 3: Model Training
            if "preprocessing" in self.status.stage_results and self.status.stage_results["preprocessing"].get("result"):
                self._run_model_training()
            
            # Stage 4: Model Evaluation
            if "training" in self.status.stage_results and self.status.stage_results["training"].get("result"):
                self._run_model_evaluation()
            
            # Stage 5: Decision Logic
            if "evaluation" in self.status.stage_results and self.status.stage_results["evaluation"].get("result"):
                should_deploy = self._should_deploy()
                self.status.mark_stage_complete("decision_logic", {
                    "should_deploy": should_deploy,
                    "reason": self._get_decision_reason()
                })
                
                # Stage 6: Deployment
                if should_deploy:
                    self._run_model_deployment()
            
            # Stage 7: Monitoring Setup
            if "decision_logic" in self.status.stage_results:
                self._setup_monitoring()
            
            # Stage 8: Dashboard Generation
            self._generate_dashboard()
            
            self.status.complete()
            
        except Exception as e:
            self.status.abort(str(e))
            logger.error("Pipeline execution failed: %s", e)
            
        # Save final status
        self._save_pipeline_status()
        
        return {
            "pipeline_name": self.config.name,
            "run_id": self.status.run_id,
            "status": self.status.status,
            "duration_seconds": self.status.total_duration_seconds,
            "stages": self.status.stage_results,
            "errors": self.status.errors,
        }
    
    def _should_collect_data(self) -> bool:
        """Check if new data collection is needed."""
        freshness_check = self.data_collector.check_freshness()
        logger.info("Data freshness check: %s (age: %.1f hours)", freshness_check["is_fresh"], freshness_check.get("age_hours", 0))
        return not freshness_check["is_fresh"]
    
    def _run_data_collection(self):
        """Execute data collection stage."""
        logger.info("Starting data collection stage")
        self._current_stage = "data_collection"
        
        result = self.data_collector.collect()
        if result.get("error"):
            self.status.mark_stage_failed("data_collection", result["error"])
            raise RuntimeError(f"Data collection failed: {result['error']}")
        
        self.status.mark_stage_complete("data_collection", result)
    
    def _run_data_preprocessing(self):
        """Execute data preprocessing stage."""
        logger.info("Starting data preprocessing stage")
        self._current_stage = "preprocessing"
        
        result = self.data_collector.preprocess()
        if result.get("error"):
            self.status.mark_stage_failed("preprocessing", result["error"])
            raise RuntimeError(f"Data preprocessing failed: {result['error']}")
        
        self.status.mark_stage_complete("preprocessing", result)
    
    def _run_model_training(self):
        """Execute model training stage."""
        logger.info("Starting model training stage")
        self._current_stage = "training"
        
        # Get the latest processed data version
        latest_version = self.data_collector.get_latest_version()
        if not latest_version:
            self.status.mark_stage_failed("training", "No processed data available")
            return
        
        # Prepare data splits
        data_splits = {
            "train": latest_version.metadata.get("splits", {}).get("train", {}).get("path"),
            "validation": latest_version.metadata.get("splits", {}).get("validation", {}).get("path"),
            "test": latest_version.metadata.get("splits", {}).get("test", {}).get("path"),
        }
        
        if self.config.training.hyperparameter_tuning:
            result = self.trainer.tune(data_splits)
        else:
            result = self.trainer.train(data_splits)
        
        if result.status == "failed":
            self.status.mark_stage_failed("training", f"Training failed: {result}")
            return
        
        self.status.mark_stage_complete("training", {
            "run_id": result.run_id,
            "model_path": result.model_path,
            "metrics": result.metrics,
            "duration_seconds": result.duration_seconds,
            "hyperparameters": result.hyperparameters,
        })
    
    def _run_model_evaluation(self):
        """Execute model evaluation stage."""
        logger.info("Starting model evaluation stage")
        self._current_stage = "evaluation"
        
        # Get the best training result
        best_training = self.trainer.get_best_run()
        if not best_training:
            self.status.mark_stage_failed("evaluation", "No trained models available")
            return
        
        # Get test data path
        latest_version = self.data_collector.get_latest_version()
        test_data_path = latest_version.metadata.get("splits", {}).get("test", {}).get("path") if latest_version else None
        
        # Get baseline metrics if available
        baseline_metrics = self._get_baseline_metrics()
        
        result = self.evaluator.evaluate(
            model_path=best_training.model_path,
            test_data_path=test_data_path,
            baseline_metrics=baseline_metrics
        )
        
        self.status.mark_stage_complete("evaluation", {
            "report_id": result.report_id,
            "model_id": result.model_id,
            "metrics": result.metrics,
            "comparison": result.comparison,
            "recommendations": result.recommendations,
            "is_better_than_baseline": result.is_better_than_baseline,
        })
    
    def _should_deploy(self) -> bool:
        """Decide whether to deploy the new model based on evaluation results."""
        eval_result = self.status.stage_results["evaluation"]["result"]
        
        # Check if model is better than baseline
        if eval_result.get("is_better_than_baseline") is False:
            return False
        
        # Check performance thresholds
        metrics = eval_result.get("metrics", {})
        primary_metric = self.config.evaluation.primary_metric
        
        if primary_metric in metrics:
            metric_value = metrics[primary_metric]
            if metric_value < 0.7:  # Minimum quality threshold
                logger.info("Model quality too low for deployment (%.3f < 0.7)", metric_value)
                return False
        
        # Check recommendation consensus
        recommendations = eval_result.get("recommendations", [])
        deployment_recommendations = len([r for r in recommendations if "deploy" in r.lower()])
        rejection_recommendations = len([r for r in recommendations if "consider not" in r.lower() or "further tuning" in r])
        
        if deployment_recommendations > rejection_recommendations:
            logger.info("Recommendations support deployment (%d vs %d)", deployment_recommendations, rejection_recommendations)
            return True
        else:
            logger.info("Recommendations do not support deployment (%d vs %d)", deployment_recommendations, rejection_recommendations)
            return False
    
    def _get_decision_reason(self) -> str:
        """Get human-readable reason for deployment decision."""
        eval_result = self.status.stage_results["evaluation"]["result"]
        
        if eval_result.get("is_better_than_baseline") is False:
            return "model does not outperform baseline"
        
        metrics = eval_result.get("metrics", {})
        primary_metric = self.config.evaluation.primary_metric
        
        if primary_metric in metrics:
            metric_value = metrics[primary_metric]
            if metric_value < 0.7:
                return f"insufficient quality ({metric_value:.3f} < 0.7)"
        
        recommendations = eval_result.get("recommendations", [])
        deployment_recommendations = len([r for r in recommendations if "deploy" in r.lower()])
        
        if deployment_recommendations > 0:
            return "recommendations support deployment"
        else:
            return "recommendations inconclusive"
    
    def _run_model_deployment(self):
        """Execute model deployment stage."""
        logger.info("Starting model deployment stage")
        self._current_stage = "deployment"
        
        # Get the best model path
        best_training = self.trainer.get_best_run()
        if not best_training:
            self.status.mark_stage_failed("deployment", "No model to deploy")
            return
        
        result = self.deployer.deploy_model(
            model_path=best_training.model_path,
            model_id=best_training.run_id,
            metadata={
                "metrics": best_training.metrics,
                "hyperparameters": best_training.hyperparameters,
                "training_duration": best_training.duration_seconds,
            }
        )
        
        if result.get("success"):
            self.status.mark_stage_complete("deployment", result)
        else:
            self.status.mark_stage_failed("deployment", result.get("error", "Deployment failed"))
    
    def _setup_monitoring(self):
        """Setup continuous monitoring for deployed model."""
        logger.info("Setting up monitoring")
        self._current_stage = "monitoring_setup"
        
        result = self.monitor.setup_model_monitoring()
        self.status.mark_stage_complete("monitoring_setup", result)
    
    def _generate_dashboard(self):
        """Generate monitoring dashboard."""
        logger.info("Generating dashboard")
        self._current_stage = "dashboard_generation"
        
        dashboard_paths = self.dashboard_gen.generate_dashboard()
        self.status.mark_stage_complete("dashboard_generation", {
            "dashboard_paths": dashboard_paths,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })
    
    def _get_baseline_metrics(self) -> Optional[Dict[str, float]]:
        """Get current production model metrics for comparison."""
        # In production, this would query the production system
        # For now, return reasonable baseline values
        return {
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.82,
            "f1_score": 0.82,
            "roc_auc": 0.88,
        }
    
    def _save_pipeline_status(self):
        """Save pipeline execution status to disk."""
        status_path = self.artifacts_dir / f"pipeline_status_{self.status.run_id}.json"
        status_data = self.status.to_dict()
        
        with open(status_path, "w") as f:
            import json
            json.dump(status_data, f, indent=2, default=str)
        
        logger.info("Pipeline status saved to %s", status_path)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return self.status.to_dict() if self.status else {}
    
    def stop(self):
        """Gracefully stop the pipeline."""
        if self.status and self.status.status == "running":
            self.status.abort("User requested stop")
            self._save_pipeline_status()