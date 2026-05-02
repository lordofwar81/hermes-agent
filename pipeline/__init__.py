"""
Automated Retraining Pipeline for Hermes Agent

End-to-end pipeline for automated model retraining, monitoring, and deployment.

Components:
  - data: Data collection, validation, and versioning
  - training: Model training orchestration with hyperparameter tuning
  - evaluation: Automated model evaluation and comparison
  - monitoring: Performance monitoring, drift detection, and alerting
  - deployment: Model deployment automation with rollback support
  - dashboards: Monitoring dashboard generation (HTML/JSON)

Quick start:
  from pipeline import RetrainingPipeline, PipelineConfig

  config = PipelineConfig.from_yaml("pipeline_config.yaml")
  pipeline = RetrainingPipeline(config)
  pipeline.run()
"""

__version__ = "1.0.0"

from pipeline.config import PipelineConfig
from pipeline.orchestrator import RetrainingPipeline

__all__ = ["RetrainingPipeline", "PipelineConfig", "__version__"]
