"""
Pipeline Configuration

Central configuration for the automated retraining pipeline.
Supports YAML loading, environment variable substitution, and validation.
"""

import os
import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses for typed config sections
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Data collection and validation settings."""
    source_dir: str = "./data/raw"
    processed_dir: str = "./data/processed"
    validation_split: float = 0.15
    test_split: float = 0.10
    min_samples: int = 100
    max_age_hours: int = 168  # 7 days
    quality_threshold: float = 0.85
    deduplication: bool = True
    augmentation: bool = False
    augmentation_factor: int = 2
    supported_formats: List[str] = field(default_factory=lambda: [".json", ".jsonl", ".csv", ".parquet"])
    schema_version: str = "1.0"


@dataclass
class TrainingConfig:
    """Model training settings."""
    model_type: str = "classifier"  # classifier | regressor | ranking
    base_model: str = "hermes-router-v1"
    hyperparameters: Dict[str, Any] = field(default_factory=lambda: {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "early_stopping_patience": 5,
        "weight_decay": 0.01,
    })
    hyperparameter_tuning: bool = True
    tuning_method: str = "bayesian"  # bayesian | grid | random
    tuning_budget: int = 20  # max trials
    cross_validation_folds: int = 5
    gpu_enabled: bool = False
    max_training_time_minutes: int = 120
    checkpoint_every_n_epochs: int = 5
    seed: int = 42


@dataclass
class EvaluationConfig:
    """Model evaluation settings."""
    primary_metric: str = "accuracy"
    secondary_metrics: List[str] = field(default_factory=lambda: [
        "precision", "recall", "f1_score", "roc_auc"
    ])
    regression_metrics: List[str] = field(default_factory=lambda: [
        "mse", "rmse", "mae", "r2"
    ])
    comparison_threshold: float = 0.02  # min improvement to accept new model
    statistical_test: str = "paired_t_test"  # paired_t_test | wilcoxon | mcnemar
    significance_level: float = 0.05
    fairness_evaluation: bool = False
    fairness_metrics: List[str] = field(default_factory=lambda: [
        "demographic_parity", "equalized_odds"
    ])
    calibration_check: bool = True
    ablation_study: bool = False


@dataclass
class MonitoringConfig:
    """Performance monitoring and drift detection settings."""
    enabled: bool = True
    check_interval_minutes: int = 60
    drift_detection_method: str = "psi"  # psi | ks_test | chi_squared | adwin
    drift_threshold: float = 0.15
    performance_degradation_threshold: float = 0.05  # 5% drop triggers retrain
    alert_channels: List[str] = field(default_factory=lambda: ["log"])
    # alert_channels can include: "log", "webhook", "email", "slack"
    webhook_url: Optional[str] = None
    slack_webhook: Optional[str] = None
    email_recipients: List[str] = field(default_factory=list)
    retention_days: int = 90
    baseline_window_days: int = 7


@dataclass
class DeploymentConfig:
    """Model deployment settings."""
    strategy: str = "canary"  # canary | blue_green | rolling | immediate
    canary_traffic_percentage: float = 10.0
    canary_duration_minutes: int = 30
    canary_success_threshold: float = 0.99
    max_rollback_versions: int = 5
    health_check_endpoint: str = "/health"
    health_check_timeout_seconds: int = 30
    model_registry_dir: str = "./models/registry"
    deployment_dir: str = "./models/deployed"
    warmup_requests: int = 10
    auto_rollback_on_failure: bool = True


@dataclass
class ScheduleConfig:
    """Pipeline scheduling settings."""
    mode: str = "drift_triggered"  # drift_triggered | scheduled | hybrid
    cron_expression: Optional[str] = None  # e.g. "0 2 * * 0" = weekly Sunday 2am
    min_retrain_interval_hours: int = 24
    max_retrain_interval_hours: int = 168  # force retrain after 1 week max
    cooldown_after_deployment_minutes: int = 60


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    name: str = "hermes-retraining-pipeline"
    version: str = "1.0.0"
    environment: str = "development"  # development | staging | production
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    logging_level: str = "INFO"
    artifacts_dir: str = "./pipeline/artifacts"

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            logger.warning("Config file %s not found, using defaults", path)
            return cls()

        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        return cls._from_dict(raw)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        """Create config from a dictionary."""
        return cls._from_dict(d)

    @classmethod
    def _from_dict(cls, d: dict) -> "PipelineConfig":
        """Internal: map a flat dict into the nested dataclass structure."""
        data_cfg = DataConfig(**d.pop("data", {}))
        train_cfg = TrainingConfig(**d.pop("training", {}))
        eval_cfg = EvaluationConfig(**d.pop("evaluation", {}))
        mon_cfg = MonitoringConfig(**d.pop("monitoring", {}))
        deploy_cfg = DeploymentConfig(**d.pop("deployment", {}))
        sched_cfg = ScheduleConfig(**d.pop("schedule", {}))

        # Remaining top-level keys
        return cls(
            data=data_cfg,
            training=train_cfg,
            evaluation=eval_cfg,
            monitoring=mon_cfg,
            deployment=deploy_cfg,
            schedule=sched_cfg,
            **{k: v for k, v in d.items() if k in cls.__dataclass_fields__},
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize config to a plain dict."""
        import dataclasses
        return dataclasses.asdict(self)

    def to_yaml(self, path: str) -> None:
        """Write config to a YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate config values. Returns list of warnings/errors."""
        issues = []

        if not 0 < self.data.validation_split < 1:
            issues.append("data.validation_split must be in (0, 1)")
        if not 0 < self.data.test_split < 1:
            issues.append("data.test_split must be in (0, 1)")
        if self.data.validation_split + self.data.test_split >= 1.0:
            issues.append("validation_split + test_split must be < 1.0")
        if self.data.min_samples < 10:
            issues.append("data.min_samples should be >= 10 for meaningful training")
        if self.training.epochs < 1:
            issues.append("training.epochs must be >= 1")
        if self.training.learning_rate <= 0:
            issues.append("training.learning_rate must be > 0")
        if self.deployment.canary_traffic_percentage <= 0 or self.deployment.canary_traffic_percentage > 100:
            issues.append("deployment.canary_traffic_percentage must be in (0, 100]")
        if self.monitoring.drift_threshold <= 0:
            issues.append("monitoring.drift_threshold must be > 0")

        return issues
