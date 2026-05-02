"""
Tests for pipeline.config module.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from pipeline.config import (
    DataConfig,
    TrainingConfig,
    EvaluationConfig,
    MonitoringConfig,
    DeploymentConfig,
    ScheduleConfig,
    PipelineConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_yaml_path(tmp_path):
    """Return a path to a temporary YAML file."""
    return str(tmp_path / "pipeline_config.yaml")


@pytest.fixture
def valid_config_dict():
    """Return a valid pipeline config dictionary."""
    return {
        "name": "test-pipeline",
        "version": "2.0.0",
        "environment": "staging",
        "data": {
            "source_dir": "/data/raw",
            "processed_dir": "/data/processed",
            "validation_split": 0.15,
            "test_split": 0.10,
            "min_samples": 50,
        },
        "training": {
            "model_type": "classifier",
            "base_model": "test-model",
            "hyperparameters": {"learning_rate": 0.01, "epochs": 10},
            "hyperparameter_tuning": False,
            "epochs": 10,
        },
        "evaluation": {
            "primary_metric": "accuracy",
            "comparison_threshold": 0.03,
        },
        "monitoring": {
            "enabled": True,
            "drift_threshold": 0.2,
        },
        "deployment": {
            "strategy": "immediate",
            "canary_traffic_percentage": 20.0,
        },
        "schedule": {
            "mode": "scheduled",
            "cron_expression": "0 2 * * 0",
        },
    }


# ---------------------------------------------------------------------------
# DataConfig defaults
# ---------------------------------------------------------------------------

class TestDataConfig:
    def test_defaults(self):
        cfg = DataConfig()
        assert cfg.source_dir == "./data/raw"
        assert cfg.processed_dir == "./data/processed"
        assert cfg.validation_split == 0.15
        assert cfg.test_split == 0.10
        assert cfg.min_samples == 100
        assert cfg.deduplication is True
        assert cfg.augmentation is False
        assert ".json" in cfg.supported_formats

    def test_custom_values(self):
        cfg = DataConfig(source_dir="/custom", min_samples=500)
        assert cfg.source_dir == "/custom"
        assert cfg.min_samples == 500


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.model_type == "classifier"
        assert cfg.hyperparameter_tuning is True
        assert cfg.tuning_method == "bayesian"
        assert cfg.seed == 42

    def test_custom_hyperparameters(self):
        hp = {"learning_rate": 0.1}
        cfg = TrainingConfig(hyperparameters=hp)
        assert cfg.hyperparameters["learning_rate"] == 0.1


class TestEvaluationConfig:
    def test_defaults(self):
        cfg = EvaluationConfig()
        assert cfg.primary_metric == "accuracy"
        assert cfg.comparison_threshold == 0.02
        assert cfg.significance_level == 0.05
        assert cfg.calibration_check is True

    def test_custom_metrics(self):
        cfg = EvaluationConfig(primary_metric="f1_score")
        assert cfg.primary_metric == "f1_score"


class TestMonitoringConfig:
    def test_defaults(self):
        cfg = MonitoringConfig()
        assert cfg.enabled is True
        assert cfg.drift_detection_method == "psi"
        assert cfg.drift_threshold == 0.15
        assert cfg.alert_channels == ["log"]
        assert cfg.webhook_url is None

    def test_custom_channels(self):
        cfg = MonitoringConfig(alert_channels=["log", "webhook"], webhook_url="http://example.com")
        assert "webhook" in cfg.alert_channels


class TestDeploymentConfig:
    def test_defaults(self):
        cfg = DeploymentConfig()
        assert cfg.strategy == "canary"
        assert cfg.canary_traffic_percentage == 10.0
        assert cfg.auto_rollback_on_failure is True
        assert cfg.warmup_requests == 10

    def test_custom_strategy(self):
        cfg = DeploymentConfig(strategy="blue_green")
        assert cfg.strategy == "blue_green"


class TestScheduleConfig:
    def test_defaults(self):
        cfg = ScheduleConfig()
        assert cfg.mode == "drift_triggered"
        assert cfg.cron_expression is None
        assert cfg.min_retrain_interval_hours == 24


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------

class TestPipelineConfigDefaults:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.name == "hermes-retraining-pipeline"
        assert cfg.version == "1.0.0"
        assert cfg.environment == "development"
        assert isinstance(cfg.data, DataConfig)
        assert isinstance(cfg.training, TrainingConfig)
        assert isinstance(cfg.evaluation, EvaluationConfig)
        assert isinstance(cfg.monitoring, MonitoringConfig)
        assert isinstance(cfg.deployment, DeploymentConfig)
        assert isinstance(cfg.schedule, ScheduleConfig)

    def test_nested_data_access(self):
        cfg = PipelineConfig()
        assert cfg.data.source_dir == "./data/raw"
        assert cfg.training.model_type == "classifier"


class TestPipelineConfigFromDict:
    def test_from_dict_full(self, valid_config_dict):
        cfg = PipelineConfig.from_dict(valid_config_dict)
        assert cfg.name == "test-pipeline"
        assert cfg.version == "2.0.0"
        assert cfg.environment == "staging"
        assert cfg.data.source_dir == "/data/raw"
        assert cfg.training.epochs == 10
        assert cfg.deployment.strategy == "immediate"

    def test_from_dict_empty(self):
        cfg = PipelineConfig.from_dict({})
        assert cfg.name == "hermes-retraining-pipeline"
        assert isinstance(cfg.data, DataConfig)

    def test_from_dict_partial(self):
        d = {"name": "partial", "data": {"min_samples": 200}}
        cfg = PipelineConfig.from_dict(d)
        assert cfg.name == "partial"
        assert cfg.data.min_samples == 200


class TestPipelineConfigFromYaml:
    def test_from_yaml_file(self, tmp_yaml_path, valid_config_dict):
        with open(tmp_yaml_path, "w") as f:
            yaml.dump(valid_config_dict, f)
        cfg = PipelineConfig.from_yaml(tmp_yaml_path)
        assert cfg.name == "test-pipeline"

    def test_from_yaml_missing_file(self, tmp_yaml_path):
        missing = tmp_yaml_path + ".nonexistent"
        cfg = PipelineConfig.from_yaml(missing)
        assert cfg.name == "hermes-retraining-pipeline"  # defaults

    def test_from_yaml_empty_file(self, tmp_yaml_path):
        Path(tmp_yaml_path).write_text("")
        cfg = PipelineConfig.from_yaml(tmp_yaml_path)
        assert isinstance(cfg, PipelineConfig)


class TestPipelineConfigSerialization:
    def test_to_dict(self):
        cfg = PipelineConfig()
        d = cfg.to_dict()
        assert "name" in d
        assert "data" in d
        assert "training" in d
        assert d["name"] == "hermes-retraining-pipeline"

    def test_to_yaml_roundtrip(self, tmp_yaml_path):
        cfg = PipelineConfig(name="roundtrip-test")
        cfg.to_yaml(tmp_yaml_path)
        assert Path(tmp_yaml_path).exists()
        loaded = PipelineConfig.from_yaml(tmp_yaml_path)
        assert loaded.name == "roundtrip-test"

    def test_to_yaml_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "nested" / "dir" / "config.yaml")
        cfg = PipelineConfig()
        cfg.to_yaml(path)
        assert Path(path).exists()


class TestPipelineConfigValidation:
    def test_valid_config_no_issues(self):
        cfg = PipelineConfig()
        issues = cfg.validate()
        # Default config should be valid
        assert isinstance(issues, list)

    def test_invalid_validation_split(self):
        cfg = PipelineConfig(data=DataConfig(validation_split=0.0))
        issues = cfg.validate()
        assert any("validation_split" in i for i in issues)

    def test_invalid_validation_split_over_one(self):
        cfg = PipelineConfig(data=DataConfig(validation_split=1.5))
        issues = cfg.validate()
        assert any("validation_split" in i for i in issues)

    def test_invalid_test_split(self):
        cfg = PipelineConfig(data=DataConfig(test_split=0.0))
        issues = cfg.validate()
        assert any("test_split" in i for i in issues)

    def test_splits_sum_too_large(self):
        cfg = PipelineConfig(data=DataConfig(validation_split=0.6, test_split=0.5))
        issues = cfg.validate()
        assert any("validation_split + test_split" in i for i in issues)

    def test_min_samples_too_low(self):
        cfg = PipelineConfig(data=DataConfig(min_samples=5))
        issues = cfg.validate()
        assert any("min_samples" in i for i in issues)

    def test_epochs_too_low(self):
        cfg = PipelineConfig()
        cfg.training.epochs = 0
        issues = cfg.validate()
        assert any("epochs" in i for i in issues)

    def test_learning_rate_negative(self):
        cfg = PipelineConfig()
        cfg.training.hyperparameters["learning_rate"] = -0.01
        issues = cfg.validate()
        assert any("learning_rate" in i for i in issues)

    def test_canary_traffic_zero(self):
        cfg = PipelineConfig(deployment=DeploymentConfig(canary_traffic_percentage=0.0))
        issues = cfg.validate()
        assert any("canary_traffic_percentage" in i for i in issues)

    def test_canary_traffic_over_100(self):
        cfg = PipelineConfig(deployment=DeploymentConfig(canary_traffic_percentage=150.0))
        issues = cfg.validate()
        assert any("canary_traffic_percentage" in i for i in issues)

    def test_drift_threshold_negative(self):
        cfg = PipelineConfig(monitoring=MonitoringConfig(drift_threshold=-0.1))
        issues = cfg.validate()
        assert any("drift_threshold" in i for i in issues)

    def test_validate_returns_list(self):
        cfg = PipelineConfig()
        result = cfg.validate()
        assert isinstance(result, list)
