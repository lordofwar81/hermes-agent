"""
Model Training Module

Orchestrates model training with hyperparameter tuning, checkpointing,
and experiment tracking.

Features:
- Training orchestration with configurable hyperparameters
- Bayesian / Grid / Random hyperparameter search
- Automatic checkpointing and resume support
- Training metrics logging
- Early stopping
"""

import json
import logging
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


class TrainingResult:
    """Container for a completed training run's results."""

    def __init__(
        self,
        run_id: str,
        model_path: str,
        metrics: Dict[str, float],
        hyperparameters: Dict[str, Any],
        training_history: List[Dict[str, float]],
        duration_seconds: float,
        status: str = "completed",
    ):
        self.run_id = run_id
        self.model_path = model_path
        self.metrics = metrics
        self.hyperparameters = hyperparameters
        self.training_history = training_history
        self.duration_seconds = duration_seconds
        self.status = status
        self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "model_path": self.model_path,
            "metrics": self.metrics,
            "hyperparameters": self.hyperparameters,
            "training_history": self.training_history,
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingResult":
        r = cls(
            run_id=d["run_id"],
            model_path=d["model_path"],
            metrics=d["metrics"],
            hyperparameters=d["hyperparameters"],
            training_history=d.get("training_history", []),
            duration_seconds=d.get("duration_seconds", 0),
            status=d.get("status", "unknown"),
        )
        r.created_at = d.get("created_at", r.created_at)
        return r


class HyperparameterSpace:
    """Defines the hyperparameter search space."""

    def __init__(self):
        self._params: Dict[str, dict] = {}

    def add_float(self, name: str, low: float, high: float, log: bool = False):
        self._params[name] = {"type": "float", "low": low, "high": high, "log": log}

    def add_int(self, name: str, low: int, high: int):
        self._params[name] = {"type": "int", "low": low, "high": high}

    def add_choice(self, name: str, choices: list):
        self._params[name] = {"type": "choice", "choices": choices}

    def sample(self) -> Dict[str, Any]:
        """Sample a random point from the space."""
        params = {}
        for name, spec in self._params.items():
            if spec["type"] == "float":
                if spec.get("log"):
                    params[name] = math.exp(random.uniform(math.log(spec["low"]), math.log(spec["high"])))
                else:
                    params[name] = random.uniform(spec["low"], spec["high"])
            elif spec["type"] == "int":
                params[name] = random.randint(spec["low"], spec["high"])
            elif spec["type"] == "choice":
                params[name] = random.choice(spec["choices"])
        return params

    def to_dict(self) -> dict:
        return dict(self._params)


class ModelTrainer:
    """Trains models with hyperparameter tuning and experiment tracking."""

    def __init__(self, config, artifacts_dir: str = "./pipeline/artifacts"):
        """
        Args:
            config: TrainingConfig instance
            artifacts_dir: Directory for storing training artifacts
        """
        self.config = config
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._run_history: List[TrainingResult] = []
        self._load_history()

    # ------------------------------------------------------------------
    # Default hyperparameter space
    # ------------------------------------------------------------------

    def _default_search_space(self) -> HyperparameterSpace:
        """Build a default hyperparameter search space."""
        space = HyperparameterSpace()
        space.add_float("learning_rate", 1e-5, 1e-2, log=True)
        space.add_int("batch_size", 16, 128)
        space.add_float("weight_decay", 1e-6, 1e-2, log=True)
        space.add_float("dropout_rate", 0.1, 0.5)
        return space

    # ------------------------------------------------------------------
    # Training Entry Points
    # ------------------------------------------------------------------

    def train(
        self,
        data_splits: Dict[str, Any],
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> TrainingResult:
        """Train a single model with given or default hyperparameters.

        Args:
            data_splits: dict with 'train', 'validation', 'test' paths
            hyperparameters: override default hyperparameters

        Returns:
            TrainingResult with metrics and model path
        """
        hp = {**self.config.hyperparameters, **(hyperparameters or {})}
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info("Starting training run %s with hyperparameters: %s", run_id, hp)

        start_time = time.time()
        run_dir = self.artifacts_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Simulate training process
            # In production, this would invoke actual model training
            metrics, history = self._execute_training(
                data_splits, hp, run_dir
            )

            duration = time.time() - start_time
            result = TrainingResult(
                run_id=run_id,
                model_path=str(run_dir / "model.bin"),
                metrics=metrics,
                hyperparameters=hp,
                training_history=history,
                duration_seconds=duration,
                status="completed",
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error("Training run %s failed: %s", run_id, e)
            result = TrainingResult(
                run_id=run_id,
                model_path="",
                metrics={},
                hyperparameters=hp,
                training_history=[],
                duration_seconds=duration,
                status="failed",
            )

        # Save result
        self._save_run_result(result, run_dir)
        self._run_history.append(result)
        self._save_history()

        logger.info(
            "Training run %s completed in %.1fs - metrics: %s",
            run_id, duration, result.metrics,
        )
        return result

    def tune(
        self,
        data_splits: Dict[str, Any],
        search_space: Optional[HyperparameterSpace] = None,
        num_trials: Optional[int] = None,
    ) -> TrainingResult:
        """Run hyperparameter tuning and return the best result.

        Args:
            data_splits: dict with 'train', 'validation', 'test' paths
            search_space: custom search space (uses default if None)
            num_trials: override number of tuning trials

        Returns:
            Best TrainingResult from all trials
        """
        space = search_space or self._default_search_space()
        budget = num_trials or self.config.tuning_budget

        logger.info(
            "Starting hyperparameter tuning: %d trials with %s method",
            budget, self.config.tuning_method,
        )

        best_result = None
        best_score = float("-inf")
        all_trials = []

        for trial_idx in range(budget):
            # Sample hyperparameters
            if self.config.tuning_method == "random" or self.config.tuning_method == "bayesian":
                hp = space.sample()
            else:
                # For grid, we'd enumerate; simplified here as random sample
                hp = space.sample()

            # Train with these hyperparameters
            result = self.train(data_splits, hyperparameters=hp)
            all_trials.append(result.to_dict())

            # Track best
            if result.status == "completed":
                score = result.metrics.get(self._primary_metric_key(), 0)
                if score > best_score:
                    best_score = score
                    best_result = result
                    logger.info(
                        "New best trial %d: %s = %.4f",
                        trial_idx, self._primary_metric_key(), score,
                    )

        # Save tuning report
        self._save_tuning_report(all_trials, best_result)

        if best_result:
            logger.info(
                "Tuning complete. Best run: %s (%s=%.4f)",
                best_result.run_id, self._primary_metric_key(), best_score,
            )
        return best_result

    # ------------------------------------------------------------------
    # Training Execution (Simulated)
    # ------------------------------------------------------------------

    def _execute_training(
        self,
        data_splits: Dict[str, Any],
        hyperparameters: Dict[str, Any],
        run_dir: Path,
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """Execute model training.

        In production, this would:
        1. Load and prepare data from split paths
        2. Initialize model architecture
        3. Run training loop with gradient updates
        4. Evaluate on validation set after each epoch
        5. Save model checkpoints

        For this pipeline framework, we simulate the training loop
        with realistic metric progression.
        """
        epochs = hyperparameters.get("epochs", self.config.epochs)
        lr = hyperparameters.get("learning_rate", self.config.hyperparameters["learning_rate"])
        patience = hyperparameters.get("early_stopping_patience", self.config.hyperparameters["early_stopping_patience"])

        # Simulate metric progression (realistic learning curve)
        base_accuracy = 0.55 + random.uniform(0, 0.1)
        best_val_metric = 0.0
        patience_counter = 0
        history = []

        for epoch in range(1, epochs + 1):
            # Simulate decreasing loss and increasing metrics
            progress = epoch / epochs
            noise = random.uniform(-0.02, 0.02)

            train_loss = max(0.01, 2.0 * math.exp(-3 * progress) + random.uniform(-0.05, 0.05))
            val_loss = train_loss + random.uniform(0.01, 0.1)

            train_acc = min(0.99, base_accuracy + 0.35 * (1 - math.exp(-4 * progress)) + noise)
            val_acc = min(0.98, train_acc - random.uniform(0.01, 0.05))

            epoch_metrics = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "train_accuracy": round(train_acc, 4),
                "val_accuracy": round(val_acc, 4),
                "learning_rate": lr,
            }
            history.append(epoch_metrics)

            # Early stopping check
            if val_acc > best_val_metric:
                best_val_metric = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
                    break

            # Checkpoint
            if epoch % self.config.checkpoint_every_n_epochs == 0:
                checkpoint_path = run_dir / f"checkpoint_epoch_{epoch}.json"
                with open(checkpoint_path, "w") as f:
                    json.dump({"epoch": epoch, "metrics": epoch_metrics}, f)

        # Final metrics
        final = history[-1] if history else {}
        metrics = {
            "accuracy": final.get("val_accuracy", 0),
            "train_loss": final.get("train_loss", 0),
            "val_loss": final.get("val_loss", 0),
            "precision": min(0.99, final.get("val_accuracy", 0) + random.uniform(-0.05, 0.03)),
            "recall": min(0.99, final.get("val_accuracy", 0) + random.uniform(-0.04, 0.04)),
            "f1_score": min(0.99, final.get("val_accuracy", 0) + random.uniform(-0.03, 0.02)),
            "best_val_accuracy": round(best_val_metric, 4),
            "epochs_trained": len(history),
        }

        # Round all metrics
        metrics = {k: round(v, 4) for k, v in metrics.items()}

        # Save model artifact (simulated)
        model_info = {
            "model_type": self.config.model_type,
            "base_model": self.config.base_model,
            "hyperparameters": hyperparameters,
            "final_metrics": metrics,
            "training_history": history,
        }
        with open(run_dir / "model.bin", "w") as f:
            json.dump(model_info, f, indent=2, default=str)

        return metrics, history

    def _primary_metric_key(self) -> str:
        """Return the primary metric key for optimization."""
        return "accuracy"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_run_result(self, result: TrainingResult, run_dir: Path):
        """Save training result metadata."""
        with open(run_dir / "result.json", "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

    def _save_tuning_report(self, trials: List[dict], best: Optional[TrainingResult]):
        """Save hyperparameter tuning report."""
        report = {
            "method": self.config.tuning_method,
            "num_trials": len(trials),
            "best_run": best.to_dict() if best else None,
            "all_trials": trials,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        report_path = self.artifacts_dir / "tuning_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

    def _load_history(self):
        """Load training run history."""
        hist_path = self.artifacts_dir / "training_history.json"
        if hist_path.exists():
            with open(hist_path, "r") as f:
                history = json.load(f)
            self._run_history = [TrainingResult.from_dict(r) for r in history]

    def _save_history(self):
        """Save training run history."""
        hist_path = self.artifacts_dir / "training_history.json"
        history = [r.to_dict() for r in self._run_history]
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2, default=str)

    def get_best_run(self) -> Optional[TrainingResult]:
        """Return the best training run by primary metric."""
        completed = [r for r in self._run_history if r.status == "completed"]
        if not completed:
            return None
        return max(completed, key=lambda r: r.metrics.get("accuracy", 0))

    def list_runs(self) -> List[dict]:
        """List all training runs."""
        return [r.to_dict() for r in self._run_history]
