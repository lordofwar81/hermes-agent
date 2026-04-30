"""Budget enforcement for Venice API usage.

Enforces daily budget limits with configurable reset times. Tracks spend across all
Venice models and provides hard rejection when budget is exhausted.

Classes:
    VeniceBudgetEnforcer: Enforces daily budget limits with reset logic
    VeniceSpendTracker: Tracks and persists Venice API spend data
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml


class VeniceSpendTracker:
    """Track and persist Venice API spend data."""

    def __init__(self, config_path: str = None):
        """Initialize tracker with config loading.

        Args:
            config_path: Path to config.yaml file (defaults to ~/.hermes/config.yaml)
        """
        if config_path is None:
            config_path = "/home/lordofwarai/.hermes/config.yaml"
        self._config_path = config_path
        self._load_config()
        self._data_file = self._get_data_file()
        self._lock = threading.RLock()

        # Initialize data file if it doesn't exist
        if not self._data_file.exists():
            self._data_file.parent.mkdir(parents=True, exist_ok=True)
            self._save_data({"spend": [], "summary": {}})

    def _load_config(self) -> None:
        """Load configuration from YAML file.

        Parses Venice budget settings from config.yaml:
        - venice_budget.daily_limit_usd: 7.40
        - venice_budget.tracking_enabled: true
        - venice_budget.reset_time: "17:00 PT" or "midnight UTC"
        """
        with open(self._config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

        budget_config = self._config.get("venice_budget", {})
        self._daily_limit_usd = budget_config.get("daily_limit_usd", 7.40)
        self._tracking_enabled = budget_config.get("tracking_enabled", True)
        self._reset_time = budget_config.get("reset_time", "midnight UTC")

        # Parse reset time
        if self._reset_time == "midnight UTC":
            self._reset_hour = 0
            self._reset_timezone = timezone.utc
        else:
            # Parse "HH:MM PT" format
            parts = self._reset_time.split()
            self._reset_hour = int(parts[0].split(":")[0])
            self._reset_timezone = timezone.utc  # Simplified for now

    def _get_data_file(self) -> Path:
        """Get path to spend tracking data file.

        Returns:
            Path to JSON data file
        """
        base_dir = Path.home() / ".hermes" / "hermes-agent" / "agent"
        return base_dir / "venice_spend_data.json"

    def _get_current_date_key(self) -> str:
        """Get current date key for tracking.

        Returns:
            Date string in YYYY-MM-DD format
        """
        now = datetime.now(self._reset_timezone)
        return now.strftime("%Y-%m-%d")

    def _load_data(self) -> dict[str, Any]:
        """Load spend data from file.

        Returns:
            Dict with spend data
        """
        with self._lock:
            with open(self._data_file, "r", encoding="utf-8") as f:
                return json.load(f)

    def _save_data(self, data: dict[str, Any]) -> None:
        """Save spend data to file.

        Args:
            data: Dict with spend data to save
        """
        with self._lock:
            self._data_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._data_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    def add_spend(self, amount_usd: float, model: str, request_id: str = None) -> None:
        """Record a new spend entry.

        Args:
            amount_usd: Amount spent in USD
            model: Venice model used
            request_id: Optional request ID for tracking
        """
        data = self._load_data()

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "amount_usd": amount_usd,
            "model": model,
            "request_id": request_id or f"req_{int(datetime.now().timestamp())}",
        }

        data["spend"].append(entry)

        # Update summary
        if "summary" not in data:
            data["summary"] = {}

        summary = data["summary"]
        current_date = self._get_current_date_key()

        # Initialize date if needed
        if current_date not in summary:
            summary[current_date] = {"total": 0, "requests": 0, "models": {}}

        # Update summary
        summary[current_date]["total"] += amount_usd
        summary[current_date]["requests"] += 1

        if model not in summary[current_date]["models"]:
            summary[current_date]["models"][model] = {"count": 0, "total": 0}

        summary[current_date]["models"][model]["count"] += 1
        summary[current_date]["models"][model]["total"] += amount_usd

        self._save_data(data)

    def get_current_spend(self) -> float:
        """Get current day's total spend.

        Returns:
            Total spend in USD for current day
        """
        data = self._load_data()
        current_date = self._get_current_date_key()

        if current_date not in data.get("summary", {}):
            return 0.0

        return data["summary"][current_date]["total"]

    def get_remaining_budget(self) -> float:
        """Get remaining budget for current day.

        Returns:
            Remaining budget in USD
        """
        current_spend = self.get_current_spend()
        return max(0, self._daily_limit_usd - current_spend)

    def get_usage_percentage(self) -> float:
        """Get usage percentage of daily budget.

        Returns:
            Percentage of budget used (0-100)
        """
        current_spend = self.get_current_spend()
        return min(100, (current_spend / self._daily_limit_usd) * 100)

    def get_model_breakdown(self) -> dict[str, Any]:
        """Get model breakdown for current day.

        Returns:
            Dict with model usage statistics
        """
        data = self._load_data()
        current_date = self._get_current_date_key()

        if current_date not in data.get("summary", {}):
            return {"models": {}, "total_requests": 0, "total_spend": 0}

        summary = data["summary"][current_date]
        return {
            "models": summary.get("models", {}),
            "total_requests": summary.get("requests", 0),
            "total_spend": summary.get("total", 0),
        }

    def should_reset(self) -> bool:
        """Check if budget should be reset.

        Returns:
            True if budget should be reset for new day
        """
        now = datetime.now(self._reset_timezone)
        current_date = now.strftime("%Y-%m-%d")
        current_hour = now.hour

        # Check if we've crossed the reset time
        if current_hour > self._reset_hour:
            return True
        elif current_hour == self._reset_hour and now.minute >= 0:
            return True

        return False

    def reset_budget(self) -> bool:
        """Reset budget for new day.

        Returns:
            True if reset was successful
        """
        if not self._tracking_enabled:
            return False

        data = self._load_data()
        current_date = self._get_current_date_key()

        # Keep only last 30 days of data
        if "summary" in data:
            data["summary"] = {
                k: v
                for k, v in data["summary"].items()
                if datetime.strptime(k, "%Y-%m-%d").date()
                >= (datetime.now() - timedelta(days=30)).date()
            }

        self._save_data(data)
        return True

    def get_spend_history(self, days: int = 7) -> list[dict[str, Any]]:
        """Get spend history for past N days.

        Args:
            days: Number of days to retrieve

        Returns:
            List of daily spend entries
        """
        data = self._load_data()
        summary = data.get("summary", {})

        history = []
        for date_str in sorted(summary.keys(), reverse=True)[:days]:
            history.append(
                {
                    "date": date_str,
                    "total": summary[date_str]["total"],
                    "requests": summary[date_str]["requests"],
                    "models": summary[date_str].get("models", {}),
                }
            )

        return history


class VeniceBudgetEnforcer:
    """Enforce Venice API budget limits with reset logic."""

    def __init__(self, config_path: str = None):
        """Initialize enforcer with config loading.

        Args:
            config_path: Path to config.yaml file (defaults to ~/.hermes/config.yaml)
        """
        if config_path is None:
            config_path = "/home/lordofwarai/.hermes/config.yaml"
        self._config_path = config_path
        self._tracker = VeniceSpendTracker(config_path)

        # Parse reset time for logging
        reset_time = self._tracker._reset_time
        if reset_time == "midnight UTC":
            self._reset_info = "midnight UTC"
        else:
            self._reset_info = reset_time

        # Thresholds for warnings
        self._warning_thresholds = [0.50, 0.75, 0.90]

    def check_budget(self, model: str = None) -> tuple[bool, str]:
        """Check if budget allows Venice routing.

        Args:
            model: Optional model name for logging

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        if not self._tracker._tracking_enabled:
            return True, "tracking disabled"

        # Check for budget exhaustion
        remaining = self._tracker.get_remaining_budget()
        if remaining <= 0:
            return False, f"budget exhausted (remaining: ${remaining:.2f})"

        # Check warning thresholds
        usage = self._tracker.get_usage_percentage()
        for threshold in self._warning_thresholds:
            if usage >= (threshold * 100) and usage < ((threshold + 0.01) * 100):
                return True, f"warning: {usage:.0f}% of daily budget used"

        return True, f"allowed (remaining: ${remaining:.2f})"

    def is_budget_exhausted(self) -> bool:
        """Check if daily budget is exhausted.

        Returns:
            True if budget is exhausted
        """
        return self._tracker.get_remaining_budget() <= 0

    def get_budget_status(self) -> dict[str, Any]:
        """Get comprehensive budget status.

        Returns:
            Dict with budget status information
        """
        return {
            "daily_limit_usd": self._tracker._daily_limit_usd,
            "spent_usd": self._tracker.get_current_spend(),
            "remaining_budget": self._tracker.get_remaining_budget(),
            "usage_percentage": self._tracker.get_usage_percentage(),
            "reset_time": self._reset_info,
            "tracking_enabled": self._tracker._tracking_enabled,
            "model_breakdown": self._tracker.get_model_breakdown(),
            "should_reset": self._tracker.should_reset(),
        }

    def log_warning(self, model: str = None) -> None:
        """Log budget warning if near threshold.

        Args:
            model: Optional model name for logging
        """
        usage = self._tracker.get_usage_percentage()

        for threshold in self._warning_thresholds:
            if usage >= (threshold * 100) and usage < ((threshold + 0.01) * 100):
                print(
                    f"[Venice Budget] Warning: {usage:.0f}% of daily budget used ({self._reset_info} reset)"
                )
                break

    def record_spend(
        self, amount_usd: float, model: str, request_id: str = None
    ) -> None:
        """Record Venice API spend.

        Args:
            amount_usd: Amount spent in USD
            model: Venice model used
            request_id: Optional request ID for tracking
        """
        self._tracker.add_spend(amount_usd, model, request_id)

    def enforce_and_log(self, model: str = None) -> tuple[bool, str]:
        """Enforce budget limits with logging.

        Args:
            model: Optional model name for logging

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        allowed, reason = self.check_budget(model)

        if not allowed:
            print(f"[Venice Budget] Rejected: {reason}")

        self.log_warning(model)

        return allowed, reason

    def should_use_venice(
        self,
        complexity: str,
        trivial: bool = False,
        local_failed: bool = False,
        zai_failed: bool = False,
    ) -> tuple[bool, str]:
        """Determine if Venice should be used based on complexity and failure chain.

        Complex logic:
        1. Trivial tasks NEVER use Venice
        2. Simple/medium tasks prefer local/Z.ai
        3. Complex tasks: try local/Z.ai first
        4. Expert tasks: use Venice if local/Z.ai failed OR truly stuck

        Args:
            complexity: Task complexity level
            trivial: Whether task is trivial
            local_failed: Whether local models failed
            zai_failed: Whether Z.ai models failed

        Returns:
            Tuple of (should_use: bool, reason: str)
        """
        # Trivial tasks never use Venice
        if trivial:
            return False, "trivial task - route to local/Z.ai"

        # Budget check
        allowed, budget_reason = self.check_budget()
        if not allowed:
            return False, budget_reason

        # Simple/medium tasks prefer free models
        if complexity in ("simple", "medium"):
            return False, f"{complexity} task - prefer local/Z.ai"

        # Complex tasks: use only if fallback chain failed
        if complexity == "complex":
            if local_failed and zai_failed:
                return True, "complex task with failed fallback chain"
            return False, "complex task - try local/Z.ai first"

        # Expert tasks: use Venice if fallback failed or truly stuck
        if complexity == "expert":
            if local_failed and zai_failed:
                return True, "expert task with failed fallback chain"
            return True, "expert task - Venice allowed"

        return False, "unknown complexity"
