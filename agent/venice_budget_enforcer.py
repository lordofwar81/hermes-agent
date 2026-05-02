"""Venice budget enforcer - prevents Venice routing when budget exhausted.

Implements:
    - Daily spend tracking with configurable reset time
    - Hard budget limits during model selection
    - Reset logic (default: midnight UTC, configurable)

Configuration (config.yaml):
    venice_budget:
        daily_limit_usd: 7.40
        tracking_enabled: true
        reset_time: "00:00 UTC"  # or "17:00 PT"
        spend_file: "~/.hermes/venice_spend.json"

Usage:
    >>> enforcer = VeniceBudgetEnforcer()
    >>> result = enforcer.check_budget("venice-deepseek-v3.2")
    >>> if not result.allowed:
    ...     print(f"Budget exhausted: {result.message}")
    ...     # Route to local/Z.ai instead
"""

import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home as get_hermes_home_func

logger = logging.getLogger(__name__)

# Thread lock for file operations
_file_lock = threading.Lock()


class VeniceBudgetEnforcer:
    """Enforces Venice budget limits with configurable reset logic."""

    def __init__(
        self,
        daily_limit_usd: float = 7.40,
        tracking_enabled: bool = True,
        reset_time: str = "00:00 UTC",
        spend_file: Optional[str] = None,
    ):
        """Initialize the budget enforcer."""
        self.daily_limit_usd = daily_limit_usd
        self.tracking_enabled = tracking_enabled
        self.reset_time = reset_time
        self.spend_file = spend_file or str(
            get_hermes_home_func() / "venice_spend.json"
        )

        # Parse reset time (format: "HH:MM TZ")
        parts = reset_time.strip().split()
        if len(parts) != 2:
            logger.warning(
                f"Invalid reset time format '{reset_time}', using default 00:00 UTC"
            )
            self.reset_hour, self.reset_minute = 0, 0
        else:
            time_str = parts[0]
            hour, minute = map(int, time_str.split(":"))
            self.reset_hour, self.reset_minute = hour, minute

        # In-memory cache for current spend
        self._current_spend: float = 0.0
        self._last_reset: Optional[datetime] = None
        self._load_spend_file()

    def _load_spend_file(self) -> None:
        """Load spend data from tracking file."""
        if not self.tracking_enabled:
            return

        try:
            path = Path(self.spend_file)
            if path.exists():
                with open(path, "r") as f:
                    data = json.load(f)
                    self._current_spend = float(data.get("daily_spend_usd", 0.0))
                    last_reset_str = data.get("last_reset")
                    if last_reset_str:
                        self._last_reset = datetime.fromisoformat(
                            last_reset_str.replace("Z", "+00:00")
                        )
                    logger.debug(f"Loaded spend: ${self._current_spend:.2f}")
            else:
                self._current_spend = 0.0
                self._last_reset = None
                logger.debug(f"Spend file not found, initializing")
        except Exception as e:
            logger.warning(f"Failed to load spend file: {e}, initializing fresh")
            self._current_spend = 0.0
            self._last_reset = None

    def _save_spend_file(self) -> None:
        """Save spend data to tracking file."""
        if not self.tracking_enabled:
            return

        try:
            path = Path(self.spend_file)
            last_reset_str = (
                self._last_reset.isoformat()
                if self._last_reset
                else datetime.now(timezone.utc).isoformat()
            )
            data = {
                "daily_spend_usd": round(self._current_spend, 2),
                "last_reset": last_reset_str,
                "requests_today": int(self._current_spend / 0.01)
                if self._current_spend > 0
                else 0,
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved spend: ${data['daily_spend_usd']:.2f}")
        except Exception as e:
            logger.error(f"Failed to save spend file: {e}")

    def _get_reset_datetime(self) -> datetime:
        """Get the next reset datetime based on current time and reset config."""
        now_utc = datetime.now(timezone.utc)
        target_utc = now_utc.replace(
            hour=self.reset_hour,
            minute=self.reset_minute,
            second=0,
            microsecond=0,
        )
        if now_utc >= target_utc:
            target_utc += timedelta(days=1)
        return target_utc

    def _check_reset_needed(self) -> bool:
        """Check if spend tracker needs to be reset."""
        if not self.tracking_enabled:
            return False
        if not self._last_reset:
            return True
        next_reset = self._get_reset_datetime()
        now_utc = datetime.now(timezone.utc)
        if now_utc >= next_reset:
            logger.info(f"Venice budget reset triggered (next reset was {next_reset})")
            return True
        return False

    def _apply_reset(self) -> None:
        """Apply budget reset - clear spend tracker."""
        with _file_lock:
            self._current_spend = 0.0
            self._last_reset = datetime.now(timezone.utc)
            self._save_spend_file()
            logger.info(f"Venice budget reset to $0.00")

    def check_budget(self, model: str) -> Dict[str, Any]:
        """Check if Venice model can be used given current budget.

        Returns:
            {
                "allowed": bool,
                "remaining_budget": float,
                "message": str
            }
        """
        with _file_lock:
            if self._check_reset_needed():
                self._apply_reset()

            remaining = self.daily_limit_usd - self._current_spend

            if remaining <= 0:
                return {
                    "allowed": False,
                    "remaining_budget": 0.0,
                    "message": f"Venice budget exhausted (spent ${self._current_spend:.2f}/${self.daily_limit_usd:.2f})",
                }

            model_costs = {
                "venice-deepseek-v3.2": 0.008,
                "venice-glm-4.7-flash": 0.007,
                "venice-glm-4.7": 0.03,
                "venice-glm-5": 0.04,
                "venice-qwen-3-6-plus": 0.05,
                "venice-claude-sonnet-4-6": 0.22,
                "venice-grok-4-20-beta": 0.10,
                "venice-qwen3-coder-480b": 0.04,
                "venice-qwen3-5-35b": 0.02,
                "venice-uncensored": 0.01,
            }
            estimated_cost = model_costs.get(model, 0.03)

            if estimated_cost > remaining:
                return {
                    "allowed": False,
                    "remaining_budget": remaining,
                    "message": f"Insufficient budget for {model} (${estimated_cost:.3f} > ${remaining:.2f} remaining)",
                }

            return {
                "allowed": True,
                "remaining_budget": remaining,
                "message": f"Budget OK: ${remaining:.2f} remaining",
            }

    def record_spend(self, model: str, amount: float) -> None:
        """Record spend for a Venice request."""
        with _file_lock:
            self._current_spend += amount
            self._save_spend_file()
            logger.debug(
                f"Recorded spend: ${amount:.3f} (total: ${self._current_spend:.2f})"
            )

    def get_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        with _file_lock:
            if self._check_reset_needed():
                self._apply_reset()

            remaining = self.daily_limit_usd - self._current_spend
            percent_used = (self._current_spend / self.daily_limit_usd) * 100

            return {
                "daily_limit_usd": self.daily_limit_usd,
                "current_spend": round(self._current_spend, 2),
                "remaining_budget": round(remaining, 2),
                "percent_used": round(percent_used, 1),
                "next_reset": self._get_reset_datetime().isoformat(),
                "tracking_enabled": self.tracking_enabled,
            }


_enforcer: Optional[VeniceBudgetEnforcer] = None


def get_budget_enforcer() -> VeniceBudgetEnforcer:
    """Get the global budget enforcer instance."""
    global _enforcer
    if _enforcer is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
            venice_config = config.get("venice_budget", {})

            _enforcer = VeniceBudgetEnforcer(
                daily_limit_usd=venice_config.get("daily_limit_usd", 7.40),
                tracking_enabled=venice_config.get("tracking_enabled", True),
                reset_time=venice_config.get("reset_time", "00:00 UTC"),
                spend_file=venice_config.get("spend_file"),
            )
        except Exception as e:
            logger.warning(f"Failed to load venice_budget config: {e}, using defaults")
            _enforcer = VeniceBudgetEnforcer()
    return _enforcer


def check_venice_budget(model: str) -> Dict[str, Any]:
    """Quick check: can we use Venice for this model?"""
    return get_budget_enforcer().check_budget(model)


def record_venice_spend(model: str, amount: float) -> None:
    """Quick record: log Venice spend."""
    get_budget_enforcer().record_spend(model, amount)


def get_venice_budget_status() -> Dict[str, Any]:
    """Quick status check."""
    return get_budget_enforcer().get_status()
