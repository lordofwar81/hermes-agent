"""
Gateway runtime status helpers.

Extracted from gateway/run.py to reduce the God file size.
Provides functions for persisting runtime health/status information,
managing per-platform pause/resume state, and logging platform failures.
"""

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def update_runtime_status(
    restart_requested: bool,
    running_agent_count: int,
    gateway_state: Optional[str] = None,
    exit_reason: Optional[str] = None,
) -> None:
    """Persist gateway-level runtime health/status information."""
    try:
        from gateway.status import write_runtime_status

        write_runtime_status(
            gateway_state=gateway_state,
            exit_reason=exit_reason,
            restart_requested=restart_requested,
            active_agents=running_agent_count,
        )
    except Exception:
        pass


def update_platform_runtime_status(
    platform: str,
    platform_state: Optional[str] = None,
    error_code: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    """Persist per-platform runtime health/status information."""
    try:
        from gateway.status import write_runtime_status

        write_runtime_status(
            platform=platform,
            platform_state=platform_state,
            error_code=error_code,
            error_message=error_message,
        )
    except Exception:
        pass


def pause_failed_platform(
    failed_platforms: Dict[Any, Dict],
    platform: Any,
    *,
    reason: str = "",
) -> None:
    """Mark a queued platform as paused — keep it in ``failed_platforms``
    but stop the reconnect watcher from hammering it.

    Used by the circuit breaker and by the /platform pause slash command.
    """
    info = failed_platforms.get(platform)
    if info is None:
        return
    if info.get("paused"):
        return
    info["paused"] = True
    info["pause_reason"] = reason or "auto-paused after repeated failures"
    info["next_retry"] = float("inf")
    try:
        update_platform_runtime_status(
            platform.value,
            platform_state="paused",
            error_code=None,
            error_message=info["pause_reason"],
        )
    except Exception:
        pass
    logger.warning(
        "%s paused after %d consecutive failures (%s) — "
        "fix the underlying issue then run `/platform resume %s` "
        "to retry, or `hermes gateway restart` to restart the gateway.",
        platform.value,
        info.get("attempts", 0),
        info["pause_reason"],
        platform.value,
    )


def resume_paused_platform(
    failed_platforms: Dict[Any, Dict],
    platform: Any,
) -> bool:
    """Unpause a platform — reset its attempt counter and schedule an
    immediate retry.  Returns True if the platform was paused and is
    now queued; False if it wasn't paused (or wasn't in the queue).
    """
    info = failed_platforms.get(platform)
    if info is None:
        return False
    if not info.get("paused"):
        return False
    info["paused"] = False
    info.pop("pause_reason", None)
    info["attempts"] = 0
    info["next_retry"] = time.monotonic()  # retry on next watcher tick
    try:
        update_platform_runtime_status(
            platform.value,
            platform_state="retrying",
            error_code=None,
            error_message=None,
        )
    except Exception:
        pass
    logger.info("%s resumed — retrying on next watcher tick", platform.value)
    return True
