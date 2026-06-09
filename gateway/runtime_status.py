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


# Sentinel placed into _running_agents immediately when a session starts
# processing, *before* any await.  Prevents a second message for the same
# session from bypassing the "already running" guard during the async gap
# between the guard check and actual agent creation.
_AGENT_PENDING_SENTINEL = object()


def running_agent_count(runner) -> int:
    """Return the number of currently running agents."""
    return len(runner._running_agents)


def status_action_label(runner) -> str:
    """Return 'restart' if restart is pending, else 'shutdown'."""
    return "restart" if runner._restart_requested else "shutdown"


def status_action_gerund(runner) -> str:
    """Return 'restarting' if restart is pending, else 'shutting down'."""
    return "restarting" if runner._restart_requested else "shutting down"


def queue_during_drain_enabled(runner) -> bool:
    """Check if queue-during-drain mode is enabled.

    Both "queue" and "steer" modes imply the user doesn't want messages
    to be lost during restart — queue them for the newly-spawned gateway
    process to pick up.  "interrupt" mode drops them (current behaviour).
    """
    return runner._restart_requested and runner._busy_input_mode in {"queue", "steer"}


def snapshot_running_agents(runner) -> Dict[str, Any]:
    """Snapshot current running agents, excluding pending sentinel."""
    return {
        session_key: agent
        for session_key, agent in runner._running_agents.items()
        if agent is not _AGENT_PENDING_SENTINEL
    }


def agent_has_active_subagents(running_agent: Any) -> bool:
    """Return True when *running_agent* is currently driving subagents.

    Background (#30170): ``AIAgent.interrupt()`` cascades through the
    parent's ``_active_children`` list and calls ``interrupt()`` on
    every child synchronously, which aborts in-flight subagent work
    and produces a fallback cascade with no actionable signal.
    Demoting ``busy_input_mode='interrupt'`` to ``queue`` semantics
    whenever this helper returns True protects subagent work from
    conversational follow-ups while leaving the explicit ``/stop``
    path (which goes through ``_interrupt_and_clear_session``)
    untouched. Safe-by-default: returns False on any attribute or
    lock error so a missing/broken parent never blocks the existing
    interrupt path.
    """
    if running_agent is None or running_agent is _AGENT_PENDING_SENTINEL:
        return False
    children = getattr(running_agent, "_active_children", None)
    # AIAgent always initialises this as a concrete list (see
    # agent/agent_init.py). Reject anything that isn't a real
    # collection — this guards against ``MagicMock()._active_children``
    # auto-creating a truthy stub in tests and triggering the demotion
    # against an agent that doesn't actually have subagents.
    if not isinstance(children, (list, tuple, set)):
        return False
    if not children:
        return False
    lock = getattr(running_agent, "_active_children_lock", None)
    try:
        if lock is not None:
            with lock:
                return bool(children)
        return bool(children)
    except Exception:
        return False
