"""Agent management helpers extracted from GatewayRunner.

Round 14 of gateway decomposition. _finalize_shutdown_agents runs session-end
hooks then cleans up agents; _release_evicted_agent_soft soft-cleans cache
evictions; _bind_adapter_run_generation tags adapter sessions; the other two
read/write runtime status and goal state. All stateless. _finalize/_release
call _cleanup_agent_resources from gateway_lifecycle.
"""

from typing import Any, Dict, Optional

import logging
logger = logging.getLogger("gateway.run")


def _finalize_shutdown_agents(active_agents: Dict[str, Any]) -> None:
    from gateway.gateway_lifecycle import _cleanup_agent_resources
    for agent in active_agents.values():
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            _invoke_hook(
                "on_session_finalize",
                session_id=getattr(agent, "session_id", None),
                platform="gateway",
                reason="shutdown",
            )
        except Exception:
            pass
        _cleanup_agent_resources(agent)


def _release_evicted_agent_soft(agent: Any) -> None:
    """Soft cleanup for cache-evicted agents — preserves session tool state.

    Called from _enforce_agent_cache_cap and _sweep_idle_cached_agents.
    Distinct from _cleanup_agent_resources (full teardown) because a
    cache-evicted session may resume at any time — its terminal
    sandbox, browser daemon, and tracked bg processes must outlive
    the Python AIAgent instance so the next agent built for the
    same task_id inherits them.
    """
    from gateway.gateway_lifecycle import _cleanup_agent_resources
    if agent is None:
        return
    try:
        if hasattr(agent, "release_clients"):
            agent.release_clients()
        else:
            # Older agent instance (shouldn't happen in practice) —
            # fall back to the legacy full-close path.
            _cleanup_agent_resources(agent)
    except Exception:
        pass
    # Free conversation history memory — can be tens of MB with tool
    # outputs (file reads, terminal output, search results) on heavy
    # 100+-tool-call sessions. release_clients() deliberately preserves
    # session tool state for resume, but the message list is rebuilt from
    # persisted session JSON on the next turn, so dropping it here is safe.
    if hasattr(agent, "_session_messages"):
        agent._session_messages = []


def _bind_adapter_run_generation(
    adapter: Any,
    session_key: str,
    generation: int | None,
) -> None:
    """Bind a gateway run generation to the adapter's active-session event."""
    if not adapter or not session_key or generation is None:
        return
    try:
        interrupt_event = getattr(adapter, "_active_sessions", {}).get(session_key)
        if interrupt_event is not None:
            setattr(interrupt_event, "_hermes_run_generation", int(generation))
    except Exception:
        pass


def _update_platform_runtime_status(
    platform: str,
    *,
    platform_state: Optional[str] = None,
    error_code: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
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


def _goal_still_active_for_session(session_id: str) -> bool:
    """Best-effort fresh DB check before running a queued continuation."""
    if not session_id:
        return False
    try:
        from hermes_cli.goals import GoalManager
        return GoalManager(session_id=session_id).is_active()
    except Exception as exc:
        logger.debug("goal continuation: active-state recheck failed: %s", exc)
        return False
