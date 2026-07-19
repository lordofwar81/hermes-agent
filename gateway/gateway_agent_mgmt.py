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
        # Persist any in-flight transcript to the SQLite session store before
        # teardown (#13121).  An agent forcibly interrupted by the drain-timeout
        # escalation may never reach ``turn_finalizer.finalize_turn`` (the only
        # place that flushes the turn to state.db) -- its in-flight tool rounds
        # live only in the in-memory ``_session_messages``, so the immediate
        # pre-restart turn is silently dropped from ``load_transcript()`` on
        # resume.  Flushing here closes that gap; the resume_pending /
        # fresh-tool-tail branches already expect a transcript whose tail may be
        # a pending tool result.  Idempotent (identity-tracked in
        # ``_flush_messages_to_session_db``), so agents that DID finish
        # gracefully re-flush nothing.  Restored from commit d19aabbf2 after the
        # gateway decomposition refactor (R47) extracted this helper and dropped
        # the flush.
        try:
            _flush = getattr(agent, "_flush_messages_to_session_db", None)
            _session_messages = getattr(agent, "_session_messages", None)
            if callable(_flush) and isinstance(_session_messages, list) and _session_messages:
                # Strip private empty-response retry scaffolding from the tail
                # first, mirroring the graceful ``_persist_session`` path, so a
                # resumed turn doesn't replay synthetic recovery nudges.
                _strip = getattr(
                    agent, "_drop_trailing_empty_response_scaffolding", None
                )
                if callable(_strip):
                    try:
                        _strip(_session_messages)
                    except Exception:
                        pass
                _flush(_session_messages)
        except Exception as _e:
            logger.debug("Shutdown transcript flush failed: %s", _e)
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
