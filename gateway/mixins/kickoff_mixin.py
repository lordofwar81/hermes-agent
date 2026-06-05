"""Mixin for GatewayRunner: lifecycle queries & message kickoff.

Small identity/query methods that return gateway status or route
messages to the correct handler.  Mixed into GatewayRunner so they
can access ``self.*`` attributes set up by ``__init__``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class KickoffMixin:
    """Lifecycle queries, session key resolution, agent snapshot helpers."""

    def should_exit_cleanly(self) -> bool:
        return getattr(self, "_exit_code", 0) == 0

    def should_exit_with_failure(self) -> bool:
        return getattr(self, "_exit_code", 0) != 0

    def exit_reason(self) -> Optional[str]:
        return getattr(self, "_exit_reason", None)

    def exit_code(self) -> Optional[int]:
        return getattr(self, "_exit_code", None)

    def _running_agent_count(self) -> int:
        agents = getattr(self, "_running_agents", None)
        if agents is None:
            return 0
        from gateway.run import _AGENT_PENDING_SENTINEL
        return sum(
            1 for a in agents.values()
            if a is not None and a is not _AGENT_PENDING_SENTINEL
        )

    def _status_action_label(self) -> str:
        return "restart" if getattr(self, "_restart_requested", False) else "shutdown"

    def _status_action_gerund(self) -> str:
        return "restarting" if getattr(self, "_restart_requested", False) else "shutting down"

    def _queue_during_drain_enabled(self) -> bool:
        return getattr(self, "_restart_requested", False) and getattr(self, "_busy_input_mode", "interrupt") in {"queue", "steer"}

    @staticmethod
    def _is_goal_continuation_event(event_or_text: Any) -> bool:
        text = getattr(event_or_text, "text", event_or_text) or ""
        return str(text).startswith("[Continuing toward your standing goal]\nGoal:")

    def _goal_still_active_for_session(self, session_id: str) -> bool:
        if not session_id:
            return False
        try:
            from hermes_cli.goals import GoalManager
            return GoalManager(session_id=session_id).is_active()
        except Exception as exc:
            logger.debug("goal continuation: active-state recheck failed: %s", exc)
            return False

    def _snapshot_running_agents(self) -> Dict[str, Any]:
        agents = getattr(self, "_running_agents", {})
        from gateway.run import _AGENT_PENDING_SENTINEL
        return {
            session_key: agent
            for session_key, agent in agents.items()
            if agent is not _AGENT_PENDING_SENTINEL
        }

    @staticmethod
    def _agent_has_active_subagents(running_agent: Any) -> bool:
        if running_agent is None:
            return False
        children = getattr(running_agent, "_active_children", None)
        if not children:
            return False
        try:
            return len(children) > 0
        except Exception:
            return False

    def _session_key_for_source(self, source: Any) -> str:
        from gateway.run import _session_key_for_source as _resolve_key
        return _resolve_key(source)

    def _clear_goal_pending_continuations(self, session_key: str, adapter: Any) -> int:
        removed = 0
        pending_slot = getattr(adapter, "_pending_messages", None) if adapter is not None else None
        if isinstance(pending_slot, dict):
            pending_event = pending_slot.get(session_key)
            if self._is_goal_continuation_event(pending_event):
                pending_slot.pop(session_key, None)
                removed += 1

        queued_events = getattr(self, "_queued_events", None)
        if isinstance(queued_events, dict):
            overflow = queued_events.get(session_key) or []
            if overflow:
                kept = []
                for queued_event in overflow:
                    if self._is_goal_continuation_event(queued_event):
                        removed += 1
                    else:
                        kept.append(queued_event)
                if kept:
                    queued_events[session_key] = kept
                else:
                    queued_events.pop(session_key, None)
        return removed

    def _enqueue_fifo(self, session_key: str, queued_event: "Any", adapter: Any) -> None:
        """Append a /queue event to the FIFO chain for a session."""
        if adapter is None:
            return
        pending_slot = getattr(adapter, "_pending_messages", None)
        if pending_slot is None:
            return
        queued_events = getattr(self, "_queued_events", None)
        if queued_events is None:
            queued_events = {}
            self._queued_events = queued_events
        if session_key in pending_slot:
            queued_events.setdefault(session_key, []).append(queued_event)
        else:
            pending_slot[session_key] = queued_event

    def _promote_queued_event(
        self,
        session_key: str,
        adapter: Any,
        pending_event: Optional["Any"],
    ) -> Optional["Any"]:
        """Promote the next overflow item after the slot was drained."""
        queued_events = getattr(self, "_queued_events", None)
        if not queued_events:
            return pending_event
        overflow = queued_events.get(session_key)
        if not overflow:
            return pending_event
        next_queued = overflow.pop(0)
        if not overflow:
            queued_events.pop(session_key, None)
        if pending_event is None:
            return next_queued
        if adapter is not None and hasattr(adapter, "_pending_messages"):
            adapter._pending_messages[session_key] = next_queued
        else:
            queued_events.setdefault(session_key, []).insert(0, next_queued)
        return pending_event

    def _queue_depth(self, session_key: str, *, adapter: Any = None) -> int:
        """Total pending /queue items for a session — slot + overflow."""
        queued_events = getattr(self, "_queued_events", None) or {}
        depth = len(queued_events.get(session_key, []))
        if adapter is not None and session_key in getattr(adapter, "_pending_messages", {}):
            depth += 1
        return depth

    def _queue_or_replace_pending_event(self, session_key: str, event: Any) -> None:
        from gateway.run import merge_pending_message_event

        adapter = self.adapters.get(event.source.platform)
        if not adapter:
            return
        merge_pending_message_event(adapter._pending_messages, session_key, event)
