"""Mixin for GatewayRunner: lifecycle queries & message kickoff.

Small identity/query methods that return gateway status or route
messages to the correct handler.  Mixed into GatewayRunner so they
can access ``self.*`` attributes set up by ``__init__``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency
# from gateway.queue_helpers import is_goal_continuation_event

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


    def _queue_or_replace_pending_event(self, session_key: str, event: Any) -> None:
        from gateway.run import merge_pending_message_event

        adapter = self.adapters.get(event.source.platform)
        if not adapter:
            return
        merge_pending_message_event(adapter._pending_messages, session_key, event)
