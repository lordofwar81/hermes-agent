"""Running-agent bookkeeping methods for ``GatewayRunner``.

Round 17 of the god-file decomposition. Extracted verbatim into a mixin —
``self.*`` references (``self._running_agents``, ``self._get_max_concurrent_sessions``)
resolve unchanged via the MRO. Behavior-neutral lift matching the existing
authz/kanban/slash/goals mixin pattern.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class GatewayRunningMixin:
    """Running-agent introspection methods for ``GatewayRunner``."""

    def _running_agent_count(self) -> int:
        return len(self._running_agents)

    def _snapshot_running_agents(self) -> Dict[str, Any]:
        # Lazy import to avoid circular dependency: gateway.run imports this
        # mixin at module top, so we can only resolve _AGENT_PENDING_SENTINEL
        # at call time (same pattern as authz_mixin's lazy logger import).
        from gateway.run import _AGENT_PENDING_SENTINEL
        return {
            session_key: agent
            for session_key, agent in self._running_agents.items()
            if agent is not _AGENT_PENDING_SENTINEL
        }

    def _active_session_limit_message(self, session_key: str) -> Optional[str]:
        """Return a user-facing rejection when starting a new session exceeds the cap."""
        max_sessions = self._get_max_concurrent_sessions()
        if max_sessions is None:
            return None
        if session_key in getattr(self, "_running_agents", {}):
            return None
        active_count = len(getattr(self, "_running_agents", {}))
        if active_count < max_sessions:
            return None
        return (
            f"Hermes is at the active session limit ({active_count}/{max_sessions}). "
            "Try again when another session finishes."
        )
