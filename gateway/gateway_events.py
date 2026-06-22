"""Event-inspection + per-turn agent init helpers extracted from GatewayRunner.

Round 10 of gateway decomposition. Pure functions over their arguments —
no instance/class state. _agent_has_active_subagents reads a run.py module
sentinel via lazy import to avoid a circular dep.
"""

import shlex
import time
from typing import Any, Optional

from gateway.platforms.base import MessageEvent

import logging
logger = logging.getLogger("gateway.run")


def _is_goal_continuation_event(event_or_text: Any) -> bool:
    """Return True for synthetic /goal continuation turns.

    Goal continuations are normal queued user-role events, so pause/clear
    must distinguish them from real user /queue messages before removing or
    suppressing them.
    """
    text = getattr(event_or_text, "text", event_or_text) or ""
    return str(text).startswith("[Continuing toward your standing goal]\nGoal:")


def _parse_reasoning_command_args(raw_args: str) -> tuple[str, bool]:
    """Parse `/reasoning` args into `(value, persist_global)`.

    `/reasoning <level>` is session-scoped by default. `--global` may be
    supplied in any position to persist the change to config.yaml.
    """
    text = str(raw_args or "").strip().replace("—", "--")
    if not text:
        return "", False
    try:
        tokens = shlex.split(text)
    except ValueError:
        tokens = text.split()

    persist_global = False
    value_tokens = []
    for token in tokens:
        if token == "--global":
            persist_global = True
        else:
            value_tokens.append(token)
    return " ".join(value_tokens).strip().lower(), persist_global


def _agent_has_active_subagents(running_agent: Any) -> bool:
    """Return True when *running_agent* is currently driving subagents
    via the ``delegate_task`` tool.

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
    from gateway.run import _AGENT_PENDING_SENTINEL
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


def _get_guild_id(event: MessageEvent) -> Optional[int]:
    """Extract Discord guild_id from the raw message object."""
    raw = getattr(event, "raw_message", None)
    if raw is None:
        return None
    # Slash command interaction
    if hasattr(raw, "guild_id") and raw.guild_id:
        return int(raw.guild_id)
    # Regular message
    if hasattr(raw, "guild") and raw.guild:
        return raw.guild.id
    return None


def _init_cached_agent_for_turn(agent: Any, interrupt_depth: int) -> None:
    """Reset per-turn state on a cached agent before a new turn starts.

    Both _last_activity_ts and _last_activity_desc are only reset for
    fresh external turns (depth 0); they are semantically paired —
    desc describes the activity *at* ts, so updating one without the
    other would make get_activity_summary() misleading.
    For interrupt-recursive turns both are preserved so the inactivity
    watchdog can accumulate stuck-turn idle time and fire the 30-min
    timeout (#15654).  The depth-0 reset is still needed: a session
    idle for 29 min would otherwise trip the watchdog before the new
    turn makes its first API call (#9051).
    """
    if interrupt_depth == 0:
        agent._last_activity_ts = time.time()
        agent._last_activity_desc = "starting new turn (cached)"
        # Reset the SessionDB flush cursor so the new turn's messages are
        # fully persisted — a stale value from the previous turn would
        # cause `_flush_messages_to_session_db` to skip new rows (#44327).
        if hasattr(agent, "_last_flushed_db_idx"):
            agent._last_flushed_db_idx = 0
    agent._api_call_count = 0
