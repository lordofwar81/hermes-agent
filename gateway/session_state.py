"""
Session run-generation tracking for the gateway.

Extracted from gateway/run.py to reduce the God file size.
Provides functions for managing monotonically-increasing run generation
tokens, binding generation to adapter events, and querying generation
currency.

Every top-level gateway turn gets a monotonically increasing token.
If a later command like /stop or /new invalidates that token while the
old worker is still unwinding, the late result can be recognized and
dropped instead of bleeding into the fresh session.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def begin_session_run_generation(
    generations: Dict[str, int],
    session_key: str,
) -> int:
    """Claim a fresh run generation token for ``session_key``."""
    if not session_key:
        return 0
    next_gen = int(generations.get(session_key, 0)) + 1
    generations[session_key] = next_gen
    return next_gen


def invalidate_session_run_generation(
    generations: Dict[str, int],
    session_key: str,
    *,
    reason: str = "",
) -> int:
    """Invalidate any in-flight run token for ``session_key``."""
    generation = begin_session_run_generation(generations, session_key)
    if reason:
        logger.info(
            "Invalidated run generation for %s → %d (%s)",
            session_key,
            generation,
            reason,
        )
    return generation


def is_session_run_current(
    generations: Dict[str, int],
    session_key: str,
    generation: int,
) -> bool:
    """Return True when ``generation`` is still current for ``session_key``."""
    if not session_key:
        return True
    return int(generations.get(session_key, 0)) == int(generation)


def bind_adapter_run_generation(
    adapter: Any,
    session_key: str,
    generation: Optional[int],
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
