"""
Adapter utility functions for GatewayRunner.

Extracted from gateway/run.py to reduce the God file size.
Provides functions for adapter-related operations.
"""

import logging

logger = logging.getLogger(__name__)


def bind_adapter_run_generation(
    runner,  # GatewayRunner instance
    adapter,  # Platform adapter
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


def adapter_disconnect_timeout_secs(
    runner,  # GatewayRunner instance
) -> float:
    """Return the per-adapter disconnect timeout used during shutdown."""
    import os

    raw = os.getenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "").strip()
    if raw:
        try:
            timeout = float(raw)
        except ValueError:
            logger.warning(
                "Ignoring invalid HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT=%r",
                raw,
            )
        else:
            return max(0.0, timeout)
    return 5.0
