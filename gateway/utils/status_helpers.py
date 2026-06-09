"""
Gateway status and notification utilities.

This module contains functions for handling gateway status messages,
notifications, and related operations.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional, List

from gateway.utils.gateway_helpers import (
    _gateway_platform_value,
    _redact_gateway_user_facing_secrets,
    _looks_like_gateway_provider_error,
    _gateway_provider_error_reply,
    _TELEGRAM_NOISY_STATUS_RE,
)

logger = logging.getLogger(__name__)


def _prepare_gateway_status_message(
    platform: Any,
    event_type: str,
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Filter/sanitize agent status callbacks before platform delivery.

    Args:
        platform: Gateway platform enum or string
        event_type: Type of status event
        message: Status message content
        metadata: Optional metadata dictionary

    Returns:
        Sanitized status message or None if should be filtered
    """
    text = str(message or "").strip()
    if not text:
        return None
    if _gateway_platform_value(platform) != "telegram":
        return text

    text = _redact_gateway_user_facing_secrets(text)
    if _TELEGRAM_NOISY_STATUS_RE.search(text):
        return None
    if _looks_like_gateway_provider_error(text):
        return _gateway_provider_error_reply(text)
    return text


async def _send_or_update_status_coro(
    adapter,
    chat_id: str,
    status_key: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Route a status message through adapter.send_or_update_status when supported.

    Issue #30045: adapters that implement send_or_update_status (currently
    Telegram) edit the previous bubble for the same status_key instead of
    appending a new one. Adapters without the method fall back to plain send.

    Args:
        adapter: Platform adapter instance
        chat_id: Chat ID to send status to
        status_key: Unique key for status identification (for updates)
        content: Status message content
        metadata: Optional metadata dictionary
    """
    sender = getattr(adapter, "send_or_update_status", None)
    if callable(sender):
        return await sender(chat_id, status_key, content, metadata=metadata or {})
    return await adapter.send(chat_id, content, metadata=metadata or {})


def _last_transcript_timestamp(messages: List[Dict[str, Any]]) -> Optional[float]:
    """Get the timestamp of the last message in a transcript.

    Args:
        messages: List of message dictionaries with 'timestamp' key

    Returns:
        Unix timestamp of last message or None if no timestamps found
    """
    if not messages:
        return None

    last_ts = None
    for msg in reversed(messages):
        ts = msg.get("timestamp")
        if ts is not None:
            try:
                last_ts = float(ts)
                break
            except (TypeError, ValueError):
                continue

    return last_ts


def _uses_telegram_observed_group_context(adapter) -> bool:
    """Check if the adapter uses Telegram observed group context feature.

    Args:
        adapter: Platform adapter instance

    Returns:
        True if adapter supports and uses observed group context
    """
    return hasattr(adapter, "use_observed_group_context") and getattr(
        adapter, "use_observed_group_context", False
    )


def _dequeue_pending_event(adapter, session_key: str) -> Optional[Any]:
    """Consume and return the full pending event for a session.

    Queued follow-ups must preserve their media metadata so they can re-enter
    the normal image/STT/document preprocessing path instead of being reduced
    to a placeholder string.

    Args:
        adapter: Platform adapter instance
        session_key: Session key to dequeue event for

    Returns:
        Pending event or None if no pending event
    """
    return adapter.get_pending_message(session_key)


def _build_status_notification(
    title: str,
    message: str,
    status_type: str = "info",
    timestamp: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a standardized status notification dictionary.

    Args:
        title: Notification title
        message: Notification message content
        status_type: Type of status (info, warning, error, success)
        timestamp: Optional Unix timestamp

    Returns:
        Dictionary with notification structure
    """
    return {
        "type": "status_notification",
        "title": title,
        "message": message,
        "status_type": status_type,
        "timestamp": timestamp or datetime.now().timestamp(),
    }
