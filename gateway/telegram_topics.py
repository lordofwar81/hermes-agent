"""Telegram DM topic management for the gateway.

Extracted from gateway/run.py to reduce the God file size.
Handles Telegram topic-mode detection, lobby/lane routing,
reminder rate-limiting, topic binding persistence, and
thread-id recovery across Telegram's sometimes-flaky
message_thread_id delivery.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Telegram's General (pinned top) topic in forum-enabled private chats.
# Bot API behavior varies: some clients omit message_thread_id for
# General, others send "1". Treat both as "root" for lobby/lane purposes.
TELEGRAM_GENERAL_TOPIC_IDS = frozenset({"", "1"})

TELEGRAM_LOBBY_REMINDER_COOLDOWN_S = 30.0


def is_telegram_topic_mode_enabled(
    source: Any,
    session_db: Any,
) -> bool:
    """Return whether Telegram DM topic mode is active for this chat."""
    try:
        from gateway.platforms.base import Platform
    except ImportError:
        return False

    if source.platform != Platform.TELEGRAM or source.chat_type != "dm":
        return False
    if session_db is None:
        return False
    try:
        raw = session_db.is_telegram_topic_mode_enabled(
            chat_id=str(source.chat_id),
            user_id=str(source.user_id),
        )
    except Exception:
        logger.debug("Failed to read Telegram topic mode state", exc_info=True)
        return False
    return raw is True


def is_telegram_topic_root_lobby(
    source: Any,
    topic_mode_enabled: bool,
) -> bool:
    """True for the main Telegram DM (or General topic) when topic mode has made it a lobby."""
    try:
        from gateway.platforms.base import Platform
    except ImportError:
        return False

    if source.platform != Platform.TELEGRAM or source.chat_type != "dm":
        return False
    if not topic_mode_enabled:
        return False
    tid = str(source.thread_id or "")
    return tid in TELEGRAM_GENERAL_TOPIC_IDS


def is_telegram_topic_lane(
    source: Any,
    topic_mode_enabled: bool,
) -> bool:
    """True for a user-created Telegram private-chat topic lane."""
    try:
        from gateway.platforms.base import Platform
    except ImportError:
        return False

    if source.platform != Platform.TELEGRAM or source.chat_type != "dm":
        return False
    if not topic_mode_enabled:
        return False
    tid = str(source.thread_id or "")
    if not tid or tid in TELEGRAM_GENERAL_TOPIC_IDS:
        return False
    return True


def should_send_telegram_lobby_reminder(
    chat_id: str,
    reminder_ts: dict,
) -> bool:
    """Rate-limit root-DM lobby reminders to one message per cooldown window."""
    if not chat_id:
        return True
    now = time.monotonic()
    last = reminder_ts.get(chat_id, 0.0)
    if now - last < TELEGRAM_LOBBY_REMINDER_COOLDOWN_S:
        return False
    reminder_ts[chat_id] = now
    return True


def telegram_topic_root_lobby_message() -> str:
    return (
        "This main chat is reserved for system commands.\n\n"
        "To start a new Hermes chat, open the All Messages topic at the top "
        "of this bot interface and send any message there. Telegram will "
        "create a new topic for that message; each topic works as an "
        "independent Hermes session."
    )


def telegram_topic_root_new_message() -> str:
    return (
        "To start a new parallel Hermes chat, open the All Messages topic "
        "at the top of this bot interface and send any message there. "
        "Telegram will create a new topic for it.\n\n"
        "Each topic is an independent Hermes session. Use /new inside an "
        "existing topic only if you want to replace that topic's current session."
    )


def telegram_topic_new_header(
    source: Any,
    is_topic_lane: bool,
) -> Optional[str]:
    if not is_topic_lane:
        return None
    return (
        "Started a new Hermes session in this topic.\n\n"
        "Tip: for parallel work, open All Messages and send a message there "
        "to create a separate topic instead of using /new here. /new replaces "
        "the session attached to the current topic."
    )


def record_telegram_topic_binding(
    source: Any,
    session_entry: Any,
    session_db: Any,
) -> None:
    """Persist the Telegram topic -> Hermes session binding for topic lanes."""
    if session_db is None or not source.chat_id or not source.thread_id:
        return
    session_db.bind_telegram_topic(
        chat_id=str(source.chat_id),
        thread_id=str(source.thread_id),
        user_id=str(source.user_id or ""),
        session_key=session_entry.session_key,
        session_id=session_entry.session_id,
    )


def recover_telegram_topic_thread_id(
    source: Any,
    topic_mode_enabled: bool,
    session_db: Any,
) -> Optional[str]:
    """Pin DM-topic routing to the user's last-active topic.

    Telegram can omit ``message_thread_id`` or surface General (``1``)
    for some topic-mode DM replies. In those lobby-shaped cases, keep the
    conversation attached to the user's most-recent bound topic.

    Do not rewrite a non-lobby, previously-unbound thread id: a newly
    created Telegram DM topic is also "unknown" until the first inbound
    message is recorded, and rewriting it would send that brand-new topic's
    answer into an older lane. Returns None to leave the source alone.
    """
    try:
        from gateway.platforms.base import Platform
    except ImportError:
        return None

    if (
        source.platform != Platform.TELEGRAM
        or source.chat_type != "dm"
        or not source.chat_id
        or not source.user_id
        or not topic_mode_enabled
    ):
        return None
    inbound = str(source.thread_id or "")
    is_lobby = not inbound or inbound in TELEGRAM_GENERAL_TOPIC_IDS
    if not is_lobby:
        return None
    if session_db is None:
        return None
    try:
        bindings = session_db.list_telegram_topic_bindings_for_chat(
            chat_id=str(source.chat_id),
        )
    except Exception:
        logger.debug("topic-recover: read failed", exc_info=True)
        return None
    if not bindings:
        return None
    user_id = str(source.user_id)
    for b in bindings:
        if str(b.get("user_id") or "") == user_id:
            recovered = str(b.get("thread_id") or "")
            if recovered and recovered != inbound:
                return recovered
            return None
    return None
