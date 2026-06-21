"""Telegram topic-mode helpers extracted from gateway/run.py GatewayRunner.

Round 7 of gateway decomposition. These methods managed Telegram's forum-topic
multi-session mode: DM-topic detection, topic title sanitization, and the
user-facing help/lobby text. None touch instance state — they were defined
on GatewayRunner for locality only. Moved here as plain module functions.

Names kept identical; call sites rewritten self._X() -> _X().
"""

import re
from typing import Any, Optional

from gateway.platforms.base import Platform

import logging
logger = logging.getLogger("gateway.run")


def _is_telegram_dm_topic_target(
    platform: Optional[Platform],
    chat_id: Optional[str],
    thread_id: Optional[str],
    *,
    chat_type: Optional[str] = None,
    adapter: Optional[Any] = None,
) -> bool:
    """Return True when a target is a Telegram private DM topic lane."""
    if platform != Platform.TELEGRAM or thread_id is None:
        return False
    if chat_type == "dm":
        return True
    # Inspect operator-declared DM topics via the adapter's lookup. Resolve
    # the method on the CLASS, not the instance: getattr() on a MagicMock
    # auto-creates a callable child for any attribute, so an instance-level
    # lookup would report a DM topic for every test double. Only a
    # dict-shaped return counts as an operator-declared topic — a bare
    # MagicMock or other sentinel must not. Mirrors the guard in
    # _rename_telegram_topic_for_session_title.
    if adapter is not None and chat_id:
        get_dm_topic_info = getattr(type(adapter), "_get_dm_topic_info", None)
        if callable(get_dm_topic_info):
            try:
                topic_info = get_dm_topic_info(adapter, str(chat_id), str(thread_id))
            except Exception:
                logger.debug("Failed to inspect Telegram DM topic metadata", exc_info=True)
            else:
                return isinstance(topic_info, dict)
    return False


def _sanitize_telegram_topic_title(title: str) -> str:
    """Return a Bot API-safe forum topic name from a generated session title."""
    cleaned = re.sub(r"\s+", " ", str(title or "")).strip()
    if not cleaned:
        return "Hermes Chat"
    # Telegram forum topic names are short (currently 1-128 chars). Keep
    # extra room for multi-byte titles and avoid trailing ellipsis churn.
    if len(cleaned) > 120:
        cleaned = cleaned[:117].rstrip() + "..."
    return cleaned


def _telegram_topic_root_lobby_message() -> str:
    return (
        "This main chat is reserved for system commands.\n\n"
        "To start a new Hermes chat, open the All Messages topic at the top "
        "of this bot interface and send any message there. Telegram will "
        "create a new topic for that message; each topic works as an "
        "independent Hermes session."
    )


def _telegram_topic_root_new_message() -> str:
    return (
        "To start a new parallel Hermes chat, open the All Messages topic "
        "at the top of this bot interface and send any message there. "
        "Telegram will create a new topic for it.\n\n"
        "Each topic is an independent Hermes session. Use /new inside an "
        "existing topic only if you want to replace that topic's current session."
    )


def _telegram_topic_help_text() -> str:
    return (
        "/topic — enable multi-session DM mode (one bot, many parallel chats)\n"
        "\n"
        "Usage:\n"
        "  /topic             Enable topic mode, or show status if already on\n"
        "  /topic help        Show this message\n"
        "  /topic off         Disable topic mode and clear topic bindings\n"
        "  /topic <id>        Inside a topic: restore a previous session by ID\n"
        "\n"
        "How it works:\n"
        "1. Run /topic once in this DM — Hermes checks BotFather Threads\n"
        "   Settings are enabled and flips on multi-session mode.\n"
        "2. Tap All Messages at the top of the bot and send any message.\n"
        "   Telegram creates a new topic for that message; each topic is\n"
        "   an independent Hermes session (fresh history, fresh context).\n"
        "3. The root DM becomes a system lobby — send /topic, /status,\n"
        "   /help, /usage there. Normal prompts go in a topic.\n"
        "4. /new inside a topic resets just that topic's session.\n"
        "5. /topic <id> inside a topic restores an old session into it."
    )
