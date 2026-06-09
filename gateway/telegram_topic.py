"""
Telegram topic mode management.

This module handles Telegram-specific topic functionality for group chats,
including topic mode detection, lobby management, and message formatting.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def telegram_topic_mode_enabled(topic_mode_states: Dict[str, bool], session_key: str) -> bool:
    """Check if Telegram topic mode is enabled for a session.

    Args:
        topic_mode_states: Dict of session topic mode states
        session_key: Current session key

    Returns:
        True if topic mode is enabled for this session
    """
    return topic_mode_states.get(session_key, False)


def is_telegram_topic_root_lobby(source: Any) -> bool:
    """Check if this is the Telegram topic root lobby.

    Args:
        source: MessageEvent source object

    Returns:
        True if this is the root lobby in topic mode
    """
    # Check if we're in Telegram and this is the root topic
    platform = getattr(source, "platform", None)
    if platform is None:
        return False

    from gateway.platforms.base import Platform
    if platform != Platform.TELEGRAM:
        return False

    # Check if this is the root topic (no thread_id)
    thread_id = getattr(source, "thread_id", None)
    is_root = thread_id is None or thread_id == "root"

    # Check if topic mode is enabled for this chat
    # (This would need access to topic_mode_states, but for now
    # we check if there's a channel_prompt indicating topic mode)
    channel_prompt = getattr(source, "channel_prompt", None)
    has_topic_mode = channel_prompt and "topic" in channel_prompt.lower()

    return is_root and has_topic_mode


def is_telegram_topic_lane(source: Any) -> bool:
    """Check if this is a Telegram topic lane.

    Args:
        source: MessageEvent source object

    Returns:
        True if this is a topic lane (not root lobby)
    """
    platform = getattr(source, "platform", None)
    if platform is None:
        return False

    from gateway.platforms.base import Platform
    if platform != Platform.TELEGRAM:
        return False

    thread_id = getattr(source, "thread_id", None)
    return thread_id is not None and thread_id != "root"


def should_send_telegram_lobby_reminder(
    source: Any,
    session_messages: int,
    reminder_sent: bool,
) -> bool:
    """Check if we should send a Telegram topic lobby reminder.

    Args:
        source: MessageEvent source object
        session_messages: Number of messages in session
        reminder_sent: Whether reminder was already sent

    Returns:
        True if reminder should be sent
    """
    if reminder_sent:
        return False

    if not is_telegram_topic_root_lobby(source):
        return False

    # Send reminder after a few messages to give context
    return session_messages >= 3


def telegram_topic_root_lobby_message() -> str:
    """Get the message for Telegram topic root lobby.

    Returns:
        Message to display in root lobby
    """
    return (
        "👋 Welcome to the topic lobby! Use /topic <name> to create a new "
        "topic conversation, or reply to an existing topic to continue."
    )


def telegram_topic_root_new_message(topic_name: str) -> str:
    """Get the message for a new Telegram topic.

    Args:
        topic_name: Name of the new topic

    Returns:
        Message to display for new topic
    """
    return f"✨ Created new topic: {topic_name}"


def telegram_topic_new_header(topic_name: str, message_count: int) -> str:
    """Get the header for a Telegram topic message.

    Args:
        topic_name: Name of the topic
        message_count: Number of messages in topic

    Returns:
        Header string for topic message
    """
    return f"📎 {topic_name} ({message_count} messages)"
