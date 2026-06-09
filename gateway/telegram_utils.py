"""
Telegram-specific utilities for GatewayRunner.

Extracted from gateway/run.py to reduce the God file size.
Provides functions for Telegram-specific operations.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

logger = logging.getLogger(__name__)


def telegram_topic_auto_rename_disabled(
    runner,  # GatewayRunner instance
    source: "SessionSource",
) -> bool:
    """Return True when operator disabled per-topic auto-rename for this Telegram chat.

    Controlled via ``gateway.platforms.telegram.extra.disable_topic_auto_rename``.
    Default is False (auto-rename enabled, preserves prior behaviour).
    """
    platform_cfg = (
        runner.config.platforms.get(source.platform)
        if getattr(runner, "config", None) and getattr(runner.config, "platforms", None)
        else None
    )
    if platform_cfg is None:
        return False
    extra = getattr(platform_cfg, "extra", None) or {}
    value = extra.get("disable_topic_auto_rename")
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def schedule_telegram_topic_title_rename(
    runner,  # GatewayRunner instance
    source: "SessionSource",
    session_id: str,
    title: str,
) -> None:
    """Schedule a topic rename from the auto-title background thread."""
    import asyncio
    import dataclasses

    from gateway.run import safe_schedule_threadsafe

    if not title or not runner._is_telegram_topic_lane(source):
        return
    if telegram_topic_auto_rename_disabled(runner, source):
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = getattr(runner, "_gateway_loop", None)
    if loop is None or loop.is_closed():
        return
    try:
        copied_source = dataclasses.replace(source)
    except Exception:
        copied_source = source
    future = safe_schedule_threadsafe(
        runner._rename_telegram_topic_for_session_title(copied_source, session_id, title),
        loop,
        logger=logger,
        log_message="Telegram topic title rename failed to schedule",
    )
    if future is None:
        return

    def _log_rename_failure(fut) -> None:
        try:
            fut.result()
        except Exception:
            logger.debug("Telegram topic title rename failed", exc_info=True)

    future.add_done_callback(_log_rename_failure)


def thread_metadata_for_target(
    runner,  # GatewayRunner instance
    platform: "Optional[Platform]",
    chat_id: "Optional[str]",
    thread_id: "Optional[str]",
    *,
    chat_type: "Optional[str]" = None,
    reply_to_message_id: "Optional[str]" = None,
    adapter: "Optional[Any]" = None,
) -> "Optional[Dict[str, Any]]":
    """Build thread metadata for synthetic sends that only have routing state."""
    from typing import Any, Dict, Optional

    if thread_id is None:
        return None
    metadata: Dict[str, Any] = {"thread_id": thread_id}
    if is_telegram_dm_topic_target(
        platform,
        chat_id,
        thread_id,
        chat_type=chat_type,
        adapter=adapter,
    ):
        metadata["telegram_dm_topic_reply_fallback"] = True
        # Telegram DM topic lanes need direct_messages_topic_id in metadata
        # so synthetic/queued messages (goal continuations, status notices)
        # route to the correct topic even when reply anchor is unavailable.
        tid = str(thread_id)
        if tid and tid not in {"", "1"}:
            metadata["direct_messages_topic_id"] = tid
        if reply_to_message_id is not None:
            metadata["telegram_reply_to_message_id"] = str(reply_to_message_id)
    return metadata


def is_telegram_dm_topic_target(
    platform: "Optional[Platform]",
    chat_id: "Optional[str]",
    thread_id: "Optional[str]",
    *,
    chat_type: "Optional[str]" = None,
    adapter: "Optional[Any]" = None,
) -> bool:
    """Return True when a target is a Telegram private DM topic lane."""
    from hermes_cli.enums import Platform

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


def record_telegram_topic_binding(
    runner,  # GatewayRunner instance
    source: "SessionSource",
    session_entry,
) -> None:
    """Persist the Telegram topic -> Hermes session binding for topic lanes."""
    session_db = getattr(runner, "_session_db", None)
    if session_db is None or not source.chat_id or not source.thread_id:
        return
    session_db.bind_telegram_topic(
        chat_id=str(source.chat_id),
        thread_id=str(source.thread_id),
        user_id=str(source.user_id or ""),
        session_key=session_entry.session_key,
        session_id=session_entry.session_id,
    )


def sync_telegram_topic_binding(
    runner,  # GatewayRunner instance
    source: "SessionSource",
    session_entry,
    *,
    reason: str,
) -> None:
    """Update the topic binding to point at ``session_entry.session_id``.

    Telegram topic lanes persist a (chat_id, thread_id) -> session_id row
    so reopening a topic in a fresh process resumes the right Hermes
    session. When compression rotates ``session_entry.session_id`` mid-turn,
    the binding goes stale and the next inbound message in that topic
    reloads the oversized parent transcript instead of the compressed
    child, retriggering preflight compression — sometimes in a loop
    (#20470, #29712, #33414).
    """
    if not runner._is_telegram_topic_lane(source):
        return
    try:
        record_telegram_topic_binding(runner, source, session_entry)
    except Exception:
        logger.debug(
            "telegram topic binding refresh failed (%s)", reason, exc_info=True,
        )


def recover_telegram_topic_thread_id(
    runner,  # GatewayRunner instance
    source: "SessionSource",
) -> "Optional[str]":
    """Pin DM-topic routing to the user's last-active topic.

    Telegram can omit ``message_thread_id`` or surface General (``1``)
    for some topic-mode DM replies. In those lobby-shaped cases, keep the
    conversation attached to the user's most-recent bound topic.

    Do not rewrite a non-lobby, previously-unbound thread id: a newly
    created Telegram DM topic is also "unknown" until the first inbound
    message is recorded, and rewriting it would send that brand-new topic's
    answer into an older lane. Returns None to leave the source alone.
    """
    from hermes_cli.enums import Platform

    if (
        source.platform != Platform.TELEGRAM
        or source.chat_type != "dm"
        or not source.chat_id
        or not source.user_id
        or not runner._telegram_topic_mode_enabled(source)
    ):
        return None
    inbound = str(source.thread_id or "")
    is_lobby = not inbound or inbound in runner._TELEGRAM_GENERAL_TOPIC_IDS
    if not is_lobby:
        # A non-lobby, unknown thread_id is most likely the first message in
        # a brand-new Telegram DM topic. Preserve it so it can be recorded
        # as a new independent lane below instead of hijacking the latest
        # existing topic binding.
        return None
    session_db = getattr(runner, "_session_db", None)
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
    for b in bindings:  # newest-first
        if str(b.get("user_id") or "") == user_id:
            recovered = str(b.get("thread_id") or "")
            if recovered and recovered != inbound:
                return recovered
            return None
    return None


def telegram_topic_mode_enabled(
    runner,  # GatewayRunner instance
    source: "SessionSource",
) -> bool:
    """Return whether Telegram DM topic mode is active for this chat."""
    from hermes_cli.enums import Platform

    if source.platform != Platform.TELEGRAM or source.chat_type != "dm":
        return False
    session_db = getattr(runner, "_session_db", None)
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
    # Only honor a real True from the SessionDB. Any other value
    # (including MagicMock instances from test fixtures that didn't
    # opt into topic mode) means topic mode is off for this chat.
    return raw is True
