"""
Shutdown notification utilities for GatewayRunner.

Extracted from gateway/run.py to reduce the God file size.
Provides functions for sending shutdown/restart notifications to active sessions.
"""

import logging
from typing import Dict, Any, Optional, Set, Tuple

from hermes_cli.enums import Platform
from gateway.session import parse_session_key as _parse_session_key

logger = logging.getLogger(__name__)


async def notify_active_sessions_of_shutdown(
    runner: "GatewayRunner",
) -> None:
    """Send shutdown/restart notifications to active chats and home channels.

    Called at the very start of stop() — adapters are still connected so
    messages can be delivered. Best-effort: individual send failures are
    logged and swallowed so they never block the shutdown sequence.

    Args:
        runner: GatewayRunner instance
    """
    active = runner._snapshot_running_agents()
    restart_source = runner._restart_command_source if runner._restart_requested else None

    action = "restarting" if runner._restart_requested else "shutting down"
    hint = (
        "Your current task will be interrupted. "
        "Send any message after restart and I'll try to resume where you left off."
        if runner._restart_requested
        else "Your current task will be interrupted."
    )
    msg = f"⚠️ Gateway {action} — {hint}"

    notified: set[Tuple[str, str, Optional[str]]] = set()

    for session_key in active:
        source = None
        try:
            if getattr(runner, "session_store", None) is not None:
                runner.session_store._ensure_loaded()
                entry = runner.session_store._entries.get(session_key)
                source = getattr(entry, "origin", None) if entry else None
        except Exception as e:
            logger.debug(
                "Failed to load session origin for shutdown notification %s: %s",
                session_key,
                e,
            )

        if source is None:
            source = runner._get_cached_session_source(session_key)

        if source is not None:
            platform_str = source.platform.value
            chat_id = str(source.chat_id)
            thread_id = source.thread_id
        else:
            # Fall back to parsing the session key when no persisted
            # origin is available (legacy sessions/tests).
            _parsed = _parse_session_key(session_key)
            if not _parsed:
                continue
            platform_str = _parsed["platform"]
            chat_id = _parsed["chat_id"]
            thread_id = _parsed.get("thread_id")

        # Deduplicate only identical delivery targets. Thread/topic-aware
        # platforms can share a parent chat while still routing to distinct
        # destinations via metadata.
        dedup_key = (platform_str, chat_id, str(thread_id) if thread_id else None)
        if dedup_key in notified:
            continue

        try:
            platform = Platform(platform_str)
            adapter = runner.adapters.get(platform)
            if not adapter:
                continue

            platform_cfg = runner.config.platforms.get(platform)
            if platform_cfg is not None and not platform_cfg.gateway_restart_notification:
                logger.info(
                    "Shutdown notification suppressed for active session: %s has gateway_restart_notification=false",
                    platform_str,
                )
                continue

            reply_to_message_id = getattr(source, "message_id", None) if source is not None else None
            if reply_to_message_id is None and restart_source is not None:
                try:
                    restart_platform = restart_source.platform.value
                    restart_chat_id = str(restart_source.chat_id)
                    restart_thread_id = str(restart_source.thread_id) if restart_source.thread_id else None
                    if (restart_platform, restart_chat_id, restart_thread_id) == dedup_key:
                        reply_to_message_id = getattr(restart_source, "message_id", None)
                except Exception:
                    pass

            metadata = runner._thread_metadata_for_target(
                platform,
                chat_id,
                thread_id,
                chat_type=getattr(source, "chat_type", None) if source is not None else None,
                reply_to_message_id=reply_to_message_id,
                adapter=adapter,
            )

            result = await adapter.send(chat_id, msg, metadata=metadata)
            if result is not None and getattr(result, "success", True) is False:
                logger.debug(
                    "Failed to send shutdown notification to %s:%s: %s",
                    platform_str,
                    chat_id,
                    getattr(result, "error", "send returned success=False"),
                )
                continue

            notified.add(dedup_key)
            logger.info(
                "Sent shutdown notification to active chat %s:%s",
                platform_str, chat_id,
            )
        except Exception as e:
            logger.debug(
                "Failed to send shutdown notification to %s:%s: %s",
                platform_str, chat_id, e,
            )

    if runner._restart_requested and restart_source is not None:
        logger.debug("Skipping home-channel shutdown notifications for in-chat restart")
        return

    # Snapshot adapters up front: adapter.send() can hit a fatal error
    # path that pops the adapter from self.adapters (see _handle_fatal
    # elsewhere), which would otherwise trigger
    # ``RuntimeError: dictionary changed size during iteration`` —
    # observed in a user report during gateway shutdown.
    for platform, adapter in list(runner.adapters.items()):
        home = runner.config.get_home_channel(platform)
        if not home or not home.chat_id:
            continue

        platform_cfg = runner.config.platforms.get(platform)
        if platform_cfg is not None and not platform_cfg.gateway_restart_notification:
            logger.info(
                "Shutdown notification suppressed for home channel: %s has gateway_restart_notification=false",
                platform.value,
            )
            continue

        dedup_key = (platform.value, str(home.chat_id), str(home.thread_id) if home.thread_id else None)
        if dedup_key in notified:
            continue

        try:
            metadata = runner._thread_metadata_for_target(
                platform,
                home.chat_id,
                home.thread_id,
                adapter=adapter,
            )
            if metadata:
                result = await adapter.send(str(home.chat_id), msg, metadata=metadata)
            else:
                result = await adapter.send(str(home.chat_id), msg)
            if result is not None and getattr(result, "success", True) is False:
                logger.debug(
                    "Failed to send shutdown notification to home channel %s:%s: %s",
                    platform.value,
                    home.chat_id,
                    getattr(result, "error", "send returned success=False"),
                )
                continue

            notified.add(dedup_key)
            logger.info(
                "Sent shutdown notification to home channel %s:%s",
                platform.value,
                home.chat_id,
            )
        except Exception as e:
            logger.debug(
                "Failed to send shutdown notification to home channel %s:%s: %s",
                platform.value,
                home.chat_id,
                e,
            )
