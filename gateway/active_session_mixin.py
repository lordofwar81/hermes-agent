"""Active-session lifecycle methods for ``GatewayRunner``.

Round 24 of the god-file decomposition. Lifted verbatim into a mixin.
Three related subsystems moved together:

1. **Cross-process slot claiming** — ``_claim_active_session_slot``
   (``self._running_agents``, ``self._active_session_limit_message``,
   ``self.config``; calls ``hermes_cli.active_sessions.try_acquire_active_session``).
2. **Busy-path message handling** — ``_handle_active_session_busy_message``
   (authorization gate, draining case, queue/interrupt/steer modes,
   busy-ack debounce, subagent protection #30170, first-touch onboarding).
3. **Shutdown/restart fan-out** — ``_notify_active_sessions_of_shutdown``
   (active chats + home channels, dedup by delivery target, per-platform
   ``gateway_restart_notification`` opt-out).

``self.*`` references resolve unchanged via the MRO. Behavior-neutral lift
matching the existing mixin pattern.

``gateway.run`` module-level globals (``logger``, ``_AGENT_PENDING_SENTINEL``,
``_thread_metadata_for_source``, ``_thread_metadata_for_target``,
``_parse_session_key``, ``_agent_has_active_subagents``,
``merge_pending_message_event``, ``_load_gateway_config``,
``_platform_config_key``, ``_hermes_home``) are lazy-imported inside each
method body to avoid a circular import (``gateway.run`` imports this mixin
at module top). Type-only imports (``SessionSource``, ``MessageEvent``,
``MessageType``, ``Platform``, ``Any``, ``Optional``) come from modules
with no circular dependency on ``gateway.run`` and are imported at module
top.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


class GatewayActiveSessionMixin:
    """Active-session lifecycle methods for ``GatewayRunner``."""

    def _claim_active_session_slot(
        self,
        session_key: str,
        source: SessionSource,
    ) -> tuple[Any, Optional[str]]:
        """Claim a cross-process active-session slot for a new gateway turn."""
        from gateway.run import logger
        if session_key in getattr(self, "_running_agents", {}):
            return None, None
        local_limit_message = self._active_session_limit_message(session_key)
        if local_limit_message is not None:
            return None, local_limit_message
        try:
            from hermes_cli.active_sessions import try_acquire_active_session

            platform = source.platform.value if source and source.platform else "gateway"
            return try_acquire_active_session(
                session_id=session_key,
                surface=f"gateway:{platform}",
                config=getattr(self, "config", None),
                metadata={
                    "platform": platform,
                    "chat_id": getattr(source, "chat_id", "") or "",
                    "user_id": getattr(source, "user_id", "") or "",
                },
            )
        except Exception as exc:
            logger.warning("Failed to claim active session slot: %s", exc)
            return None, None
    async def _handle_active_session_busy_message(self, event: MessageEvent, session_key: str) -> bool:
        # --- Authorization gate (#17775) ---
        # The cold path (_handle_message) checks _is_user_authorized before
        # creating a session.  The busy path must enforce the same check;
        # otherwise unauthorized users in shared threads (Slack/Telegram/Discord)
        # can inject messages into an active session they don't own.
        from gateway.run import logger, _AGENT_PENDING_SENTINEL, _agent_has_active_subagents, _hermes_home, _load_gateway_config, _platform_config_key, _thread_metadata_for_source, merge_pending_message_event
        if not self._is_user_authorized(event.source):
            logger.warning(
                "Dropping message from unauthorized user in active session: "
                "user=%s (%s), platform=%s, session=%s",
                event.source.user_id,
                event.source.user_name,
                event.source.platform.value if event.source.platform else "unknown",
                session_key,
            )
            return True  # handled (silently dropped); do not fall through

        # --- Draining case (gateway restarting/stopping) ---
        if self._draining:
            adapter = self.adapters.get(event.source.platform)
            if not adapter:
                return True

            reply_anchor = self._reply_anchor_for_event(event)
            thread_meta = _thread_metadata_for_source(event.source, reply_anchor)
            if self._queue_during_drain_enabled():
                self._queue_or_replace_pending_event(session_key, event)
                message = f"⏳ Gateway {self._status_action_gerund()} — queued for the next turn after it comes back."
            else:
                message = f"⏳ Gateway is {self._status_action_gerund()} and is not accepting another turn right now."

            await adapter._send_with_retry(
                chat_id=event.source.chat_id,
                content=message,
                reply_to=(
                    reply_anchor
                    if event.source.platform == Platform.TELEGRAM
                    and event.source.chat_type == "dm"
                    and event.source.thread_id
                    else (None if event.source.platform == Platform.TELEGRAM and event.source.thread_id else event.message_id)
                ),
                metadata=thread_meta,
            )
            return True

        # Normal busy case (agent actively running a task)
        adapter = self.adapters.get(event.source.platform)
        if not adapter:
            return False  # let default path handle it

        running_agent = self._running_agents.get(session_key)

        effective_mode = self._busy_input_mode
        busy_text_mode = getattr(self, "_busy_text_mode", "interrupt")
        if (
            event.message_type == MessageType.TEXT
            and busy_text_mode == "queue"
            and effective_mode != "steer"
        ):
            return False

        # Steer mode: inject mid-run via running_agent.steer() instead of
        # queueing + interrupting.  If the agent isn't running yet
        # (sentinel) or lacks steer(), or the payload is empty, fall back
        # to queue semantics so nothing is lost.
        # #30170 — Subagent protection. ``AIAgent.interrupt()`` cascades
        # to every entry in the parent's ``_active_children`` list and
        # aborts in-flight ``delegate_task`` work. Demote ``interrupt``
        # to ``queue`` when the parent is currently driving subagents so
        # a conversational follow-up doesn't destroy minutes of subagent
        # work. Explicit ``/stop`` and ``/new`` slash commands go through
        # ``_interrupt_and_clear_session`` and are unaffected — the
        # operator still has a way to force-cancel everything.
        demoted_for_subagents = (
            effective_mode == "interrupt"
            and _agent_has_active_subagents(running_agent)
        )
        if demoted_for_subagents:
            logger.info(
                "Demoting busy_input_mode 'interrupt' to 'queue' for session %s "
                "because the running agent has active subagents (#30170)",
                session_key,
            )
            effective_mode = "queue"
        steered = False
        if effective_mode == "steer":
            steer_text = (event.text or "").strip()
            can_steer = (
                steer_text
                and running_agent is not None
                and running_agent is not _AGENT_PENDING_SENTINEL
                and hasattr(running_agent, "steer")
            )
            if can_steer:
                try:
                    steered = bool(running_agent.steer(steer_text))
                except Exception as exc:
                    logger.warning("Gateway steer failed for session %s: %s", session_key, exc)
                    steered = False
            if not steered:
                # Fall back to queue (merge into pending messages, no interrupt)
                effective_mode = "queue"

        # Store the message so it's processed as the next turn after the
        # current run finishes (or is interrupted).  Skip this for a
        # successful steer — the text already landed inside the run and
        # must NOT also be replayed as a next-turn user message.
        if not steered:
            merge_pending_message_event(
                adapter._pending_messages,
                session_key,
                event,
                merge_text=event.message_type == MessageType.TEXT,
            )

        is_queue_mode = effective_mode == "queue"
        is_steer_mode = effective_mode == "steer"

        # If not in queue/steer mode, interrupt the running agent immediately.
        # This aborts in-flight tool calls and causes the agent loop to exit
        # at the next check point.
        if effective_mode == "interrupt" and running_agent and running_agent is not _AGENT_PENDING_SENTINEL:
            try:
                running_agent.interrupt(event.text)
            except Exception:
                pass  # don't let interrupt failure block the ack

        # Check if busy ack is disabled — skip sending but still process the input.
        # Placed before debounce so we don't stamp a "last ack" timestamp that was
        # never actually delivered.
        busy_ack_enabled = os.environ.get("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true").lower() == "true"
        if not busy_ack_enabled:
            logger.debug("Busy ack suppressed for session %s", session_key)
            return True  # input still processed, just no ack sent

        # Debounce: only send an acknowledgment once every 30 seconds per session
        # to avoid spamming the user when they send multiple messages quickly
        _BUSY_ACK_COOLDOWN = 30
        now = time.time()
        last_ack = self._busy_ack_ts.get(session_key, 0)
        if now - last_ack < _BUSY_ACK_COOLDOWN:
            return True  # interrupt sent (if not queue), ack already delivered recently

        self._busy_ack_ts[session_key] = now

        # Build a status-rich acknowledgment. Mobile chat defaults keep this
        # terse; detailed iteration/tool state is still available in logs and
        # can be opted in per platform via display.platforms.<platform>.busy_ack_detail.
        from gateway.display_config import resolve_display_setting
        status_parts = []
        busy_ack_detail_enabled = bool(
            resolve_display_setting(
                _load_gateway_config(),
                _platform_config_key(event.source.platform),
                "busy_ack_detail",
                True,
            )
        )

        if busy_ack_detail_enabled and running_agent and running_agent is not _AGENT_PENDING_SENTINEL:
            try:
                summary = running_agent.get_activity_summary()
                iteration = summary.get("api_call_count", 0)
                max_iter = summary.get("max_iterations", 0)
                current_tool = summary.get("current_tool")
                start_ts = self._running_agents_ts.get(session_key, 0)
                if start_ts:
                    elapsed_min = int((now - start_ts) / 60)
                    if elapsed_min > 0:
                        status_parts.append(f"{elapsed_min} min elapsed")
                if max_iter:
                    status_parts.append(f"iteration {iteration}/{max_iter}")
                if current_tool:
                    status_parts.append(f"running: {current_tool}")
            except Exception:
                pass

        status_detail = f" ({', '.join(status_parts)})" if status_parts else ""
        if is_steer_mode:
            message = (
                f"⏩ Steered into current run{status_detail}. "
                f"Your message arrives after the next tool call."
            )
        elif is_queue_mode and demoted_for_subagents:
            # #30170 — explain the demotion so the user knows their
            # follow-up didn't accidentally kill the subagent and
            # discovers `/stop` as the explicit escape hatch.
            message = (
                f"⏳ Subagent working{status_detail} — your message is queued for "
                f"when it finishes (use /stop to cancel everything)."
            )
        elif is_queue_mode:
            message = (
                f"⏳ Queued for the next turn{status_detail}. "
                f"I'll respond once the current task finishes."
            )
        else:
            message = (
                f"⚡ Interrupting current task{status_detail}. "
                f"I'll respond to your message shortly."
            )

        # First-touch onboarding: the very first time a user sends a message
        # while the agent is busy, append a one-time hint explaining the
        # queue/interrupt knob.  Flag is persisted to config.yaml so it never
        # fires again on this install.
        try:
            from agent.onboarding import (
                BUSY_INPUT_FLAG,
                busy_input_hint_gateway,
                is_seen,
                mark_seen,
            )
            _user_cfg = _load_gateway_config()
            if not is_seen(_user_cfg, BUSY_INPUT_FLAG):
                if is_steer_mode:
                    _hint_mode = "steer"
                elif is_queue_mode:
                    _hint_mode = "queue"
                else:
                    _hint_mode = "interrupt"
                message = (
                    f"{message}\n\n"
                    f"{busy_input_hint_gateway(_hint_mode)}"
                )
                mark_seen(_hermes_home / "config.yaml", BUSY_INPUT_FLAG)
        except Exception as _onb_err:
            logger.debug("Failed to apply busy-input onboarding hint: %s", _onb_err)

        reply_anchor = self._reply_anchor_for_event(event)
        thread_meta = _thread_metadata_for_source(event.source, reply_anchor)
        try:
            await adapter._send_with_retry(
                chat_id=event.source.chat_id,
                content=message,
                reply_to=(
                    reply_anchor
                    if event.source.platform == Platform.TELEGRAM
                    and event.source.chat_type == "dm"
                    and event.source.thread_id
                    else (None if event.source.platform == Platform.TELEGRAM and event.source.thread_id else event.message_id)
                ),
                metadata=thread_meta,
            )
        except Exception as e:
            logger.debug("Failed to send busy-ack: %s", e)

        return True
    async def _notify_active_sessions_of_shutdown(self) -> None:
        """Send shutdown/restart notifications to active chats and home channels.

        Called at the very start of stop() — adapters are still connected so
        messages can be delivered. Best-effort: individual send failures are
        logged and swallowed so they never block the shutdown sequence.
        """
        from gateway.run import logger, _parse_session_key, _thread_metadata_for_target
        active = self._snapshot_running_agents()
        restart_source = self._restart_command_source if self._restart_requested else None

        action = "restarting" if self._restart_requested else "shutting down"
        hint = (
            "Your current task will be interrupted. "
            "Send any message after restart and I'll try to resume where you left off."
            if self._restart_requested
            else "Your current task will be interrupted."
        )
        msg = f"⚠️ Gateway {action} — {hint}"

        notified: set[tuple[str, str, Optional[str]]] = set()
        for session_key in active:
            source = None
            try:
                if getattr(self, "session_store", None) is not None:
                    self.session_store._ensure_loaded()
                    entry = self.session_store._entries.get(session_key)
                    source = getattr(entry, "origin", None) if entry else None
            except Exception as e:
                logger.debug(
                    "Failed to load session origin for shutdown notification %s: %s",
                    session_key,
                    e,
                )

            if source is None:
                source = self._get_cached_session_source(session_key)

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
                adapter = self.adapters.get(platform)
                if not adapter:
                    continue

                platform_cfg = self.config.platforms.get(platform)
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

                metadata = _thread_metadata_for_target(
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

        if self._restart_requested and restart_source is not None:
            logger.debug("Skipping home-channel shutdown notifications for in-chat restart")
            return

        # Snapshot adapters up front: adapter.send() can hit a fatal error
        # path that pops the adapter from self.adapters (see _handle_fatal
        # elsewhere), which would otherwise trigger
        # ``RuntimeError: dictionary changed size during iteration`` —
        # observed in a user report during gateway shutdown.
        for platform, adapter in list(self.adapters.items()):
            home = self.config.get_home_channel(platform)
            if not home or not home.chat_id:
                continue

            platform_cfg = self.config.platforms.get(platform)
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
                metadata = _thread_metadata_for_target(
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
