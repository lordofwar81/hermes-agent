"""Background-process / CLI handoff methods for ``GatewayRunner``.

Round 21 of the god-file decomposition. Extracted verbatim into a mixin.
``self.*`` references (``self._running``, ``self._session_db``,
``self.adapters``, ``self.config``, ``self.session_store``,
``self._evict_cached_agent``, ``self._release_running_agent_state``,
``self._handle_message``, ``self._get_cached_session_source``,
``self._build_process_event_source``, ``self._process_handoff``) resolve
unchanged via the MRO. Behavior-neutral lift matching the existing mixin
pattern.

``logger`` and ``_parse_session_key`` are lazy-imported inside the methods
that use them to avoid a circular import (gateway.run imports this mixin at
module top). The session/platform type helpers (``SessionSource``,
``build_session_key``, ``Platform``, ``MessageEvent``, ``MessageType``,
``_BUILTIN_PLATFORM_VALUES``, ``_load_background_notifications_mode``) are
imported at module top from their owning modules with no circular dependency
on gateway.run. (``_process_handoff`` and ``_build_process_event_source``
retain their original in-body ``from gateway... import`` statements — now
redundant but kept verbatim for a behavior-neutral lift.)
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

from gateway.config import Platform, _BUILTIN_PLATFORM_VALUES
from gateway.gateway_config_loaders import _load_background_notifications_mode
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


class GatewayProcessMixin:
    """Background-process / CLI handoff methods for ``GatewayRunner``."""

    async def _handoff_watcher(self, interval: float = 2.0) -> None:
        """Background task that processes pending CLI→gateway session handoffs.

        Polls ``state.db`` for sessions in ``handoff_state='pending'`` and,
        for each one:

        1. Atomically claims it (pending → running).
        2. Resolves the destination platform's configured home channel.
        3. Re-binds the gateway's session_key for that home channel to the
           CLI's existing session_id via ``session_store.switch_session`` so
           the full role-aware transcript replays on the next agent turn.
        4. Forges a synthetic ``MessageEvent`` (``internal=True``) with a
           handoff-notice text and dispatches through the normal gateway
           message pipeline so the agent runs and replies on the platform.
        5. Marks the row ``completed`` (or ``failed`` with ``handoff_error``).

        The CLI process is poll-blocked on the row's terminal state and
        prints the result to the user.
        """
        from gateway.run import logger
        # Initial delay so the gateway is fully connected to its platforms
        # before we try to dispatch handoffs through them.
        await asyncio.sleep(5)
        while self._running:
            try:
                if self._session_db is None:
                    await asyncio.sleep(interval)
                    continue
                pending = self._session_db.list_pending_handoffs()
                for row in pending:
                    session_id = row.get("id")
                    if not session_id:
                        continue
                    if not self._session_db.claim_handoff(session_id):
                        # Another tick or another gateway already claimed it.
                        continue
                    try:
                        await self._process_handoff(row)
                        self._session_db.complete_handoff(session_id)
                    except Exception as exc:
                        logger.warning(
                            "Handoff for session %s failed: %s",
                            session_id, exc, exc_info=True,
                        )
                        self._session_db.fail_handoff(session_id, str(exc))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Handoff watcher tick error: %s", exc, exc_info=True)
            await asyncio.sleep(interval)

    async def _process_handoff(self, row: Dict[str, Any]) -> None:
        """Execute one handoff row. Raises on failure (caller marks failed)."""
        from gateway.config import Platform
        from gateway.session import SessionSource, build_session_key
        from gateway.platforms.base import MessageEvent

        cli_session_id = row["id"]
        platform_name = (row.get("handoff_platform") or "").strip().lower()
        if not platform_name:
            raise RuntimeError("handoff_platform is empty")

        # Resolve platform enum
        try:
            platform = Platform(platform_name)
        except (ValueError, KeyError):
            raise RuntimeError(f"unknown platform '{platform_name}'")

        # Adapter must be live
        adapter = self.adapters.get(platform)
        if not adapter:
            raise RuntimeError(
                f"platform '{platform_name}' is not active in this gateway"
            )

        # Home channel must be configured
        home = self.config.get_home_channel(platform)
        if not home or not home.chat_id:
            raise RuntimeError(
                f"no home channel configured for {platform_name}; "
                f"run /sethome on the desired chat first"
            )

        cli_title = row.get("title") or cli_session_id[:8]

        # Try to create a fresh thread on the destination so the handoff
        # has its own scrollback. Adapter returns None if threading isn't
        # supported (Matrix/WhatsApp/Signal/SMS) or if creation failed
        # (no permission, topics-mode off, parent is a DM, etc.). When
        # None we fall through to using the home channel directly — the
        # synthetic turn still lands; just without thread isolation.
        thread_name = f"Hermes — {cli_title}"
        try:
            new_thread_id = await adapter.create_handoff_thread(
                str(home.chat_id), thread_name,
            )
        except Exception as exc:
            from gateway.run import logger
            logger.debug(
                "Handoff: create_handoff_thread raised on %s: %s",
                platform_name, exc, exc_info=True,
            )
            new_thread_id = None

        # Use the new thread if the adapter created one; otherwise fall
        # back to whatever thread (if any) the home channel was configured
        # with.
        effective_thread_id = new_thread_id or (
            str(home.thread_id) if home.thread_id else None
        )

        # Determine chat_type for the destination source. If we created a
        # thread, key the session_key as a thread (build_session_key sets
        # thread sessions to user-shared by default, which is what we
        # want — the synthetic turn and any later real-user message both
        # land on the same key without needing a user_id).
        if new_thread_id:
            dest_chat_type = "thread"
        else:
            # No thread — assume DM-style for the home channel. For
            # group/channel home channels without thread support
            # (Matrix/WhatsApp/Signal), the platform's own keying makes
            # the synthetic turn shared anyway (single-DM platforms).
            dest_chat_type = "dm"

        dest_source = SessionSource(
            platform=platform,
            chat_id=str(home.chat_id),
            chat_name=home.name,
            chat_type=dest_chat_type,
            user_id="system:handoff",
            user_name="Handoff",
            thread_id=effective_thread_id,
        )

        # Compute the gateway's session_key for that destination using the
        # same rules its adapters use, so switch_session targets the right
        # entry. For thread destinations build_session_key keys without
        # user_id (thread_sessions_per_user defaults to False) — so the
        # next real user message in the thread shares this same session.
        platform_cfg = self.config.platforms.get(platform)
        extra = platform_cfg.extra if platform_cfg else {}
        session_key = build_session_key(
            dest_source,
            group_sessions_per_user=extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=extra.get("thread_sessions_per_user", False),
        )

        # Make sure there's an entry in the session_store for this key. If
        # the home channel has never been used, get_or_create_session
        # creates one; switch_session then re-points it.
        self.session_store.get_or_create_session(dest_source)

        # Re-bind the destination key to the CLI session_id. switch_session
        # ends the prior session in SQLite and reopens the CLI session under
        # the new key. The CLI's transcript becomes the active one for the
        # gateway from this moment on.
        switched = self.session_store.switch_session(session_key, cli_session_id)
        if switched is None:
            raise RuntimeError(
                f"could not switch session key {session_key} → {cli_session_id}"
            )

        # Evict any cached AIAgent for this session_key so the next dispatch
        # rebuilds it against the CLI session_id (mirrors /resume / /branch).
        self._evict_cached_agent(session_key)

        # Cancel any in-flight running-agent state for the destination key
        # so the synthetic turn isn't queued behind a stale running flag.
        self._release_running_agent_state(session_key)

        synthetic_text = (
            f"[Session was just handed off from CLI (\"{cli_title}\") to this "
            f"channel. The full prior conversation history is loaded above. "
            f"Briefly confirm you're working here and summarize what we were "
            f"working on, so the user can continue from this device.]"
        )

        synthetic_event = MessageEvent(
            text=synthetic_text,
            source=dest_source,
            internal=True,
        )

        from gateway.run import logger
        logger.info(
            "Handoff: dispatching synthetic turn for CLI session %s → %s "
            "(home=%s, thread=%s, session_key=%s)",
            cli_session_id, platform_name, home.chat_id, effective_thread_id,
            session_key,
        )

        # Dispatch through the runner directly. Going through
        # adapter.handle_message would spawn a background task and we'd
        # lose synchronous error visibility; calling _handle_message inline
        # keeps the success/failure path observable for the watcher.
        response_text = await self._handle_message(synthetic_event)
        if not response_text:
            # Streaming may have already delivered the response inline.
            # Either way, agent ran without raising — count as success.
            return

        # Send the agent's reply to the destination. Route to the new
        # thread if we created one; otherwise the configured home channel
        # (which may itself carry a thread_id).
        send_metadata: Dict[str, Any] = {}
        if effective_thread_id:
            send_metadata["thread_id"] = effective_thread_id
        try:
            result = await adapter.send(
                chat_id=str(home.chat_id),
                content=response_text,
                metadata=send_metadata or None,
            )
        except Exception as exc:
            raise RuntimeError(f"adapter.send failed: {exc}") from exc

        if not getattr(result, "success", True):
            err = getattr(result, "error", "send returned success=False")
            raise RuntimeError(f"adapter.send failed: {err}")

    def _build_process_event_source(self, evt: dict):
        """Resolve the canonical source for a synthetic background-process event.

        Prefer the persisted session-store origin for the event's session key.
        Falling back to the currently active foreground event is what causes
        cross-topic bleed, so don't do that.
        """
        from gateway.session import SessionSource

        session_key = str(evt.get("session_key") or "").strip()
        derived_platform = ""
        derived_chat_type = ""
        derived_chat_id = ""

        if session_key:
            try:
                self.session_store._ensure_loaded()
                entry = self.session_store._entries.get(session_key)
                if entry and getattr(entry, "origin", None):
                    return entry.origin
            except Exception as exc:
                from gateway.run import logger
                logger.debug(
                    "Synthetic process-event session-store lookup failed for %s: %s",
                    session_key,
                    exc,
                )

            cached_source = self._get_cached_session_source(session_key)
            if cached_source is not None:
                return cached_source

            from gateway.run import _parse_session_key
            _parsed = _parse_session_key(session_key)
            if _parsed:
                derived_platform = _parsed["platform"]
                derived_chat_type = _parsed["chat_type"]
                derived_chat_id = _parsed["chat_id"]

        platform_name = str(evt.get("platform") or derived_platform or "").strip().lower()
        chat_type = str(evt.get("chat_type") or derived_chat_type or "").strip().lower()
        chat_id = str(evt.get("chat_id") or derived_chat_id or "").strip()
        if not platform_name or not chat_type or not chat_id:
            return None

        try:
            platform = Platform(platform_name)
            # Reject arbitrary strings that create dynamic pseudo-members.
            # Built-in platforms are always valid; plugin platforms must be
            # registered in the platform registry.
            if platform.value not in _BUILTIN_PLATFORM_VALUES:
                try:
                    from gateway.platform_registry import platform_registry
                    if not platform_registry.is_registered(platform.value):
                        raise ValueError(platform_name)
                except Exception:
                    raise ValueError(platform_name)
        except Exception:
            from gateway.run import logger
            logger.warning(
                "Synthetic process event has invalid platform metadata: %r",
                platform_name,
            )
            return None

        return SessionSource(
            platform=platform,
            chat_id=chat_id,
            chat_type=chat_type,
            thread_id=str(evt.get("thread_id") or "").strip() or None,
            user_id=str(evt.get("user_id") or "").strip() or None,
            user_name=str(evt.get("user_name") or "").strip() or None,
        )

    async def _run_process_watcher(self, watcher: dict) -> None:
        """
        Periodically check a background process and push updates to the user.

        Runs as an asyncio task. Stays silent when nothing changed.
        Auto-removes when the process exits or is killed.

        Notification mode (from ``display.background_process_notifications``):
          - ``all``    — running-output updates + final message
          - ``result`` — final completion message only
          - ``error``  — final message only when exit code != 0
          - ``off``    — no messages at all
        """
        from tools.process_registry import process_registry

        session_id = watcher["session_id"]
        interval = watcher["check_interval"]
        session_key = watcher.get("session_key", "")
        platform_name = watcher.get("platform", "")
        chat_id = watcher.get("chat_id", "")
        thread_id = watcher.get("thread_id", "")
        user_id = watcher.get("user_id", "")
        user_name = watcher.get("user_name", "")
        message_id = str(watcher.get("message_id") or "").strip() or None
        agent_notify = watcher.get("notify_on_complete", False)
        notify_mode = _load_background_notifications_mode()

        from gateway.run import logger
        logger.debug("Process watcher started: %s (every %ss, notify=%s, agent_notify=%s)",
                      session_id, interval, notify_mode, agent_notify)

        if notify_mode == "off" and not agent_notify:
            # Still wait for the process to exit so we can log it, but don't
            # push any messages to the user.
            while True:
                await asyncio.sleep(interval)
                session = process_registry.get(session_id)
                if session is None or session.exited:
                    break
            logger.debug("Process watcher ended (silent): %s", session_id)
            return

        last_output_len = 0
        while True:
            await asyncio.sleep(interval)

            session = process_registry.get(session_id)
            if session is None:
                break

            current_output_len = len(session.output_buffer)
            has_new_output = current_output_len > last_output_len
            last_output_len = current_output_len

            if session.exited:
                # --- Agent-triggered completion: inject synthetic message ---
                # Skip if the agent already consumed the result via wait/poll/log
                from tools.process_registry import format_process_notification, process_registry as _pr_check
                if agent_notify and not _pr_check.is_completion_consumed(session_id):
                    from tools.ansi_strip import strip_ansi
                    _raw = strip_ansi(session.output_buffer) if session.output_buffer else ""
                    # Truncate at line boundaries so notifications never start
                    # mid-line (fixes #23284). Keep the last ~2000 chars but
                    # snap to the nearest preceding newline, then prepend a
                    # truncation marker when output was cut.
                    _LIMIT = 2000
                    if len(_raw) > _LIMIT:
                        _tail = _raw[-_LIMIT:]
                        _nl = _tail.find("\n")
                        _tail = _tail[_nl + 1:] if _nl != -1 else _tail
                        _out = f"[… output truncated — showing last {len(_tail)} chars]\n{_tail}"
                    else:
                        _out = _raw
                    synth_text = format_process_notification({
                        "type": "completion",
                        "session_id": session_id,
                        "command": session.command,
                        "exit_code": session.exit_code,
                        "completion_reason": getattr(session, "completion_reason", "exited"),
                        "termination_source": getattr(session, "termination_source", ""),
                        "output": _out,
                    })
                    if not synth_text:
                        break
                    source = self._build_process_event_source({
                        "session_id": session_id,
                        "session_key": session_key,
                        "platform": platform_name,
                        "chat_id": chat_id,
                        "thread_id": thread_id,
                        "user_id": user_id,
                        "user_name": user_name,
                    })
                    if not source:
                        logger.warning(
                            "Dropping completion notification with no routing metadata for process %s",
                            session_id,
                        )
                        break

                    adapter = None
                    for p, a in self.adapters.items():
                        if p == source.platform:
                            adapter = a
                            break
                    if adapter and source.chat_id:
                        try:
                            synth_event = MessageEvent(
                                text=synth_text,
                                message_type=MessageType.TEXT,
                                source=source,
                                internal=True,
                                message_id=message_id,
                            )
                            logger.info(
                                "Process %s finished — injecting agent notification for session %s chat=%s thread=%s",
                                session_id,
                                session_key,
                                source.chat_id,
                                source.thread_id,
                            )
                            await adapter.handle_message(synth_event)
                        except Exception as e:
                            logger.error("Agent notify injection error: %s", e)
                    break

                # --- Normal text-only notification ---
                # Decide whether to notify based on mode
                should_notify = (
                    notify_mode in {"all", "result"}
                    or (notify_mode == "error" and session.exit_code not in {0, None})
                )
                if should_notify:
                    new_output = session.output_buffer[-1000:] if session.output_buffer else ""
                    message_text = (
                        f"[Background process {session_id} finished with exit code {session.exit_code}~ "
                        f"Here's the final output:\n{new_output}]"
                    )
                    adapter = None
                    for p, a in self.adapters.items():
                        if p.value == platform_name:
                            adapter = a
                            break
                    if adapter and chat_id:
                        try:
                            send_meta = {"thread_id": thread_id} if thread_id else None
                            await adapter.send(chat_id, message_text, metadata=send_meta)
                        except Exception as e:
                            logger.error("Watcher delivery error: %s", e)
                break

            elif has_new_output and notify_mode == "all" and not agent_notify:
                # New output available -- deliver status update (only in "all" mode)
                # Skip periodic updates for agent_notify watchers (they only care about completion)
                new_output = session.output_buffer[-500:] if session.output_buffer else ""
                message_text = (
                    f"[Background process {session_id} is still running~ "
                    f"New output:\n{new_output}]"
                )
                adapter = None
                for p, a in self.adapters.items():
                    if p.value == platform_name:
                        adapter = a
                        break
                if adapter and chat_id:
                    try:
                        send_meta = {"thread_id": thread_id} if thread_id else None
                        await adapter.send(chat_id, message_text, metadata=send_meta)
                    except Exception as e:
                        logger.error("Watcher delivery error: %s", e)

        logger.debug("Process watcher ended: %s", session_id)
