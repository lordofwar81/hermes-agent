"""Busy-agent message dispatch — round 50 of the gateway god-file decomposition.

Extracts the PRIORITY running-agent region from ``GatewayRunner._handle_message``
(run.py L2747-3117) into ``BusyAgentDispatchMixin._dispatch_busy_agent_message``.
This is a **sub-region extraction**, not a whole-method lift: ``_handle_message``
is a command dispatcher (long if/elif chain), and the busy-agent region is the one
cohesive, self-contained block within it — every branch returns.

Semantics preserved verbatim:

  * Called only when ``_quick_key in self._running_agents`` (the caller keeps
    that guard and delegates the body here).
  * Routes slash commands that must bypass the running-agent interrupt
    (``/stop``, ``/reset``, ``/new``, ``/queue``, ``/steer``, ``/model``,
    ``/approve``, ``/deny``, ``/background``, ``/kanban``, ``/goal``, etc.) to
    their dedicated handlers.
  * For non-command follow-ups: applies the platform-specific queueing policy
    (Telegram follow-up grace, photo burst batching, busy-input-mode
    queue/steer/interrupt), with subagent protection (#30170) demoting
    interrupts to queue semantics.

``gateway.run`` module-level runtime symbols (``logger``, ``_AGENT_PENDING_SENTINEL``,
``_INTERRUPT_REASON_RESET``, ``_INTERRUPT_REASON_STOP``) are lazy-imported at the
top of the method body to avoid the circular import (``gateway.run`` imports this
mixin at module top). Stdlib and third-party top-level imports are at module top.
Every other name in the body is either an in-body lazy import (kept verbatim from
source) or a ``self.*`` reference that resolves unchanged through the MRO.
"""

from __future__ import annotations

import os
import time

from agent.i18n import t
from gateway.config import Platform
from gateway.gateway_events import _agent_has_active_subagents
from gateway.platforms.base import (
    EphemeralReply,
    MessageEvent,
    MessageType,
    merge_pending_message_event,
)


class BusyAgentDispatchMixin:
    async def _dispatch_busy_agent_message(self, event, source, _quick_key: str):
        """Handle a message that arrived while an agent is already running.

        Called from ``_handle_message`` only when ``_quick_key`` is present in
        ``self._running_agents``. Returns the command/follow-up result (which
        ``_handle_message`` returns to its caller). Every branch returns — there
        is no fall-through; if none of the branches match, the final
        ``running_agent.interrupt(event.text)`` path returns ``None``.
        """
        from gateway.run import (
            _AGENT_PENDING_SENTINEL,
            _INTERRUPT_REASON_RESET,
            _INTERRUPT_REASON_STOP,
            logger,
        )

        if event.get_command() == "status":
            return await self._handle_status_command(event)

        # Resolve the command once for all early-intercept checks below.
        from hermes_cli.commands import (
            ACTIVE_SESSION_BYPASS_COMMANDS as _DEDICATED_HANDLERS,
            resolve_command as _resolve_cmd_inner,
        )
        _evt_cmd = event.get_command()
        _cmd_def_inner = _resolve_cmd_inner(_evt_cmd) if _evt_cmd else None

        # Slash command access control on the running-agent fast-path.
        # Mirrors the cold-path gate further below so non-admin users
        # can't bypass gating just because an agent happens to be busy.
        # /status above is intentionally pre-gate so users always see
        # session state. /help and /whoami fall under the always-allowed
        # floor inside _check_slash_access.
        if _evt_cmd and _cmd_def_inner is not None:
            _denied = self._check_slash_access(source, _cmd_def_inner.name)
            if _denied is not None:
                return _denied

        # Telegram sends /start for bot launches/deep-links. Treat it as a
        # platform ping, not a user command: no help dump, no agent
        # interrupt, no queued text.
        if _cmd_def_inner and _cmd_def_inner.name == "start":
            logger.info("Ignoring /start platform ping for active session %s", _quick_key)
            return ""

        if _cmd_def_inner and _cmd_def_inner.name == "restart":
            return await self._handle_restart_command(event)

        # /stop must hard-kill the session when an agent is running.
        # A soft interrupt (agent.interrupt()) doesn't help when the agent
        # is truly hung — the executor thread is blocked and never checks
        # _interrupt_requested.  Force-clean _running_agents so the session
        # is unlocked and subsequent messages are processed normally.
        if _cmd_def_inner and _cmd_def_inner.name == "stop":
            await self._interrupt_and_clear_session(
                _quick_key,
                source,
                interrupt_reason=_INTERRUPT_REASON_STOP,
                invalidation_reason="stop_command",
            )
            logger.info("STOP for session %s — agent interrupted, session lock released", _quick_key)
            return EphemeralReply(t("gateway.stop.stopped"))

        # /reset and /new must bypass the running-agent guard so they
        # actually dispatch as commands instead of being queued as user
        # text (which would be fed back to the agent with the same
        # broken history — #2170).  Interrupt the agent first, then
        # clear the adapter's pending queue so the stale "/reset" text
        # doesn't get re-processed as a user message after the
        # interrupt completes.
        if _cmd_def_inner and _cmd_def_inner.name == "new":
            # Clear any pending messages so the old text doesn't replay
            await self._interrupt_and_clear_session(
                _quick_key,
                source,
                interrupt_reason=_INTERRUPT_REASON_RESET,
                invalidation_reason="new_command",
            )
            # Clean up the running agent entry so the reset handler
            # doesn't think an agent is still active.
            return await self._handle_reset_command(event)

        # /queue <prompt> — queue without interrupting.
        # Semantics: each /queue invocation produces its own full agent
        # turn, processed in FIFO order after the current run (and any
        # earlier /queue items) finishes.  Messages are NOT merged.
        if event.get_command() in {"queue", "q"}:
            queued_text = event.get_command_args().strip()
            if not queued_text:
                return "Usage: /queue <prompt>"
            adapter = self.adapters.get(source.platform)
            if adapter:
                queued_event = MessageEvent(
                    text=queued_text,
                    message_type=MessageType.TEXT,
                    source=event.source,
                    message_id=event.message_id,
                    channel_prompt=event.channel_prompt,
                )
                self._enqueue_fifo(_quick_key, queued_event, adapter)
            depth = self._queue_depth(_quick_key, adapter=self.adapters.get(source.platform))
            if depth <= 1:
                return "Queued for the next turn."
            return f"Queued for the next turn. ({depth} queued)"

        # /steer <prompt> — inject mid-run after the next tool call.
        # Unlike /queue (turn boundary), /steer lands BETWEEN tool-call
        # iterations inside the same agent run, by appending to the
        # last tool result's content. No interrupt, no new user turn,
        # no role-alternation violation.
        if _cmd_def_inner and _cmd_def_inner.name == "steer":
            steer_text = event.get_command_args().strip()
            if not steer_text:
                return "Usage: /steer <prompt>"
            running_agent = self._running_agents.get(_quick_key)
            if running_agent is _AGENT_PENDING_SENTINEL:
                # Agent hasn't started yet — queue as turn-boundary fallback.
                adapter = self.adapters.get(source.platform)
                if adapter:
                    queued_event = MessageEvent(
                        text=steer_text,
                        message_type=MessageType.TEXT,
                        source=event.source,
                        message_id=event.message_id,
                        channel_prompt=event.channel_prompt,
                    )
                    adapter._pending_messages[_quick_key] = queued_event
                return "Agent still starting — /steer queued for the next turn."
            if running_agent and hasattr(running_agent, "steer"):
                try:
                    accepted = running_agent.steer(steer_text)
                except Exception as exc:
                    logger.warning("Steer failed for session %s: %s", _quick_key, exc)
                    return f"⚠️ Steer failed: {exc}"
                if accepted:
                    preview = steer_text[:60] + ("..." if len(steer_text) > 60 else "")
                    return f"⏩ Steer queued — arrives after the next tool call: '{preview}'"
                return "Steer rejected (empty payload)."
            # Running agent is missing or lacks steer() — fall back to queue.
            adapter = self.adapters.get(source.platform)
            if adapter:
                queued_event = MessageEvent(
                    text=steer_text,
                    message_type=MessageType.TEXT,
                    source=event.source,
                    message_id=event.message_id,
                    channel_prompt=event.channel_prompt,
                )
                adapter._pending_messages[_quick_key] = queued_event
            return "No active agent — /steer queued for the next turn."

        # /model must not be used while the agent is running.
        if _cmd_def_inner and _cmd_def_inner.name == "model":
            return "Agent is running — wait or /stop first, then switch models."

        # /codex-runtime must not be used while the agent is running.
        # Switching mid-turn would split a turn across two transports.
        if _cmd_def_inner and _cmd_def_inner.name == "codex-runtime":
            return ("Agent is running — wait or /stop first, then "
                    "change runtime.")

        # /approve and /deny must bypass the running-agent interrupt path.
        # The agent thread is blocked on a threading.Event inside
        # tools/approval.py — sending an interrupt won't unblock it.
        # Route directly to the approval handler so the event is signalled.
        if _cmd_def_inner and _cmd_def_inner.name in {"approve", "deny"}:
            if _cmd_def_inner.name == "approve":
                return await self._handle_approve_command(event)
            return await self._handle_deny_command(event)

        # /agents (/tasks alias) should be query-only and never interrupt.
        if _cmd_def_inner and _cmd_def_inner.name == "agents":
            return await self._handle_agents_command(event)

        # /background must bypass the running-agent guard — it starts a
        # parallel task and must never interrupt the active conversation.
        # /btw is an alias of /background and resolves to the same canonical
        # name, so this branch handles both commands.
        if _cmd_def_inner and _cmd_def_inner.name == "background":
            return await self._handle_background_command(event)

        # /kanban must bypass the guard. It writes to a profile-agnostic
        # DB (kanban.db), not to the running agent's state. In fact
        # /kanban unblock is often the only way to free a worker that
        # has blocked waiting for a peer — letting that be dispatched
        # mid-run is the whole point of the board.
        if _cmd_def_inner and _cmd_def_inner.name == "kanban":
            return await self._handle_kanban_command(event)

        # /goal is safe mid-run for status/pause/clear (inspection and
        # control-plane only — doesn't interrupt the running turn).
        # Setting a new goal text mid-run is rejected with the same
        # "wait or /stop" message as /model so we don't race a second
        # continuation prompt against the current turn.
        if _cmd_def_inner and _cmd_def_inner.name == "goal":
            _goal_arg = (event.get_command_args() or "").strip().lower()
            if not _goal_arg or _goal_arg in {"status", "pause", "resume", "clear", "stop", "done"}:
                return await self._handle_goal_command(event)
            return "Agent is running — use /goal status / pause / clear mid-run, or /stop before setting a new goal."

        # /subgoal is safe mid-run — it only modifies the goal's
        # subgoals list, which the judge reads at the next turn
        # boundary. No race with the running turn.
        if _cmd_def_inner and _cmd_def_inner.name == "subgoal":
            return await self._handle_subgoal_command(event)
        # /optimize <prompt> — rewrite the user's prompt using the
        # prompt-optimizer skill, then execute it.  Delegates to the
        # skill-invocation system for prompt-optimizer.
        if _cmd_def_inner and _cmd_def_inner.name == "optimize":
            optimize_text = event.get_command_args().strip()
            if not optimize_text:
                return "Usage: /optimize <prompt>"
            try:
                from agent.skill_commands import (
                    get_skill_commands,
                    build_skill_invocation_message,
                    resolve_skill_command_key,
                )
                skill_cmds = get_skill_commands()
                cmd_key = resolve_skill_command_key("prompt-optimizer")
                if cmd_key is None:
                    return "The prompt-optimizer skill is not installed."
                msg = build_skill_invocation_message(
                    cmd_key, optimize_text, task_id=_quick_key
                )
                if msg:
                    event.text = msg
                    _cmd_def_inner = None  # Prevent catch-all rejection below
                else:
                    return "Failed to load the prompt-optimizer skill."
            except Exception as exc:
                logger.warning("Optimize handler failed for session %s: %s", _quick_key, exc)
                return f"⚠️ Optimize failed: {exc}"

        # Session-level toggles that are safe to run mid-agent —
        # /yolo can unblock a pending approval prompt, /verbose cycles
        # the tool-progress display mode for the ongoing stream.
        # Both modify session state without needing agent interaction
        # and must not be queued (the safety net would discard them).
        # /fast and /reasoning are config-only and take effect next
        # message, so they fall through to the catch-all busy response
        # below — users should wait and set them between turns.
        if _cmd_def_inner and _cmd_def_inner.name in {"yolo", "verbose"}:
            if _cmd_def_inner.name == "yolo":
                return await self._handle_yolo_command(event)
            if _cmd_def_inner.name == "verbose":
                return await self._handle_verbose_command(event)
            if _cmd_def_inner.name == "footer":
                return await self._handle_footer_command(event)

        # Gateway-handled info/control commands with dedicated
        # running-agent handlers.
        if _cmd_def_inner and _cmd_def_inner.name in _DEDICATED_HANDLERS:
            if _cmd_def_inner.name == "help":
                return await self._handle_help_command(event)
            if _cmd_def_inner.name == "commands":
                return await self._handle_commands_command(event)
            if _cmd_def_inner.name == "profile":
                return await self._handle_profile_command(event)
            if _cmd_def_inner.name == "update":
                return await self._handle_update_command(event)
            if _cmd_def_inner.name == "version":
                return await self._handle_version_command(event)

        # Catch-all: any other recognized slash command reached the
        # running-agent guard. Reject gracefully rather than falling
        # through to interrupt + discard. Without this, commands
        # like /model, /reasoning, /voice, /insights, /title,
        # /resume, /retry, /undo, /compress, /usage,
        # /reload-mcp, /sethome, /reset (all registered as Discord
        # slash commands) would interrupt the agent AND get
        # silently discarded by the slash-command safety net,
        # producing a zero-char response. See #5057, #6252, #10370.
        if _cmd_def_inner:
            return (
                f"⏳ Agent is running — `/{_cmd_def_inner.name}` can't run "
                f"mid-turn. Wait for the current response or `/stop` first."
            )

        if event.message_type == MessageType.PHOTO:
            logger.debug("PRIORITY photo follow-up for session %s — queueing without interrupt", _quick_key)
            adapter = self.adapters.get(source.platform)
            if adapter:
                merge_pending_message_event(adapter._pending_messages, _quick_key, event)
            return None

        _telegram_followup_grace = float(
            os.getenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", "3.0")
        )
        _started_at = self._running_agents_ts.get(_quick_key, 0)
        if (
            source.platform == Platform.TELEGRAM
            and event.message_type == MessageType.TEXT
            and _telegram_followup_grace > 0
            and _started_at
            and (time.time() - _started_at) <= _telegram_followup_grace
        ):
            logger.debug(
                "Telegram follow-up arrived %.2fs after run start for %s — queueing without interrupt",
                time.time() - _started_at,
                _quick_key,
            )
            adapter = self.adapters.get(source.platform)
            if adapter:
                if self._busy_input_mode == "queue":
                    self._enqueue_fifo(_quick_key, event, adapter)
                else:
                    merge_pending_message_event(
                        adapter._pending_messages,
                        _quick_key,
                        event,
                        merge_text=True,
                    )
            return None

        running_agent = self._running_agents.get(_quick_key)
        if running_agent is _AGENT_PENDING_SENTINEL:
            # Agent is being set up but not ready yet.
            if event.get_command() == "stop":
                # Force-clean the sentinel so the session is unlocked.
                self._release_running_agent_state(_quick_key)
                logger.info("HARD STOP (pending) for session %s — sentinel cleared", _quick_key)
                return EphemeralReply("⚡ Force-stopped. The agent was still starting — session unlocked.")
            # Queue the message so it will be picked up after the
            # agent starts.
            adapter = self.adapters.get(source.platform)
            if adapter:
                merge_pending_message_event(
                    adapter._pending_messages,
                    _quick_key,
                    event,
                    merge_text=True,
                )
            return None
        if self._draining:
            if self._queue_during_drain_enabled():
                self._queue_or_replace_pending_event(_quick_key, event)
            return (
                f"⏳ Gateway {self._status_action_gerund()} — queued for the next turn after it comes back."
                if self._queue_during_drain_enabled()
                else f"⏳ Gateway is {self._status_action_gerund()} and is not accepting another turn right now."
            )
        if self._busy_input_mode == "queue":
            logger.debug("PRIORITY queue follow-up for session %s", _quick_key)
            self._queue_or_replace_pending_event(_quick_key, event)
            return None
        if self._busy_input_mode == "steer":
            # Steer mode: inject text into the running agent mid-run via
            # agent.steer().  Falls back to queue semantics if the payload
            # is empty, the agent lacks steer(), or steer() rejects.
            steer_text = (event.text or "").strip()
            steered = False
            if steer_text and hasattr(running_agent, "steer"):
                try:
                    steered = bool(running_agent.steer(steer_text))
                except Exception as exc:
                    logger.warning("PRIORITY steer failed for session %s: %s", _quick_key, exc)
                    steered = False
            if steered:
                logger.debug("PRIORITY steer for session %s", _quick_key)
                return None
            logger.debug("PRIORITY steer-fallback-to-queue for session %s", _quick_key)
            self._queue_or_replace_pending_event(_quick_key, event)
            return None
        # #30170 — Subagent protection (PRIORITY path). Same rationale
        # as ``_handle_active_session_busy_message``: an interrupt
        # cascades through ``_active_children`` and aborts in-flight
        # delegate_task work. Demote to queue semantics when the
        # parent is currently driving subagents so a conversational
        # follow-up doesn't destroy minutes of subagent progress.
        # /stop reaches its dedicated handler above, so the operator
        # still has a clean escape hatch.
        if _agent_has_active_subagents(running_agent):
            logger.info(
                "PRIORITY interrupt demoted to queue for session %s "
                "because the running agent has active subagents (#30170)",
                _quick_key,
            )
            self._queue_or_replace_pending_event(_quick_key, event)
            return None
        logger.debug("PRIORITY interrupt for session %s", _quick_key)
        running_agent.interrupt(event.text)
        # NOTE: self._pending_messages was write-only (never consumed).
        # The actual interrupt message is delivered via adapter._pending_messages
        # which is read by _run_agent. Removed to prevent unbounded growth.
        return None
