"""
Message pipeline processing for inbound message preparation and busy session handling.

Extracted from gateway/run.py:
- GatewayRunner._prepare_inbound_message_text
- GatewayRunner._handle_active_session_busy_message
"""
import asyncio
import logging
import os
import re
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, Platform, SessionSource
    from agent.onboarding import BUSY_INPUT_FLAG

logger = logging.getLogger(__name__)

_AGENT_PENDING_SENTINEL = object()
_BUSY_ACK_COOLDOWN = 30


async def handle_active_session_busy_message(
    runner,
    event: "MessageEvent",
    session_key: str,
) -> bool:
    """
    Handle a message received while a session is already active.

    This function processes messages that arrive when the gateway already has
    an active agent running for the given session. It implements busy-mode
    semantics (queue, steer, or interrupt) and sends acknowledgment messages.

    Args:
        runner: GatewayRunner instance with necessary state and methods
        event: The incoming message event
        session_key: Unique key for the active session

    Returns:
        True if the message was handled, False if it should fall through
        to the default message handler.
    """
    # --- Authorization gate (#17775) ---
    # The cold path (_handle_message) checks _is_user_authorized before
    # creating a session.  The busy path must enforce the same check;
    # otherwise unauthorized users in shared threads (Slack/Telegram/Discord)
    # can inject messages into an active session they don't own.
    if not runner._is_user_authorized(event.source):
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
    if runner._draining:
        adapter = runner.adapters.get(event.source.platform)
        if not adapter:
            return True

        reply_anchor = runner._reply_anchor_for_event(event)
        thread_meta = runner._thread_metadata_for_source(event.source, reply_anchor)
        if runner._queue_during_drain_enabled():
            runner._queue_or_replace_pending_event(session_key, event)
            message = f"⏳ Gateway {runner._status_action_gerund()} — queued for the next turn after it comes back."
        else:
            message = f"⏳ Gateway is {runner._status_action_gerund()} and is not accepting another turn right now."

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
    adapter = runner.adapters.get(event.source.platform)
    if not adapter:
        return False  # let default path handle it

    running_agent = runner._running_agents.get(session_key)

    effective_mode = runner._busy_input_mode
    busy_text_mode = getattr(runner, "_busy_text_mode", "queue")
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
        and runner._agent_has_active_subagents(running_agent)
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
        from gateway.platforms.base import merge_pending_message_event

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
    now = time.time()
    last_ack = runner._busy_ack_ts.get(session_key, 0)
    if now - last_ack < _BUSY_ACK_COOLDOWN:
        return True  # interrupt sent (if not queue), ack already delivered recently

    runner._busy_ack_ts[session_key] = now

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
            start_ts = runner._running_agents_ts.get(session_key, 0)
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
        from hermes_constants import get_hermes_home

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
            mark_seen(get_hermes_home() / "config.yaml", BUSY_INPUT_FLAG)
    except Exception as _onb_err:
        logger.debug("Failed to apply busy-input onboarding hint: %s", _onb_err)

    reply_anchor = runner._reply_anchor_for_event(event)
    thread_meta = runner._thread_metadata_for_source(event.source, reply_anchor)
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


def _load_gateway_config() -> dict:
    """Load the gateway configuration."""
    from gateway.config import _load_gateway_config as _cfg
    return _cfg()


def _platform_config_key(platform: "Platform") -> str:
    """Get the config key for a platform."""
    from gateway.config import _platform_config_key as _key
    return _key(platform)


# ----------------------------------------------------------------------
# Inbound message text preparation
# ----------------------------------------------------------------------

_PENDING_NATIVE_IMAGE_SESSIONS: Dict[str, List[str]] = {}


async def prepare_inbound_message_text(
    *,
    event: "MessageEvent",
    source: "SessionSource",
    history: list[dict[str, any]],
    config: any,
    adapters: dict[str, any],
    model: str,
    base_url: str | None,
    session_key_for_source: callable,
    consume_pending_native_image_paths: callable,
    decide_image_input_mode: callable,
    enrich_message_with_vision: callable,
    enrich_message_with_transcription: callable,
    reply_anchor_for_event: callable,
    thread_metadata_for_source: callable,
    has_setup_skill: bool,
) -> str | None:
    """Prepare inbound event text for the agent.

    Keep the normal inbound path and the queued follow-up path on the same
    preprocessing pipeline so sender attribution, image enrichment, STT,
    document notes, reply context, and @ references all behave the same.

    Side effect: buffers per-session native image paths when the active
    model supports native vision AND the user has images attached. The
    caller consumes and clears that session-scoped buffer at the
    ``run_conversation`` site to build a multimodal user turn. When the
    list is empty, the ``_enrich_message_with_vision`` text path has
    already run and images are represented in-text.

    Args:
        event: The inbound message event
        source: Session source metadata
        history: Conversation history (optional)
        config: Gateway config object
        adapters: Platform adapters dict
        model: Current model name
        base_url: Current base URL
        session_key_for_source: Function to build session key from source
        consume_pending_native_image_paths: Function to consume buffered image paths
        decide_image_input_mode: Function to determine native vs text image routing
        enrich_message_with_vision: Async function to analyze images and return text
        enrich_message_with_transcription: Async function to transcribe audio
        reply_anchor_for_event: Function to get reply anchor from event
        thread_metadata_for_source: Function to build thread metadata
        has_setup_skill: Whether setup skill is available

    Returns:
        The prepared message text, or None if blocked (e.g., @-reference rejection)
    """
    from gateway.run import (
        _load_gateway_config,
        _probe_audio_duration,
        _resolve_runtime_agent_kwargs,
    )
    from gateway.session import is_shared_multi_user_session

    history = history or []
    message_text = event.text or ""
    _group_sessions_per_user = getattr(config, "group_sessions_per_user", True)
    _thread_sessions_per_user = getattr(config, "thread_sessions_per_user", False)
    # Use the same helper every other call site uses so the write key here
    # matches the consume key at the run_conversation site — even if the
    # session store overrides build_session_key's default behavior.
    session_key = session_key_for_source(source)
    # Reset only this session's per-call buffer; other sessions may be
    # concurrently preparing multimodal turns on the same runner.
    consume_pending_native_image_paths(session_key)

    _is_shared_multi_user = is_shared_multi_user_session(
        source,
        group_sessions_per_user=_group_sessions_per_user,
        thread_sessions_per_user=_thread_sessions_per_user,
    )
    if _is_shared_multi_user and source.user_name:
        message_text = f"[{source.user_name}] {message_text}"

    # Prepend channel context from history backfill (if any).  This
    # happens after sender-prefix so the prefix only applies to the
    # trigger message, not the backfill block.
    if getattr(event, "channel_context", None):
        message_text = f"{event.channel_context}\n\n[New message]\n{message_text}"

    # Declare at outer scope so the audio-file-paths handling block below
    # remains safe when ``event.media_urls`` is empty (no inner block runs).
    audio_file_paths: list[str] = []

    if event.media_urls:
        image_paths = []
        audio_paths = []
        for i, path in enumerate(event.media_urls):
            mtype = event.media_types[i] if i < len(event.media_types) else ""
            if mtype.startswith("image/") or event.message_type == MessageType.PHOTO:
                image_paths.append(path)
            # MessageType.AUDIO = audio file attachment (e.g. .mp3, .m4a) — never STT
            # MessageType.VOICE = voice message (Opus/OGG) — always STT
            if event.message_type == MessageType.AUDIO:
                audio_file_paths.append(path)
            elif event.message_type == MessageType.VOICE or (
                mtype.startswith("audio/")
                and event.message_type not in {MessageType.AUDIO, MessageType.DOCUMENT}
            ):
                audio_paths.append(path)

        if image_paths:
            # Decide routing: native (attach pixels) vs text (vision_analyze
            # pre-run + prepend description).  See agent/image_routing.py.
            _img_mode = decide_image_input_mode()
            if _img_mode == "native":
                # Defer attachment to the run_conversation call site.
                _PENDING_NATIVE_IMAGE_SESSIONS[session_key] = list(image_paths)
                logger.info(
                    "Image routing: native (model supports vision). %d image(s) will be attached inline.",
                    len(image_paths),
                )
            else:
                logger.info(
                    "Image routing: text (mode=%s). Pre-analyzing %d image(s) via vision_analyze.",
                    _img_mode, len(image_paths),
                )
                message_text = await enrich_message_with_vision(
                    message_text,
                    image_paths,
                )

        if audio_paths:
            message_text = await enrich_message_with_transcription(
                message_text,
                audio_paths,
            )
            _stt_fail_markers = (
                "No STT provider",
                "STT is disabled",
                "can't listen",
                "VOICE_TOOLS_OPENAI_KEY",
            )
            if any(marker in message_text for marker in _stt_fail_markers):
                _stt_adapter = adapters.get(source.platform)
                _stt_meta = thread_metadata_for_source(source, reply_anchor_for_event(event))
                if _stt_adapter:
                    try:
                        _stt_msg = (
                            "🎤 I received your voice message but can't transcribe it — "
                            "no speech-to-text provider is configured.\n\n"
                            "To enable voice: install faster-whisper "
                            "(`uv pip install faster-whisper` in the Hermes venv; "
                            "`pip install faster-whisper` also works if pip is on PATH) "
                            "and set `stt.enabled: true` in config.yaml, "
                            "then /restart the gateway."
                        )
                        if has_setup_skill:
                            _stt_msg += "\n\nFor full setup instructions, type: `/skill hermes-agent-setup`"
                        await _stt_adapter.send(
                            source.chat_id,
                            _stt_msg,
                            metadata=_stt_meta,
                        )
                    except Exception:
                        pass

    if audio_file_paths:
        from tools.credential_files import to_agent_visible_cache_path as _to_agent_path
        for _apath in audio_file_paths:
            _basename = os.path.basename(_apath)
            _parts = _basename.split("_", 2)
            _display = _parts[2] if len(_parts) >= 3 else _basename
            _display = re.sub(r'[^\w.\- ]', '_', _display)
            _agent_path = _to_agent_path(_apath)
            _note = (
                f"[The user sent an audio file attachment: '{_display}'. "
                f"It is saved at: {_agent_path}. "
                f"Ask the user what they'd like you to do with it, or pass the path to a transcription or media tool.]"
            )
            message_text = f"{_note}\n\n{message_text}"

    if event.media_urls and event.message_type == MessageType.DOCUMENT:
        import mimetypes as _mimetypes
        from tools.credential_files import to_agent_visible_cache_path

        _TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".log", ".json", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg"}
        for i, path in enumerate(event.media_urls):
            mtype = event.media_types[i] if i < len(event.media_types) else ""
            if mtype in {"", "application/octet-stream"}:
                _ext = os.path.splitext(path)[1].lower()
                if _ext in _TEXT_EXTENSIONS:
                    mtype = "text/plain"
                else:
                    guessed, _ = _mimetypes.guess_type(path)
                    if guessed:
                        mtype = guessed
            if not mtype.startswith(("application/", "text/")):
                continue

            basename = os.path.basename(path)
            parts = basename.split("_", 2)
            display_name = parts[2] if len(parts) >= 3 else basename
            display_name = re.sub(r'[^\w.\- ]', '_', display_name)

            # Translate host cache path to in-container path if running under Docker backend.
            # This ensures the agent receives a path it can open inside its sandbox, as the
            # cache directories are auto-mounted at /root/.hermes/cache/* by get_cache_directory_mounts().
            agent_path = to_agent_visible_cache_path(path)

            if mtype.startswith("text/"):
                context_note = (
                    f"[The user sent a text document: '{display_name}'. "
                    f"Its content has been included below. "
                    f"The file is also saved at: {agent_path}]"
                )
            else:
                context_note = (
                    f"[The user sent a document: '{display_name}'. "
                    f"The file is saved at: {agent_path}. "
                    f"Ask the user what they'd like you to do with it.]"
                )
            message_text = f"{context_note}\n\n{message_text}"

    if getattr(event, "reply_to_text", None) and event.reply_to_message_id:
        # Always inject the reply-to pointer — even when the quoted text
        # already appears in history. The prefix isn't deduplication, it's
        # disambiguation: it tells the agent *which* prior message the user
        # is referencing. History can contain the same or similar text
        # multiple times, and without an explicit pointer the agent has to
        # guess (or answer for both subjects). Token overhead is minimal.
        reply_snippet = event.reply_to_text[:500]
        message_text = f'[Replying to: "{reply_snippet}"]\n\n{message_text}'

    if "@" in message_text:
        try:
            from agent.context_references import preprocess_context_references_async
            from agent.model_metadata import get_model_context_length

            _msg_cwd = os.environ.get("TERMINAL_CWD", os.path.expanduser("~"))
            _msg_runtime = _resolve_runtime_agent_kwargs()
            _msg_config_ctx = None
            try:
                _msg_cfg = _load_gateway_config()
                _msg_model_cfg = _msg_cfg.get("model", {})
                if isinstance(_msg_model_cfg, dict):
                    _msg_raw_ctx = _msg_model_cfg.get("context_length")
                    if _msg_raw_ctx is not None:
                        _msg_config_ctx = int(_msg_raw_ctx)
            except Exception:
                pass
            _msg_ctx_len = get_model_context_length(
                model,
                base_url=base_url or _msg_runtime.get("base_url") or "",
                api_key=_msg_runtime.get("api_key") or "",
                config_context_length=_msg_config_ctx,
            )
            _ctx_result = await preprocess_context_references_async(
                message_text,
                cwd=_msg_cwd,
                context_length=_msg_ctx_len,
                allowed_root=_msg_cwd,
            )
            if _ctx_result.blocked:
                _adapter = adapters.get(source.platform)
                if _adapter:
                    await _adapter.send(
                        source.chat_id,
                        "\n".join(_ctx_result.warnings) or "Context injection refused.",
                    )
                return None
            if _ctx_result.expanded:
                message_text = _ctx_result.message
        except Exception as exc:
            logger.debug("@ context reference expansion failed: %s", exc)

    return message_text


def consume_pending_native_image_paths_impl(session_key: str) -> list[str]:
    """Consume and return pending native image paths for a session.

    Args:
        session_key: The session key to consume paths for

    Returns:
        List of image paths for this session (empty if none)
    """
    return list(_PENDING_NATIVE_IMAGE_SESSIONS.pop(session_key, []) or [])
