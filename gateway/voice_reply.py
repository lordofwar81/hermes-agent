"""
Voice channel and TTS reply handling for GatewayRunner.

Extracted from gateway/run.py to reduce the God file size.
Provides functions for managing Discord voice channel connections,
handling voice input transcripts, and sending TTS voice replies.
"""

import asyncio
import json
import logging
import os
import re
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Dict

from gateway.message_schema import MessageEvent, MessageType, SessionSource
from gateway.platforms.base import BasePlatformAdapter
from hermes_cli.enums import Platform

logger = logging.getLogger(__name__)

# Import voice_mode utilities
from gateway import voice_mode as _voice_mode
from gateway.authorization import is_user_authorized


async def handle_voice_channel_join(
    runner: Any,
    event: MessageEvent,
) -> str:
    """Join the user's current Discord voice channel."""
    adapter = runner.adapters.get(event.source.platform)
    if not hasattr(adapter, "join_voice_channel"):
        return "Voice channels are not supported on this platform."

    guild_id = _get_guild_id(runner, event)
    if not guild_id:
        return "This command only works in a Discord server."

    voice_channel = await adapter.get_user_voice_channel(
        guild_id, event.source.user_id
    )
    if not voice_channel:
        return "You need to be in a voice channel first."

    # Wire callbacks BEFORE join so voice input arriving immediately
    # after connection is not lost.
    if hasattr(adapter, "_voice_input_callback"):
        adapter._voice_input_callback = lambda *args, **kwargs: handle_voice_channel_input(
            runner, *args, **kwargs
        )
    if hasattr(adapter, "_on_voice_disconnect"):
        adapter._on_voice_disconnect = lambda chat_id: handle_voice_timeout_cleanup(
            runner, chat_id
        )

    try:
        success = await adapter.join_voice_channel(voice_channel)
    except Exception as e:
        logger.warning("Failed to join voice channel: %s", e)
        adapter._voice_input_callback = None
        err_lower = str(e).lower()
        if "pynacl" in err_lower or "nacl" in err_lower or "davey" in err_lower:
            return (
                "Voice dependencies are missing (PyNaCl / davey). "
                f"Install with: `python -m pip install PyNaCl`"
            )
        return f"Failed to join voice channel: {e}"

    if success:
        adapter._voice_text_channels[guild_id] = int(event.source.chat_id)
        if hasattr(adapter, "_voice_sources"):
            adapter._voice_sources[guild_id] = event.source.to_dict()
        runner._voice_mode[_voice_mode.voice_key(event.source.platform, event.source.chat_id)] = "all"
        _voice_mode.save_voice_modes()
        _voice_mode.set_adapter_auto_tts_enabled(adapter, event.source.chat_id, enabled=True)
        return (
            f"Joined voice channel **{voice_channel.name}**.\n"
            f"I'll speak my replies and listen to you. Use /voice leave to disconnect."
        )
    # Join failed — clear callback
    adapter._voice_input_callback = None
    return "Failed to join voice channel. Check bot permissions (Connect + Speak)."


async def handle_voice_channel_leave(
    runner: Any,
    event: MessageEvent,
) -> str:
    """Leave the Discord voice channel."""
    adapter = runner.adapters.get(event.source.platform)
    guild_id = _get_guild_id(runner, event)

    if not guild_id or not hasattr(adapter, "leave_voice_channel"):
        return "Not in a voice channel."

    if not hasattr(adapter, "is_in_voice_channel") or not adapter.is_in_voice_channel(guild_id):
        return "Not in a voice channel."

    try:
        await adapter.leave_voice_channel(guild_id)
    except Exception as e:
        logger.warning("Error leaving voice channel: %s", e)
    # Always clean up state even if leave raised an exception
    runner._voice_mode[_voice_mode.voice_key(event.source.platform, event.source.chat_id)] = "off"
    _voice_mode.save_voice_modes()
    _voice_mode.set_adapter_auto_tts_disabled(adapter, event.source.chat_id, disabled=True)
    if hasattr(adapter, "_voice_input_callback"):
        adapter._voice_input_callback = None
    return "Left voice channel."


def handle_voice_timeout_cleanup(runner: Any, chat_id: str) -> None:
    """Called by the adapter when a voice channel times out.

    Cleans up runner-side voice_mode state that the adapter cannot reach.
    """
    runner._voice_mode[_voice_mode.voice_key(Platform.DISCORD, chat_id)] = "off"
    _voice_mode.save_voice_modes()
    adapter = runner.adapters.get(Platform.DISCORD)
    _voice_mode.set_adapter_auto_tts_disabled(adapter, chat_id, disabled=True)


def is_duplicate_voice_transcript(
    runner: Any,
    guild_id: int,
    user_id: int,
    transcript: str,
) -> bool:
    """Suppress repeated STT outputs for the same recent utterance.

    Voice capture can occasionally emit the same utterance twice a few
    seconds apart, which creates a second queued agent run and overlapping
    spoken replies. Dedup exact and near-exact repeats per guild/user over a
    short window while allowing genuinely new turns through.
    """
    from difflib import SequenceMatcher

    normalized = re.sub(r"\s+", " ", transcript).strip().lower()
    normalized = re.sub(r"[^\w\s]", "", normalized)
    if not normalized:
        return False

    now = time.monotonic()
    window_seconds = 12.0
    key = (guild_id, user_id)
    recent_store = getattr(runner, "_recent_voice_transcripts", None)
    if not isinstance(recent_store, dict):
        recent_store = {}
        runner._recent_voice_transcripts = recent_store
    recent = [
        (ts, txt)
        for ts, txt in recent_store.get(key, [])
        if now - ts <= window_seconds
    ]

    for _, prior in recent:
        if prior == normalized:
            recent_store[key] = recent
            return True
        if len(prior) >= 16 and len(normalized) >= 16:
            if SequenceMatcher(None, prior, normalized).ratio() >= 0.95:
                recent_store[key] = recent
                return True

    recent.append((now, normalized))
    recent_store[key] = recent[-5:]
    return False


async def handle_voice_channel_input(
    runner: Any,
    guild_id: int,
    user_id: int,
    transcript: str,
) -> None:
    """Handle transcribed voice from a user in a voice channel.

    Creates a synthetic MessageEvent and processes it through the
    adapter's full message pipeline (session, typing, agent, TTS reply).
    """
    adapter = runner.adapters.get(Platform.DISCORD)
    if not adapter:
        return

    text_ch_id = adapter._voice_text_channels.get(guild_id)
    if not text_ch_id:
        return

    # Build source — reuse the linked text channel's metadata when available
    # so voice input shares the same session as the bound text conversation.
    source_data = getattr(adapter, "_voice_sources", {}).get(guild_id)
    if source_data:
        source = SessionSource.from_dict(source_data)
        source.user_id = str(user_id)
        source.user_name = str(user_id)
    else:
        source = SessionSource(
            platform=Platform.DISCORD,
            chat_id=str(text_ch_id),
            user_id=str(user_id),
            user_name=str(user_id),
            chat_type="channel",
        )

    # Check authorization before processing voice input
    if not is_user_authorized(runner, source):
        logger.debug("Unauthorized voice input from user %d, ignoring", user_id)
        return

    if is_duplicate_voice_transcript(runner, guild_id, user_id, transcript):
        logger.info(
            "Suppressing duplicate voice transcript for guild=%s user=%s: %s",
            guild_id,
            user_id,
            transcript[:100],
        )
        return

    # Show transcript in text channel (after auth, with mention sanitization)
    try:
        channel = adapter._client.get_channel(text_ch_id)
        if channel:
            safe_text = transcript[:2000].replace("@everyone", "@​everyone").replace("@here", "@​here")
            await channel.send(f"**[Voice]** <@{user_id}>: {safe_text}")
    except Exception:
        pass

    # Build a synthetic MessageEvent and feed through the normal pipeline
    # Use SimpleNamespace as raw_message so _get_guild_id() can extract
    # guild_id and send_voice_reply() plays audio in the voice channel.
    event = MessageEvent(
        source=source,
        text=transcript,
        message_type=MessageType.VOICE,
        raw_message=SimpleNamespace(guild_id=guild_id, guild=None),
    )

    await adapter.handle_message(event)


def should_send_voice_reply(
    runner: Any,
    event: MessageEvent,
    response: str,
    agent_messages: list,
    already_sent: bool = False,
) -> bool:
    """Decide whether the runner should send a TTS voice reply.

    Returns False when:
    - voice_mode is off for this chat
    - response is empty or an error
    - agent already called text_to_speech tool (dedup)
    - voice input and base adapter auto-TTS already handled it (skip_double)
      UNLESS streaming already consumed the response (already_sent=True),
      in which case the base adapter won't have text for auto-TTS so the
      runner must handle it.
    """
    if not response or response.startswith("Error:"):
        return False

    chat_id = event.source.chat_id
    voice_mode_val = runner._voice_mode.get(_voice_mode.voice_key(event.source.platform, chat_id), "off")
    is_voice_input = (event.message_type == MessageType.VOICE)

    should = (
        (voice_mode_val == "all")
        or (voice_mode_val == "voice_only" and is_voice_input)
    )
    if not should:
        return False

    # Dedup: agent already called TTS tool
    has_agent_tts = any(
        msg.get("role") == "assistant"
        and any(
            tc.get("function", {}).get("name") == "text_to_speech"
            for tc in (msg.get("tool_calls") or [])
        )
        for msg in agent_messages
    )
    if has_agent_tts:
        return False

    # Dedup: base adapter auto-TTS already handles voice input
    # (play_tts plays in VC when connected, so runner can skip).
    # When streaming already delivered the text (already_sent=True),
    # the base adapter will receive None and can't run auto-TTS,
    # so the runner must take over.
    if is_voice_input and not already_sent:
        return False

    return True


async def send_voice_reply(runner: Any, event: MessageEvent, text: str) -> None:
    """Generate TTS audio and send as a voice message before the text reply."""
    import uuid as _uuid
    audio_path = None
    actual_path = None
    try:
        from tools.tts_tool import text_to_speech_tool, _strip_markdown_for_tts

        tts_text = _strip_markdown_for_tts(text[:4000])
        if not tts_text:
            return

        # Use .mp3 extension so edge-tts conversion to opus works correctly.
        # The TTS tool may convert to .ogg — use file_path from result.
        audio_path = os.path.join(
            tempfile.gettempdir(), "hermes_voice",
            f"tts_reply_{_uuid.uuid4().hex[:12]}.mp3",
        )
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)

        result_json = await asyncio.to_thread(
            text_to_speech_tool, text=tts_text, output_path=audio_path
        )
        try:
            result = json.loads(result_json)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Auto voice reply TTS returned invalid JSON: %s", result_json[:200] if result_json else result_json)
            return

        # Use the actual file path from result (may differ after opus conversion)
        actual_path = result.get("file_path", audio_path)
        if not result.get("success") or not os.path.isfile(actual_path):
            logger.warning("Auto voice reply TTS failed: %s", result.get("error"))
            return

        adapter = runner.adapters.get(event.source.platform)

        # If connected to a voice channel, play there instead of sending a file
        guild_id = _get_guild_id(runner, event)
        if (guild_id
                and hasattr(adapter, "play_in_voice_channel")
                and hasattr(adapter, "is_in_voice_channel")
                and adapter.is_in_voice_channel(guild_id)):
            await adapter.play_in_voice_channel(guild_id, actual_path)
        elif adapter and hasattr(adapter, "send_voice"):
            reply_anchor = reply_anchor_for_event(runner, event)
            thread_meta = thread_metadata_for_source(
                runner, event.source, reply_anchor
            )
            # Mark the auto voice reply as notify-worthy. Mirrors the
            # final-text path in gateway/platforms/base.py which sets
            # ``notify=True`` so platform adapters that gate push
            # notifications (Telegram "important" mode) deliver the
            # final voice reply as a normal notification instead of a
            # silent message. Clone first so we don't mutate metadata
            # shared with concurrent typing-indicator state.
            if thread_meta is not None:
                thread_meta = dict(thread_meta)
                thread_meta["notify"] = True
            else:
                thread_meta = {"notify": True}
            send_kwargs: Dict[str, Any] = {
                "chat_id": event.source.chat_id,
                "audio_path": actual_path,
                "reply_to": reply_anchor,
                "metadata": thread_meta,
            }
            await adapter.send_voice(**send_kwargs)
    except Exception as e:
        logger.warning("Auto voice reply failed: %s", e, exc_info=True)
    finally:
        for p in {audio_path, actual_path} - {None}:
            try:
                os.unlink(p)
            except OSError:
                pass


# Helper functions (these may need to be extracted from GatewayRunner too)

def _get_guild_id(runner: Any, event: MessageEvent) -> int | None:
    """Extract guild_id from event source or raw_message."""
    if hasattr(event.source, "guild_id"):
        return event.source.guild_id
    if hasattr(event, "raw_message") and event.raw_message:
        raw = event.raw_message
        if hasattr(raw, "guild_id"):
            return getattr(raw, "guild_id", None)
        if hasattr(raw, "guild") and raw.guild:
            return getattr(raw.guild, "id", None)
    return None


def reply_anchor_for_event(runner: Any, event: MessageEvent) -> Any | None:
    """Get the reply anchor for an event."""
    # This is typically the message ID for threading/reply context
    # Implementation depends on the platform adapter
    adapter = runner.adapters.get(event.source.platform)
    if adapter and hasattr(adapter, "get_reply_anchor"):
        return adapter.get_reply_anchor(event)
    return None


def thread_metadata_for_source(
    runner: Any, source: SessionSource, reply_anchor: Any
) -> Dict[str, Any] | None:
    """Get thread metadata for a message source."""
    adapter = runner.adapters.get(source.platform)
    if adapter and hasattr(adapter, "get_thread_metadata"):
        return adapter.get_thread_metadata(source, reply_anchor)
    return None
