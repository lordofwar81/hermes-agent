"""Transcription enrichment methods for ``GatewayRunner``.

Round 29 of the god-file decomposition. Lifted verbatim into a mixin.
Three transcription-related methods moved together:

1. **Consume pending native image paths** —
   ``_consume_pending_native_image_paths`` (sync, drains a per-event
   list of native image paths awaiting attachment).
2. **Enrich with transcription** — ``_enrich_message_with_transcription``
   (async; transcribes attached audio via tools.transcription_tools,
   probes duration, respects the setup-skill gate).
3. **Dequeue pending with transcription** —
   ``_dequeue_pending_with_transcription`` (async; pulls a queued event
   and runs transcription enrichment before delivery).

``self.*`` references resolve unchanged via the MRO. Behavior-neutral
lift matching the existing mixin pattern.

``gateway.run`` module-level globals (``logger``, ``MessageType``,
``_probe_audio_duration``, ``_build_media_placeholder``) are lazy-imported
inside each method body to avoid a circular import (``gateway.run``
imports this mixin at module top). Stdlib (``asyncio``, ``json``, ``re``,
``os``), types (``List``, ``Optional``, ``MessageEvent``), and the
non-circular helpers ``_has_setup_skill`` (from
``gateway.gateway_gateway_env``) and ``_build_media_placeholder`` (from
``gateway.gateway_message_builders``) are imported at module top.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import List, Optional

from gateway.gateway_gateway_env import _has_setup_skill
from gateway.gateway_message_builders import _build_media_placeholder
from gateway.platforms.base import MessageEvent


class GatewayTranscriptionMixin:
    """Transcription enrichment methods for ``GatewayRunner``."""

    def _consume_pending_native_image_paths(self, session_key: str) -> List[str]:
        pending_native = getattr(self, "_pending_native_image_paths_by_session", None)
        if not pending_native:
            return []
        return list(pending_native.pop(session_key, []) or [])
    async def _enrich_message_with_transcription(
        self,
        user_text: str,
        audio_paths: List[str],
    ) -> tuple[str, List[str]]:
        """
        Auto-transcribe user voice/audio messages using the configured STT provider
        and prepend the transcript to the message text.

        Args:
            user_text:   The user's original caption / message text.
            audio_paths: List of local file paths to cached audio files.

        Returns:
            A tuple of ``(enriched_text, successful_transcripts)``:
              - ``enriched_text``: the message string with transcription wrappers
                prepended (same as before).
              - ``successful_transcripts``: the raw transcript strings for audio
                clips that were successfully transcribed, in input order. Empty
                list if every clip failed or STT is disabled. Callers can use
                this to echo transcripts back to the user before the agent loop.
        """
        from gateway.run import _probe_audio_duration, logger
        if not getattr(self.config, "stt_enabled", True):
            notes = []
            for path in audio_paths:
                abs_path = os.path.abspath(path)
                duration_str = await _probe_audio_duration(abs_path)
                if duration_str:
                    notes.append(
                        f"[The user sent a voice message: {abs_path} (duration: {duration_str})]"
                    )
                else:
                    notes.append(f"[The user sent a voice message: {abs_path}]")
            if not notes:
                return user_text, []
            prefix = "\n\n".join(notes)
            _placeholder = "(The user sent a message with no text content)"
            if user_text and user_text.strip() == _placeholder:
                return prefix, []
            if user_text:
                return f"{prefix}\n\n{user_text}", []
            return prefix, []

        from tools.transcription_tools import transcribe_audio

        enriched_parts = []
        successful_transcripts: List[str] = []
        for path in audio_paths:
            try:
                logger.debug("Transcribing user voice: %s", path)
                result = await asyncio.to_thread(transcribe_audio, path)
                if result["success"]:
                    transcript = result["transcript"]
                    successful_transcripts.append(transcript)
                    enriched_parts.append(
                        f'[The user sent a voice message~ '
                        f'Here\'s what they said: "{transcript}"]'
                    )
                else:
                    error = result.get("error", "unknown error")
                    if (
                        "No STT provider" in error
                        or error.startswith("Neither VOICE_TOOLS_OPENAI_KEY nor OPENAI_API_KEY is set")
                    ):
                        _no_stt_note = (
                            "[The user sent a voice message but I can't listen "
                            "to it right now — no STT provider is configured. "
                            "A direct message has already been sent to the user "
                            "with setup instructions."
                        )
                        if _has_setup_skill():
                            _no_stt_note += (
                                " You have a skill called hermes-agent-setup "
                                "that can help users configure Hermes features "
                                "including voice, tools, and more."
                            )
                        _no_stt_note += "]"
                        enriched_parts.append(_no_stt_note)
                    else:
                        enriched_parts.append(
                            "[The user sent a voice message but I had trouble "
                            f"transcribing it~ ({error})]"
                        )
            except Exception as e:
                logger.error("Transcription error: %s", e)
                enriched_parts.append(
                    "[The user sent a voice message but something went wrong "
                    "when I tried to listen to it~ Let them know!]"
                )

        if enriched_parts:
            prefix = "\n\n".join(enriched_parts)
            # Strip the empty-content placeholder from the Discord adapter
            # when we successfully transcribed the audio — it's redundant.
            _placeholder = "(The user sent a message with no text content)"
            if user_text and user_text.strip() == _placeholder:
                return prefix, successful_transcripts
            if user_text:
                return f"{prefix}\n\n{user_text}", successful_transcripts
            return prefix, successful_transcripts
        return user_text, successful_transcripts
    async def _dequeue_pending_with_transcription(
        self,
        adapter,
        session_key: str,
        source,
    ) -> str | None:
        """Dequeue a pending queued message, auto-transcribing audio media.

        When a voice/audio message arrives during an active agent run, the
        adapter stores the event in its pending queue and signals an interrupt
        (see base.BaseAdapter.handle_message). The adapter path bypasses
        _handle_message entirely, so the normal STT pipeline at message-receive
        time never runs.

        This helper fills that gap: when the dequeued event has audio media,
        we transcribe inline, echo the raw transcript back to the user (same
        "🎙️" format as the fresh-message path), and return enriched text.
        Non-audio events fall back to _build_media_placeholder, matching the
        original _dequeue_pending_text behavior.
        """
        from gateway.run import MessageType, _build_media_placeholder, logger
        event = adapter.get_pending_message(session_key)
        if not event:
            return None

        text = event.text or ""

        audio_paths: List[str] = []
        media_urls = getattr(event, "media_urls", None) or []
        media_types = getattr(event, "media_types", None) or []
        for i, path in enumerate(media_urls):
            mtype = media_types[i] if i < len(media_types) else ""
            is_audio = (
                mtype.startswith("audio/")
                or getattr(event, "message_type", None) in (MessageType.VOICE, MessageType.AUDIO)
            )
            if is_audio:
                audio_paths.append(path)

        if audio_paths:
            enriched_text, successful_transcripts = await self._enrich_message_with_transcription(
                text, audio_paths,
            )
            # Echo raw transcripts back to the user so voice interrupts
            # feel identical to fresh voice messages.
            if successful_transcripts:
                echo_adapter = self.adapters.get(source.platform)
                echo_meta = {"thread_id": source.thread_id} if source.thread_id else None
                if echo_adapter:
                    for tx in successful_transcripts:
                        try:
                            await echo_adapter.send(
                                source.chat_id,
                                f'🎙️ "{tx}"',
                                metadata=echo_meta,
                            )
                        except Exception as echo_exc:
                            logger.debug(
                                "Transcript echo failed (non-fatal): %s", echo_exc,
                            )
            return enriched_text or None

        # Non-audio fallback: preserve original _dequeue_pending_text semantics.
        if not text and media_urls:
            text = _build_media_placeholder(event)
        return text or None
