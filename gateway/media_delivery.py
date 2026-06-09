"""
Media delivery utilities for gateway responses.

This module handles:
- Extraction and delivery of MEDIA: tags from agent responses
- Native image path consumption from platform adapters
- Destructive slash command confirmation
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote as _quote

from gateway.platforms.base import BasePlatformAdapter, should_send_media_as_audio


logger = logging.getLogger(__name__)


# Regex pattern for matching MEDIA: tags in tool outputs
_TOOL_MEDIA_RE = __import__("re").compile(
    r'MEDIA:((?:[A-Za-z]:[/\\]|/|~\/)\S+\.(?:png|jpe?g|gif|webp|'
    r'mp4|mov|avi|mkv|webm|ogg|opus|mp3|wav|m4a|'
    r'flac|epub|pdf|zip|rar|7z|docx?|xlsx?|pptx?|'
    r'txt|csv|apk|ipa))',
    __import__("re").IGNORECASE
)

# Tool names that are allowed to auto-append MEDIA tags (TTS tools)
_AUTO_APPEND_MEDIA_TOOL_NAMES = {
    "tts",
    "text_to_speech",
    "speak",
    "say",
    "voice_synthesis",
}

# Video file extensions for routing
_VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.3gp'}

# Image file extensions for batch routing
_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}


def collect_auto_append_media_tags(
    messages: List[Dict[str, Any]],
    history_offset: int = 0,
    history_media_paths: Optional[set] = None,
) -> tuple[List[str], bool]:
    """Collect real media tags from current-turn producer-tool results only.

    Two layered guards keep stale/example MEDIA: strings out of the reply:

    1. Producer-tool allowlist: only tools that intentionally emit deliverable
       artifacts (TTS) are eligible. Documentation, logs, and search results can
       contain example strings such as MEDIA:/absolute/path/to/file, which must
       never be delivered as attachments. (Fixes the original report behind #16721.)
    2. Current-turn isolation: only messages produced this turn are scanned, so a
       tool result from an earlier turn (still present in the full message list)
       cannot leak onto a later text-only reply (#34608).

    Mid-run context compression can rewrite/shrink the message list below the
    original history length. When that happens the slice boundary is no longer
    trustworthy, so fall back to scanning every message and rely on
    ``history_media_paths`` for dedup, preserving the compression-safe behaviour
    of #160. The producer-tool allowlist still applies on the fallback path.

    Args:
        messages: Full message list from the turn
        history_offset: Number of history messages at the start of the list
        history_media_paths: Set of media paths already seen in history

    Returns:
        Tuple of (media_tags list, has_voice_directive bool)
    """
    history_media_paths = history_media_paths or set()
    # Only trust the slice boundary when the message list still contains the
    # full history prefix. Otherwise scan everything (compression-safe fallback).
    if history_offset and len(messages) >= history_offset:
        new_messages = messages[history_offset:]
    else:
        new_messages = messages

    tool_name_by_call_id: Dict[str, str] = {}
    for msg in new_messages:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            call_id = call.get("id") or call.get("call_id")
            fn = call.get("function") or {}
            name = str(fn.get("name") or call.get("name") or "")
            if call_id and name:
                tool_name_by_call_id[str(call_id)] = name

    media_tags: List[str] = []
    has_voice_directive = False
    for msg in new_messages:
        if msg.get("role") not in ("tool", "function"):
            continue
        call_id = str(msg.get("tool_call_id") or msg.get("call_id") or "")
        if tool_name_by_call_id.get(call_id) not in _AUTO_APPEND_MEDIA_TOOL_NAMES:
            continue
        content = str(msg.get("content") or "")
        if "MEDIA:" not in content:
            continue
        for match in _TOOL_MEDIA_RE.finditer(content):
            path = match.group(1).strip().rstrip('\",}')
            if path and path not in history_media_paths:
                media_tags.append(f"MEDIA:{path}")
        if "[[audio_as_voice]]" in content:
            has_voice_directive = True

    return media_tags, has_voice_directive


async def deliver_media_from_response(
    response: str,
    event: "MessageEvent",
    adapter: BasePlatformAdapter,
    thread_metadata_fn,
    reply_anchor_fn,
) -> None:
    """Extract MEDIA: tags and local file paths from a response and deliver them.

    Called after streaming has already sent the text to the user, so the
    text itself is already delivered — this only handles file attachments
    that the normal _process_message_background path would have caught.

    Args:
        response: The agent response text (may contain MEDIA: tags)
        event: The message event with source info
        adapter: The platform adapter for delivery
        thread_metadata_fn: Function to get thread metadata for replies
        reply_anchor_fn: Function to get reply anchor for event
    """
    try:
        # Capture [[as_document]] before extract_media strips it, so the
        # dispatch partition below can route image-extension files
        # through send_document (preserving bytes) instead of
        # send_multiple_images (Telegram sendPhoto recompresses to ~1280px).
        force_document_attachments = "[[as_document]]" in response

        media_files, cleaned = adapter.extract_media(response)
        media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
        # Chain the cleaned text through each extractor (extract_media →
        # extract_images → extract_local_files) so MEDIA: tags and image URLs
        # are removed before the bare-path auto-detect runs. Previously the
        # cleaned text from extract_media was dropped (``_``) and
        # extract_local_files scanned text that still contained MEDIA: tags,
        # producing false-positive bare-path matches with the MEDIA: prefix
        # glued on. This matches the chain order in gateway/platforms/base.py.
        _, cleaned = adapter.extract_images(cleaned)
        local_files, _ = adapter.extract_local_files(cleaned)
        local_files = BasePlatformAdapter.filter_local_delivery_paths(local_files)

        _thread_meta = thread_metadata_fn(event.source, reply_anchor_fn(event))

        # Partition out images so they can be sent as a single batch
        # (e.g. Signal's multi-attachment RPC). When [[as_document]] was
        # set, image-extension files skip the photo path and route to
        # send_document below — preserving original bytes.
        image_paths: list = []
        non_image_media: list = []
        for media_path, is_voice in media_files:
            ext = Path(media_path).suffix.lower()
            if (ext in _IMAGE_EXTS
                    and not is_voice
                    and not force_document_attachments):
                image_paths.append(media_path)
            else:
                non_image_media.append((media_path, is_voice))

        non_image_local: list = []
        for file_path in local_files:
            if (Path(file_path).suffix.lower() in _IMAGE_EXTS
                    and not force_document_attachments):
                image_paths.append(file_path)
            else:
                non_image_local.append(file_path)

        if image_paths:
            try:
                images = [(f"file://{_quote(p)}", "") for p in image_paths]
                await adapter.send_multiple_images(
                    chat_id=event.source.chat_id,
                    images=images,
                    metadata=_thread_meta,
                )
            except Exception as e:
                logger.warning("[%s] Post-stream image batch delivery failed: %s", adapter.name, e)

        for media_path, is_voice in non_image_media:
            try:
                ext = Path(media_path).suffix.lower()
                if should_send_media_as_audio(event.source.platform, ext, is_voice=is_voice):
                    await adapter.send_voice(
                        chat_id=event.source.chat_id,
                        audio_path=media_path,
                        metadata=_thread_meta,
                    )
                elif ext in _VIDEO_EXTS:
                    await adapter.send_video(
                        chat_id=event.source.chat_id,
                        video_path=media_path,
                        metadata=_thread_meta,
                    )
                else:
                    await adapter.send_document(
                        chat_id=event.source.chat_id,
                        file_path=media_path,
                        metadata=_thread_meta,
                    )
            except Exception as e:
                logger.warning("[%s] Post-stream media delivery failed: %s", adapter.name, e)

        for file_path in non_image_local:
            try:
                ext = Path(file_path).suffix.lower()
                if ext in _VIDEO_EXTS:
                    await adapter.send_video(
                        chat_id=event.source.chat_id,
                        video_path=file_path,
                        metadata=_thread_meta,
                    )
                else:
                    await adapter.send_document(
                        chat_id=event.source.chat_id,
                        file_path=file_path,
                        metadata=_thread_meta,
                    )
            except Exception as e:
                logger.warning("[%s] Post-stream file delivery failed: %s", adapter.name, e)

    except Exception as e:
        logger.warning("Post-stream media extraction failed: %s", e)


class SlashConfirmHandler:
    """Handler for destructive slash command confirmation UI.

    This provides the UI layer for commands like /new, /reset, /undo that
    can destroy conversation state. Platforms with native button support
    (Telegram, Discord, Slack) show interactive yes/no buttons; others
    fall back to text-based confirmation.
    """

    def __init__(self, runner):
        """Initialize the handler with a GatewayRunner instance.

        Args:
            runner: The GatewayRunner instance for config and session access
        """
        self.runner = runner

    async def maybe_confirm_destructive_slash(
        self,
        *,
        event: "MessageEvent",
        command: str,
        title: str,
        detail: str,
        execute,
    ) -> Union[str, "EphemeralReply", None]:
        """Gate a destructive session slash command (/new, /reset, /undo).

        ``execute`` is an async callable ``execute() -> str | EphemeralReply``
        that performs the destructive action. If the
        ``approvals.destructive_slash_confirm`` config gate is off, ``execute``
        runs immediately (returning its result). Otherwise this routes
        through ``_request_slash_confirm`` — native yes/no buttons on
        Telegram/Discord/Slack, text fallback elsewhere.

        Three-option resolution:

          - ``once``  — run ``execute`` and return its result
          - ``always`` — persist ``approvals.destructive_slash_confirm: false``,
                        then run ``execute``
          - ``cancel`` — return a "cancelled" message; do not run ``execute``

        Args:
            event: The message event triggering the command
            command: Command name (e.g., "reset", "new")
            title: Short title for the confirmation dialog
            detail: Detailed explanation of what will be destroyed
            execute: Async callable that performs the destructive action

        Returns:
            The result of execute (str or EphemeralReply), or a cancellation message
        """
        # Gate check.
        confirm_required = True
        try:
            cfg = self.runner._read_user_config()
            approvals = cfg.get("approvals") if isinstance(cfg, dict) else None
            if isinstance(approvals, dict):
                confirm_required = bool(approvals.get("destructive_slash_confirm", True))
        except Exception:
            pass

        if not confirm_required:
            return await execute()

        session_key = self.runner._session_key_for_source(event.source)

        async def _on_confirm(choice: str):
            if choice == "cancel":
                return f"🟡 /{command} cancelled. Conversation unchanged."
            if choice == "always":
                try:
                    from cli import save_config_value
                    save_config_value("approvals.destructive_slash_confirm", False)
                    logger.info(
                        "User opted out of destructive slash confirm (session=%s)",
                        session_key,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to persist destructive_slash_confirm=false: %s", exc,
                    )
            result = await execute()
            if choice == "always":
                note = (
                    "\n\nℹ️ Future /clear, /new, /reset, and /undo will run "
                    "without confirmation. Re-enable via "
                    "`approvals.destructive_slash_confirm: true` in config.yaml."
                )
                if isinstance(result, str):
                    return result + note
                # EphemeralReply or other — leave untouched; the opt-out note
                # would otherwise mangle structured replies. The persist itself
                # already happened above; user gets the same UX next time.
                return result
            return result

        prompt_message = (
            f"⚠️ **Confirm /{command}**\n\n"
            f"{detail}\n\n"
            "Choose:\n"
            "• **Approve Once** — proceed this time only\n"
            "• **Always Approve** — proceed and silence this prompt permanently\n"
            "• **Cancel** — keep current conversation\n\n"
            "_Text fallback: reply `/approve`, `/always`, or `/cancel`._"
        )
        return await self._request_slash_confirm(
            event=event,
            command=command,
            title=title,
            message=prompt_message,
            handler=_on_confirm,
        )

    async def _request_slash_confirm(
        self,
        *,
        event: "MessageEvent",
        command: str,
        title: str,
        message: str,
        handler,
    ) -> Optional[str]:
        """Ask the user to confirm an expensive slash command.

        ``handler`` is an async callable ``handler(choice: str) -> str``
        where ``choice`` is ``"once"``, ``"always"``, or ``"cancel"``.
        The handler runs on the event loop when the user responds; its
        return value is sent back as a gateway message.

        Returns a short acknowledgment string to send immediately (before
        the user's response). If buttons rendered successfully the ack
        is ``None`` (buttons are self-explanatory); if we fell back to
        text the message itself IS the ack.
        """
        from tools import slash_confirm as _slash_confirm_mod

        source = event.source
        session_key = self.runner._session_key_for_source(source)
        # Bare-runner test harnesses (object.__new__(GatewayRunner)) skip
        # __init__ and don't have the counter attribute — fall back to a
        # local counter so tests don't AttributeError. Real runs always
        # have the instance attribute.
        counter = getattr(self.runner, "_slash_confirm_counter", None)
        if counter is None:
            import itertools as _itertools
            counter = _itertools.count(1)
            self.runner._slash_confirm_counter = counter
        confirm_id = f"{next(counter)}"

        # Register the pending confirm FIRST so a super-fast button click
        # cannot race the send_slash_confirm return.
        _slash_confirm_mod.register(session_key, confirm_id, command, handler)

        adapter = self.runner.adapters.get(source.platform)
        metadata = self.runner._thread_metadata_for_source(source, self.runner._reply_anchor_for_event(event))

        used_buttons = False
        if adapter is not None:
            try:
                button_result = await adapter.send_slash_confirm(
                    chat_id=source.chat_id,
                    title=title,
                    message=message,
                    session_key=session_key,
                    confirm_id=confirm_id,
                    metadata=metadata,
                )
                if button_result and getattr(button_result, "success", False):
                    used_buttons = True
            except Exception as exc:
                logger.debug(
                    "send_slash_confirm failed for %s on %s: %s",
                    command, source.platform, exc,
                )

        if used_buttons:
            # Buttons rendered — no redundant text ack.
            return None
        # Text fallback — return the prompt message as the direct reply.
        return message


def consume_pending_native_image_paths(
    runner: "GatewayRunner",
    session_key: str,
) -> List[str]:
    """Consume and return pending native image paths for a session.

    Platform adapters may queue native image paths (e.g., from photo uploads)
    that need to be delivered with the next agent response. This function
    retrieves and clears those paths for the given session.

    Args:
        runner: The GatewayRunner instance
        session_key: The session key to retrieve paths for

    Returns:
        List of file paths to images that were pending
    """
    pending_native = getattr(runner, "_pending_native_image_paths_by_session", None)
    if not pending_native:
        return []
    return list(pending_native.pop(session_key, []) or [])
