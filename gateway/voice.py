"""
Voice mode utilities for GatewayRunner.

Extracted from gateway/run.py to reduce the God file size.
Provides functions for managing voice mode state across adapters.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gateway.run import GatewayRunner

logger = logging.getLogger(__name__)


def sync_voice_mode_state_to_adapter(
    runner,  # GatewayRunner instance
    adapter,  # Platform adapter
) -> None:
    """Restore persisted /voice state into a live platform adapter.

    Populates three fields from config + ``runner._voice_mode``:
      - ``_auto_tts_default``: global default from ``voice.auto_tts``
      - ``_auto_tts_enabled_chats``: chats with mode ``voice_only``/``all``
      - ``_auto_tts_disabled_chats``: chats with mode ``off``
    """
    from hermes_cli.enums import Platform

    platform = getattr(adapter, "platform", None)
    if not isinstance(platform, Platform):
        return

    disabled_chats = getattr(adapter, "_auto_tts_disabled_chats", None)
    enabled_chats = getattr(adapter, "_auto_tts_enabled_chats", None)
    if not isinstance(disabled_chats, set) and not isinstance(enabled_chats, set):
        return

    # Push the global voice.auto_tts default (config.yaml) onto the adapter.
    # Lazy import to avoid adding a module-level dep from gateway → hermes_cli.
    try:
        from hermes_cli.config import load_config as _load_full_config
        _full_cfg = _load_full_config()
        _auto_tts_default = bool(
            (_full_cfg.get("voice") or {}).get("auto_tts", False)
        )
    except Exception:
        _auto_tts_default = False
    if hasattr(adapter, "_auto_tts_default"):
        adapter._auto_tts_default = _auto_tts_default

    prefix = f"{platform.value}:"
    if isinstance(disabled_chats, set):
        disabled_chats.clear()
        disabled_chats.update(
            key[len(prefix):] for key, mode in runner._voice_mode.items()
            if mode == "off" and key.startswith(prefix)
        )
    if isinstance(enabled_chats, set):
        enabled_chats.clear()
        enabled_chats.update(
            key[len(prefix):] for key, mode in runner._voice_mode.items()
            if mode in {"voice_only", "all"} and key.startswith(prefix)
        )


def is_duplicate_voice_transcript(
    runner,  # GatewayRunner instance
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
    import re
    import time
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


def load_voice_modes(runner) -> Dict[str, str]:
    """Load persisted voice mode settings from disk."""
    import json

    try:
        data = json.loads(runner._VOICE_MODE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}

    if not isinstance(data, dict):
        return {}

    valid_modes = {"off", "voice_only", "all"}
    result = {}
    for chat_id, mode in data.items():
        if mode not in valid_modes:
            continue
        key = str(chat_id)
        # Skip legacy unprefixed keys (warn and skip)
        if ":" not in key:
            logger.warning(
                "Skipping legacy unprefixed voice mode key %r during migration. "
                "Re-enable voice mode on that chat to rebuild the prefixed key.",
                key,
            )
            continue
        result[key] = mode
    return result
