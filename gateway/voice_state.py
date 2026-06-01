"""
Voice mode state management for the gateway.

Extracted from gateway/run.py to reduce the God file size.
Provides functions for managing per-chat voice mode state, auto-TTS
settings, and image input routing.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

VALID_VOICE_MODES = {"off", "voice_only", "all"}


def voice_key(platform: Any, chat_id: str) -> str:
    """Return a platform-namespaced key for voice mode state."""
    return f"{platform.value}:{chat_id}"


def load_voice_modes(voice_mode_path: Path) -> Dict[str, str]:
    """Load persisted voice mode state from a JSON file."""
    try:
        data = json.loads(voice_mode_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}

    if not isinstance(data, dict):
        return {}

    result = {}
    for chat_id, mode in data.items():
        if mode not in VALID_VOICE_MODES:
            continue
        key = str(chat_id)
        if ":" not in key:
            logger.warning(
                "Skipping legacy unprefixed voice mode key %r during migration. "
                "Re-enable voice mode on that chat to rebuild the prefixed key.",
                key,
            )
            continue
        result[key] = mode
    return result


def save_voice_modes(voice_mode: Dict[str, str], voice_mode_path: Path) -> None:
    """Persist voice mode state to a JSON file."""
    try:
        voice_mode_path.parent.mkdir(parents=True, exist_ok=True)
        voice_mode_path.write_text(json.dumps(voice_mode, indent=2))
    except OSError as e:
        logger.warning("Failed to save voice modes: %s", e)


def set_adapter_auto_tts_disabled(adapter: Any, chat_id: str, disabled: bool) -> None:
    """Update an adapter's in-memory auto-TTS suppression set if present."""
    disabled_chats = getattr(adapter, "_auto_tts_disabled_chats", None)
    if not isinstance(disabled_chats, set):
        return
    if disabled:
        disabled_chats.add(chat_id)
        enabled_chats = getattr(adapter, "_auto_tts_enabled_chats", None)
        if isinstance(enabled_chats, set):
            enabled_chats.discard(chat_id)
    else:
        disabled_chats.discard(chat_id)


def set_adapter_auto_tts_enabled(adapter: Any, chat_id: str, enabled: bool) -> None:
    """Update an adapter's per-chat auto-TTS opt-in set if present.

    Used for /voice on / /voice tts where the user explicitly wants
    auto-TTS even when voice.auto_tts is False globally.
    """
    enabled_chats = getattr(adapter, "_auto_tts_enabled_chats", None)
    if not isinstance(enabled_chats, set):
        return
    if enabled:
        enabled_chats.add(chat_id)
        disabled_chats = getattr(adapter, "_auto_tts_disabled_chats", None)
        if isinstance(disabled_chats, set):
            disabled_chats.discard(chat_id)
    else:
        enabled_chats.discard(chat_id)


def sync_voice_mode_state_to_adapter(
    adapter: Any, voice_mode: Dict[str, str], platform: Any
) -> None:
    """Restore persisted /voice state into a live platform adapter.

    Populates three fields from config + ``voice_mode``:
      - _auto_tts_default: global default from voice.auto_tts
      - _auto_tts_enabled_chats: chats with mode voice_only/all
      - _auto_tts_disabled_chats: chats with mode off
    """
    disabled_chats = getattr(adapter, "_auto_tts_disabled_chats", None)
    enabled_chats = getattr(adapter, "_auto_tts_enabled_chats", None)
    if not isinstance(disabled_chats, set) and not isinstance(enabled_chats, set):
        return

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
            key[len(prefix):]
            for key, mode in voice_mode.items()
            if mode == "off" and key.startswith(prefix)
        )
    if isinstance(enabled_chats, set):
        enabled_chats.clear()
        enabled_chats.update(
            key[len(prefix):]
            for key, mode in voice_mode.items()
            if mode in {"voice_only", "all"} and key.startswith(prefix)
        )


def decide_image_input_mode() -> str:
    """Resolve the image-input routing for the currently active model.

    Returns "native" (attach pixels on the user turn) or "text"
    (pre-analyze with vision_analyze and prepend the description).
    """
    try:
        from agent.image_routing import decide_image_input_mode as _decide
        from agent.auxiliary_client import _read_main_model, _read_main_provider
        from hermes_cli.config import load_config

        cfg = load_config()
        provider = _read_main_provider()
        model = _read_main_model()
        return _decide(provider, model, cfg)
    except Exception as exc:
        logger.debug("image_routing: decision failed, falling back to text — %s", exc)
        return "text"
