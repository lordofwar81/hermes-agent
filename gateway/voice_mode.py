"""
Voice mode management for GatewayRunner.

This module handles voice configuration and TTS (text-to-speech) settings
for platform adapters that support voice input/output.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def voice_key(source: Any) -> str:
    """Generate a unique key for voice mode storage.

    Args:
        source: MessageEvent source object

    Returns:
        A string key suitable for storing voice mode state
    """
    # Use platform and user/chat ID to create a unique key
    platform = getattr(source, "platform", None)
    user_id = getattr(source, "user_id", None)
    chat_id = getattr(source, "chat_id", None)

    parts = []
    if platform:
        parts.append(str(platform.value))
    if user_id:
        parts.append(str(user_id))
    if chat_id:
        parts.append(str(chat_id))

    return ":".join(parts) if parts else "default"


def load_voice_modes(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Load voice modes from gateway config.

    Args:
        cfg: Gateway configuration dict

    Returns:
        Dict mapping voice mode keys to mode settings
    """
    voice_cfg = cfg.get("voice_modes", {})
    if isinstance(voice_cfg, dict):
        return voice_cfg
    return {}


def save_voice_modes(
    voice_modes: Dict[str, Dict[str, Any]],
    cfg_path: Optional[str] = None,
) -> None:
    """Save voice modes to gateway config.

    Args:
        voice_modes: Dict of voice mode settings
        cfg_path: Optional path to config file
    """
    # In a real implementation, this would write to config.yaml
    # For now, voice modes are ephemeral and stored in memory only
    logger.debug("Voice modes saved (ephemeral): %d entries", len(voice_modes))


def set_adapter_auto_tts_disabled(
    adapter: Any,
    chat_id: Any = None,
    disabled: bool = True,
) -> None:
    """Set adapter auto-TTS disabled state.

    Args:
        adapter: Platform adapter instance
        chat_id: Optional chat ID (for compatibility, not used)
        disabled: Whether auto-TTS should be disabled
    """
    if hasattr(adapter, "auto_tts_disabled"):
        adapter.auto_tts_disabled = disabled


def set_adapter_auto_tts_enabled(
    adapter: Any,
    chat_id: Any = None,
    enabled: bool = True,
) -> None:
    """Set adapter auto-TTS enabled state.

    Args:
        adapter: Platform adapter instance
        chat_id: Optional chat ID (for compatibility, not used)
        enabled: Whether auto-TTS should be enabled
    """
    if hasattr(adapter, "auto_tts_enabled"):
        adapter.auto_tts_enabled = enabled


def sync_voice_mode_state_to_adapter(
    adapter: Any,
    voice_mode: Optional[Dict[str, Any]],
) -> None:
    """Synchronize voice mode state to adapter.

    Args:
        adapter: Platform adapter instance
        voice_mode: Voice mode settings dict
    """
    if not voice_mode:
        return

    # Sync auto-TTS settings
    if voice_mode.get("input"):
        set_adapter_auto_tts_enabled(adapter, True)
    elif voice_mode.get("input") is False:
        set_adapter_auto_tts_disabled(adapter, True)

    # Sync other voice settings if adapter supports them
    if hasattr(adapter, "set_voice_mode"):
        try:
            adapter.set_voice_mode(voice_mode)
        except Exception as e:
            logger.debug("Failed to set voice mode on adapter: %s", e)
