"""Voice / auto-TTS adapter helpers extracted from gateway/run.py GatewayRunner.

Round 9 of gateway decomposition. These manage per-adapter voice-mode and
auto-TTS opt-in/opt-out sets. All stateless on the gateway side — they
mutate the ADAPTER's attributes, never self. Moved as plain functions.
"""

from gateway.platforms.base import Platform


def _voice_key(platform: Platform, chat_id: str) -> str:
    """Return a platform-namespaced key for voice mode state."""
    return f"{platform.value}:{chat_id}"


def _set_adapter_auto_tts_disabled(adapter, chat_id: str, disabled: bool) -> None:
    """Update an adapter's in-memory auto-TTS suppression set if present."""
    disabled_chats = getattr(adapter, "_auto_tts_disabled_chats", None)
    if not isinstance(disabled_chats, set):
        return
    if disabled:
        disabled_chats.add(chat_id)
        # ``/voice off`` also clears any explicit enable — it's a hard override.
        enabled_chats = getattr(adapter, "_auto_tts_enabled_chats", None)
        if isinstance(enabled_chats, set):
            enabled_chats.discard(chat_id)
    else:
        disabled_chats.discard(chat_id)


def _set_adapter_auto_tts_enabled(adapter, chat_id: str, enabled: bool) -> None:
    """Update an adapter's per-chat auto-TTS opt-in set if present.

    Used for ``/voice on``/``/voice tts`` where the user explicitly wants
    auto-TTS even when ``voice.auto_tts`` is False globally.
    """
    enabled_chats = getattr(adapter, "_auto_tts_enabled_chats", None)
    if not isinstance(enabled_chats, set):
        return
    if enabled:
        enabled_chats.add(chat_id)
        # An explicit opt-in clears any stale /voice off for this chat.
        disabled_chats = getattr(adapter, "_auto_tts_disabled_chats", None)
        if isinstance(disabled_chats, set):
            disabled_chats.discard(chat_id)
    else:
        enabled_chats.discard(chat_id)
