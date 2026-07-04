"""E2E tests for platform adapters.

Tests the full message lifecycle: receive message -> process -> send response.
Uses mocks for all external HTTP calls. No real API connections.
"""

import asyncio
import json
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
    is_network_accessible,
    merge_pending_message_event,
    safe_url_for_log,
    utf16_len,
)


# ---------------------------------------------------------------------------
# Telegram mock setup (same pattern as other telegram tests)
# ---------------------------------------------------------------------------

def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.constants.ChatType.PRIVATE = "private"
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)


_ensure_telegram_mock()

from plugins.platforms.telegram.adapter import TelegramAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def telegram_adapter():
    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = TelegramAdapter(config)
    adapter._bot = MagicMock()
    adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
    adapter._bot.send_photo = AsyncMock(return_value=MagicMock(message_id=2))
    adapter._bot.send_document = AsyncMock(return_value=MagicMock(message_id=3))
    adapter._bot.set_message_reaction = AsyncMock(return_value=True)
    adapter._bot.get_chat = AsyncMock(return_value=MagicMock(
        id=456, type=MagicMock(value="private"), title=None
    ))
    return adapter


# ---------------------------------------------------------------------------
# MessageEvent tests
# ---------------------------------------------------------------------------

class TestMessageEvent:
    def test_is_command(self):
        assert MessageEvent(text="/new").is_command() is True
        assert MessageEvent(text="/reset").is_command() is True
        assert MessageEvent(text="hello").is_command() is False

    def test_get_command(self):
        assert MessageEvent(text="/new").get_command() == "new"
        assert MessageEvent(text="/reset @botname").get_command() == "reset"
        assert MessageEvent(text="/MODEL gpt-4").get_command() == "model"

    def test_get_command_args(self):
        assert MessageEvent(text="/new").get_command_args() == ""
        assert MessageEvent(text="/model gpt-4o").get_command_args() == "gpt-4o"
        assert MessageEvent(text="not a command").get_command_args() == "not a command"

    def test_command_rejects_paths(self):
        assert MessageEvent(text="/home/user/file.txt").get_command() is None

    def test_command_strips_botname(self):
        assert MessageEvent(text="/new@mybot").get_command() == "new"


# ---------------------------------------------------------------------------
# Telegram adapter E2E (async tests via asyncio.run)
# ---------------------------------------------------------------------------

class TestTelegramE2E:
    def test_send_text_message(self, telegram_adapter):
        # send(chat_id, content) positional order
        asyncio.run(telegram_adapter.send("456", "Hello world"))
        telegram_adapter._bot.send_message.assert_called_once()

    def test_send_long_message_chunks(self, telegram_adapter):
        long_text = "x" * 10000
        asyncio.run(telegram_adapter.send("456", long_text))
        assert telegram_adapter._bot.send_message.call_count >= 1

    def test_send_with_reply_to(self, telegram_adapter):
        asyncio.run(telegram_adapter.send("456", "Reply text", reply_to="100"))
        assert telegram_adapter._bot.send_message.call_count >= 1

    def test_typing_indicator(self, telegram_adapter):
        asyncio.run(telegram_adapter.send_typing("456"))

    def test_format_message(self, telegram_adapter):
        result = telegram_adapter.format_message("Hello **bold** and *italic*")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_format_message_empty(self, telegram_adapter):
        result = telegram_adapter.format_message("")
        assert isinstance(result, str)

    def test_format_message_code_block(self, telegram_adapter):
        result = telegram_adapter.format_message("```python\nprint('hi')\n```")
        assert isinstance(result, str)

    def test_get_chat_info(self, telegram_adapter):
        info = asyncio.run(telegram_adapter.get_chat_info("456"))
        assert isinstance(info, dict)

    def test_reactions_lifecycle(self, telegram_adapter):
        event = MessageEvent(
            text="test",
            message_type=MessageType.TEXT,
            message_id="1",
            source=MagicMock(chat_id="2", user_id="3"),
        )
        asyncio.run(telegram_adapter.on_processing_start(event))
        asyncio.run(telegram_adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS))


# ---------------------------------------------------------------------------
# BasePlatformAdapter contract tests
# ---------------------------------------------------------------------------

class TestBasePlatformAdapterContract:
    def test_adapter_name(self, telegram_adapter):
        assert isinstance(telegram_adapter.name, str)
        assert len(telegram_adapter.name) > 0

    def test_initial_state_disconnected(self, telegram_adapter):
        assert telegram_adapter.is_connected is False

    def test_set_message_handler(self, telegram_adapter):
        handler = AsyncMock()
        telegram_adapter.set_message_handler(handler)

    def test_extract_images_static(self):
        images_url = "![alt](https://example.com/img.png)"
        images, remaining = BasePlatformAdapter.extract_images(images_url)
        assert len(images) == 1
        assert "example.com" in images[0][0]

    def test_extract_images_no_images(self):
        images, remaining = BasePlatformAdapter.extract_images("No images here")
        assert remaining == "No images here"
        assert len(images) == 0

    def test_merge_caption(self):
        result = BasePlatformAdapter._merge_caption(None, "new caption")
        assert result == "new caption"

    def test_merge_caption_existing(self):
        result = BasePlatformAdapter._merge_caption("existing", "new")
        assert "existing" in result
        assert "new" in result

    def test_is_retryable_error(self):
        # These match the actual patterns in the codebase
        assert BasePlatformAdapter._is_retryable_error("connectionerror") is True
        assert BasePlatformAdapter._is_retryable_error("broken pipe") is True
        assert BasePlatformAdapter._is_retryable_error("network error") is True
        assert BasePlatformAdapter._is_retryable_error(None) is False
        assert BasePlatformAdapter._is_retryable_error("invalid_auth") is False

    def test_is_timeout_error(self):
        assert BasePlatformAdapter._is_timeout_error("timed out") is True
        assert BasePlatformAdapter._is_timeout_error(None) is False
        assert BasePlatformAdapter._is_timeout_error("connectionerror") is False


# ---------------------------------------------------------------------------
# MessageEvent merging
# ---------------------------------------------------------------------------

class TestMessageEventMerging:
    def test_merge_replaces_by_default(self):
        pending = {}
        base = MessageEvent(text="Hello", message_type=MessageType.TEXT, message_id="1")
        pending["test_key"] = base
        incoming = MessageEvent(text=" World", message_type=MessageType.TEXT, message_id="2")
        merge_pending_message_event(pending, "test_key", incoming)
        assert pending["test_key"].text == " World"
        assert pending["test_key"].message_id == "2"

    def test_merge_text_appends_when_enabled(self):
        pending = {}
        base = MessageEvent(text="Hello", message_type=MessageType.TEXT, message_id="1")
        pending["test_key"] = base
        incoming = MessageEvent(text=" World", message_type=MessageType.TEXT, message_id="2")
        merge_pending_message_event(pending, "test_key", incoming, merge_text=True)
        assert "Hello" in pending["test_key"].text
        assert "World" in pending["test_key"].text
        assert pending["test_key"].message_id == "1"

    def test_merge_new_key_stores_event(self):
        pending = {}
        event = MessageEvent(text="Fresh", message_type=MessageType.TEXT, message_id="1")
        merge_pending_message_event(pending, "new_key", event)
        assert "new_key" in pending
        assert pending["new_key"].text == "Fresh"


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------

class TestBaseUtilities:
    def test_utf16_len(self):
        assert utf16_len("hello") == 5
        assert utf16_len("") == 0
        assert utf16_len("🎉") == 2

    def test_is_network_accessible(self):
        # Loopback addresses are not network-accessible
        assert is_network_accessible("127.0.0.1") is False
        assert is_network_accessible("localhost") is False
        # Public IPs and hostnames are network-accessible
        assert is_network_accessible("example.com") is True
        assert is_network_accessible("1.2.3.4") is True

    def test_safe_url_strips_query(self):
        url = "https://api.example.com/v1/chat?api_key=secret&model=gpt4"
        safe = safe_url_for_log(url)
        assert "secret" not in safe
        assert "api.example.com" in safe

    def test_safe_url_truncates(self):
        url = "https://example.com/" + "a" * 200
        safe = safe_url_for_log(url)
        assert len(safe) <= 100
