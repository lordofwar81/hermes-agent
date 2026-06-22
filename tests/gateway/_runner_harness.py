"""Shared test harness for GatewayRunner integration tests.

Consolidates the ~65 duplicated ``_make_runner()`` / ``_make_event()`` /
``_make_source()`` helpers scattered across ``tests/gateway/`` into one
importable surface. Every primitive here is lifted from a proven per-file
pattern — this module just makes them shared.

Established patterns reused:
  - ``object.__new__(GatewayRunner)`` + hand-set attrs (from
    ``restart_test_helpers.make_restart_runner``)
  - ``BasePlatformAdapter`` subclass recording sends (from
    ``RestartTestAdapter`` / ``StubAdapter``)
  - ``sys.modules["run_agent"]`` swap to fake the 30-kwarg ``AIAgent``
    (from ``test_session_hygiene.py``)
  - minimal ``GatewayConfig`` (from ``test_runner_startup_failures.py``)

Usage::

    from tests.gateway._runner_harness import (
        build_runner, build_event, RecordingAdapter, swap_run_agent,
    )

    runner = build_runner(tmp_path=tmp_path)
    runner.adapters = {Platform.TELEGRAM: RecordingAdapter()}
    event = build_event(text="/hello")
    await runner._handle_message(event)
"""

from __future__ import annotations

import asyncio
import sys
import threading
import types
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.restart import DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
from gateway.run import GatewayRunner
from gateway.session import SessionSource


# ---------------------------------------------------------------------------
# RecordingAdapter — the canonical fake platform adapter.
# Lifted from restart_test_helpers.RestartTestAdapter (fullest existing fake)
# with added connect-behavior parametrization from test_runner_startup_failures.
# ---------------------------------------------------------------------------


class RecordingAdapter(BasePlatformAdapter):
    """Fake platform adapter that records sends and typing calls.

    Connect behavior is parametrizable so the same class covers the
    success / retryable-fatal / disabled cases that
    test_runner_startup_failures.py needed three separate classes for.

    Attributes:
        sent: list of content strings passed to ``send()``.
        sent_calls: list of ``(chat_id, content, metadata)`` tuples.
        typing_calls: list of ``(chat_id, metadata)`` tuples.
    """

    def __init__(
        self,
        platform: Platform = Platform.TELEGRAM,
        *,
        connect_result: bool = True,
        fatal_error_code: str | None = None,
        fatal_error_message: str | None = None,
        fatal_error_retryable: bool = False,
    ):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent: list[str] = []
        self.sent_calls: list[tuple[str, object, object]] = []
        self.typing_calls: list[tuple[str, object]] = []
        self._connect_result = connect_result
        self._fatal_code = fatal_error_code
        self._fatal_msg = fatal_error_message
        self._fatal_retryable = fatal_error_retryable

    async def connect(self) -> bool:
        if self._fatal_code:
            self._set_fatal_error(
                self._fatal_code,
                self._fatal_msg or "connect failed",
                retryable=self._fatal_retryable,
            )
        return self._connect_result

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        self.sent_calls.append((chat_id, content, metadata))
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id, metadata=None):
        self.typing_calls.append((chat_id, metadata))
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


# ---------------------------------------------------------------------------
# build_runner — construct a GatewayRunner without running __init__.
# Lifted from restart_test_helpers.make_restart_runner. Methods resolve
# normally through MRO (no rebind needed — object.__new__ preserves class
# dispatch). Override any attr via kwargs.
# ---------------------------------------------------------------------------


def build_runner(
    *,
    config: GatewayConfig | None = None,
    tmp_path=None,
    adapters: dict | None = None,
    session_store=None,
    platform: Platform = Platform.TELEGRAM,
    real_init: bool = False,
) -> GatewayRunner:
    """Build a GatewayRunner with sensible test defaults.

    By default bypasses ``__init__`` (fast, isolated) and hand-sets the
    ~30 attributes every runner test needs. Pass ``real_init=True`` to
    run the real 333-line ``__init__`` instead (heavier, realistic —
    matches test_runner_startup_failures.py).

    Args:
        config: GatewayConfig. If None, uses minimal_config(platform, tmp_path).
        tmp_path: pytest tmp_path for sessions_dir. Required if config is None.
        adapters: dict of {Platform: adapter}. If None, empty dict.
        session_store: SessionStore or MagicMock. If None, MagicMock with
            ``_entries={}`` (the established stub pattern).
        platform: Platform for the minimal config (when config is None).
        real_init: If True, construct via ``GatewayRunner(config)`` and
            return immediately (ignores adapters/session_store overrides —
            set them after construction).
    """
    if config is None:
        config = minimal_config(platform=platform, sessions_dir=tmp_path)

    if real_init:
        return GatewayRunner(config)

    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner._running = True
    runner._gateway_loop = None
    runner._shutdown_event = asyncio.Event()
    runner._exit_reason = None
    runner._exit_code = None
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()
    runner._draining = False
    runner._stop_task = None
    runner._restart_requested = False
    runner._signal_initiated_shutdown = False
    runner._restart_task_started = False
    runner._restart_detached = False
    runner._restart_via_service = False
    runner._restart_command_source = None
    runner._restart_drain_timeout = DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    runner._busy_input_mode = "interrupt"
    runner._busy_text_mode = "normal"
    runner._update_prompt_pending = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._session_sources = OrderedDict()
    runner._session_sources_max = 512
    runner._failed_platforms = {}
    runner._agent_cache = {}
    # Match production (gateway/run.py __init__): _agent_cache_lock is a
    # threading.Lock, entered via a sync `with`. Several mixins (stop_mixin,
    # agent_cache_mixin) and the idle-cache teardown path use it sync-style;
    # an asyncio.Lock here raises "does not support the context manager
    # protocol" the first time a test exercises stop() or cache eviction
    # end-to-end. threading.Lock is the faithful, drop-in choice.
    runner._agent_cache_lock = threading.Lock()
    runner._active_session_leases = {}
    runner._startup_restore_in_progress = False
    runner._startup_restore_queue = []
    runner._startup_restore_tasks = []
    runner._shutdown_all_gateway_honcho = lambda: None
    runner._update_runtime_status = MagicMock()
    runner._is_user_authorized = lambda _source: True
    runner._model = "test-model"
    runner._base_url = None
    runner._service_tier = None

    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.discover_and_load = MagicMock()
    runner.pairing_store = MagicMock()
    runner.delivery_router = MagicMock()

    if session_store is not None:
        runner.session_store = session_store
    else:
        runner.session_store = MagicMock()
        runner.session_store._entries = {}

    runner.adapters = adapters if adapters is not None else {}
    return runner


# ---------------------------------------------------------------------------
# build_source / build_event — consolidate the ~5 duplicated builders.
# Only platform + chat_id are required for SessionSource; only text for
# MessageEvent (confirmed: gateway/session.py:71, gateway/platforms/base.py:1416).
# ---------------------------------------------------------------------------


def build_source(
    *,
    platform: Platform = Platform.TELEGRAM,
    chat_id: str = "c1",
    user_id: str | None = "u1",
    user_name: str | None = None,
    chat_type: str = "dm",
    chat_name: str | None = None,
    thread_id: str | None = None,
    guild_id: str | None = None,
    role_authorized: bool = False,
) -> SessionSource:
    """Build a minimal-but-real SessionSource."""
    return SessionSource(
        platform=platform,
        chat_id=chat_id,
        user_id=user_id,
        user_name=user_name,
        chat_type=chat_type,
        chat_name=chat_name,
        thread_id=thread_id,
        guild_id=guild_id,
        role_authorized=role_authorized,
    )


def build_event(
    *,
    text: str = "hi",
    source: SessionSource | None = None,
    message_id: str = "1",
    message_type: MessageType = MessageType.TEXT,
    media_urls: list | None = None,
    media_types: list | None = None,
    reply_to_text: str | None = None,
    reply_to_message_id: str | None = None,
    channel_context: str | None = None,
    internal: bool = False,
    platform: Platform = Platform.TELEGRAM,
) -> MessageEvent:
    """Build a minimal-but-real MessageEvent. Auto-builds a source if none given."""
    if source is None:
        source = build_source(platform=platform)
    return MessageEvent(
        text=text,
        source=source,
        message_id=message_id,
        message_type=message_type,
        media_urls=media_urls or [],
        media_types=media_types or [],
        reply_to_text=reply_to_text,
        reply_to_message_id=reply_to_message_id,
        channel_context=channel_context,
        internal=internal,
    )


# ---------------------------------------------------------------------------
# FakeAgent + swap_run_agent — fake the 30-kwarg AIAgent via sys.modules swap.
# Lifted from test_session_hygiene.py:307 / test_run_progress_topics.py:125.
# The swap sidesteps needing to model AIAgent.__init__'s full signature.
# ---------------------------------------------------------------------------


class FakeAgent:
    """Minimal agent double. __init__ swallows all kwargs so the 30-kwarg
    AIAgent.__init__ signature doesn't need modeling.

    Captures the callbacks _run_agent wires (tool_progress_callback, etc.)
    and returns a configurable result from run_conversation matching the
    real contract: result["final_response"] (str) + result["messages"] (list).
    """

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.session_id = kwargs.get("session_id", "fake-session")
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.tools = kwargs.get("tools", [])
        self.valid_tool_names = set()
        self.enabled_toolsets = kwargs.get("enabled_toolsets")
        self.disabled_toolsets = kwargs.get("disabled_toolsets")
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tool_start_callback = kwargs.get("tool_start_callback")
        self.tool_complete_callback = kwargs.get("tool_complete_callback")
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self._final_response = kwargs.get("final_response", "fake agent response")
        self._messages = kwargs.get("messages", [])

    async def run_conversation(self, message, conversation_history=None, task_id=None):
        return {
            "final_response": self._final_response,
            "messages": self._messages,
            "api_calls": 1,
        }

    async def interrupt(self, message=None):
        return True

    async def get_activity_summary(self):
        return "fake activity summary"

    def shutdown_memory_provider(self):
        pass

    async def close(self):
        pass


def swap_run_agent(monkeypatch, agent_cls=FakeAgent):
    """Swap sys.modules["run_agent"] so ``from run_agent import AIAgent``
    resolves to ``agent_cls``. Returns the fake module for further customization.

    This is the established pattern from test_session_hygiene.py:307. The
    swap replaces the module globally for the test's duration (monkeypatch
    auto-undoes it).

    Usage::

        def test_something(monkeypatch):
            fake_mod = swap_run_agent(monkeypatch)
            # optionally: fake_mod.AIAgent = MyCustomAgent
            ...
    """
    fake_mod = types.ModuleType("run_agent")
    fake_mod.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_mod)
    return fake_mod


# ---------------------------------------------------------------------------
# minimal_config — the canonical minimal GatewayConfig seen in every
# runner test. GatewayConfig has all fields defaulted; this just sets the
# two tests commonly need: platforms + sessions_dir.
# ---------------------------------------------------------------------------


def minimal_config(
    *,
    platform: Platform = Platform.TELEGRAM,
    sessions_dir=None,
    enabled: bool = True,
) -> GatewayConfig:
    """Build a minimal valid GatewayConfig with one platform enabled."""
    return GatewayConfig(
        platforms={platform: PlatformConfig(enabled=enabled, token="***")},
        sessions_dir=sessions_dir,
    )
