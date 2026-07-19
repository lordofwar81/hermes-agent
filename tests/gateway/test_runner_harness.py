"""Self-tests for the shared runner harness.

Catches harness regressions: if build_runner stops setting an attr that
downstream tests rely on, or build_event drops a required field, these
fail before any dependent test does.
"""

import asyncio

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource

from tests.gateway._runner_harness import (
    RecordingAdapter,
    build_event,
    build_runner,
    build_source,
    minimal_config,
    swap_run_agent,
    FakeAgent,
)


def test_build_runner_produces_usable_instance(tmp_path):
    runner = build_runner(tmp_path=tmp_path)
    # Core attrs every test relies on
    assert runner._running is True
    assert runner.adapters == {}
    assert runner._failed_platforms == {}
    assert runner._agent_cache == {}
    assert runner.config is not None
    # Methods resolve through MRO (the object.__new__ bypass must not
    # break class-level dispatch). After the god-file decomposition (R47-R57),
    # _handle_message lives on HandleMessageMixin — qualname no longer starts
    # with "Gateway", but MRO resolution is the real invariant.
    assert callable(runner._handle_message)
    assert callable(runner._session_key_for_source)
    assert type(runner)._handle_message is GatewayRunner._handle_message


def test_build_runner_accepts_overrides(tmp_path):
    custom_store = type("S", (), {"_entries": {}})()
    adapter = RecordingAdapter()
    runner = build_runner(
        tmp_path=tmp_path,
        adapters={Platform.TELEGRAM: adapter},
        session_store=custom_store,
    )
    assert runner.adapters[Platform.TELEGRAM] is adapter
    assert runner.session_store is custom_store


def test_build_runner_real_init_path(tmp_path):
    """real_init=True runs the actual __init__ (heavier but realistic)."""
    runner = build_runner(tmp_path=tmp_path, real_init=True)
    assert isinstance(runner.config, GatewayConfig)
    assert hasattr(runner, "session_store")


@pytest.mark.asyncio
async def test_recording_adapter_records(tmp_path):
    adapter = RecordingAdapter()
    assert await adapter.connect() is True
    result = await adapter.send("chat-1", "hello world", metadata={"k": "v"})
    assert result.success is True
    assert adapter.sent == ["hello world"]
    assert adapter.sent_calls == [("chat-1", "hello world", {"k": "v"})]
    await adapter.send_typing("chat-1", metadata={"t": 1})
    assert adapter.typing_calls == [("chat-1", {"t": 1})]


@pytest.mark.asyncio
async def test_recording_adapter_fatal_connect(tmp_path):
    """Parametrized fatal-error path mirrors test_runner_startup_failures."""
    adapter = RecordingAdapter(
        connect_result=False,
        fatal_error_code="test_error",
        fatal_error_message="boom",
        fatal_error_retryable=True,
    )
    assert await adapter.connect() is False
    assert adapter.has_fatal_error is True
    assert adapter.fatal_error_retryable is True
    assert adapter.fatal_error_code == "test_error"


def test_build_source_minimal():
    src = build_source()
    assert src.platform == Platform.TELEGRAM
    assert src.chat_id == "c1"
    assert isinstance(src, SessionSource)


def test_build_source_custom():
    src = build_source(
        platform=Platform.DISCORD,
        chat_id="d1",
        user_name="alice",
        thread_id="t1",
    )
    assert src.platform == Platform.DISCORD
    assert src.user_name == "alice"
    assert src.thread_id == "t1"


def test_build_event_minimal():
    ev = build_event()
    assert ev.text == "hi"
    assert ev.message_type == MessageType.TEXT
    assert isinstance(ev, MessageEvent)
    assert ev.source is not None  # auto-built
    assert ev.source.platform == Platform.TELEGRAM


def test_build_event_custom_with_source():
    src = build_source(platform=Platform.SLACK)
    ev = build_event(text="/help", source=src, message_id="m99")
    assert ev.text == "/help"
    assert ev.source is src
    assert ev.message_id == "m99"
    assert ev.source.platform == Platform.SLACK


def test_minimal_config_default():
    cfg = minimal_config()
    assert isinstance(cfg, GatewayConfig)
    tg = cfg.platforms[Platform.TELEGRAM]
    assert tg.enabled is True


def test_minimal_config_custom(tmp_path):
    cfg = minimal_config(platform=Platform.DISCORD, sessions_dir=tmp_path / "s")
    assert cfg.platforms[Platform.DISCORD].enabled is True
    assert Platform.TELEGRAM not in cfg.platforms


def test_swap_run_agent(monkeypatch):
    fake_mod = swap_run_agent(monkeypatch)
    # The swap must make `from run_agent import AIAgent` resolve to FakeAgent
    import run_agent  # noqa: F401  (exercises the swap)
    assert run_agent.AIAgent is FakeAgent
    # Custom agent class
    class MyAgent(FakeAgent):
        pass
    fake_mod2 = swap_run_agent(monkeypatch, agent_cls=MyAgent)
    import run_agent as ra2  # noqa: F401
    assert ra2.AIAgent is MyAgent


@pytest.mark.asyncio
async def test_fake_agent_run_conversation_contract():
    """FakeAgent.run_conversation must return final_response + messages keys
    matching what gateway/run.py's _run_agent consumes."""
    agent = FakeAgent(final_response="hello back", messages=[{"role": "user", "content": "hi"}])
    result = await agent.run_conversation("hi")
    assert "final_response" in result
    assert "messages" in result
    assert result["final_response"] == "hello back"
    assert isinstance(result["messages"], list)


def test_fake_agent_swallows_kwargs():
    """__init__ must accept arbitrary kwargs — the real AIAgent takes ~30."""
    agent = FakeAgent(
        model="m", provider="p", api_key="k", base_url="u",
        session_id="s1", max_iterations=5, quiet_mode=True,
        some_future_kwarg_we_havent_added_yet="ok",
    )
    assert agent.session_id == "s1"
    assert agent.init_kwargs["some_future_kwarg_we_havent_added_yet"] == "ok"
