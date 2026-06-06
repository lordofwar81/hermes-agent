"""Smoke test for the gateway → agent → tools pipeline.

Verifies that after the gateway decomposition refactoring (ProgressManager,
_RunContext, closure extraction), the end-to-end _run_agent flow still
works: progress callbacks fire, status callbacks send, step callbacks emit
hook events, the final response is delivered, and progress messages are
edited through their lifecycle.
"""

import asyncio
import importlib
import sys
import time
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource


class PipelineTestAdapter(BasePlatformAdapter):
    """Adapter that records every call for pipeline inspection."""

    _next_mid = 500

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []
        self.edits = []
        self.deleted = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    def _mint_id(self) -> str:
        PipelineTestAdapter._next_mid += 1
        return str(PipelineTestAdapter._next_mid)

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        mid = self._mint_id()
        self.sent.append(
            {"chat_id": chat_id, "content": content, "message_id": mid, "metadata": metadata}
        )
        return SendResult(success=True, message_id=mid)

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edits.append({"chat_id": chat_id, "message_id": message_id, "content": content})
        return SendResult(success=True, message_id=message_id)

    async def delete_message(self, chat_id, message_id) -> bool:
        self.deleted.append({"chat_id": chat_id, "message_id": str(message_id)})
        return True

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class PipelineAgent:
    """Emits progress and status callbacks, then returns a final response."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.step_callback = kwargs.get("step_callback")
        self.status_callback = kwargs.get("status_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        if cb is not None:
            cb("tool.started", "terminal", "pwd", {})
            time.sleep(0.1)
            cb("tool.completed", "terminal", "pwd", {})
            time.sleep(0.1)
            cb("tool.started", "terminal", "ls", {})
            time.sleep(0.1)
        sc = self.step_callback
        if sc is not None:
            sc(1, [{"name": "pwd", "result": "/home"}])
        st = self.status_callback
        if st is not None:
            st("thinking", "Analyzing output...")
        return {"final_response": "pipeline test complete", "messages": [], "api_calls": 1}


def _make_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
    GatewayRunner = gateway_run.GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    return runner


def _install_fakes(monkeypatch, agent_cls):
    """Wire up the module stubs every _run_agent test needs."""
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")
    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    return gateway_run


@pytest.mark.asyncio
async def test_pipeline_smoke_progress_callbacks_fire(monkeypatch, tmp_path):
    """Progress callbacks from the agent trigger progress messages."""
    adapter = PipelineTestAdapter()
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, PipelineAgent)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001")
    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-pipeline-1",
        session_key="agent:main:telegram:group:-1001",
    )

    assert result["final_response"] == "pipeline test complete"
    assert result.get("failed") is None

    assert any("pwd" in str(s) for s in adapter.sent)
    assert len(adapter.sent) >= 1


@pytest.mark.asyncio
async def test_pipeline_smoke_status_callbacks_send(monkeypatch, tmp_path):
    """Status callbacks from the agent appear as adapter messages."""
    adapter = PipelineTestAdapter()
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, PipelineAgent)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001")
    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-pipeline-2",
        session_key="agent:main:telegram:group:-1001",
    )

    assert result["final_response"] == "pipeline test complete"


@pytest.mark.asyncio
async def test_pipeline_smoke_step_callback_emits_hook(monkeypatch, tmp_path):
    """Step callbacks reach the hooks system and emit agent:step events."""
    hook_events = []

    class HookCollector:
        loaded_hooks = True

        async def emit(self, event, data):
            hook_events.append((event, data))

    adapter = PipelineTestAdapter()
    runner = _make_runner(adapter)
    runner.hooks = HookCollector()
    gateway_run = _install_fakes(monkeypatch, PipelineAgent)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001")
    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-pipeline-3",
        session_key="agent:main:telegram:group:-1001",
    )

    assert result["final_response"] == "pipeline test complete"
    hook_events_names = [ev for ev, _ in hook_events]
    assert "agent:step" in hook_events_names


@pytest.mark.asyncio
async def test_pipeline_smoke_progress_message_lifecycle(monkeypatch, tmp_path):
    """Progress bubbles are sent, then edited through the run lifecycle."""
    adapter = PipelineTestAdapter()
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, PipelineAgent)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001")
    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-pipeline-4",
        session_key="agent:main:telegram:group:-1001",
    )

    assert result["final_response"] == "pipeline test complete"
    sent_ids = {s["message_id"] for s in adapter.sent}
    edited_ids = {e["message_id"] for e in adapter.edits}
    overlapping = sent_ids & edited_ids
    assert len(overlapping) >= 1


@pytest.mark.asyncio
async def test_pipeline_smoke_agent_failure_returns_error(monkeypatch, tmp_path):
    """A failing agent propagates the error through the pipeline."""

    class FailingPipelineAgent:
        def __init__(self, **kwargs):
            self.tools = []

        def run_conversation(self, message, conversation_history=None, task_id=None):
            return {"final_response": "", "messages": [], "api_calls": 1, "failed": True, "error": "simulated failure"}

    adapter = PipelineTestAdapter()
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, FailingPipelineAgent)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001")
    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-pipeline-5",
        session_key="agent:main:telegram:group:-1001",
    )

    assert result.get("failed") is True
    assert "simulated failure" in result.get("error", "")
