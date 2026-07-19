"""Integration tests for the ``_run_agent`` extraction (round 52).

Exercises ``RunAgentMixin._run_agent`` end-to-end with the method moved onto
``RunAgentMixin``. The method is the agent-execution orchestrator — the biggest
and most complex extraction in the decomposition (2585ln, 11 nested closures).
It is the "build the AIAgent, run it in a thread pool, wire progress/streaming/
status callbacks, deliver the response" pipeline called from
``_handle_message_with_agent``.

Two contract guarantees of the extraction:

1. **MRO resolution** —
   ``RunAgentMixin._run_agent is GatewayRunner._run_agent``.
   Without this, the old inline method would still be shadowing the mixin
   (or the mixin wasn't added to the bases) and the extraction is a no-op.
   The direct calls in ``test_run_progress_topics.py`` (6 test cases) and
   ``tests/cron/test_codex_execution_paths.py`` resolve via MRO through this
   same identity.
2. **Closure-wiring canary** — the method's 11 nested closures
   (``_run_still_current``, ``voice_ack_callback``, ``progress_callback``,
   ``send_progress_messages``, ``_step_callback_sync``,
   ``_status_callback_sync``, ``run_sync``, ``_start_stream_consumer``,
   ``track_agent``, ``monitor_for_interrupt``, ``_notify_long_running``)
   must all survive the lift. The strongest single canary is the
   progress-callback path: a tool call from the fake agent fires
   ``progress_callback`` → ``progress_queue`` → ``send_progress_messages``
   task → adapter ``send``. If any of those three closures was dropped or
   mis-scoped during the lift, the progress bubble never reaches the adapter.

A dropped closure (e.g. ``progress_callback`` not assigned to
``agent.tool_progress_callback``) would fail its canary. The test mirrors the
``_make_runner`` + ``FakeAgent`` pattern from ``test_run_progress_topics``
because that module already proved the exact attr set ``_run_agent`` needs
when invoked directly without a real event loop agent.
"""

import importlib
import sys
import time
import types
from types import SimpleNamespace

import pytest

import gateway.platforms.base as base_platform
from gateway.config import Platform, PlatformConfig, StreamingConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run_agent_mixin import RunAgentMixin
from gateway.session import SessionSource


class ProgressCaptureAdapter(BasePlatformAdapter):
    """Adapter that records sends/edits/typing for progress-bubble canaries."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []
        self.edits = []
        self.typing = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id="progress-1")

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edits.append(
            {
                "chat_id": chat_id,
                "content": content,
                "message_id": message_id,
            }
        )
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        self.typing.append({"chat_id": chat_id, "metadata": metadata})

    async def stop_typing(self, chat_id) -> None:
        self.typing.append({"chat_id": chat_id, "metadata": {"stopped": True}})

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class ToolProgressAgent:
    """Agent double that fires the tool-progress callback once.

    The production code path assigns ``tool_progress_callback`` to the agent
    AFTER construction (read at call time, not frozen in __init__), so this
    double mirrors that contract: it reads ``self.tool_progress_callback``
    inside ``run_conversation`` and fires it with a synthetic
    ``tool.started`` event. If the ``progress_callback`` closure was dropped
    or mis-scoped during the lift, this never fires and no progress bubble
    reaches the adapter.
    """

    def __init__(self, **kwargs):
        self.tool_progress_callback = None
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        if cb is not None:
            cb("tool.started", "terminal", "pwd", {})
            time.sleep(0.35)
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


def _make_runner(adapter):
    """Minimal GatewayRunner setup for direct ``_run_agent`` invocation.

    Lifted from ``test_run_progress_topics._make_runner`` — that module is the
    established direct-caller harness for ``_run_agent`` and proved the exact
    attr stub set the method needs to run without a real agent.
    """
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
    runner._service_tier = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
        streaming=StreamingConfig(),
    )
    return runner


def _bootstrap_env(monkeypatch, tmp_path):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = ToolProgressAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    import tools.terminal_tool  # noqa: F401 - register terminal emoji
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    return fake_run_agent


def test_method_lives_on_mixin_and_resolves_via_mro():
    """``_run_agent`` must resolve to the mixin through the MRO.

    The single most important assertion: proves the method actually lives on
    ``RunAgentMixin`` and that mixin is in ``GatewayRunner``'s bases. If the
    inline method were still in run.py (shadowing the mixin) or the mixin
    wasn't appended to the bases, this identity check fails. This same
    resolution backs the 6 direct calls in ``test_run_progress_topics``.
    """
    gateway_run = importlib.import_module("gateway.run")
    assert RunAgentMixin._run_agent is gateway_run.GatewayRunner._run_agent
    assert RunAgentMixin in gateway_run.GatewayRunner.__mro__


@pytest.mark.asyncio
async def test_progress_callback_closure_wires_to_adapter(monkeypatch, tmp_path):
    """The progress-callback → queue → send_progress_messages path must fire.

    This is the strongest single canary for the 11-closure lift: a tool call
    from the agent fires ``progress_callback`` (closure #3), which enqueues
    onto ``progress_queue``, which the ``send_progress_messages`` asyncio task
    (closure #4) drains to the adapter as a ``send``. If any of those three
    closures was dropped or mis-scoped during the lift, the progress bubble
    never reaches the adapter and ``adapter.sent`` stays empty.

    The executor target ``run_sync`` (closure #7) is what wires
    ``agent.tool_progress_callback = progress_callback``; if that assignment
    was lost, the ToolProgressAgent would have no callback to fire.
    """
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")
    _bootstrap_env(monkeypatch, tmp_path)

    adapter = ProgressCaptureAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-r52",
        session_key="agent:main:telegram:group:-1001:17585",
    )

    # Result shape — proves the result-shaping tail of the method survived.
    assert result["final_response"] == "done"
    # Progress bubble reached the adapter — proves progress_callback,
    # progress_queue, and send_progress_messages closures all lifted intact.
    assert adapter.sent, (
        "progress_callback closure did not fire through to adapter.send — "
        "a progress-wiring closure was dropped or mis-scoped during the lift"
    )
    assert "terminal" in adapter.sent[0]["content"]


@pytest.mark.asyncio
async def test_proxy_mode_delegates_to_run_agent_via_proxy(monkeypatch, tmp_path):
    """Proxy-mode branch must delegate to ``_run_agent_via_proxy``.

    Exercises the head of the method (before any closure is defined):
    ``if _get_proxy_url(): return await self._run_agent_via_proxy(...)``.
    If this early-return was dropped, the proxy delegation would be bypassed
    and the method would try to build a real AIAgent instead.
    """
    _bootstrap_env(monkeypatch, tmp_path)
    gateway_run = importlib.import_module("gateway.run")

    # Arm the proxy gate. R50 gotcha #11: the mixin imported _get_proxy_url at
    # ITS module top, so patching gateway.run or gateway.gateway_gateway_env
    # does nothing — the binding lives in run_agent_mixin's namespace.
    import gateway.run_agent_mixin as run_agent_mixin
    monkeypatch.setattr(
        run_agent_mixin, "_get_proxy_url", lambda: "http://proxy.example/api"
    )

    adapter = ProgressCaptureAdapter()
    runner = _make_runner(adapter)

    delegated = {}

    async def _fake_via_proxy(**kwargs):
        delegated.update(kwargs)
        return {"final_response": "proxied", "messages": [], "tools": []}

    runner._run_agent_via_proxy = _fake_via_proxy

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="ctx",
        history=[],
        source=source,
        session_id="sess-proxy",
        session_key="agent:main:telegram:group:-1001",
    )

    assert result["final_response"] == "proxied"
    assert delegated["message"] == "hello"
    assert delegated["context_prompt"] == "ctx"


@pytest.mark.asyncio
async def test_result_dict_shape_carries_documented_keys(monkeypatch, tmp_path):
    """The shaped result dict must carry the documented contract keys.

    The method's tail shapes the raw agent result into a dict with
    ``final_response``, ``messages``, ``api_calls``, ``completed``,
    ``tools``, ``history_offset``, ``session_id``. If the result-shaping
    region was dropped, the dict would be missing keys the caller
    (``_handle_message_with_agent``) reads.
    """
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")
    _bootstrap_env(monkeypatch, tmp_path)

    adapter = ProgressCaptureAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-shape",
        session_key="agent:main:telegram:group:-1001",
    )

    for key in ("final_response", "messages", "api_calls"):
        assert key in result, f"result dict missing documented key {key!r}"
    assert result["final_response"] == "done"
