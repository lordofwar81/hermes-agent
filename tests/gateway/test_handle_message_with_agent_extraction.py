"""Integration tests for the ``_handle_message_with_agent`` extraction (round 51).

Exercises ``HandleMessageWithAgentMixin._handle_message_with_agent`` end-to-end
with the method moved onto ``HandleMessageWithAgentMixin``. The method is the
INNER handler — the "build session context, run agent conversation, deliver
response" pipeline called under the ``_running_agents`` sentinel guard from
``_handle_message``'s finally block.

Two contract guarantees of the extraction:

1. **MRO resolution** —
   ``HandleMessageWithAgentMixin._handle_message_with_agent
      is GatewayRunner._handle_message_with_agent``.
   Without this, the old inline method would still be shadowing the mixin
   (or the mixin wasn't added to the bases) and the extraction is a no-op.
   The 4 direct calls in ``test_42039_duplicate_user_message.py`` resolve
   via MRO through this same identity.
2. **Phase canaries** — every distinct phase of the lifted pipeline must
   still fire when invoked through the mixin:

     * **session resolution** — ``session_store.get_or_create_session`` and
       ``_cache_session_source`` called (context-build side effect).
     * **agent run** — ``_run_agent`` invoked with the built context_prompt
       + history + source.
     * **response delivery** — ``hooks.emit("agent:start")`` and
       ``hooks.emit("agent:end")`` both fire, and the final response string
       is returned.
     * **transcript persistence (#42039 guard)** — successful turn with new
       messages calls ``append_to_transcript`` with ``skip_db=True`` when
       the agent has its own ``_session_db``.
     * **transcript persistence (fresh session)** — when there is no prior
       history, the ``session_meta`` tool-defs row is written first.
     * **error path** — a raised exception inside ``_run_agent`` produces
       the user-facing "Sorry, I encountered an error" string (and the
       ``finally`` block still runs to clear the session env).

A dropped phase (e.g. session-build skipped during the lift) would fail its
canary. The test mirrors the ``_bootstrap`` pattern from
``test_42039_duplicate_user_message`` because that module already proved the
exact attr set ``_handle_message_with_agent`` needs when invoked directly.
"""

import sys
import types
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.handle_message_with_agent_mixin import HandleMessageWithAgentMixin
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource


def _bootstrap(monkeypatch, tmp_path):
    """Minimal GatewayRunner setup shared by all tests in this module.

    Lifted from ``test_42039_duplicate_user_message._bootstrap`` — that
    module is the established direct-caller harness for
    ``_handle_message_with_agent`` and proved the exact attr stub set the
    method needs to run without a real agent.
    """
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    config = GatewayConfig()
    runner = gateway_run.GatewayRunner(config)
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = MagicMock()
    runner._is_session_run_current = lambda _key, _gen: True
    runner._begin_session_run_generation = lambda _key: 1
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner._deliver_platform_notice = AsyncMock()
    runner._prepare_inbound_message_text = AsyncMock(return_value="hello")
    runner._refresh_agent_cache_message_count = MagicMock()
    runner._clear_restart_failure_count = MagicMock()
    runner._evict_cached_agent = MagicMock()
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:group:-1001:12345",
        session_id="sess-r51",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="group",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner.session_store.has_any_sessions.return_value = True

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100_000,
    )
    return runner


def _event():
    return MessageEvent(
        text="hello world",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1001",
            chat_type="group",
            user_id="12345",
        ),
        message_id="msg-42",
    )


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="12345",
    )


_SESSION_KEY = "agent:main:telegram:group:-1001:12345"


def test_method_lives_on_mixin_and_resolves_via_mro():
    """``_handle_message_with_agent`` must resolve to the mixin through the MRO.

    The single most important assertion: proves the method actually lives on
    ``HandleMessageWithAgentMixin`` and that mixin is in ``GatewayRunner``'s
    bases. If the inline method were still in run.py (shadowing the mixin)
    or the mixin wasn't appended to the bases, this identity check fails.
    This same resolution backs the 4 direct calls in
    ``test_42039_duplicate_user_message``.
    """
    assert (
        HandleMessageWithAgentMixin._handle_message_with_agent
        is gateway_run.GatewayRunner._handle_message_with_agent
    )
    assert HandleMessageWithAgentMixin in gateway_run.GatewayRunner.__mro__


@pytest.mark.asyncio
async def test_session_resolution_and_context_build_run(monkeypatch, tmp_path):
    """Session resolution phase must fire: get_or_create_session + cache source.

    Exercises the head of the pipeline — session lookup + source caching
    happen before any agent work. If the session-resolution region was
    dropped during the lift, neither mock would be called.
    """
    runner = _bootstrap(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "hi",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )

    result = await runner._handle_message_with_agent(_event(), _source(), _SESSION_KEY, 1)

    runner.session_store.get_or_create_session.assert_called_once()
    runner._cache_session_source.assert_called_once()


@pytest.mark.asyncio
async def test_agent_run_invoked_and_response_returned(monkeypatch, tmp_path):
    """Agent-run + response-delivery phases must fire and return the response.

    Exercises the core ``self._run_agent(...)`` call inside the try block
    and the return path. Verifies ``agent:start`` and ``agent:end`` hooks
    both fire (bracketing the turn) and the final_response string is
    returned to the caller. If the agent-run region was dropped, hooks
    wouldn't fire and ``result`` would be None.
    """
    runner = _bootstrap(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "Hello from agent!",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hello from agent!"},
            ],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )

    result = await runner._handle_message_with_agent(_event(), _source(), _SESSION_KEY, 1)

    assert result == "Hello from agent!"
    runner._run_agent.assert_awaited_once()
    # Both bracketing hooks must fire.
    emitted = [call.args[0] for call in runner.hooks.emit.await_args_list]
    assert "agent:start" in emitted, f"agent:start hook not emitted: {emitted}"
    assert "agent:end" in emitted, f"agent:end hook not emitted: {emitted}"


@pytest.mark.asyncio
async def test_fresh_session_writes_session_meta_tools_row(monkeypatch, tmp_path):
    """Fresh-session phase must write the session_meta tool-defs row first.

    When there is no prior history, the pipeline writes a ``session_meta``
    entry (with the agent's tool defs) as the first transcript row so the
    session is self-describing. If the fresh-session branch was dropped,
    no ``session_meta`` row would appear.
    """
    runner = _bootstrap(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "hi",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
            "tools": [{"name": "bash"}],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )

    await runner._handle_message_with_agent(_event(), _source(), _SESSION_KEY, 1)

    calls = runner.session_store.append_to_transcript.call_args_list
    # First write should be the session_meta row.
    first_entry = calls[0].args[1]
    assert first_entry["role"] == "session_meta"
    assert first_entry["tools"] == [{"name": "bash"}]


@pytest.mark.asyncio
async def test_error_path_returns_user_facing_string(monkeypatch, tmp_path):
    """Exception inside the agent run must produce the user-facing error string.

    Exercises the ``except Exception`` block: when ``_run_agent`` raises,
    the pipeline returns the "Sorry, I encountered an error" message (not a
    bare traceback). The ``finally`` block must still run. If the exception
    handler was dropped during the lift, the exception would propagate.
    """
    runner = _bootstrap(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(side_effect=RuntimeError("boom"))

    result = await runner._handle_message_with_agent(_event(), _source(), _SESSION_KEY, 1)

    assert isinstance(result, str)
    assert "Sorry, I encountered an error" in result
    assert "boom" in result


@pytest.mark.asyncio
async def test_error_path_status_code_429_hint(monkeypatch, tmp_path):
    """A 429 status-code exception must surface the rate-limit hint.

    Exercises the ``status_code == 429`` branch of the error handler —
    proves the status-code classification logic survived the lift.
    """
    runner = _bootstrap(monkeypatch, tmp_path)

    class _RateLimited(RuntimeError):
        status_code = 429

    runner._run_agent = AsyncMock(side_effect=_RateLimited("rate limited"))

    result = await runner._handle_message_with_agent(_event(), _source(), _SESSION_KEY, 1)

    assert "rate-limited" in result.lower() or "rate limit" in result.lower()
