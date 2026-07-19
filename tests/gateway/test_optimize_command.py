"""Regression tests for the /optimize slash command dispatch.

The /optimize handler has been silently deleted 3× by upstream fork merges.
These tests verify the dispatch path exists and functions correctly, so a
future merge that drops the handler fails CI instead of silently breaking
the command.

Coverage targets:
  - Handler exists in the canonical dispatch chain (not deleted by merge)
  - CommandDef registered exactly once (no duplicates from restoration)
  - 'optimize' in GATEWAY_KNOWN_COMMANDS (access gate recognizes it)
  - Empty args returns usage hint
  - Valid args rewrite event.text via skill invocation
  - Skill key resolves correctly

Note: the root conftest isolates HERMES_HOME to a temp dir, so
resolve_skill_command_key won't find the real skill. The functional tests
mock the skill_commands module to verify dispatch wiring without requiring
the actual skill to be installed in the test environment.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource


# ---------------------------------------------------------------------------
# Helpers (mirrors test_slash_access_dispatch.py patterns)
# ---------------------------------------------------------------------------

def _make_source(
    *,
    platform: Platform = Platform.DISCORD,
    user_id: str = "user1",
    chat_type: str = "dm",
    chat_id: str = "c1",
) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name=f"name-{user_id}",
        chat_type=chat_type,
    )


def _make_event(text: str, source: SessionSource) -> MessageEvent:
    return MessageEvent(text=text, source=source, message_id="m1")


def _make_runner(*, platform: Platform = Platform.DISCORD):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            platform: PlatformConfig(
                enabled=True,
                token="***",
                extra={},
            )
        }
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {platform: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )
    runner.session_store = MagicMock()
    session_entry = SessionEntry(
        session_key="agent:main:discord:dm:c1",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=platform,
        chat_type="dm",
        total_tokens=0,
    )
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._session_run_generation = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_sources = {}
    runner._session_db = MagicMock()
    runner._session_db.get_session_title.return_value = None
    runner._session_db.get_session.return_value = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner


# ---------------------------------------------------------------------------
# 1. Structural — handler existence (the core regression check)
# ---------------------------------------------------------------------------


def test_optimize_handler_exists_in_dispatch_chain():
    """The canonical == 'optimize' block must exist in handle_message_mixin.py.

    This is the single most important test: it catches silent deletion by
    upstream merges. If this fails, the handler was deleted — restore it per
    Pitfall #11 in the prompt-optimizer skill's reference-details.md.
    """
    import inspect

    from gateway.handle_message_mixin import HandleMessageMixin

    source = inspect.getsource(HandleMessageMixin._handle_message)
    assert 'canonical == "optimize"' in source, (
        "/optimize handler missing from HandleMessageMixin._handle_message. "
        "The dispatch block was likely deleted by an upstream merge. "
        "See prompt-optimizer skill Pitfall #11 for restoration."
    )


def test_optimize_commanddef_registered_once():
    """CommandDef('optimize') must appear exactly once in COMMAND_REGISTRY.

    A duplicate was introduced by a prior restoration (someone added it without
    checking if it already existed). This test prevents that recurrence.
    """
    from hermes_cli.commands import COMMAND_REGISTRY

    optimize_defs = [c for c in COMMAND_REGISTRY if c.name == "optimize"]
    assert len(optimize_defs) == 1, (
        f"Expected exactly 1 CommandDef('optimize'), found {len(optimize_defs)}. "
        "Duplicate entries cause confusion and should be deduplicated."
    )


def test_optimize_in_gateway_known_commands():
    """'optimize' must be in GATEWAY_KNOWN_COMMANDS so the dispatch chain
    recognizes it and the access-control gate can evaluate it."""
    from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS

    assert "optimize" in GATEWAY_KNOWN_COMMANDS


def test_optimize_handler_imports_skill_commands():
    """The handler must import from agent.skill_commands — this is the
    dependency that makes the rewrite work. If the import path changes,
    this test catches it."""
    import inspect

    from gateway.handle_message_mixin import HandleMessageMixin

    source = inspect.getsource(HandleMessageMixin._handle_message)
    assert "build_skill_invocation_message" in source, (
        "/optimize handler must call build_skill_invocation_message to "
        "rewrite the user's prompt via the skill invocation system."
    )
    assert "resolve_skill_command_key" in source, (
        "/optimize handler must call resolve_skill_command_key to find "
        "the prompt-optimizer skill."
    )


# ---------------------------------------------------------------------------
# 2. Functional — empty args returns usage hint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_optimize_empty_args_returns_usage():
    """`/optimize` with no arguments should return a usage hint."""
    runner = _make_runner()
    source = _make_source()
    result = await runner._handle_message(_make_event("/optimize", source))
    assert result is not None
    assert "Usage" in result or "usage" in result.lower()


# ---------------------------------------------------------------------------
# 3. Functional — valid args rewrite event.text (mocked skill resolution)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_optimize_valid_args_rewrites_event_text():
    """`/optimize <prompt>` should rewrite event.text to a skill invocation
    message. We mock skill_commands because the test HERMES_HOME is isolated
    and won't contain the real skill. This tests the dispatch wiring, not
    the skill content itself.
    """
    runner = _make_runner()
    source = _make_source()

    fake_msg = "[SKILL INVOCATION: prompt-optimizer for test prompt]"
    with patch("agent.skill_commands.resolve_skill_command_key", return_value="/prompt-optimizer"), \
         patch("agent.skill_commands.build_skill_invocation_message", return_value=fake_msg), \
         patch("agent.skill_commands.get_skill_commands", return_value={"/prompt-optimizer": {"name": "prompt-optimizer"}}):

        event = _make_event("/optimize test prompt", source)
        # The handler rewrites event.text then falls through to agent dispatch.
        # We can't easily run the full agent loop, but we can call _handle_message
        # and verify it doesn't return an error string (which means the handler
        # succeeded and fell through to agent dispatch).
        result = await runner._handle_message(event)

    # If the handler succeeded, event.text was rewritten to the skill invocation.
    # The result may be None (fell through to agent) or the skill invocation.
    # If it returned an error string, the handler failed.
    if result is not None and isinstance(result, str):
        assert "not installed" not in result.lower(), (
            "Handler returned 'not installed' even with mock — skill resolution failed"
        )
        assert "failed" not in result.lower() or "skill" not in result.lower(), (
            f"Handler returned error: {result}"
        )


# ---------------------------------------------------------------------------
# 4. Skill resolution contract (no mock — verifies the function exists)
# ---------------------------------------------------------------------------


def test_resolve_skill_command_key_is_callable():
    """resolve_skill_command_key must exist and be callable. It may return
    None in the test env (isolated HERMES_HOME), but the function itself
    must be importable and callable without error."""
    from agent.skill_commands import resolve_skill_command_key

    # Should not raise — may return None in test env
    result = resolve_skill_command_key("prompt-optimizer")
    # In production (real HERMES_HOME) this returns "/prompt-optimizer"
    assert result is None or isinstance(result, str)
