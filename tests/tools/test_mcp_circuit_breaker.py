"""Tests for MCP tool-handler circuit-breaker recovery.

The circuit breaker in ``tools/mcp_tool.py`` is intended to short-circuit
calls to an MCP server that has failed ``_CIRCUIT_BREAKER_THRESHOLD``
consecutive times, then *transition back to a usable state* once the
server has had time to recover (or an explicit reconnect succeeds).

The original implementation only had two states — closed and open — with
no mechanism to transition back to closed, so a tripped breaker stayed
tripped for the lifetime of the process. These tests lock in the
half-open / cooldown / reconnect-resets-breaker behavior that fixes
that.
"""
import asyncio
import json
from unittest.mock import MagicMock

import pytest


pytest.importorskip("mcp.client.auth.oauth2")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _install_stub_server(mcp_tool_module, name: str, call_tool_impl,
                        liveness_ok: bool = True):
    """Install a fake MCP server in the module's registry.

    ``call_tool_impl`` is an async function stored at ``session.call_tool``
    (it's what the tool handler invokes).

    ``liveness_ok`` controls whether the session mock responds to
    ``initialize()`` — True (default) makes the liveness probe succeed,
    False makes it raise so the probe fails.
    """
    server = MagicMock()
    server.name = name
    session = MagicMock()
    session.call_tool = call_tool_impl

    if liveness_ok:
        async def _init_ok(*a, **kw):
            return MagicMock()
        session.initialize = _init_ok
    else:
        async def _init_fails(*a, **kw):
            raise ConnectionError("probe fails")
        session.initialize = _init_fails

    server.session = session
    server._reconnect_event = MagicMock()
    server._ready = MagicMock()
    server._ready.is_set.return_value = True

    # Use a real asyncio.Lock so ``async with server._rpc_lock:`` works
    # correctly with _run_on_mcp_loop (MagicMock doesn't support the
    # async context manager descriptor protocol reliably).
    server._rpc_lock = asyncio.Lock()

    mcp_tool_module._servers[name] = server
    mcp_tool_module._server_error_counts.pop(name, None)
    if hasattr(mcp_tool_module, "_server_breaker_opened_at"):
        mcp_tool_module._server_breaker_opened_at.pop(name, None)
    return server


def _cleanup(mcp_tool_module, name: str) -> None:
    mcp_tool_module._servers.pop(name, None)
    mcp_tool_module._server_error_counts.pop(name, None)
    if hasattr(mcp_tool_module, "_server_breaker_opened_at"):
        mcp_tool_module._server_breaker_opened_at.pop(name, None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_circuit_breaker_half_opens_after_cooldown(monkeypatch, tmp_path):
    """After a tripped breaker's cooldown elapses, the *next* call must
    actually execute against the session (half-open probe). When the
    probe succeeds, the breaker resets to fully closed.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    call_count = {"n": 0}

    async def _call_tool_success(*a, **kw):
        call_count["n"] += 1
        result = MagicMock()
        result.isError = False
        block = MagicMock()
        block.text = "ok"
        result.content = [block]
        result.structuredContent = None
        return result

    _install_stub_server(mcp_tool, "srv", _call_tool_success, liveness_ok=True)
    mcp_tool._ensure_mcp_loop()

    try:
        # Trip the breaker by setting the count at/above threshold and
        # stamping the open-time to "now".
        mcp_tool._server_error_counts["srv"] = mcp_tool._CIRCUIT_BREAKER_THRESHOLD
        fake_now = [1000.0]

        def _fake_monotonic():
            return fake_now[0]

        monkeypatch.setattr(mcp_tool.time, "monotonic", _fake_monotonic)
        if hasattr(mcp_tool, "_server_breaker_opened_at"):
            mcp_tool._server_breaker_opened_at["srv"] = fake_now[0]
        cooldown = getattr(mcp_tool, "_CIRCUIT_BREAKER_COOLDOWN_SEC", 60.0)

        handler = _make_tool_handler("srv", "tool1", 10.0)

        # Before cooldown: must short-circuit (no session call).
        result = handler({})
        parsed = json.loads(result)
        assert "error" in parsed, parsed
        assert "unreachable" in parsed["error"].lower()
        assert call_count["n"] == 0, (
            "breaker should short-circuit before cooldown elapses"
        )

        # Advance past cooldown → liveness probe fires first.
        # With liveness_ok=True, the probe succeeds, breaker resets,
        # then the actual tool call goes through.
        fake_now[0] += cooldown + 1.0

        result = handler({})
        parsed = json.loads(result)
        assert parsed.get("result") == "ok", parsed
        assert call_count["n"] == 1, "actual tool call should proceed after probe success"

        # On success the breaker must be fully closed (count reset to 0).
        assert mcp_tool._server_error_counts.get("srv", 0) == 0
    finally:
        _cleanup(mcp_tool, "srv")


def test_circuit_breaker_reopens_on_probe_failure(monkeypatch, tmp_path):
    """If the half-open probe fails, the breaker must re-arm the
    cooldown (not let every subsequent call through).
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    call_count = {"n": 0}

    async def _call_tool_fails(*a, **kw):
        call_count["n"] += 1
        raise RuntimeError("still broken")

    _install_stub_server(mcp_tool, "srv", _call_tool_fails, liveness_ok=False)
    mcp_tool._ensure_mcp_loop()

    try:
        mcp_tool._server_error_counts["srv"] = mcp_tool._CIRCUIT_BREAKER_THRESHOLD
        fake_now = [1000.0]

        def _fake_monotonic():
            return fake_now[0]

        monkeypatch.setattr(mcp_tool.time, "monotonic", _fake_monotonic)
        if hasattr(mcp_tool, "_server_breaker_opened_at"):
            mcp_tool._server_breaker_opened_at["srv"] = fake_now[0]
        cooldown = getattr(mcp_tool, "_CIRCUIT_BREAKER_COOLDOWN_SEC", 60.0)

        handler = _make_tool_handler("srv", "tool1", 10.0)

        # Advance past cooldown, run liveness probe — it fails because
        # liveness_ok=False. The breaker re-arms the cooldown and
        # returns an error — the actual tool call is never attempted.
        fake_now[0] += cooldown + 1.0
        result = handler({})
        parsed = json.loads(result)
        assert "error" in parsed
        assert "liveness probe" in parsed["error"].lower()
        assert call_count["n"] == 0, (
            "liveness probe failure should prevent the actual tool call"
        )

        # Another immediate call should still short-circuit (cooldown
        # was re-armed by the failed probe).
        result = handler({})
        parsed = json.loads(result)
        assert "unreachable" in parsed.get("error", "").lower()
        assert call_count["n"] == 0, (
            "breaker should block calls after probe failure re-armed cooldown"
        )
    finally:
        _cleanup(mcp_tool, "srv")


def test_circuit_breaker_cleared_on_reconnect(monkeypatch, tmp_path):
    """When the auth-recovery path successfully reconnects the server,
    the breaker should be cleared so subsequent calls aren't gated on a
    stale failure count — even if the post-reconnect retry itself fails.

    This locks in the fix-#2 contract: a successful reconnect is
    sufficient evidence that the server is viable again. Under the old
    implementation, reset only happened on retry *success*, so a
    reconnect+retry-failure left the counter pinned above threshold
    forever.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_oauth_manager import get_manager, reset_manager_for_tests
    from mcp.client.auth import OAuthFlowError

    reset_manager_for_tests()

    async def _call_tool_unused(*a, **kw):  # pragma: no cover
        raise AssertionError("session.call_tool should not be reached in this test")

    _install_stub_server(mcp_tool, "srv", _call_tool_unused)
    mcp_tool._ensure_mcp_loop()

    # Open the breaker well above threshold, with a recent open-time so
    # it would short-circuit everything without a reset.
    mcp_tool._server_error_counts["srv"] = mcp_tool._CIRCUIT_BREAKER_THRESHOLD + 2
    if hasattr(mcp_tool, "_server_breaker_opened_at"):
        import time as _time
        mcp_tool._server_breaker_opened_at["srv"] = _time.monotonic()

    # Force handle_401 to claim recovery succeeded.
    mgr = get_manager()

    async def _h401(name, token=None):
        return True

    monkeypatch.setattr(mgr, "handle_401", _h401)

    try:
        # Retry fails *after* the successful reconnect. Under the old
        # implementation this bumps an already-tripped counter even
        # higher. Under fix #2 the reset happens on successful
        # reconnect, and the post-retry bump only raises the fresh
        # count to 1 — still below threshold.
        def _retry_call():
            raise OAuthFlowError("still failing post-reconnect")

        result = mcp_tool._handle_auth_error_and_retry(
            "srv",
            OAuthFlowError("initial"),
            _retry_call,
            "tools/call test",
        )
        # The call as a whole still surfaces needs_reauth because the
        # retry itself didn't succeed, but the breaker state must
        # reflect the successful reconnect.
        assert result is not None
        parsed = json.loads(result)
        assert parsed.get("needs_reauth") is True, parsed

        # Post-reconnect count was reset to 0, then the failing retry
        # bumped it to exactly 1 — well below threshold.
        count = mcp_tool._server_error_counts.get("srv", 0)
        assert count < mcp_tool._CIRCUIT_BREAKER_THRESHOLD, (
            f"successful reconnect must reset the breaker below threshold; "
            f"got count={count}, threshold={mcp_tool._CIRCUIT_BREAKER_THRESHOLD}"
        )
    finally:
        _cleanup(mcp_tool, "srv")
