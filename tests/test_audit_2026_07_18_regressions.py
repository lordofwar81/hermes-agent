"""Regression tests for the 2026-07-18 Hermes audit fixes.

These close the test-coverage gaps that allowed the audited defects to persist:
- H-2: EmbedClient alive-caching TTL (was caching False forever)
- D1: repetition guard tripping on successful identical tool calls
- H-3: trust-score inflation (facts locking at 1.0)
- H-1: reranker sticky-disable (no recovery after transient failure)
- D2: BudgetTracker bypassing HERMES_HOME

Run: venv/bin/python -m pytest tests/test_audit_2026_07_18_regressions.py -v
"""
import sys
import os
import time
import hashlib
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─── H-2: EmbedClient alive TTL ───────────────────────────────────────────────

def test_embed_client_alive_false_expires_after_ttl():
    """A cached alive=False must expire after the TTL and re-probe (audit H-2)."""
    from plugins.memory.holographic.store import EmbedClient
    c = EmbedClient()
    # simulate a recent failure
    c._alive = False
    c._alive_false_ts = time.time()  # just now
    assert c.alive is False, "within TTL, should stay False"
    # simulate TTL expiry
    c._alive = False
    c._alive_false_ts = time.time() - c._alive_ttl_seconds - 1
    # alive should re-probe (the real embed server is up, so this returns True)
    assert c.alive is True, "after TTL expiry, should re-probe and recover"


def test_embed_client_alive_true_stays_cached():
    """A cached alive=True should stay True (no needless re-probing)."""
    from plugins.memory.holographic.store import EmbedClient
    c = EmbedClient()
    c._alive = True
    assert c.alive is True


# ─── D1: repetition guard respects success/failure ────────────────────────────

def test_repetition_guard_does_not_break_on_successful_reads():
    """3 successful identical tool calls must NOT trip the guard (audit D1)."""
    # This replicates the guard logic from conversation_loop.py post-fix.
    # We test the decision function directly.
    from agent.display import _detect_tool_failure

    class FakeToolCall:
        def __init__(self, name, args):
            self.function = types.SimpleNamespace(name=name, arguments=args)

    streak = []
    tool_call = FakeToolCall("read_file", '{"path": "/etc/hosts"}')
    sig = (tool_call.function.name,
           hashlib.md5(tool_call.function.arguments.encode()).hexdigest()[:8])

    # 3 successful results
    success_result = [{"role": "tool", "content": '{"content": "127.0.0.1 localhost"}'}]

    broke = False
    for _ in range(3):
        # the guard checks if the PRIOR identical call failed
        prior_failed = False
        if streak and streak[-1] == sig:
            for m in reversed(success_result):
                if m.get("role") == "tool":
                    prior_failed, _ = _detect_tool_failure(
                        tool_call.function.name, m["content"])
                    break
        if prior_failed:
            streak.append(sig)
            if len(streak) >= 3:
                broke = True
                break
        else:
            streak = [sig]

    assert not broke, "guard broke on 3 successful reads (the D1 false positive)"


def test_repetition_guard_breaks_on_repeated_failures():
    """3 failed identical tool calls SHOULD trip the guard (preserves intent)."""
    from agent.display import _detect_tool_failure

    class FakeToolCall:
        def __init__(self, name, args):
            self.function = types.SimpleNamespace(name=name, arguments=args)

    streak = []
    tool_call = FakeToolCall("terminal", '{"command": "grep x /missing"}')
    sig = (tool_call.function.name,
           hashlib.md5(tool_call.function.arguments.encode()).hexdigest()[:8])
    fail_result = [{"role": "tool", "content": '{"exit_code": 1, "error": "No such file"}'}]

    broke = False
    for _ in range(3):
        prior_failed = False
        if streak and streak[-1] == sig:
            for m in reversed(fail_result):
                if m.get("role") == "tool":
                    prior_failed, _ = _detect_tool_failure(
                        tool_call.function.name, m["content"])
                    break
        if prior_failed:
            streak.append(sig)
            if len(streak) >= 3:
                broke = True
                break
        else:
            streak = [sig]

    assert broke, "guard should break on 3 failed identical calls"


# ─── H-3: trust-score ceiling ─────────────────────────────────────────────────

def test_trust_max_capped_below_1():
    """_TRUST_MAX must be < 1.0 so no fact can lock at the ceiling (audit H-3)."""
    from plugins.memory.holographic.store import _TRUST_MAX, _clamp_trust
    assert _TRUST_MAX < 1.0, f"_TRUST_MAX={_TRUST_MAX}, should be < 1.0"
    assert _clamp_trust(1.0) == _TRUST_MAX, "clamp(1.0) should hit the cap"
    assert _clamp_trust(1.5) == _TRUST_MAX, "clamp(1.5) should hit the cap"
    assert _clamp_trust(0.5) == 0.5, "clamp(0.5) should pass through"


# ─── H-1: reranker cooldown recovery ──────────────────────────────────────────

def test_reranker_recovers_after_cooldown():
    """After a load failure, is_available() should recover once cooldown expires (H-1)."""
    import importlib
    import plugins.memory.holographic.reranker as rr
    importlib.reload(rr)

    # simulate a fresh failure
    rr._model = None
    rr._last_fail_ts = time.time()
    assert not rr.is_available(), "within cooldown, should be unavailable"

    # simulate cooldown expiry
    rr._model = None
    rr._last_fail_ts = time.time() - rr._RETRY_COOLDOWN_SECONDS - 1
    # the real model can load (transformers installed), so this recovers
    assert rr.is_available(), "after cooldown, should recover"


def test_reranker_reset_cache_clears_failure():
    """reset_cache() should clear failure state for explicit reset (H-1)."""
    import importlib
    import plugins.memory.holographic.reranker as rr
    importlib.reload(rr)

    rr._model = None
    rr._last_fail_ts = time.time()
    rr.reset_cache()
    assert rr._last_fail_ts == 0.0, "reset_cache should clear _last_fail_ts"
    assert rr.is_available(), "should re-probe after reset"


# ─── D2: BudgetTracker respects HERMES_HOME ───────────────────────────────────

def test_budget_tracker_respects_hermes_home(tmp_path, monkeypatch):
    """BudgetTracker should write under HERMES_HOME, not hardcoded Path.home() (D2)."""
    fake_home = tmp_path / "fake-hermes"
    fake_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(fake_home))

    from agent.routing import BudgetTracker
    bt = BudgetTracker()
    assert str(bt._file).startswith(str(fake_home)), \
        f"BudgetTracker file {bt._file} not under HERMES_HOME {fake_home}"
