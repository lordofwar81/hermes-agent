"""Regression tests for the 2026-07-18 Hermes audit fixes.

These close the test-coverage gaps that allowed the audited defects to persist.
Run: venv/bin/python -m pytest tests/test_audit_2026_07_18_regressions.py -v
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─── H-2: EmbedClient alive TTL ───────────────────────────────────────────────

def test_embed_client_alive_false_expires_after_ttl():
    """A cached alive=False must expire after the TTL and re-probe (audit H-2)."""
    from plugins.memory.holographic.store import EmbedClient
    c = EmbedClient()
    c._alive = False
    c._alive_false_ts = time.time()
    assert c.alive is False, "within TTL, should stay False"
    c._alive = False
    c._alive_false_ts = time.time() - c._alive_ttl_seconds - 1
    assert c.alive is True, "after TTL expiry, should re-probe and recover"


def test_embed_client_alive_true_stays_cached():
    """A cached alive=True should stay True (no needless re-probing)."""
    from plugins.memory.holographic.store import EmbedClient
    c = EmbedClient()
    c._alive = True
    assert c.alive is True


# ─── D1: inline repetition guard removed (existing controller handles it) ─────

def test_no_inline_repetition_guard_in_conversation_loop():
    """The crude inline repetition guard that tripped on successful re-reads
    must be gone (audit D1). The existing ToolCallGuardrailController handles
    loop detection correctly (failure-aware, respects hard_stop config)."""
    cl_path = os.path.join(os.path.dirname(__file__), "..", "agent", "conversation_loop.py")
    content = open(cl_path).read()
    assert "_recent_tool_calls" not in content, \
        "inline repetition guard (_recent_tool_calls) still present (D1)"
    assert "_repetition_break_count" not in content, \
        "dead _repetition_break_count still present (D1)"
    assert "_failed_repeat_streak" not in content, \
        "inline guard (_failed_repeat_streak) still present (D1)"
    assert "_tool_guardrail_halt_decision" in content, \
        "existing ToolCallGuardrailController check missing"


def test_existing_guardrail_controller_is_failure_aware():
    """The ToolCallGuardrailController must only count failed calls (the D1
    invariant that prevents false positives on legitimate re-reads)."""
    from agent.tool_guardrails import ToolCallGuardrailController
    ctrl = ToolCallGuardrailController()
    success_result = '{"content": "127.0.0.1 localhost"}'
    for _ in range(3):
        ctrl.after_call("read_file", '{"path":"/etc/hosts"}',
                        result=success_result, failed=False)
    # After 3 successful calls, the controller should NOT have produced a halt decision.
    assert ctrl.halt_decision is None, "controller halted on 3 successful reads (D1 regression)"


# ─── H-3: trust-score ceiling ─────────────────────────────────────────────────

def test_trust_max_capped_below_1():
    """_TRUST_MAX must be < 1.0 so no fact can lock at the ceiling (audit H-3)."""
    from plugins.memory.holographic.store import _TRUST_MAX, _clamp_trust
    assert _TRUST_MAX < 1.0, f"_TRUST_MAX={_TRUST_MAX}, should be < 1.0"
    assert _clamp_trust(1.0) == _TRUST_MAX
    assert _clamp_trust(1.5) == _TRUST_MAX
    assert _clamp_trust(0.5) == 0.5


# ─── H-1: reranker cooldown recovery ──────────────────────────────────────────

def test_reranker_recovers_after_cooldown():
    """After a load failure, is_available() should recover once cooldown expires (H-1)."""
    import importlib
    import plugins.memory.holographic.reranker as rr
    importlib.reload(rr)
    rr._model = None
    rr._last_fail_ts = time.time()
    assert not rr.is_available(), "within cooldown, should be unavailable"
    rr._model = None
    rr._last_fail_ts = time.time() - rr._RETRY_COOLDOWN_SECONDS - 1
    assert rr.is_available(), "after cooldown, should recover"


def test_reranker_reset_cache_clears_failure():
    """reset_cache() should clear failure state for explicit reset (H-1)."""
    import importlib
    import plugins.memory.holographic.reranker as rr
    importlib.reload(rr)
    rr._model = None
    rr._last_fail_ts = time.time()
    rr.reset_cache()
    assert rr._last_fail_ts == 0.0
    assert rr.is_available()


# ─── D2: BudgetTracker respects HERMES_HOME ───────────────────────────────────

def test_budget_tracker_respects_hermes_home(tmp_path, monkeypatch):
    """BudgetTracker should write under HERMES_HOME, not hardcoded Path.home() (D2)."""
    fake_home = tmp_path / "fake-hermes"
    fake_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(fake_home))
    from agent.routing import BudgetTracker
    bt = BudgetTracker()
    assert str(bt._file).startswith(str(fake_home)), \
        f"BudgetTracker file {bt._file} not under HERMES_HOME"
