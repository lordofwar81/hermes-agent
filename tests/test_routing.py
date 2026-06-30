"""Unit tests for agent/routing.py — the single-source routing engine.

Tests cover:
- TaskClassifier: message classification into 6 categories
- CircuitBreaker: failure threshold, auto-recovery
- BudgetTracker: daily spend tracking, reset
- HealthChecker: local endpoint pre-flight
- ProviderRegistry: config loading, chain resolution, key resolution
- Router: full routing pipeline with fallback, suppression, chain exhaustion
- Singleton interface: init_router, route_turn, status
"""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.routing import (
    BudgetTracker,
    Category,
    CircuitBreaker,
    HealthChecker,
    ProviderRegistry,
    ProviderSlot,
    RouteResult,
    Router,
    TaskClassifier,
    init_router,
    get_router,
    record_routing_failure,
    record_routing_success,
    route_turn,
    routing_status,
    FAILURE_THRESHOLD,
    RECOVERY_TIMEOUT_SECONDS,
    SESSION_ACTIVE_WINDOW_S,
    SESSION_MIN_PRIOR_TURNS,
    SESSION_UPGRADE_CATEGORY,
    SUBSTANTIVE_CATEGORIES,
)


# ─── Singleton cleanup ──────────────────────────────────────────────────
# Several test classes initialize the global router singleton (init_router).
# Without resetting it after each test, a leaked _instance pollutes downstream
# test modules (e.g. gateway tests) whose _resolve_turn_agent_config would
# then route against a stale test router. This autouse fixture guarantees the
# singleton is cleared after every test in this module, regardless of which
# class/method touched it.
@pytest.fixture(autouse=True)
def _reset_router_singleton():
    yield
    import agent.routing as routing_mod
    routing_mod._instance = None


# ─── Test Config ─────────────────────────────────────────────────────────

def make_test_config() -> dict:
    """Minimal routing config for testing. All providers have keys."""
    return {
        "routing": {
            "providers": [
                {
                    "id": "zai-main",
                    "provider": "zai",
                    "model": "glm-5",
                    "base_url": "https://api.zai.ai/v1",
                    "api_key": "test-key-123",
                    "context_length": 200000,
                    "local": False,
                    "cost_input": 0.002,
                    "cost_output": 0.008,
                },
                {
                    "id": "strix-local",
                    "provider": "strix",
                    "model": "qwen3.6-35b",
                    "base_url": "http://192.168.1.229:8199/v1",
                    "api_key": "strix-key-456",
                    "local": True,
                },
                {
                    "id": "venice-ds",
                    "provider": "venice",
                    "model": "deepseek-v4-flash",
                    "base_url": "https://api.venice.ai/v1",
                    "api_key": "venice-key-789",
                    "local": False,
                },
            ],
            "chains": {
                "code": ["zai-main", "strix-local", "venice-ds"],
                "expert": ["zai-main", "venice-ds"],
                "greeting": ["strix-local", "zai-main"],
            },
            "default_chain": ["zai-main"],
            "venice_daily_budget": 7.40,
        }
    }


# ─── TaskClassifier Tests ───────────────────────────────────────────────

class TestTaskClassifier:
    """Deterministic, zero-dependency classifier — every case is exact."""

    def test_greeting_short_hello(self):
        assert TaskClassifier.classify("hello") == Category.GREETING

    def test_greeting_hi(self):
        assert TaskClassifier.classify("hi") == Category.GREETING

    def test_greeting_thanks(self):
        assert TaskClassifier.classify("thanks") == Category.GREETING

    def test_greeting_ok(self):
        assert TaskClassifier.classify("ok") == Category.GREETING

    def test_greeting_goodbye(self):
        assert TaskClassifier.classify("goodbye") == Category.GREETING

    def test_not_greeting_long_message(self):
        """Messages >25 chars should not be classified as GREETING even with greeting keywords."""
        assert TaskClassifier.classify("ok, now I need you to debug this thing") != Category.GREETING

    def test_code_backticks(self):
        assert TaskClassifier.classify("fix this `import error`") == Category.CODE

    def test_code_triple_backticks(self):
        assert TaskClassifier.classify("```python\nprint('hello')\n```") == Category.CODE

    def test_code_debug_keyword(self):
        assert TaskClassifier.classify("debug the memory leak") == Category.CODE

    def test_code_implement_keyword(self):
        assert TaskClassifier.classify("implement the auth flow") == Category.CODE

    def test_code_refactor_keyword(self):
        assert TaskClassifier.classify("refactor the database layer") == Category.CODE

    def test_code_deploy_keyword(self):
        assert TaskClassifier.classify("deploy to production") == Category.CODE

    def test_code_test_keyword(self):
        assert TaskClassifier.classify("test the API endpoint") == Category.CODE

    def test_code_security_keyword(self):
        # "security" is in _CODE_KW
        assert TaskClassifier.classify("security audit needed") == Category.CODE

    def test_code_continue_keyword(self):
        assert TaskClassifier.classify("continue") == Category.SIMPLE

    def test_code_go_ahead_keyword(self):
        assert TaskClassifier.classify("go ahead") == Category.SIMPLE

    def test_code_continuation_with_code_keyword(self):
        """Continuation phrases become CODE only when a real code keyword co-occurs."""
        assert TaskClassifier.classify("continue with the refactor") == Category.CODE
        assert TaskClassifier.classify("go ahead and fix the bug") == Category.CODE
        assert TaskClassifier.classify("proceed with the implementation") == Category.CODE

    def test_code_continuation_without_code_keyword(self):
        """Continuation phrases alone (no code keyword) → SIMPLE."""
        assert TaskClassifier.classify("continue the story") == Category.SIMPLE
        assert TaskClassifier.classify("proceed with the plan") == Category.SIMPLE
        assert TaskClassifier.classify("move forward with the discussion") == Category.SIMPLE
        assert TaskClassifier.classify("keep going with the explanation") == Category.SIMPLE

    def test_code_infrastructure_keywords(self):
        """Infrastructure terms (cron, docker, container, pipeline) → CODE."""
        assert TaskClassifier.classify("set up a cron job") == Category.CODE
        assert TaskClassifier.classify("write a docker compose file") == Category.CODE
        assert TaskClassifier.classify("create a container") == Category.CODE
        assert TaskClassifier.classify("configure the ci pipeline") == Category.CODE

    def test_expert_system_design(self):
        assert TaskClassifier.classify("design a system for real-time analytics") == Category.EXPERT

    def test_expert_architecture(self):
        assert TaskClassifier.classify("design an architecture for microservices") == Category.EXPERT

    def test_expert_end_to_end(self):
        assert TaskClassifier.classify("build a complete end-to-end pipeline") == Category.EXPERT

    def test_expert_from_scratch(self):
        assert TaskClassifier.classify("build from scratch") == Category.EXPERT

    def test_expert_production_ready(self):
        assert TaskClassifier.classify("make it production-ready") == Category.EXPERT

    def test_expert_beats_code(self):
        """EXPERT phrases should win over CODE keywords (implement is in both)."""
        assert TaskClassifier.classify("implement a complete system") == Category.EXPERT

    def test_reasoning_explain_why(self):
        assert TaskClassifier.classify("explain why the database is slow") == Category.REASONING

    def test_reasoning_how_does(self):
        assert TaskClassifier.classify("how does the circuit breaker work") == Category.REASONING

    def test_reasoning_tradeoff(self):
        assert TaskClassifier.classify("what's the tradeoff here") == Category.REASONING

    def test_analysis_analyze(self):
        assert TaskClassifier.classify("analyze the performance metrics") == Category.ANALYSIS

    def test_analysis_compare(self):
        assert TaskClassifier.classify("compare these two approaches") == Category.ANALYSIS

    # "security" is in CODE_KW so "audit the security config" → CODE, not ANALYSIS
    def test_analysis_audit_pure(self):
        assert TaskClassifier.classify("audit the performance") == Category.ANALYSIS

    def test_analysis_benchmark(self):
        assert TaskClassifier.classify("benchmark the new model") == Category.ANALYSIS

    def test_simple_default(self):
        assert TaskClassifier.classify("what time is it") == Category.SIMPLE

    def test_simple_long_question(self):
        assert TaskClassifier.classify("can you tell me about the weather today") == Category.SIMPLE

    def test_empty_string(self):
        assert TaskClassifier.classify("") == Category.SIMPLE

    def test_none_message(self):
        assert TaskClassifier.classify(None) == Category.SIMPLE

    def test_code_beats_greeting(self):
        """Code indicators (backticks) should beat greeting classification."""
        assert TaskClassifier.classify("hi, fix this `error`") == Category.CODE

    def test_all_categories_covered(self):
        """Every Category enum value should be reachable."""
        messages = [
            ("hello", Category.GREETING),
            ("what time is it", Category.SIMPLE),
            ("debug the crash", Category.CODE),
            ("explain why this happens", Category.REASONING),
            ("compare the performance", Category.ANALYSIS),
            # "design a system" is in _EXPERT_PHRASES; "design a complete system" is not
            ("design a system for analytics", Category.EXPERT),
        ]
        for msg, expected in messages:
            assert TaskClassifier.classify(msg) == expected, f"{msg} should be {expected}"

    # ── Edge case fixes (hydra/classifier-edge-case-fixes) ──────────

    def test_word_boundary_fix_vs_fixed(self):
        """'fix' should not match 'fixed' as substring (word boundary)."""
        # 'fix' as standalone word → CODE
        assert TaskClassifier.classify("fix this bug") == Category.CODE
        # 'fixed' past tense in greeting → GREETING (not CODE)
        assert TaskClassifier.classify("thanks that fixed it") == Category.GREETING

    def test_analysis_overrides_code_keywords(self):
        """Strong analysis keywords should override weak code keywords."""
        assert TaskClassifier.classify("compare test results") == Category.ANALYSIS
        assert TaskClassifier.classify("evaluate the test output") == Category.ANALYSIS

    def test_code_keyword_not_overridden_by_weak_analysis(self):
        """'audit' alone doesn't override 'security' (code keyword)."""
        assert TaskClassifier.classify("security audit needed") == Category.CODE

    def test_design_as_code_keyword(self):
        """'design' (without 'a system/architecture') routes to CODE."""
        assert TaskClassifier.classify("design the login page") == Category.CODE
        assert TaskClassifier.classify("design a new feature") == Category.CODE
        assert TaskClassifier.classify("design a user flow") == Category.CODE

    def test_production_ready_no_hyphen(self):
        """Both 'production-ready' and 'production ready' should map to EXPERT."""
        assert TaskClassifier.classify("make it production-ready") == Category.EXPERT
        assert TaskClassifier.classify("make it production ready") == Category.EXPERT
        assert TaskClassifier.classify("production ready code") == Category.EXPERT


# ─── CircuitBreaker Tests ────────────────────────────────────────────────

class TestCircuitBreaker:
    def test_initially_available(self):
        cb = CircuitBreaker()
        assert cb.is_available("provider-a")

    def test_single_failure_still_available(self):
        cb = CircuitBreaker()
        cb.record_failure("provider-a")
        assert cb.is_available("provider-a")

    def test_threshold_trips_circuit(self):
        cb = CircuitBreaker()
        for _ in range(FAILURE_THRESHOLD):
            cb.record_failure("provider-a")
        assert not cb.is_available("provider-a")

    def test_different_providers_independent(self):
        cb = CircuitBreaker()
        for _ in range(FAILURE_THRESHOLD):
            cb.record_failure("provider-a")
        assert not cb.is_available("provider-a")
        assert cb.is_available("provider-b")

    def test_success_resets_counter(self):
        cb = CircuitBreaker()
        cb.record_failure("provider-a")
        cb.record_failure("provider-a")
        cb.record_success("provider-a")
        for _ in range(FAILURE_THRESHOLD - 1):
            cb.record_failure("provider-a")
        assert cb.is_available("provider-a")

    def test_recovery_after_timeout(self):
        cb = CircuitBreaker()
        for _ in range(FAILURE_THRESHOLD):
            cb.record_failure("provider-a")
        assert not cb.is_available("provider-a")
        with patch("agent.routing.time.time", return_value=time.time() + RECOVERY_TIMEOUT_SECONDS + 1):
            assert cb.is_available("provider-a")

    def test_blocked_providers_list(self):
        cb = CircuitBreaker()
        for _ in range(FAILURE_THRESHOLD):
            cb.record_failure("provider-a")
        blocked = cb.blocked_providers()
        assert "provider-a" in blocked
        assert "provider-b" not in blocked


# ─── BudgetTracker Tests ─────────────────────────────────────────────────

class TestBudgetTracker:
    def _isolated_tracker(self, daily_limit=10.0, tmpdir=None):
        """Create a BudgetTracker isolated from the real venice_budget.json."""
        import tempfile
        tmp = tmpdir or tempfile.mkdtemp()
        bt = BudgetTracker(daily_limit=daily_limit)
        bt._file = Path(tmp) / "test_budget.json"
        bt._cache = None
        return bt

    def test_initially_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bt = self._isolated_tracker(10.0, tmpdir)
            assert bt.is_available()

    def test_spent_ratio_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bt = self._isolated_tracker(10.0, tmpdir)
            assert bt.spent_ratio() == 0.0

    def test_record_and_check(self):
        """Record spend in a temp file and verify budget tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bt = self._isolated_tracker(10.0, tmpdir)
            bt.record(5.0)
            bt._cache = None
            assert bt.spent_ratio() == 0.5
            assert bt.is_available()
            bt.record(5.0)
            bt._cache = None
            assert not bt.is_available()

    def test_daily_reset(self):
        # Cycle label changes → budget resets even if spend was maxed.
        with tempfile.TemporaryDirectory() as tmpdir:
            bt = self._isolated_tracker(10.0, tmpdir)
            bt.record(10.0)
            bt._cache = None
            assert not bt.is_available()
            # Next cycle label → tracker treats it as a fresh cycle.
            with patch.object(bt, "_budget_cycle_label", return_value="2099-01-01"):
                bt._cache = None
                assert bt.is_available()

    # ── 5pm PT boundary (BUDGET_RESET_HOUR_PST = 17) ────────────────────
    # Venice resets at 5pm Pacific, not midnight. These tests pin the cycle
    # label logic so spend recorded just before vs just after 5pm lands in
    # the correct cycle. Uses real datetime objects (no MagicMock strftime).

    def _dt(self, y, mo, d, h, mi=0):
        from zoneinfo import ZoneInfo
        from datetime import datetime
        return datetime(y, mo, d, h, mi, tzinfo=ZoneInfo("America/Los_Angeles"))

    def test_before_5pm_belongs_to_prior_days_cycle(self):
        # 2026-06-26 16:00 PT → still in the cycle that started 06-25 17:00.
        assert BudgetTracker._budget_cycle_label(self._dt(2026, 6, 26, 16)) == "2026-06-25"

    def test_at_exactly_5pm_starts_new_cycle(self):
        # 2026-06-26 17:00 PT → new cycle, label = 06-26.
        assert BudgetTracker._budget_cycle_label(self._dt(2026, 6, 26, 17)) == "2026-06-26"

    def test_after_5pm_same_cycle_as_tomorrow_morning(self):
        # 06-26 23:00 and 06-27 04:00 are both in the 06-26 cycle.
        assert BudgetTracker._budget_cycle_label(self._dt(2026, 6, 26, 23)) == "2026-06-26"
        assert BudgetTracker._budget_cycle_label(self._dt(2026, 6, 27, 4)) == "2026-06-26"

    def test_spend_before_5pm_not_reset_by_midnight_rollover(self):
        # Regression guard for the original bug: a file stamped "today" at
        # 4pm must NOT be considered a new cycle just because the calendar
        # date rolled past midnight. With the 5pm rule, the file written at
        # 4pm on the 26th has last_reset = cycle_label(26th 4pm) = "25th".
        # A read at 11pm same calendar day sees cycle_label(26th 23:00) = "26th"
        # → different label → correctly resets (the 5pm boundary crossed).
        with tempfile.TemporaryDirectory() as tmpdir:
            bt = self._isolated_tracker(10.0, tmpdir)
            with patch.object(BudgetTracker, "_now_pst", return_value=self._dt(2026, 6, 26, 16)):
                bt.record(10.0)
                bt._cache = None
                assert not bt.is_available()
            # 11pm same calendar day — midnight-rollover rule would NOT reset
            # (same date string), but 5pm rule SHOULD (boundary crossed).
            with patch.object(BudgetTracker, "_now_pst", return_value=self._dt(2026, 6, 26, 23)):
                bt._cache = None
                assert bt.is_available()

    def test_zero_limit_always_unavailable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bt = self._isolated_tracker(0.0, tmpdir)
            assert not bt.is_available()

    def test_spent_ratio_infinite_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bt = self._isolated_tracker(0.0, tmpdir)
            assert bt.spent_ratio() == 1.0


# ─── ProviderRegistry Tests ──────────────────────────────────────────────

class TestProviderRegistry:
    def test_loads_all_providers(self):
        reg = ProviderRegistry(make_test_config())
        assert len(reg.all_providers()) == 3

    def test_get_provider_by_id(self):
        reg = ProviderRegistry(make_test_config())
        slot = reg.get("zai-main")
        assert slot is not None
        assert slot.model == "glm-5"

    def test_get_nonexistent(self):
        reg = ProviderRegistry(make_test_config())
        assert reg.get("nonexistent") is None

    def test_skips_no_key_providers(self):
        config = make_test_config()
        config["routing"]["providers"][0]["api_key"] = ""
        reg = ProviderRegistry(config)
        assert reg.get("zai-main") is None

    def test_resolves_env_var_key(self):
        env = {"CUSTOM_KEY": "resolved-key"}
        config = {
            "routing": {
                "providers": [{
                    "id": "env-provider",
                    "provider": "custom",
                    "model": "test-model",
                    "base_url": "http://test/v1",
                    "api_key": "${CUSTOM_KEY}",
                }],
            }
        }
        reg = ProviderRegistry(config, env=env)
        slot = reg.get("env-provider")
        assert slot.api_key == "resolved-key"

    def test_chain_resolution(self):
        reg = ProviderRegistry(make_test_config())
        chain = reg.chain(Category.CODE)
        ids = [s.id for s in chain]
        assert ids == ["zai-main", "strix-local", "venice-ds"]

    def test_default_chain(self):
        config = make_test_config()
        config["routing"]["chains"] = {}
        reg = ProviderRegistry(config)
        chain = reg.chain(Category.ANALYSIS)
        ids = [s.id for s in chain]
        assert ids == ["zai-main"]  # from default_chain

    def test_chain_deduplication(self):
        config = make_test_config()
        config["routing"]["chains"]["simple"] = ["zai-main", "zai-main", "venice-ds"]
        reg = ProviderRegistry(config)
        chain = reg.chain(Category.SIMPLE)
        ids = [s.id for s in chain]
        assert ids == ["zai-main", "venice-ds"]

    def test_chain_by_string(self):
        reg = ProviderRegistry(make_test_config())
        chain = reg.chain("code")
        assert len(chain) == 3

    def test_chain_invalid_string(self):
        reg = ProviderRegistry(make_test_config())
        chain = reg.chain("nonexistent")
        assert chain == []

    def test_invalid_provider_entry(self):
        config = {"routing": {"providers": [{"bad": "entry"}]}}
        reg = ProviderRegistry(config)
        assert len(reg.all_providers()) == 0

    def test_unknown_category_in_chains(self):
        config = make_test_config()
        config["routing"]["chains"]["nonexistent_cat"] = ["zai-main"]
        reg = ProviderRegistry(config)
        assert reg is not None  # Should not raise


# ─── HealthChecker Tests ─────────────────────────────────────────────────

class TestHealthChecker:
    def test_non_local_always_healthy(self):
        hc = HealthChecker()
        slot = ProviderSlot(
            id="test", provider="cloud", model="gpt-4",
            base_url="https://api.openai.com/v1", api_key="key",
            is_local=False,
        )
        assert hc.check(slot)

    def test_healthy_local_endpoint(self):
        hc = HealthChecker()
        slot = ProviderSlot(
            id="test", provider="strix", model="qwen",
            base_url="http://localhost:9999/v1", api_key="key",
            is_local=True,
        )
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_urlopen.return_value.__enter__ = MagicMock(return_value=mock_resp)
            assert hc.check(slot)

    def test_unhealthy_local_endpoint(self):
        hc = HealthChecker()
        slot = ProviderSlot(
            id="test", provider="strix", model="qwen",
            base_url="http://localhost:9999/v1", api_key="key",
            is_local=True,
        )
        with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
            assert not hc.check(slot)

    def test_cache_ttl(self):
        hc = HealthChecker()
        slot = ProviderSlot(
            id="test", provider="strix", model="qwen",
            base_url="http://localhost:9999/v1", api_key="key",
            is_local=True,
        )
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_urlopen.return_value.__enter__ = MagicMock(return_value=mock_resp)
            hc.check(slot)
            hc.check(slot)  # cached
            assert mock_urlopen.call_count == 1


# ─── Router Tests ─────────────────────────────────────────────────────────

class TestRouter:
    def _make_router(self, config=None):
        return Router(config or make_test_config())

    def _primary_config(self):
        return {
            "model": "glm-5-turbo",
            "base_url": "https://api.zai.ai/v1",
            "api_key": "primary-key",
            "provider": "primary",
        }

    def test_routes_to_first_in_chain(self):
        router = self._make_router()
        result = router.route("debug the crash", self._primary_config())
        assert result.model == "glm-5"
        assert result.provider == "zai"
        assert result.category == Category.CODE

    def test_greeting_suppresses_tools(self):
        router = self._make_router()
        result = router.route("hello", self._primary_config())
        assert result.suppress_tools

    def test_simple_no_tool_suppression(self):
        router = self._make_router()
        result = router.route("what time is it", self._primary_config())
        assert result.suppress_tools  # SIMPLE now suppresses tools too

    def test_tool_suppression_override(self):
        router = self._make_router()
        result = router.route("hello", self._primary_config(), suppress_tools_override=False)
        assert not result.suppress_tools

    def test_falls_to_primary_on_empty_chain(self):
        config = make_test_config()
        config["routing"]["chains"] = {}
        config["routing"]["default_chain"] = []
        router = self._make_router(config)
        result = router.route("debug this", self._primary_config())
        assert result.provider == "primary"

    def test_fallback_on_circuit_break(self):
        router = self._make_router()
        for _ in range(FAILURE_THRESHOLD):
            router.record_failure("zai")
        result = router.route("debug this", self._primary_config())
        assert result.provider != "zai"

    def test_chain_exhaustion_falls_to_primary(self):
        router = self._make_router()
        for slot in router._registry.all_providers():
            for _ in range(FAILURE_THRESHOLD):
                router.record_failure(slot.provider)
        result = router.route("debug this", self._primary_config())
        assert result.provider == "primary"

    def test_fallback_count(self):
        router = self._make_router()
        for _ in range(FAILURE_THRESHOLD):
            router.record_failure("zai")
        result = router.route("debug this", self._primary_config())
        assert result.fallback_count >= 1

    def test_record_success_resets(self):
        router = self._make_router()
        router.record_failure("zai")
        router.record_failure("zai")
        router.record_success("zai")
        assert router._breaker.is_available("zai")

    def test_status_dict(self):
        router = self._make_router()
        status = router.status()
        assert "providers" in status
        assert "chains" in status
        assert "blocked" in status
        assert "budget" in status

    def test_route_result_dataclass(self):
        result = RouteResult(
            model="test", base_url="http://test", api_key="key",
            provider="test", category=Category.CODE, is_local=False,
            suppress_tools=False, label="test", fallback_count=0,
        )
        assert result.model == "test"
        assert result.fallback_count == 0

    def test_local_health_check_skips_unhealthy(self):
        """Unhealthy local models should be skipped in chain."""
        router = self._make_router()
        # Inject unhealthy local model (cache keyed by base_url)
        router._health._cache = {"http://192.168.1.229:8199/v1": (False, time.time())}
        # Route greeting → chain is [strix-local, zai-main]
        # strix is local and unhealthy, should fall to zai
        result = router.route("hello", self._primary_config())
        assert result.provider == "zai"

    def test_venice_budget_gate_skips_provider(self):
        """Venice providers should be skipped when daily budget is exhausted."""
        router = self._make_router()
        # Exhaust Venice budget
        router._budget.record(router._budget._daily_limit)
        router._budget._cache = None  # force reload
        # Route code → chain is [zai-main, strix-local, venice-ds]
        # venice should be skipped, zai is first so it wins
        result = router.route("debug this code", self._primary_config())
        assert result.provider == "zai"
        assert result.model == "glm-5"


# ─── Session-Aware Routing Guard Tests ────────────────────────────────────

class TestSessionAwareRouting:
    """The session-aware guard: once a session has had substantive work,
    a later SIMPLE/GREETING turn is upgraded so it is not routed to a
    tool-less local model mid-conversation. Regression coverage for the
    2026-06-25 incident where turn 8 of a 51-tool-call session got
    dumped to a slow local model because 'estradiol was high...' hit the
    keyword default."""

    def teardown_method(self):
        # Reset the router singleton so tests that go through route_turn
        # (the gateway entry point) don't leak a test router into other
        # test modules that rely on _instance being None by default.
        import agent.routing as routing_mod
        routing_mod._instance = None

    def _make_router(self, config=None):
        return Router(config or make_test_config())

    def _primary_config(self):
        return {
            "model": "glm-5-turbo",
            "base_url": "https://api.zai.ai/v1",
            "api_key": "primary-key",
            "provider": "primary",
        }

    def test_no_session_key_preserves_original_behavior(self):
        """Omitting session_key (cron, one-shot CLI) must behave exactly as
        before — no upgrade, no recording."""
        router = self._make_router()
        result = router.route("hello", self._primary_config())
        assert result.category == Category.GREETING
        assert result.suppress_tools

    def test_inactive_session_not_upgraded(self):
        """A fresh session_key with no prior substantive turns must NOT
        upgrade — the first message of a chat can still be a greeting."""
        router = self._make_router()
        result = router.route("hello", self._primary_config(), session_key="s1")
        assert result.category == Category.GREETING
        assert result.suppress_tools

    def test_substantive_turn_activates_session(self):
        """A CODE turn marks the session active."""
        router = self._make_router()
        router.route("debug the crash", self._primary_config(), session_key="s1")
        assert router._session_is_active("s1")

    def test_trivial_turn_does_not_activate_session(self):
        """A GREETING/SIMPLE turn must NOT mark the session active — a lone
        'hello' shouldn't make the next turn eligible for upgrade."""
        router = self._make_router()
        router.route("hello", self._primary_config(), session_key="s1")
        assert not router._session_is_active("s1")

    def test_active_session_upgrades_simple_to_code(self):
        """The core fix: after a substantive turn, a SIMPLE message is
        upgraded to CODE (API-first, tools enabled)."""
        router = self._make_router()
        # turn 1: substantive
        router.route("debug the crash", self._primary_config(), session_key="s1")
        # turn 2: would-be SIMPLE
        result = router.route("what time is it", self._primary_config(), session_key="s1")
        assert result.category == Category.CODE
        # upgraded turn must keep tools enabled (CODE is not suppressed)
        assert not result.suppress_tools

    def test_active_session_upgrades_greeting_to_code(self):
        """Same guard for GREETING — the estradiol incident was a short
        reply misread as trivial."""
        router = self._make_router()
        router.route("implement the feature", self._primary_config(), session_key="s1")
        result = router.route("ok", self._primary_config(), session_key="s1")
        assert result.category == Category.CODE
        assert not result.suppress_tools

    def test_sessions_are_isolated(self):
        """Activity in session A must not upgrade a turn in session B."""
        router = self._make_router()
        router.route("debug this", self._primary_config(), session_key="A")
        # session B has no history
        result = router.route("ok", self._primary_config(), session_key="B")
        assert result.category == Category.GREETING

    def test_window_expiry_deactivates_session(self):
        """After SESSION_ACTIVE_WINDOW_S with no substantive turn, the
        session is no longer active and turns are not upgraded."""
        router = self._make_router()
        router.route("debug this", self._primary_config(), session_key="s1")
        # Backdate the recorded timestamp past the window
        stale = time.time() - SESSION_ACTIVE_WINDOW_S - 1
        router._session_activity["s1"] = [stale]
        assert not router._session_is_active("s1")
        result = router.route("ok", self._primary_config(), session_key="s1")
        assert result.category == Category.GREETING

    def test_route_turn_threads_session_key(self):
        """The gateway entry point must honor session_key end-to-end."""
        import agent.routing as routing_mod
        routing_mod._instance = None
        init_router(make_test_config())
        # prime the session
        route_turn("debug the crash", self._primary_config(), session_key="sx")
        result = route_turn("ok", self._primary_config(), session_key="sx")
        assert result is not None
        assert result.category == Category.CODE
        routing_mod._instance = None

    def test_activity_buffer_is_bounded(self):
        """Very long sessions must not grow the activity list unbounded."""
        router = self._make_router()
        for _ in range(200):
            router.route("debug this", self._primary_config(), session_key="long")
        assert len(router._session_activity["long"]) <= 64


# ─── Singleton Interface Tests ────────────────────────────────────────────

class TestSingletonInterface:
    def setup_method(self):
        import agent.routing as routing_mod
        routing_mod._instance = None

    def test_init_router_returns_router(self):
        import agent.routing as routing_mod
        router = init_router(make_test_config())
        assert isinstance(router, Router)
        assert routing_mod._instance is router

    def test_get_router_returns_none_before_init(self):
        import agent.routing as routing_mod
        routing_mod._instance = None
        assert get_router() is None

    def test_get_router_returns_instance_after_init(self):
        init_router(make_test_config())
        assert get_router() is not None

    def test_route_turn_returns_none_before_init(self):
        import agent.routing as routing_mod
        routing_mod._instance = None
        result = route_turn("hello", {})
        assert result is None

    def test_route_turn_works_after_init(self):
        init_router(make_test_config())
        result = route_turn("hello", {
            "model": "test", "base_url": "http://t", "api_key": "k", "provider": "p"
        })
        assert result is not None
        assert isinstance(result, RouteResult)

    def test_record_routing_failure_no_crash_before_init(self):
        import agent.routing as routing_mod
        routing_mod._instance = None
        record_routing_failure("anything")

    def test_record_routing_success_no_crash_before_init(self):
        import agent.routing as routing_mod
        routing_mod._instance = None
        record_routing_success("anything")

    def test_routing_status_before_init(self):
        import agent.routing as routing_mod
        routing_mod._instance = None
        assert routing_status() is None

    def test_routing_status_after_init(self):
        init_router(make_test_config())
        status = routing_status()
        assert isinstance(status, dict)
        assert "providers" in status


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Semantic classifier (classify_semantic — LLM-based)
# ─────────────────────────────────────────────────────────────────────────────
# Tests mock call_llm — no network required. Verify:
#   1. classify_semantic parses the LLM response into the right Category.
#   2. Failure contract: exception/None/unparseable → classify_semantic None.
#   3. The keyword classify() path is byte-identical (regression guard).
#
# Note: the embedding-centroid approach was built first and rejected —
# Qwen3-Embedding-8B doesn't discriminate between the 6 intent categories
# (same-cat ~0.948, diff-cat ~0.962). See tests/eval/diagnose.py for proof.
# The LLM approach replaces it entirely.


def _mock_llm_response(category_word: str):
    """Build a fake call_llm return value whose .choices[0].message.content
    is the given category word."""
    mock_msg = MagicMock()
    mock_msg.message.content = category_word
    mock_choice = MagicMock()
    mock_choice.message = mock_msg.message
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    return mock_resp


class TestClassifySemantic:
    """LLM-based classifier — additive path, never breaks keyword classify."""

    @pytest.fixture(autouse=True)
    def _clear_semantic_cache(self):
        """Clear lru_cache between tests so each gets a clean classify."""
        TaskClassifier.classify_semantic.cache_clear()
        yield
        TaskClassifier.classify_semantic.cache_clear()

    def test_returns_none_on_empty_message(self):
        assert TaskClassifier.classify_semantic("") is None
        assert TaskClassifier.classify_semantic("   ") is None

    def test_returns_none_on_llm_exception(self):
        """call_llm raises → classify_semantic returns None, never raises."""
        with patch("agent.auxiliary_client.call_llm", side_effect=RuntimeError("network down")):
            result = TaskClassifier.classify_semantic("fix this bug")
        assert result is None

    def test_returns_none_on_import_failure(self):
        """auxiliary_client can't be imported → None, never raises."""
        with patch("agent.auxiliary_client.call_llm", side_effect=ImportError("no module")):
            result = TaskClassifier.classify_semantic("fix this bug")
        assert result is None

    def test_returns_none_on_unparseable_response(self):
        """LLM returns gibberish → None (caller falls back to keyword)."""
        with patch("agent.auxiliary_client.call_llm", return_value=_mock_llm_response("banana")):
            result = TaskClassifier.classify_semantic("fix this bug")
        assert result is None

    def test_returns_none_on_empty_response(self):
        with patch("agent.auxiliary_client.call_llm", return_value=_mock_llm_response("")):
            result = TaskClassifier.classify_semantic("fix this bug")
        assert result is None

    def test_classifies_code_message(self):
        """LLM says 'code' → CODE."""
        with patch("agent.auxiliary_client.call_llm", return_value=_mock_llm_response("code")):
            result = TaskClassifier.classify_semantic("the function throws KeyError")
        assert result == Category.CODE

    def test_classifies_reasoning_message(self):
        with patch("agent.auxiliary_client.call_llm", return_value=_mock_llm_response("reasoning")):
            result = TaskClassifier.classify_semantic("why does this happen")
        assert result == Category.REASONING

    def test_classifies_greeting_message(self):
        with patch("agent.auxiliary_client.call_llm", return_value=_mock_llm_response("greeting")):
            result = TaskClassifier.classify_semantic("thanks that worked")
        assert result == Category.GREETING

    def test_classifies_expert_message(self):
        with patch("agent.auxiliary_client.call_llm", return_value=_mock_llm_response("expert")):
            result = TaskClassifier.classify_semantic("design a distributed system")
        assert result == Category.EXPERT

    def test_classifies_analysis_message(self):
        with patch("agent.auxiliary_client.call_llm", return_value=_mock_llm_response("analysis")):
            result = TaskClassifier.classify_semantic("compare these options")
        assert result == Category.ANALYSIS

    def test_parses_response_with_extra_text(self):
        """LLM wraps the word ('The answer is: code.') → still parsed."""
        with patch("agent.auxiliary_client.call_llm", return_value=_mock_llm_response("The category is: code.")):
            result = TaskClassifier.classify_semantic("fix the bug")
        assert result == Category.CODE

    def test_parses_response_with_trailing_punctuation(self):
        with patch("agent.auxiliary_client.call_llm", return_value=_mock_llm_response("reasoning.")):
            result = TaskClassifier.classify_semantic("explain why")
        assert result == Category.REASONING

    def test_keyword_classify_unchanged(self):
        """Regression guard: the keyword path still works identically.

        These are a subset of TestTaskClassifier's assertions, repeated here to
        prove the semantic additions didn't touch classify() byte-for-byte.
        """
        assert TaskClassifier.classify("hello") == Category.GREETING
        assert TaskClassifier.classify("debug the memory leak") == Category.CODE
        assert TaskClassifier.classify("design a complete system") != Category.GREETING
        assert TaskClassifier.classify("compare the performance") == Category.ANALYSIS

    def test_parse_llm_category_all_six(self):
        """_parse_llm_category handles all 6 category values."""
        for cat in Category:
            assert TaskClassifier._parse_llm_category(cat.value) == cat

    def test_parse_llm_category_none_on_garbage(self):
        assert TaskClassifier._parse_llm_category("xyzzy") is None
        assert TaskClassifier._parse_llm_category("") is None
        assert TaskClassifier._parse_llm_category(None) is None


