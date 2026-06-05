"""Tests for agent.routing — deterministic routing, circuit breaker, budget tracking.

Run with: python3 -m pytest tests/agent/test_routing.py -v
"""

import os
import sys
import tempfile
import json
import time
import pytest

# ── Fixtures ───────────────────────────────────────────────────────────────

SAMPLE_CONFIG = {
    "routing": {
        "venice_daily_budget": 7.40,
        "providers": [
            {"id": "zai-5.1", "provider": "zai", "model": "glm-5.1",
             "base_url": "https://z.ai/v1", "api_key": "zai-key",
             "context_length": 200000, "timeout": 300},
            {"id": "zai-turbo", "provider": "zai", "model": "glm-5-turbo",
             "base_url": "https://z.ai/v1", "api_key": "zai-key",
             "context_length": 200000, "timeout": 300},
            {"id": "strix-qwen36", "provider": "strix", "model": "qwen3.6",
             "base_url": "http://127.0.0.1:8199/v1", "api_key": "llama-key",
             "context_length": 262144, "timeout": 120, "local": True},
            {"id": "mac-qwen36", "provider": "mac_studio", "model": "qwen3.6-mlx",
             "base_url": "http://192.168.1.149:8000/v1", "api_key": "mac-key",
             "context_length": 32768, "timeout": 120, "local": True},
            {"id": "venice-ds4", "provider": "venice", "model": "deepseek-v4-flash",
             "base_url": "https://api.venice.ai/v1", "api_key": "venice-key",
             "context_length": 1000000, "timeout": 300},
        ],
        "chains": {
            "greeting": ["strix-qwen36", "mac-qwen36", "zai-turbo"],
            "simple": ["strix-qwen36", "mac-qwen36", "zai-turbo"],
            "code": ["zai-5.1", "strix-qwen36", "venice-ds4"],
            "reasoning": ["zai-5.1", "zai-turbo", "strix-qwen36", "venice-ds4"],
            "analysis": ["zai-turbo", "strix-qwen36", "mac-qwen36"],
            "expert": ["zai-5.1", "strix-qwen36", "venice-ds4"],
        },
        "default_chain": ["zai-turbo", "strix-qwen36"],
    },
}


@pytest.fixture
def router():
    from agent.routing import Router
    return Router(SAMPLE_CONFIG)


@pytest.fixture
def config():
    return SAMPLE_CONFIG


# ── TaskClassifier Tests ─────────────────────────────────────────────────


class TestTaskClassifier:
    def test_greeting_short(self):
        from agent.routing import TaskClassifier, Category
        assert TaskClassifier.classify("hi") == Category.GREETING
        assert TaskClassifier.classify("hello") == Category.GREETING
        assert TaskClassifier.classify("thanks") == Category.GREETING
        assert TaskClassifier.classify("ok") == Category.GREETING
        assert TaskClassifier.classify("got it") == Category.GREETING

    def test_greeting_not_false_positive(self):
        from agent.routing import TaskClassifier, Category
        assert TaskClassifier.classify("high quality output") == Category.SIMPLE
        assert TaskClassifier.classify("this is a hit") == Category.SIMPLE

    def test_code_classification(self):
        from agent.routing import TaskClassifier, Category
        assert TaskClassifier.classify("fix this bug") == Category.CODE
        assert TaskClassifier.classify("debug the TypeError") == Category.CODE
        assert TaskClassifier.classify("```python\nprint('hi')\n```") == Category.CODE
        assert TaskClassifier.classify("implement a merge sort") == Category.CODE
        assert TaskClassifier.classify("refactor this function") == Category.CODE

    def test_analysis_classification(self):
        from agent.routing import TaskClassifier, Category
        assert TaskClassifier.classify("compare react vs vue") == Category.ANALYSIS
        assert TaskClassifier.classify("evaluate these options") == Category.ANALYSIS
        assert TaskClassifier.classify("analyze performance metrics") == Category.ANALYSIS

    def test_reasoning_classification(self):
        from agent.routing import TaskClassifier, Category
        assert TaskClassifier.classify("explain why distributed systems are hard") == Category.REASONING
        assert TaskClassifier.classify("how does garbage collection work") == Category.REASONING
        assert TaskClassifier.classify("redesign the data pipeline") == Category.REASONING

    def test_expert_classification(self):
        from agent.routing import TaskClassifier, Category
        assert TaskClassifier.classify("design a system for real-time analytics") == Category.EXPERT
        assert TaskClassifier.classify("implement a complete microservice architecture") == Category.EXPERT
        assert TaskClassifier.classify("build an end-to-end CI/CD pipeline from scratch") == Category.EXPERT

    def test_simple_default(self):
        from agent.routing import TaskClassifier, Category
        assert TaskClassifier.classify("what time is it") == Category.SIMPLE
        assert TaskClassifier.classify("tell me a joke") == Category.SIMPLE
        assert TaskClassifier.classify("who won the game") == Category.SIMPLE

    def test_empty_message(self):
        from agent.routing import TaskClassifier, Category
        assert TaskClassifier.classify("") == Category.SIMPLE
        assert TaskClassifier.classify(None) == Category.SIMPLE


# ── CircuitBreaker Tests ────────────────────────────────────────────────


class TestCircuitBreaker:
    def test_initial_state(self):
        from agent.routing import CircuitBreaker
        cb = CircuitBreaker()
        assert cb.is_available("zai")
        assert cb.blocked_providers() == []

    def test_trips_after_threshold(self):
        from agent.routing import CircuitBreaker, FAILURE_THRESHOLD
        cb = CircuitBreaker()
        for _ in range(FAILURE_THRESHOLD):
            cb.record_failure("zai")
        assert not cb.is_available("zai")
        assert "zai" in cb.blocked_providers()

    def test_success_resets(self):
        from agent.routing import CircuitBreaker
        cb = CircuitBreaker()
        cb.record_failure("zai")
        cb.record_success("zai")
        assert cb.is_available("zai")

    def test_recovery_after_timeout(self):
        from agent.routing import CircuitBreaker, RECOVERY_TIMEOUT_SECONDS
        cb = CircuitBreaker()
        cb.record_failure("zai")
        cb.record_failure("zai")
        cb.record_failure("zai")
        assert not cb.is_available("zai")
        # Manually expire the breaker
        cb._tripped_until["zai"] = time.time() - 1
        assert cb.is_available("zai")


# ── BudgetTracker Tests ──────────────────────────────────────────────────


class TestBudgetTracker:
    def test_initially_available(self):
        from agent.routing import BudgetTracker
        import tempfile, os
        bt = BudgetTracker(daily_limit=7.40)
        fd, tmp = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        bt._file = type(bt._file)(tmp)
        bt._cache = None
        assert bt.is_available()
        assert bt.spent_ratio() < 1.0
        os.unlink(tmp)

    def test_exhausted(self):
        from agent.routing import BudgetTracker
        import tempfile, os
        bt = BudgetTracker(daily_limit=1.00)
        # Isolate from production budget file
        fd, tmp = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        bt._file = type(bt._file)(tmp)
        bt._cache = None
        bt.record(0.80)
        assert bt.is_available()
        bt.record(0.30)
        assert not bt.is_available()
        os.unlink(tmp)

    def test_persistence(self, tmp_path):
        from agent.routing import BudgetTracker
        budget_file = tmp_path / "budget.json"
        bt = BudgetTracker(daily_limit=5.00)
        bt._file = budget_file
        bt.record(2.50)
        assert bt.spent_ratio() == pytest.approx(0.5)

        # Reload
        bt2 = BudgetTracker(daily_limit=5.00)
        bt2._file = budget_file
        bt2._cache = None  # force reload
        assert bt2.spent_ratio() == pytest.approx(0.5)


# ── ProviderRegistry Tests ────────────────────────────────────────────────


class TestProviderRegistry:
    def test_loads_all_providers(self):
        from agent.routing import ProviderRegistry
        reg = ProviderRegistry(SAMPLE_CONFIG)
        assert len(reg.all_providers()) == 5

    def test_chain_order(self):
        from agent.routing import ProviderRegistry
        reg = ProviderRegistry(SAMPLE_CONFIG)
        chain = reg.chain("code")
        ids = [p.id for p in chain]
        assert ids == ["zai-5.1", "strix-qwen36", "venice-ds4"]

    def test_default_chain(self):
        from agent.routing import ProviderRegistry, Category
        config = {"routing": {"providers": [], "default_chain": ["zai-turbo"]}}
        reg = ProviderRegistry(config)
        chain = reg.chain(Category.SIMPLE)
        assert len(chain) == 0  # zai-turbo not in providers

    def test_env_var_resolution(self):
        from agent.routing import ProviderRegistry
        config = {
            "routing": {
                "providers": [
                    {"id": "test", "provider": "test", "model": "m",
                     "base_url": "http://localhost", "api_key": "${MY_KEY}"},
                ],
                "chains": {},
            },
        }
        reg = ProviderRegistry(config, env={"MY_KEY": "resolved123"})
        p = reg.get("test")
        assert p.api_key == "resolved123"

    def test_missing_key_skipped(self):
        from agent.routing import ProviderRegistry
        config = {
            "routing": {
                "providers": [
                    {"id": "nokey", "provider": "none", "model": "m",
                     "base_url": "http://localhost", "api_key": "${MISSING_KEY}"},
                ],
                "chains": {},
            },
        }
        reg = ProviderRegistry(config)
        assert reg.get("nokey") is None


# ── Router Integration Tests ─────────────────────────────────────────────


class TestRouter:
    def test_code_routes_to_zai(self, router):
        primary = {"model": "glm-5-turbo", "base_url": "https://z.ai/v1",
                   "api_key": "zai-key", "provider": "zai"}
        result = router.route("fix this bug in my code", primary)
        assert result.model == "glm-5.1"
        assert result.provider == "zai"
        assert result.category.value == "code"
        assert not result.suppress_tools

    def test_greeting_routes_to_strix(self, router):
        primary = {"model": "glm-5-turbo", "base_url": "https://z.ai/v1",
                   "api_key": "zai-key", "provider": "zai"}
        result = router.route("hi", primary)
        # Strix is local-first for greeting but needs health check
        # Health check will fail in test env (no local server)
        # So it falls to next in chain or primary
        assert result.category.value == "greeting"
        assert result.suppress_tools is True

    def test_circuit_breaker_fallback(self, router):
        primary = {"model": "glm-5-turbo", "base_url": "https://z.ai/v1",
                   "api_key": "zai-key", "provider": "zai"}
        # Break ZAI
        for _ in range(3):
            router.record_failure("zai")
        result = router.route("fix this bug", primary)
        # Should not route to zai
        assert result.model != "glm-5.1" or result.provider != "zai"

    def test_primary_fallback_when_chain_exhausted(self, router):
        primary = {"model": "glm-4.7", "base_url": "https://z.ai/v1",
                   "api_key": "zai-key", "provider": "zai"}
        # Break all providers
        for p in ["zai", "strix", "mac_studio", "venice"]:
            for _ in range(3):
                router.record_failure(p)
        result = router.route("fix this bug", primary)
        assert result.model == "glm-4.7"  # primary fallback

    def test_status_report(self, router):
        status = router.status()
        assert "providers" in status
        assert "chains" in status
        assert "blocked" in status
        assert "budget" in status
        assert len(status["providers"]) == 5


# ── Gateway Integration ──────────────────────────────────────────────────


class TestGatewayInterface:
    def test_init_and_route(self):
        from agent.routing import init_router, route_turn, routing_status
        init_router(SAMPLE_CONFIG)
        result = route_turn(
            "implement a sorting algorithm",
            {"model": "glm-5-turbo", "base_url": "https://z.ai/v1",
             "api_key": "zai-key", "provider": "zai"},
        )
        assert result is not None
        assert result.model == "glm-5.1"
        assert result.category.value == "code"

        status = routing_status()
        assert status is not None
        assert len(status["providers"]) == 5

    def test_uninitialized_returns_none(self):
        from agent.routing import route_turn
        # Save and clear global
        import agent.routing as m
        old = m._instance
        m._instance = None
        try:
            result = route_turn("test", {"model": "m", "base_url": "", "api_key": "", "provider": ""})
            assert result is None
        finally:
            m._instance = old
