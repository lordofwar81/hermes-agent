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


# Reset the global router singleton after every test. Several tests here
# call init_router(); without cleanup the leaked _instance pollutes other
# test modules (e.g. gateway session-override tests) whose routing path
# would then resolve against this stale test router.
@pytest.fixture(autouse=True)
def _reset_router_singleton():
    yield
    import agent.routing as routing_mod
    routing_mod._instance = None


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




class TestGatewayInterface:
    def test_route_turn_is_retired_noop(self):
        """[2026-08-14] Legacy routing excised — route_turn is an unconditional
        no-op; the :4090 microservice owns all decisions."""
        from agent.routing import init_router, route_turn
        init_router(SAMPLE_CONFIG)
        result = route_turn(
            "implement a sorting algorithm",
            {"model": "glm-5-turbo", "base_url": "https://z.ai/v1",
             "api_key": "zai-key", "provider": "zai"},
        )
        assert result is None


class TestRoutingObservability:
    """Verify routing decisions are recorded to routing_history.jsonl."""

    def test_decision_recorded_on_success(self, tmp_path):
        from agent.routing import Router
        history_file = tmp_path / "routing_history.jsonl"
        router = Router(SAMPLE_CONFIG)
        router._history_file = history_file
        primary = {"model": "glm-5-turbo", "base_url": "https://z.ai/v1",
                   "api_key": "zai-key", "provider": "zai"}
        router.route("fix this bug", primary)
        assert history_file.exists()
        lines = history_file.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["category"] == "code"
        assert entry["provider"] == "zai"
        assert entry["model"] == "glm-5.1"
        assert "fix this bug" in entry["message_preview"]


    def test_no_history_file_no_crash(self, tmp_path):
        """A read-only or unwritable path must never break routing."""
        from agent.routing import Router
        router = Router(SAMPLE_CONFIG)
        # Point at an impossible path — routing must still succeed.
        router._history_file = tmp_path / "nonexistent_dir" / "deep" / "x.jsonl"
        primary = {"model": "m", "base_url": "", "api_key": "k", "provider": "zai"}
        # Note: mkdir(parents=True) in _record_decision means this WILL create
        # the dir. To truly test failure, point at a path under a file.
        router._history_file = tmp_path / "blocker"  # tmp_path/blocker as a dir later
        (tmp_path / "blocker").mkdir()
        # Now _history_file is a directory — open() for write will fail.
        result = router.route("fix this bug", primary)
        assert result is not None  # routing succeeded despite log failure
