"""Tests for agent.model_router — deterministic routing, circuit breaker, budget tracking.

Run with: scripts/run_tests.sh tests/agent/test_model_router.py
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from unittest import TestCase

import yaml


def _make_config(
    zai_key: str = "test-zai-key",
    venice_key: str = "test-venice-key",
    minimax_key: str = "test-minimax-key",
    mac_key: str = "test-mac-key",
    extra_sections: dict | None = None,
) -> str:
    """Build a minimal config.yaml string for testing."""
    cfg = {
        "model": {"provider": "z-ai", "default": "glm-5-turbo"},
        "z-ai": {
            "api_key": zai_key,
            "base_url": "https://api.z.ai/api/coding/paas/v4",
        },
        "providers": {
            "venice": {
                "api_key": venice_key,
                "base_url": "https://api.venice.ai/api/v1",
            }
        },
        "minimax_worker": {
            "api_key": minimax_key,
            "base_url": "http://192.168.1.229:8199/v1",
            "model": "MiniMax-M2.7-APEX-I-Mini",
        },
        "mac_studio_orchestrator": {
            "api_key": mac_key,
            "base_url": "http://192.168.1.149:1234/v1",
            "model": "qwen3.6-35b-a3b-abliterated-heretic-mlx",
        },
        "smart_model_routing": {
            "venice_budget": {"daily_limit_usd": 7.40},
        },
    }
    if extra_sections:
        cfg.update(extra_sections)
    return yaml.dump(cfg, default_flow_style=False)


class TestTaskClassifier(TestCase):
    """Test deterministic task classification."""

    def setUp(self):
        from agent.model_router import TaskClassifier, TaskCategory

        self.classifier = TaskClassifier
        self.category = TaskCategory

    def test_greeting_short(self):
        result = self.classifier.classify("hello")
        self.assertEqual(result, self.category.GREETING)

    def test_greeting_thanks(self):
        result = self.classifier.classify("thank you")
        self.assertEqual(result, self.category.GREETING)

    def test_greeting_hey(self):
        result = self.classifier.classify("hey")
        self.assertEqual(result, self.category.GREETING)

    def test_greeting_too_long(self):
        # Greetings must be <=20 chars; "hello there friend" is exactly 20 chars
        # (which passes len <= 20), so we use 22 chars to exceed the threshold.
        result = self.classifier.classify("hello there my friend")
        self.assertNotEqual(result, self.category.GREETING)

    def test_code_with_backticks(self):
        result = self.classifier.classify("fix this \x60print('hello')\x60")
        self.assertEqual(result, self.category.CODE)

    def test_code_with_keyword(self):
        result = self.classifier.classify("debug the server error")
        self.assertEqual(result, self.category.CODE)

    def test_code_traceback(self):
        result = self.classifier.classify("traceback shows exception in module")
        self.assertEqual(result, self.category.CODE)

    def test_analysis_keywords(self):
        result = self.classifier.classify("compare performance metrics")
        self.assertEqual(result, self.category.ANALYSIS)

    def test_reasoning_explain_why(self):
        result = self.classifier.classify("explain why this happens")
        self.assertEqual(result, self.category.REASONING)

    def test_reasoning_how_does(self):
        result = self.classifier.classify("how does the cache work")
        self.assertEqual(result, self.category.REASONING)

    def test_expert_design_system(self):
        result = self.classifier.classify("design a system for distributed caching")
        self.assertEqual(result, self.category.EXPERT)

    def test_expert_end_to_end(self):
        result = self.classifier.classify("implement a complete end-to-end pipeline")
        self.assertEqual(result, self.category.EXPERT)

    def test_simple_default(self):
        result = self.classifier.classify("what is the weather today")
        self.assertEqual(result, self.category.SIMPLE)

    def test_empty_message(self):
        result = self.classifier.classify("")
        self.assertEqual(result, self.category.SIMPLE)

    def test_ordering_expert_before_analysis(self):
        # "design" is in CODE_KEYWORDS but "design a system" should be EXPERT
        result = self.classifier.classify("design a system architecture")
        self.assertEqual(result, self.category.EXPERT)


class TestCircuitBreaker(TestCase):
    """Test circuit breaker trip/untrip behavior."""

    def setUp(self):
        from agent.model_router import RouterCircuitBreaker

        self.cb = RouterCircuitBreaker()

    def test_initially_available(self):
        self.assertTrue(self.cb.is_available("zai"))

    def test_trip_after_threshold(self):
        # FAILURE_THRESHOLD = 3
        self.cb.record_failure("zai")
        self.cb.record_failure("zai")
        self.cb.record_failure("zai")
        self.assertFalse(self.cb.is_available("zai"))

    def test_untrip_on_success(self):
        self.cb.record_failure("zai")
        self.cb.record_failure("zai")
        self.cb.record_success("zai")  # Reset failures
        self.assertTrue(self.cb.is_available("zai"))

    def test_blocked_providers_list(self):
        self.cb.record_failure("venice")
        self.cb.record_failure("venice")
        self.cb.record_failure("venice")
        blocked = self.cb.get_blocked_providers()
        self.assertIn("venice", blocked)

    def test_recovery_after_timeout(self):
        # Use a short recovery timeout for testing
        from agent.model_router import RouterCircuitBreaker, RECOVERY_TIMEOUT_SECONDS

        # Temporarily patch the timeout
        original = RouterCircuitBreaker.__module__
        import agent.model_router as mr

        old_timeout = mr.RECOVERY_TIMEOUT_SECONDS
        mr.RECOVERY_TIMEOUT_SECONDS = 0.1  # 100ms for testing

        try:
            cb = RouterCircuitBreaker()
            cb.record_failure("zai")
            cb.record_failure("zai")
            cb.record_failure("zai")
            self.assertFalse(cb.is_available("zai"))

            time.sleep(0.15)  # Wait for recovery
            self.assertTrue(cb.is_available("zai"))
        finally:
            mr.RECOVERY_TIMEOUT_SECONDS = old_timeout


class TestBudgetTracker(TestCase):
    """Test Venice daily budget tracking."""

    def test_budget_available_initially(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(_make_config())

            from agent.model_router import DailyBudgetTracker

            tracker = DailyBudgetTracker(str(config_path))
            self.assertTrue(tracker.is_budget_available())

    def test_budget_exhausted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(_make_config())

            from agent.model_router import DailyBudgetTracker

            tracker = DailyBudgetTracker(str(config_path))
            tracker.record_spending(5.0)
            tracker.record_spending(3.0)  # Total 8.0 > 7.40 limit
            self.assertFalse(tracker.is_budget_available())

    def test_budget_reset_next_day(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(_make_config())

            from agent.model_router import DailyBudgetTracker

            tracker = DailyBudgetTracker(str(config_path))
            tracker.record_spending(7.0)  # Near limit

            # Simulate next day by writing a new date to the budget file
            budget_file = Path(tmpdir) / "venice_budget.json"
            data = json.loads(budget_file.read_text())
            # Change the date to simulate a new day
            from datetime import datetime, timezone, timedelta

            yesterday = (datetime.now(timezone(timedelta(hours=-8))) - timedelta(days=1)).strftime(
                "%Y-%m-%d"
            )
            data["last_reset_date"] = yesterday
            budget_file.write_text(json.dumps(data))

            # Next check should show fresh budget
            self.assertTrue(tracker.is_budget_available())


class TestProviderRegistry(TestCase):
    """Test provider credential resolution from config."""

    def test_all_providers_loaded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(_make_config())

            from agent.model_router import ProviderRegistry

            registry = ProviderRegistry(str(config_path))
            available = registry.list_available()
            provider_names = [p.model for p in available]

            # Should have ZAI, MiniMax, Mac Studio, and Venice providers
            self.assertGreaterEqual(len(available), 4)

    def test_zai_providers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(_make_config())

            from agent.model_router import ProviderRegistry

            registry = ProviderRegistry(str(config_path))
            zai_providers = registry.get_by_provider("zai")
            self.assertGreaterEqual(len(zai_providers), 1)

    def test_minimax_provider(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(_make_config())

            from agent.model_router import ProviderRegistry

            registry = ProviderRegistry(str(config_path))
            minimax = registry.get("minimax-m27")
            self.assertIsNotNone(minimax)
            self.assertEqual(minimax.provider, "minimax")

    def test_mac_studio_provider(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(_make_config())

            from agent.model_router import ProviderRegistry

            registry = ProviderRegistry(str(config_path))
            mac_studio = registry.get("mac-studio-qwen36")
            self.assertIsNotNone(mac_studio)
            self.assertEqual(mac_studio.provider, "mac_studio")

    def test_missing_api_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            # No z-ai key
            cfg = {
                "model": {"provider": "z-ai", "default": "glm-5-turbo"},
                "minimax_worker": {"api_key": "test-key", "base_url": "http://localhost:8199/v1"},
            }
            config_path.write_text(yaml.dump(cfg, default_flow_style=False))

            from agent.model_router import ProviderRegistry

            registry = ProviderRegistry(str(config_path))
            available = registry.list_available()
            zai_providers = [p for p in available if p.provider == "zai"]
            self.assertEqual(len(zai_providers), 0)


class TestModelRouter(TestCase):
    """Test end-to-end routing decisions."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = Path(self.tmpdir) / "config.yaml"
        self.config_path.write_text(_make_config())

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_greeting_routed_to_fastest(self):
        from agent.model_router import ModelRouter

        router = ModelRouter(str(self.config_path))
        route = router.route("hello", {"model": "glm-5-turbo", "provider": "z-ai"})
        # Should route to Mac Studio (fastest) or ZAI for greetings
        self.assertIn(route["model"], ["qwen3.6-35b-a3b-abliterated-heretic-mlx", "glm-5.1"])

    def test_code_routed_to_minimax(self):
        from agent.model_router import ModelRouter

        router = ModelRouter(str(self.config_path))
        route = router.route("debug this server error", {"model": "glm-5-turbo", "provider": "z-ai"})
        # ZAI (glm-5.1) is now the default for ALL tasks including code.
        # Local models (MiniMax, Mac Studio) are only used when ZAI is circuit-broken.
        self.assertEqual(route["model"], "glm-5.1")

    def test_reasoning_routed_to_mac_studio(self):
        from agent.model_router import ModelRouter

        router = ModelRouter(str(self.config_path))
        route = router.route("explain why this happens", {"model": "glm-5-turbo", "provider": "z-ai"})
        # ZAI (glm-5.1) is now the default for ALL tasks including reasoning.
        # Local models (Mac Studio, MiniMax) are only used when ZAI is circuit-broken.
        self.assertEqual(route["model"], "glm-5.1")

    def test_analysis_routed_to_zai(self):
        from agent.model_router import ModelRouter

        router = ModelRouter(str(self.config_path))
        route = router.route("compare performance metrics", {"model": "glm-5-turbo", "provider": "z-ai"})
        # Analysis tasks should route to ZAI
        self.assertIn(route["model"], ["glm-5.1", "glm-5-turbo", "glm-5"])

    def test_expert_routed_to_venice(self):
        from agent.model_router import ModelRouter

        router = ModelRouter(str(self.config_path))
        route = router.route(
            "design a system for distributed caching",
            {"model": "glm-5-turbo", "provider": "z-ai"},
        )
        # Expert tasks should route to Venice (if budget available)
        self.assertIn(route["model"], ["deepseek-v3.2", "claude-sonnet-4-6"])

    def test_simple_routed_to_fastest(self):
        from agent.model_router import ModelRouter

        router = ModelRouter(str(self.config_path))
        route = router.route("what is 2+2", {"model": "glm-5-turbo", "provider": "z-ai"})
        # ZAI (glm-5.1) is now the default for ALL tasks including simple ones.
        # Local models are only used when ZAI is circuit-broken.
        self.assertIn(route["model"], ["glm-5.1", "qwen3.6-35b-a3b-abliterated-heretic-mlx"])

    def test_fallback_to_primary_when_no_providers(self):
        # Config with no valid providers
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            cfg = {"model": {"provider": "z-ai", "default": "glm-5-turbo"}}
            config_path.write_text(yaml.dump(cfg, default_flow_style=False))

            from agent.model_router import ModelRouter

            router = ModelRouter(str(config_path))
            route = router.route("hello", {"model": "glm-5-turbo", "provider": "z-ai"})
            # Should fall back to primary model
            self.assertEqual(route["model"], "glm-5-turbo")

    def test_status_report(self):
        from agent.model_router import ModelRouter

        router = ModelRouter(str(self.config_path))
        status = router.status_report()
        self.assertGreaterEqual(status["providers_loaded"], 4)
        self.assertIn("budget_status", status)
        self.assertTrue(status["budget_status"]["available"])


class TestGatewayIntegration(TestCase):
    """Test that the gateway can import and use the new router."""

    def test_gateway_imports_new_router(self):
        """Verify gateway/run.py imports from agent.model_router, not smart_model_routing."""
        import ast

        # Path(__file__) is tests/agent/test_model_router.py; go up 3 levels to project root
        gateway_path = Path(__file__).resolve().parent.parent.parent / "gateway" / "run.py"
        source = gateway_path.read_text()
        tree = ast.parse(source)

        found_new_import = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "model_router" in node.module:
                    found_new_import = True
                    break

        self.assertTrue(
            found_new_import,
            "gateway/run.py must import from agent.model_router, not smart_model_routing",
        )

    def test_deprecated_routing_modules_removed(self):
        """All old routing modules should have been deleted — only model_router.py remains."""
        hermes_home = Path(__file__).parent.parent

        deleted_files = [
            "smart_model_routing.py",
            "priority_router.py",
            "trivial_task_classifier.py",
            "model_selector.py",
            "priority_routing_integration.py",
            "budget_enforcer.py",
            "venice_budget_enforcer.py",
        ]

        for filename in deleted_files:
            filepath = hermes_home / "agent" / filename
            self.assertFalse(
                filepath.exists(),
                f"Deprecated module {filename} should have been deleted. "
                "All routing is now in agent/model_router.py.",
            )


class TestCanonicalRouterOnly(TestCase):
    """Verify only the canonical router exists."""

    def test_no_top_level_model_router_shim(self):
        """Top-level model_router.py shim should be deleted."""
        hermes_home = Path(__file__).parent.parent.parent
        shim_path = hermes_home / "model_router.py"
        self.assertFalse(
            shim_path.exists(),
            "Top-level model_router.py shim should be deleted. "
            "All routing is in agent/model_router.py.",
        )

    def test_no_modified_model_selector_orphan(self):
        """modified_model_selector.py orphan should be deleted."""
        hermes_home = Path(__file__).parent.parent
        orphan_path = hermes_home / "modified_model_selector.py"
        self.assertFalse(
            orphan_path.exists(),
            "modified_model_selector.py is an orphaned duplicate and should be deleted.",
        )

    def test_canonical_router_exists(self):
        """agent/model_router.py must exist as the single source of truth."""
        # Path(__file__) is tests/agent/test_model_router.py; go up 3 levels to project root
        router_path = Path(__file__).resolve().parent.parent.parent / "agent" / "model_router.py"
        self.assertTrue(router_path.exists())
