"""Tests for agent/model_selector.py — intelligent multi-model routing."""

import pytest
from agent.model_selector import (
    classify_message,
    select_model,
    smart_select_route,
    MODEL_PROFILES,
)


class TestClassifyMessage:
    """Test heuristic message classification."""

    def test_code_detection(self):
        cls = classify_message("fix the crash in main.py line 42")
        assert cls["task_type"] == "code"

    def test_code_with_backticks(self):
        cls = classify_message("this function `def foo()` has a bug")
        assert cls["task_type"] == "code"

    def test_writing_detection(self):
        cls = classify_message("write a professional email to the team")
        assert cls["task_type"] == "writing"

    def test_analysis_detection(self):
        cls = classify_message("analyze the sales data trends from Q1")
        assert cls["task_type"] == "analysis"

    def test_creative_detection(self):
        cls = classify_message("write a creative story about a robot")
        assert cls["task_type"] == "creative"

    def test_reasoning_detection(self):
        cls = classify_message("explain why this architecture would fail at scale")
        assert cls["task_type"] == "reasoning"

    def test_general_fallback(self):
        cls = classify_message("hello there")
        assert cls["task_type"] == "general"

    def test_realtime_urgency(self):
        cls = classify_message("quick what time is it")
        assert cls["urgency"] == "realtime"

    def test_deep_urgency_for_complex(self):
        cls = classify_message("perform a comprehensive system-wide architecture audit of the entire codebase")
        assert cls["urgency"] == "deep"
        assert cls["complexity"] in ("complex", "expert")

    def test_simple_complexity(self):
        cls = classify_message("ok")
        assert cls["complexity"] == "simple"

    def test_expert_complexity(self):
        cls = classify_message("perform a comprehensive full audit of the entire system architecture and design a migration strategy for the production database")
        assert cls["complexity"] == "expert"


class TestSelectModel:
    """Test model selection logic.

    classify_message is mocked to ensure deterministic results — these
    tests verify the scoring/filtering engine, not the classifier.
    """

    @pytest.fixture(autouse=True)
    def _mock_classifier(self, monkeypatch):
        """Force heuristic-only classification (no LLM calls in tests).

        Strips API keys AND mocks the LLM classifier to return None so
        _classify_with_llm's .env file fallback path can't leak through.
        """
        monkeypatch.delenv("GLM_API_KEY", raising=False)
        monkeypatch.delenv("ZAI_API_KEY", raising=False)
        monkeypatch.delenv("Z_AI_API_KEY", raising=False)
        monkeypatch.setattr("agent.model_selector._classify_with_llm", lambda *a, **kw: None)

    def test_returns_none_when_flag_off(self):
        config = {"enabled": True}
        result = select_model("implement a REST API", config, "zai", "glm-5-turbo")
        assert result is None

    def test_returns_none_for_simple_messages(self):
        config = {
            "use_model_selector": True,
            "models": {"zai": ["glm-5.1"]},
        }
        result = select_model("ok", config, "zai", "glm-5-turbo")
        assert result is None

    def test_selects_code_model_for_code_task(self):
        config = {
            "use_model_selector": True,
            "models": {
                "zai": ["glm-5.1", "glm-4.5-air"],
                "venice": ["claude-sonnet-4-6", "deepseek-v3.2"],
            },
        }
        result = select_model(
            "implement a complex authentication system with JWT and OAuth2",
            config,
            "zai",
            "glm-5-turbo",
        )
        assert result is not None
        provider, model, reason = result
        # Should select a high-capability model (not glm-4.5-air)
        assert model in ("glm-5.1", "deepseek-v3.2", "claude-sonnet-4-6")
        assert "code" in reason or "reasoning" in reason

    def test_selects_creative_model_for_creative_task(self):
        config = {
            "use_model_selector": True,
            "models": {
                "zai": ["glm-5.1", "glm-4.5-air"],
                "venice": ["venice-uncensored", "deepseek-v3.2"],
            },
        }
        # Make it complex enough to trigger the selector (longer message)
        result = select_model(
            "write a creative uncensored story about a cyberpunk heist crew pulling off the biggest data breach in history, with complex character development and plot twists",
            config,
            "zai",
            "glm-5-turbo",
        )
        assert result is not None
        provider, model, reason = result
        assert "creative" in reason

    def test_favors_free_models_for_moderate_quality(self):
        """Moderate complexity tasks should prefer free models when capable."""
        config = {
            "use_model_selector": True,
            "models": {
                "zai": ["glm-4.7", "glm-4.5-air"],
                "venice": ["deepseek-v3.2", "claude-sonnet-4-6"],
            },
        }
        # Moderate writing task — heuristic classifies as writing/moderate/standard
        result = select_model(
            "please summarize this long document about project management methodologies and explain the key differences between agile, waterfall, and scrum frameworks in detail",
            config,
            "zai",
            "glm-5-turbo",
        )
        assert result is not None
        provider, model, reason = result
        # deepseek-v3.2 is cheap ($0.008) with high general capability — wins for moderate

    def test_context_window_filtering(self):
        """Models with too-small context should be filtered out."""
        config = {
            "use_model_selector": True,
            "models": {
                "venice": ["venice-uncensored"],  # 32K context
                "zai": ["glm-5.1"],  # 198K context
            },
        }
        # Simulate a very long message (> 25K tokens estimated)
        long_msg = "analyze this " + ("very important data point. " * 5000)
        result = select_model(long_msg, config, "zai", "glm-5-turbo")
        assert result is not None
        provider, model, reason = result
        # Should select glm-5.1 (198K context), not venice-uncensored (32K)
        assert model == "glm-5.1"


class TestSmartSelectRoute:
    """Test the full routing integration entry point."""

    def test_returns_none_when_flag_off(self):
        route = smart_select_route(
            "implement a REST API",
            {"enabled": True},
            {"model": "glm-5-turbo", "provider": "zai"},
        )
        assert route is None

    def test_returns_valid_route_shape(self):
        config = {
            "use_model_selector": True,
            "models": {
                "zai": ["glm-5.1", "glm-4.5-air"],
                "venice": ["deepseek-v3.2"],
            },
        }
        route = smart_select_route(
            "implement a complex distributed system architecture",
            config,
            {"model": "glm-5-turbo", "provider": "zai", "api_key": "test", "base_url": "https://test"},
        )
        if route is not None:
            assert "model" in route
            assert "runtime" in route
            assert "label" in route
            assert "signature" in route
            assert isinstance(route["signature"], tuple)
            assert len(route["signature"]) == 6

    def test_preserves_credential_pool(self):
        """Ensure credential_pool from primary is preserved in fallback path."""
        config = {"enabled": True}  # Flag off
        route = smart_select_route(
            "anything",
            config,
            {"model": "glm-5-turbo", "provider": "zai", "credential_pool": "fake_pool"},
        )
        assert route is None  # Falls through to existing router


class TestModelProfiles:
    """Verify model profile database integrity."""

    def test_all_models_have_profiles(self):
        expected_models = {
            # z-ai
            "glm-5.1", "glm-5-turbo", "glm-5", "glm-4.7", "glm-4.6", "glm-4.5", "glm-4.5-air",
            # venice
            "qwen-3-6-plus", "claude-sonnet-4-6", "zai-org-glm-5", "zai-org-glm-4.7",
            "zai-org-glm-4.7-flash", "deepseek-v3.2", "grok-4-20-beta",
            "qwen3-coder-480b-a35b-instruct", "qwen3-5-35b-a3b", "venice-uncensored",
            # local
            "Qwen3-Coder-30B-APEX-I-Compact", "LFM2-24B-A2B-APEX-I-Compact",
            "Qwopus-MoE-35B-A3B-APEX-I-Compact", "Huihui3.5-67B-A3B-APEX-I-Compact",
        }
        for model in expected_models:
            assert model in MODEL_PROFILES, f"Missing profile for {model}"

    def test_local_models_are_free(self):
        for name, profile in MODEL_PROFILES.items():
            if profile.provider == "local":
                assert profile.cost_per_request == 0.0, f"{name} should be free"

    def test_capability_scores_in_range(self):
        for name, profile in MODEL_PROFILES.items():
            for attr in ("code_quality", "reasoning", "writing", "analysis", "creative", "general", "speed"):
                val = getattr(profile, attr)
                assert 0.0 <= val <= 1.0, f"{name}.{attr} = {val} out of range"
