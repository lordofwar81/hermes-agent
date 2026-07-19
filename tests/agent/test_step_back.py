"""Tests for agent/step_back.py — step-back prompting (Gulli Appendix A, Tier 2).

Tests mock call_llm (no network) and verify:
  1. Step-back is skipped when the step_back flag is off.
  2. Step-back is skipped for GREETING/SIMPLE/CODE turns (category gate).
  3. Step-back fires for REASONING/ANALYSIS/EXPERT turns.
  4. Principle parsing handles valid PRINCIPLE: format.
  5. Principle parsing handles N/A (no deeper principle) → None.
  6. Failure contract: any exception → None, never raises.
"""
import os
from unittest.mock import MagicMock, patch

import pytest

from agent.routing import Category
from agent.step_back import run_step_back, _parse_step_back_response


# ─── Helpers ──────────────────────────────────────────────────────────────

def _mock_agent(provider="zai", model="glm-4.7"):
    """Build a minimal mock agent with the attributes run_step_back reads."""
    agent = MagicMock()
    agent._current_main_runtime.return_value = {
        "model": model, "provider": provider,
        "base_url": "http://test/v1", "api_key": "test-key", "api_mode": "",
    }
    return agent


def _mock_llm_response(content: str):
    mock_msg = MagicMock()
    mock_msg.message.content = content
    mock_choice = MagicMock()
    mock_choice.message = mock_msg.message
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    return mock_resp


def _patch_flag(value: bool):
    """Patch the step_back flag to the given value."""
    return patch("agent.feature_flags.step_back_enabled", return_value=value)


def _patch_classify(category: Category):
    """Patch TaskClassifier.classify to always return the given category."""
    return patch("agent.routing.TaskClassifier.classify", return_value=category)


def _patch_semantic_off():
    """Force the semantic classifier off so the keyword classify mock is used."""
    return patch("agent.feature_flags.semantic_classifier_enabled",
                 return_value=False)


PRINCIPLE_RESPONSE = """\
PRINCIPLE:
The time value of money: a dollar today is worth more than a dollar tomorrow \
due to its earning capacity."""

NA_RESPONSE = "PRINCIPLE:\nN/A"


# ─── Flag gating ──────────────────────────────────────────────────────────

class TestStepBackFlagGate:
    def test_skipped_when_flag_off(self):
        with _patch_flag(False), _patch_classify(Category.ANALYSIS), \
             _patch_semantic_off(), \
             patch("agent.auxiliary_client.call_llm") as mock_call:
            result = run_step_back(_mock_agent(), "complex question")
        assert result is None
        mock_call.assert_not_called()

    def test_fires_when_flag_on(self):
        with _patch_flag(True), _patch_classify(Category.ANALYSIS), \
             _patch_semantic_off(), \
             patch("agent.auxiliary_client.call_llm",
                   return_value=_mock_llm_response(PRINCIPLE_RESPONSE)) as mock_call:
            result = run_step_back(_mock_agent(), "complex question")
        assert result is not None
        assert "time value of money" in result.lower()
        mock_call.assert_called_once()


# ─── Category gating ──────────────────────────────────────────────────────

class TestStepBackCategoryGate:
    def test_skipped_for_greeting(self):
        with _patch_flag(True), _patch_classify(Category.GREETING), \
             _patch_semantic_off(), \
             patch("agent.auxiliary_client.call_llm") as mock_call:
            result = run_step_back(_mock_agent(), "hi")
        assert result is None
        mock_call.assert_not_called()

    def test_skipped_for_simple(self):
        with _patch_flag(True), _patch_classify(Category.SIMPLE), \
             _patch_semantic_off(), \
             patch("agent.auxiliary_client.call_llm") as mock_call:
            result = run_step_back(_mock_agent(), "what is 2+2")
        assert result is None
        mock_call.assert_not_called()

    def test_skipped_for_code(self):
        with _patch_flag(True), _patch_classify(Category.CODE), \
             _patch_semantic_off(), \
             patch("agent.auxiliary_client.call_llm") as mock_call:
            result = run_step_back(_mock_agent(), "write a function")
        assert result is None
        mock_call.assert_not_called()

    def test_fires_for_reasoning(self):
        with _patch_flag(True), _patch_classify(Category.REASONING), \
             _patch_semantic_off(), \
             patch("agent.auxiliary_client.call_llm",
                   return_value=_mock_llm_response(PRINCIPLE_RESPONSE)):
            result = run_step_back(_mock_agent(), "why does X happen")
        assert result is not None

    def test_fires_for_analysis(self):
        with _patch_flag(True), _patch_classify(Category.ANALYSIS), \
             _patch_semantic_off(), \
             patch("agent.auxiliary_client.call_llm",
                   return_value=_mock_llm_response(PRINCIPLE_RESPONSE)):
            result = run_step_back(_mock_agent(), "analyze X")
        assert result is not None

    def test_fires_for_expert(self):
        with _patch_flag(True), _patch_classify(Category.EXPERT), \
             _patch_semantic_off(), \
             patch("agent.auxiliary_client.call_llm",
                   return_value=_mock_llm_response(PRINCIPLE_RESPONSE)):
            result = run_step_back(_mock_agent(), "design X")
        assert result is not None


# ─── Response parsing ─────────────────────────────────────────────────────

class TestStepBackParsing:
    def test_parse_valid_principle(self):
        result = _parse_step_back_response(PRINCIPLE_RESPONSE)
        assert result is not None
        assert "time value of money" in result.lower()

    def test_parse_na_returns_none(self):
        result = _parse_step_back_response(NA_RESPONSE)
        assert result is None

    def test_parse_empty_returns_none(self):
        assert _parse_step_back_response("") is None
        assert _parse_step_back_response(None) is None

    def test_parse_no_marker_returns_none(self):
        result = _parse_step_back_response("just some text without the marker")
        assert result is None

    def test_parse_principle_on_same_line(self):
        result = _parse_step_back_response("PRINCIPLE: Supply and demand curves intersect at equilibrium.")
        assert result is not None
        assert "equilibrium" in result.lower()

    def test_parse_multiline_principle(self):
        resp = "PRINCIPLE:\nFirst sentence.\nSecond sentence."
        result = _parse_step_back_response(resp)
        assert result is not None
        assert "First sentence" in result
        assert "Second sentence" in result


# ─── Failure contract ─────────────────────────────────────────────────────

class TestStepBackFailureContract:
    def test_llm_exception_returns_none(self):
        with _patch_flag(True), _patch_classify(Category.ANALYSIS), \
             _patch_semantic_off(), \
             patch("agent.auxiliary_client.call_llm",
                   side_effect=RuntimeError("network down")):
            result = run_step_back(_mock_agent(), "complex question")
        assert result is None  # never raises, returns None

    def test_empty_user_message_returns_none(self):
        with _patch_flag(True):
            result = run_step_back(_mock_agent(), "")
        assert result is None

    def test_whitespace_user_message_returns_none(self):
        with _patch_flag(True):
            result = run_step_back(_mock_agent(), "   ")
        assert result is None

    def test_na_response_returns_none(self):
        """When the aux model says N/A (no deeper principle), skip injection."""
        with _patch_flag(True), _patch_classify(Category.ANALYSIS), \
             _patch_semantic_off(), \
             patch("agent.auxiliary_client.call_llm",
                   return_value=_mock_llm_response(NA_RESPONSE)):
            result = run_step_back(_mock_agent(), "what year did X happen")
        assert result is None
