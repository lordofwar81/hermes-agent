"""Tests for agent/reflection.py — the pre-delivery critic (Phase 3).

Tests mock call_llm (no network) and verify:
  1. Critic is skipped when the reflection flag is off.
  2. Critic is skipped for GREETING/SIMPLE/CODE turns (category gate).
  3. Critic fires for REASONING/ANALYSIS/EXPERT turns.
  4. Revise-once: ACCEPT → original returned; REVISE → revised returned.
  5. Failure contract: any exception → original returned, never raises.
  6. Response parsing handles ACCEPT, REVISE, and unparseable.
"""
import os
from unittest.mock import MagicMock, patch

import pytest

from agent.routing import Category
from agent.reflection import run_critic, _parse_critic_response


# ─── Helpers ──────────────────────────────────────────────────────────────

def _mock_agent(provider="zai", model="glm-4.5-flash"):
    """Build a minimal mock agent with the attributes run_critic reads."""
    agent = MagicMock()
    agent._current_main_runtime.return_value = {
        "model": model, "provider": provider,
        "base_url": "http://test/v1", "api_key": "test-key", "api_mode": "",
    }
    agent._buffer_status = MagicMock()
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
    """Patch the reflection flag to the given value."""
    return patch("agent.feature_flags.reflection_enabled",
                 return_value=value)


def _patch_classify(category: Category):
    """Patch TaskClassifier.classify to always return the given category."""
    return patch("agent.routing.TaskClassifier.classify",
                 return_value=category)


ACCEPT_RESPONSE = """\
VERDICT: ACCEPT"""

REVISE_RESPONSE = """\
VERDICT: REVISE
REVISED_ANSWER:
This is the revised answer. It is better than the original because it \
addresses the completeness issue the original had."""

ORIGINAL_ANSWER = "This is the original answer from the main model."


# ─── Flag gating ──────────────────────────────────────────────────────────

class TestCriticFlagGate:
    def test_skipped_when_flag_off(self):
        agent = _mock_agent()
        with _patch_flag(False), _patch_classify(Category.EXPERT):
            with patch("agent.auxiliary_client.call_llm") as mock_call:
                result = run_critic(agent, ORIGINAL_ANSWER, "design a system")
        assert result == ORIGINAL_ANSWER
        mock_call.assert_not_called()  # never made the LLM call

    def test_flag_module_unavailable_skips_critic(self):
        """If feature_flags can't be imported → original returned, no raise."""
        agent = _mock_agent()
        with patch("agent.feature_flags.reflection_enabled",
                   side_effect=ImportError("no module")):
            result = run_critic(agent, ORIGINAL_ANSWER, "design a system")
        assert result == ORIGINAL_ANSWER


# ─── Category gating ──────────────────────────────────────────────────────

class TestCriticCategoryGate:
    def test_skipped_for_greeting(self):
        agent = _mock_agent()
        with _patch_flag(True), _patch_classify(Category.GREETING):
            with patch("agent.auxiliary_client.call_llm") as mock_call:
                result = run_critic(agent, ORIGINAL_ANSWER, "hello")
        assert result == ORIGINAL_ANSWER
        mock_call.assert_not_called()

    def test_skipped_for_simple(self):
        agent = _mock_agent()
        with _patch_flag(True), _patch_classify(Category.SIMPLE):
            with patch("agent.auxiliary_client.call_llm") as mock_call:
                result = run_critic(agent, ORIGINAL_ANSWER, "what is 2+2")
        assert result == ORIGINAL_ANSWER
        mock_call.assert_not_called()

    def test_skipped_for_code(self):
        """CODE turns skip the critic (correctness depends on execution)."""
        agent = _mock_agent()
        with _patch_flag(True), _patch_classify(Category.CODE):
            with patch("agent.auxiliary_client.call_llm") as mock_call:
                result = run_critic(agent, ORIGINAL_ANSWER, "fix the bug")
        assert result == ORIGINAL_ANSWER
        mock_call.assert_not_called()


# ─── Critic fires + revise-once ───────────────────────────────────────────

class TestCriticFires:
    @pytest.mark.parametrize("category", [Category.EXPERT, Category.REASONING, Category.ANALYSIS])
    def test_fires_for_eligible_categories(self, category):
        agent = _mock_agent()
        with _patch_flag(True), _patch_classify(category):
            with patch("agent.auxiliary_client.call_llm",
                       return_value=_mock_llm_response(REVISE_RESPONSE)) as mock_call:
                result = run_critic(agent, ORIGINAL_ANSWER, "test question")
        assert result != ORIGINAL_ANSWER
        assert "revised answer" in result.lower()
        mock_call.assert_called_once()  # exactly one LLM call (revise-once)

    def test_accept_returns_original(self):
        agent = _mock_agent()
        with _patch_flag(True), _patch_classify(Category.EXPERT):
            with patch("agent.auxiliary_client.call_llm",
                       return_value=_mock_llm_response(ACCEPT_RESPONSE)):
                result = run_critic(agent, ORIGINAL_ANSWER, "design a system")
        assert result == ORIGINAL_ANSWER

    def test_revise_returns_revised_text(self):
        agent = _mock_agent()
        with _patch_flag(True), _patch_classify(Category.EXPERT):
            with patch("agent.auxiliary_client.call_llm",
                       return_value=_mock_llm_response(REVISE_RESPONSE)):
                result = run_critic(agent, ORIGINAL_ANSWER, "design a system")
        assert "This is the revised answer" in result

    def test_revise_emits_status_buffer(self):
        """When the critic revises, a status message is buffered for the user."""
        agent = _mock_agent()
        with _patch_flag(True), _patch_classify(Category.EXPERT):
            with patch("agent.auxiliary_client.call_llm",
                       return_value=_mock_llm_response(REVISE_RESPONSE)):
                run_critic(agent, ORIGINAL_ANSWER, "design a system")
        agent._buffer_status.assert_called_once()
        status_arg = agent._buffer_status.call_args[0][0]
        assert "revised" in status_arg.lower() or "critic" in status_arg.lower()


# ─── Failure contract ─────────────────────────────────────────────────────

class TestCriticFailureContract:
    def test_llm_exception_returns_original(self):
        agent = _mock_agent()
        with _patch_flag(True), _patch_classify(Category.EXPERT):
            with patch("agent.auxiliary_client.call_llm",
                       side_effect=RuntimeError("network down")):
                result = run_critic(agent, ORIGINAL_ANSWER, "design a system")
        assert result == ORIGINAL_ANSWER

    def test_import_error_returns_original(self):
        agent = _mock_agent()
        with _patch_flag(True), _patch_classify(Category.EXPERT):
            with patch("agent.auxiliary_client.call_llm",
                       side_effect=ImportError("no module")):
                result = run_critic(agent, ORIGINAL_ANSWER, "design a system")
        assert result == ORIGINAL_ANSWER

    def test_empty_response_returns_original(self):
        agent = _mock_agent()
        with _patch_flag(True), _patch_classify(Category.EXPERT):
            with patch("agent.auxiliary_client.call_llm",
                       return_value=_mock_llm_response("")):
                result = run_critic(agent, ORIGINAL_ANSWER, "design a system")
        assert result == ORIGINAL_ANSWER

    def test_empty_final_response_skips_critic(self):
        """No point critiquing an empty answer."""
        agent = _mock_agent()
        with _patch_flag(True), _patch_classify(Category.EXPERT):
            with patch("agent.auxiliary_client.call_llm") as mock_call:
                result = run_critic(agent, "", "design a system")
        assert result == ""
        mock_call.assert_not_called()


# ─── Response parsing ─────────────────────────────────────────────────────

class TestParseCriticResponse:
    def test_accept_returns_none(self):
        assert _parse_critic_response("VERDICT: ACCEPT") is None

    def test_revise_returns_text(self):
        result = _parse_critic_response(REVISE_RESPONSE)
        assert result is not None
        assert "revised answer" in result.lower()

    def test_unparseable_returns_none(self):
        """Garbage response → None (treated as accept, don't block delivery)."""
        assert _parse_critic_response("banana") is None
        assert _parse_critic_response("") is None
        assert _parse_critic_response(None) is None

    def test_revise_without_answer_text_returns_none(self):
        """REVISE but no REVISED_ANSWER body → None (can't use empty revision)."""
        assert _parse_critic_response("VERDICT: REVISE\nREVISED_ANSWER:\n") is None

    def test_accept_case_insensitive(self):
        assert _parse_critic_response("verdict: accept") is None
        assert _parse_critic_response("Verdict: Accept") is None
