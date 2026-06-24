"""Tests for agent/planning_gate.py — the planning gate (Phase 4, Gulli Ch6).

Tests mock call_llm (no network) and verify:
  1. Plan is skipped when the planning flag is off.
  2. Plan is skipped for GREETING/SIMPLE/REASONING turns (category gate).
  3. Plan fires for EXPERT/CODE/ANALYSIS turns.
  4. Plan parsing handles valid PLAN:/DONE_CRITERIA: format.
  5. Plan parsing handles edge cases (single step, missing criteria).
  6. Failure contract: any exception → None, never raises.
"""
from unittest.mock import MagicMock, patch

import pytest

from agent.routing import Category
from agent.planning_gate import build_plan, _parse_plan, Plan


# ─── Helpers ──────────────────────────────────────────────────────────────

def _mock_agent():
    agent = MagicMock()
    agent._current_main_runtime.return_value = {
        "model": "glm-5-turbo", "provider": "zai",
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
    return patch("agent.feature_flags.planning_gate_enabled", return_value=value)


def _patch_classify(category: Category):
    return patch("agent.routing.TaskClassifier.classify", return_value=category)


VALID_PLAN_RESPONSE = """\
PLAN:
1. Research the existing rate limiting approaches
2. Choose between token bucket and sliding window
3. Implement the chosen algorithm in Redis
4. Write tests for the rate limiter
5. Add monitoring and alerting
DONE_CRITERIA:
- Rate limiter correctly limits requests per window
- Tests pass for all edge cases
- Monitoring dashboard shows rate limit hits"""

SINGLE_STEP_RESPONSE = """\
PLAN:
1. Look up the capital of France
DONE_CRITERIA:
- Answer is correct"""


# ─── Flag gating ──────────────────────────────────────────────────────────

class TestPlanFlagGate:
    def test_skipped_when_flag_off(self):
        agent = _mock_agent()
        with _patch_flag(False), _patch_classify(Category.EXPERT):
            with patch("agent.auxiliary_client.call_llm") as mock_call:
                result = build_plan(agent, "design a system")
        assert result is None
        mock_call.assert_not_called()

    def test_flag_module_unavailable_skips(self):
        agent = _mock_agent()
        with patch("agent.feature_flags.planning_gate_enabled",
                   side_effect=ImportError("no module")):
            result = build_plan(agent, "design a system")
        assert result is None


# ─── Category gating ──────────────────────────────────────────────────────

class TestPlanCategoryGate:
    def test_skipped_for_greeting(self):
        agent = _mock_agent()
        with _patch_flag(True), _patch_classify(Category.GREETING):
            with patch("agent.auxiliary_client.call_llm") as mock_call:
                result = build_plan(agent, "hello")
        assert result is None
        mock_call.assert_not_called()

    def test_skipped_for_simple(self):
        agent = _mock_agent()
        with _patch_flag(True), _patch_classify(Category.SIMPLE):
            with patch("agent.auxiliary_client.call_llm") as mock_call:
                result = build_plan(agent, "what is 2+2")
        assert result is None
        mock_call.assert_not_called()

    def test_skipped_for_reasoning(self):
        """REASONING turns are explanatory — no multi-step execution to plan."""
        agent = _mock_agent()
        with _patch_flag(True), _patch_classify(Category.REASONING):
            with patch("agent.auxiliary_client.call_llm") as mock_call:
                result = build_plan(agent, "explain how X works")
        assert result is None
        mock_call.assert_not_called()


# ─── Plan fires + output ──────────────────────────────────────────────────

class TestPlanFires:
    @pytest.mark.parametrize("category", [Category.EXPERT, Category.CODE, Category.ANALYSIS])
    def test_fires_for_eligible_categories(self, category):
        agent = _mock_agent()
        with _patch_flag(True), _patch_classify(category):
            with patch("agent.auxiliary_client.call_llm",
                       return_value=_mock_llm_response(VALID_PLAN_RESPONSE)) as mock_call:
                result = build_plan(agent, "test question")
        assert result is not None
        assert len(result.steps) == 5
        assert len(result.done_criteria) == 3
        mock_call.assert_called_once()

    def test_plan_surfaced_to_user(self):
        """When a plan is generated, it's buffered to the status output."""
        agent = _mock_agent()
        with _patch_flag(True), _patch_classify(Category.EXPERT):
            with patch("agent.auxiliary_client.call_llm",
                       return_value=_mock_llm_response(VALID_PLAN_RESPONSE)):
                build_plan(agent, "design a system")
        agent._buffer_status.assert_called_once()
        rendered = agent._buffer_status.call_args[0][0]
        assert "Plan" in rendered or "📋" in rendered

    def test_single_step_plan(self):
        """A simple task gets a 1-step plan."""
        agent = _mock_agent()
        with _patch_flag(True), _patch_classify(Category.CODE):
            with patch("agent.auxiliary_client.call_llm",
                       return_value=_mock_llm_response(SINGLE_STEP_RESPONSE)):
                result = build_plan(agent, "look up capital")
        assert result is not None
        assert len(result.steps) == 1

    def test_plan_render_format(self):
        plan = Plan(steps=["step 1", "step 2"], done_criteria=["criterion"])
        rendered = plan.render()
        assert "1." in rendered
        assert "step 1" in rendered
        assert "step 2" in rendered
        assert "criterion" in rendered


# ─── Failure contract ─────────────────────────────────────────────────────

class TestPlanFailureContract:
    def test_llm_exception_returns_none(self):
        agent = _mock_agent()
        with _patch_flag(True), _patch_classify(Category.EXPERT):
            with patch("agent.auxiliary_client.call_llm",
                       side_effect=RuntimeError("network down")):
                result = build_plan(agent, "design a system")
        assert result is None

    def test_unparseable_response_returns_none(self):
        agent = _mock_agent()
        with _patch_flag(True), _patch_classify(Category.EXPERT):
            with patch("agent.auxiliary_client.call_llm",
                       return_value=_mock_llm_response("banana")):
                result = build_plan(agent, "design a system")
        assert result is None

    def test_empty_message_returns_none(self):
        agent = _mock_agent()
        with _patch_flag(True), _patch_classify(Category.EXPERT):
            with patch("agent.auxiliary_client.call_llm") as mock_call:
                result = build_plan(agent, "")
        assert result is None
        mock_call.assert_not_called()


# ─── Response parsing ─────────────────────────────────────────────────────

class TestParsePlan:
    def test_valid_plan_with_criteria(self):
        result = _parse_plan(VALID_PLAN_RESPONSE)
        assert result is not None
        assert len(result.steps) == 5
        assert len(result.done_criteria) == 3
        assert "Research" in result.steps[0]
        assert "Tests pass" in result.done_criteria[1]

    def test_plan_without_criteria_section(self):
        """If DONE_CRITERIA is missing, steps are still parsed."""
        result = _parse_plan("PLAN:\n1. Do thing\n2. Do other thing")
        assert result is not None
        assert len(result.steps) == 2
        assert len(result.done_criteria) == 0

    def test_no_plan_marker_returns_none(self):
        assert _parse_plan("random text without plan marker") is None
        assert _parse_plan("") is None
        assert _parse_plan(None) is None

    def test_bullets_instead_of_numbers(self):
        """Parser handles bulleted steps, not just numbered."""
        result = _parse_plan("PLAN:\n- First step\n- Second step")
        assert result is not None
        assert len(result.steps) == 2
        assert "First step" in result.steps[0]
