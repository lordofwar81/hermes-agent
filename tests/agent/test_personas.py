"""Tests for agent/personas.py + delegate persona injection (Phase 5, Gulli Ch7).

Tests verify:
  1. Persona registry: get_persona, list_personas, resolve_persona_toolsets.
  2. _build_child_system_prompt injects persona prefix when flag on, skips when off.
  3. Unknown persona falls back gracefully (warn + no prefix).
  4. DELEGATE_TASK_SCHEMA accepts the persona field.
  5. Persona is orthogonal to role (leaf/orchestrator unaffected).
"""
from unittest.mock import patch

import pytest

from agent.personas import (
    Persona, get_persona, list_personas, resolve_persona_toolsets,
)
from agent.routing import Category  # not used directly but confirms no import cycle


# ─── Persona registry ────────────────────────────────────────────────────

class TestPersonaRegistry:
    def test_get_persona_coder(self):
        p = get_persona("coder")
        assert p is not None
        assert p.name == "coder"
        assert "senior software engineer" in p.prompt_prefix.lower()
        assert "terminal" in p.default_toolsets

    def test_get_persona_researcher(self):
        p = get_persona("researcher")
        assert p is not None
        assert "web" in p.default_toolsets

    def test_get_persona_critic(self):
        p = get_persona("critic")
        assert p is not None
        assert "evaluate" in p.prompt_prefix.lower()

    def test_get_persona_case_insensitive(self):
        assert get_persona("CODER") is not None
        assert get_persona("Researcher") is not None

    def test_get_persona_unknown_returns_none(self):
        assert get_persona("nonexistent") is None
        assert get_persona("") is None
        assert get_persona(None) is None

    def test_list_personas_has_three(self):
        names = list_personas()
        assert "coder" in names
        assert "researcher" in names
        assert "critic" in names
        assert len(names) == 3

    def test_persona_is_frozen(self):
        """Persona dataclass is frozen — can't mutate after creation."""
        p = get_persona("coder")
        with pytest.raises((AttributeError, Exception)):
            p.name = "changed"


# ─── Toolset resolution ───────────────────────────────────────────────────

class TestResolvePersonaToolsets:
    def test_caller_toolsets_win(self):
        """Explicit caller toolsets override persona defaults."""
        result = resolve_persona_toolsets("coder", caller_toolsets=["web"])
        assert result == ["web"]

    def test_persona_default_when_no_caller(self):
        result = resolve_persona_toolsets("coder", caller_toolsets=None)
        assert result == ["terminal", "file"]

    def test_none_when_no_persona_no_caller(self):
        result = resolve_persona_toolsets(None, caller_toolsets=None)
        assert result is None

    def test_none_when_unknown_persona_no_caller(self):
        result = resolve_persona_toolsets("nonexistent", caller_toolsets=None)
        assert result is None


# ─── Prompt injection ─────────────────────────────────────────────────────

class TestPromptInjection:
    def test_no_prefix_when_flag_off(self):
        """Flag off → persona is ignored, standard prompt only."""
        from tools.delegate_tool import _build_child_system_prompt
        with patch("agent.feature_flags.personas_enabled", return_value=False):
            prompt = _build_child_system_prompt("do thing", persona="coder")
        assert "senior software engineer" not in prompt
        assert "You are a focused subagent" in prompt

    def test_prefix_injected_when_flag_on(self):
        """Flag on + known persona → prefix prepended to prompt."""
        from tools.delegate_tool import _build_child_system_prompt
        with patch("agent.feature_flags.personas_enabled", return_value=True):
            prompt = _build_child_system_prompt("do thing", persona="coder")
        assert "senior software engineer" in prompt
        assert "You are a focused subagent" in prompt  # standard part still there
        # Persona prefix comes BEFORE the standard prompt.
        assert prompt.index("senior software engineer") < prompt.index("You are a focused subagent")

    def test_unknown_persona_no_prefix_even_when_flag_on(self):
        """Flag on but unknown persona → warn + no prefix, standard prompt."""
        from tools.delegate_tool import _build_child_system_prompt
        with patch("agent.feature_flags.personas_enabled", return_value=True):
            prompt = _build_child_system_prompt("do thing", persona="banana")
        assert "You are a focused subagent" in prompt  # standard prompt intact
        # No banana text leaked in.
        assert "banana" not in prompt

    def test_no_persona_no_prefix(self):
        """No persona set → standard prompt, no prefix (unchanged behavior)."""
        from tools.delegate_tool import _build_child_system_prompt
        with patch("agent.feature_flags.personas_enabled", return_value=True):
            prompt = _build_child_system_prompt("do thing", persona=None)
        assert "You are a focused subagent" in prompt

    def test_persona_with_orchestrator_role(self):
        """Persona + orchestrator role coexist — persona prefix + delegation block."""
        from tools.delegate_tool import _build_child_system_prompt
        with patch("agent.feature_flags.personas_enabled", return_value=True):
            prompt = _build_child_system_prompt(
                "do thing", role="orchestrator", persona="coder",
                max_spawn_depth=3, child_depth=1,
            )
        assert "senior software engineer" in prompt  # persona
        assert "Subagent Spawning" in prompt  # orchestrator block


# ─── Schema acceptance ────────────────────────────────────────────────────

class TestSchemaAcceptsPersona:
    def test_schema_has_persona_field(self):
        """DELEGATE_TASK_SCHEMA includes persona at top level."""
        from tools.delegate_tool import DELEGATE_TASK_SCHEMA
        props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        assert "persona" in props
        assert props["persona"]["type"] == "string"

    def test_schema_task_item_has_persona(self):
        """Per-task item in the tasks array includes persona."""
        from tools.delegate_tool import DELEGATE_TASK_SCHEMA
        task_props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]
        assert "persona" in task_props

    def test_persona_not_required(self):
        """persona is optional — required list should not contain it."""
        from tools.delegate_tool import DELEGATE_TASK_SCHEMA
        required = DELEGATE_TASK_SCHEMA["parameters"].get("required", [])
        assert "persona" not in required
