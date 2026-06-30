"""Tests for _repair_tool_call_arguments — malformed JSON repair pipeline."""

import json

from run_agent import _repair_tool_call_arguments


class TestRepairToolCallArguments:
    """Verify each repair stage in the pipeline."""

    # -- Stage 1: empty / whitespace-only --

    def test_empty_string_returns_empty_object(self):
        assert _repair_tool_call_arguments("", "t") == "{}"

    def test_whitespace_only_returns_empty_object(self):
        assert _repair_tool_call_arguments("   \n\t  ", "t") == "{}"

    def test_none_type_returns_empty_object(self):
        """Non-string input (e.g. None from a broken model response)."""
        assert _repair_tool_call_arguments(None, "t") == "{}"

    # -- Stage 2: Python None literal --

    def test_python_none_literal(self):
        assert _repair_tool_call_arguments("None", "t") == "{}"

    def test_python_none_with_whitespace(self):
        assert _repair_tool_call_arguments("  None  ", "t") == "{}"

    # -- Stage 3: trailing comma repair --

    def test_trailing_comma_in_object(self):
        result = _repair_tool_call_arguments('{"key": "value",}', "t")
        assert json.loads(result) == {"key": "value"}

    def test_trailing_comma_in_array(self):
        result = _repair_tool_call_arguments('{"a": [1, 2,]}', "t")
        parsed = json.loads(result)
        assert parsed == {"a": [1, 2]}

    def test_multiple_trailing_commas(self):
        result = _repair_tool_call_arguments('{"a": 1, "b": 2,}', "t")
        parsed = json.loads(result)
        assert parsed["a"] == 1
        assert parsed["b"] == 2

    # -- Stage 4: unclosed brackets --

    def test_unclosed_brace(self):
        result = _repair_tool_call_arguments('{"key": "value"', "t")
        parsed = json.loads(result)
        assert parsed == {"key": "value"}

    def test_unclosed_bracket_and_brace(self):
        result = _repair_tool_call_arguments('{"a": [1, 2', "t")
        # Bracket counting adds ']' then '}', producing {"a": [1, 2]}
        # which is valid JSON.  But the naive count can't always recover
        # complex nesting — verify we at least get valid JSON.
        json.loads(result)

    # -- Stage 5: excess closing delimiters --

    def test_extra_closing_brace(self):
        result = _repair_tool_call_arguments('{"key": "value"}}', "t")
        parsed = json.loads(result)
        assert parsed == {"key": "value"}

    def test_extra_closing_bracket(self):
        result = _repair_tool_call_arguments('{"a": [1]]}', "t")
        # Should produce valid JSON
        json.loads(result)

    # -- Stage 6: last resort --

    def test_unrepairable_garbage_returns_empty_object(self):
        assert _repair_tool_call_arguments("totally not json", "t") == "{}"

    def test_unrepairable_partial_returns_empty_object(self):
        # Truncated in the middle of a string key — bracket closing won't help
        assert _repair_tool_call_arguments('{"truncated": "val', "t") == "{}"

    # -- Valid JSON passthrough (this path is via except, but still works) --

    def test_already_valid_json_passes_through(self):
        """When json.loads fails for a non-JSON reason (shouldn't normally
        happen), but the repair pipeline still produces valid output."""
        raw = '{"path": "/tmp/foo", "content": "hello"}'
        result = _repair_tool_call_arguments(raw, "t")
        parsed = json.loads(result)
        assert parsed["path"] == "/tmp/foo"

    # -- Combined repairs --

    def test_trailing_comma_plus_unclosed_brace(self):
        result = _repair_tool_call_arguments('{"a": 1, "b": 2,', "t")
        # Trailing comma stripped first, then closing brace added.
        # May or may not fully recover — verify valid JSON at minimum.
        json.loads(result)

    def test_real_world_glm_truncation(self):
        """Simulates GLM-5.1 truncating mid-argument."""
        raw = '{"command": "ls -la /tmp", "timeout": 30, "background":'
        result = _repair_tool_call_arguments(raw, "terminal")
        # Should at least be valid JSON, even if background is lost
        json.loads(result)

    # -- Stage 0: strict=False (literal control chars in strings) --
    # llama.cpp backends sometimes emit literal tabs/newlines inside JSON
    # string values. strict=False accepts these; we re-serialise to the
    # canonical wire form (#12068).

    def test_literal_newline_inside_string_value(self):
        raw = '{"summary": "line one\nline two"}'
        result = _repair_tool_call_arguments(raw, "t")
        parsed = json.loads(result)
        assert parsed == {"summary": "line one\nline two"}

    def test_literal_tab_inside_string_value(self):
        raw = '{"summary": "col1\tcol2"}'
        result = _repair_tool_call_arguments(raw, "t")
        parsed = json.loads(result)
        assert parsed == {"summary": "col1\tcol2"}

    def test_literal_control_char_reserialised_to_wire_form(self):
        """After repair, the output must parse under strict=True."""
        raw = '{"msg": "has\tliteral\ttabs"}'
        result = _repair_tool_call_arguments(raw, "t")
        # strict=True must now accept this
        parsed = json.loads(result)
        assert parsed["msg"] == "has\tliteral\ttabs"

    # -- Stage 4: control-char escape fallback --

    def test_control_chars_with_trailing_comma(self):
        """strict=False fails due to trailing comma, but brace-count pass
        + control-char escape rescues it."""
        raw = '{"msg": "line\none",}'
        result = _repair_tool_call_arguments(raw, "t")
        parsed = json.loads(result)
        assert "line" in parsed["msg"]

    # -- Stage 7: concatenated JSON objects --
    # GLM-5.x (z.ai gateway) and local Gemma sometimes emit two-or-more
    # complete JSON objects smashed together for a single tool_call.
    # Without this repair the whole blob is nuked to {} (see errors.log
    # "Unrepairable tool_call arguments"). The merge pass parses each
    # object and merges into one dict (later keys win).

    def test_two_objects_concatenated_merged(self):
        raw = '{"path": "/x"}{"command": "ls"}'
        result = _repair_tool_call_arguments(raw, "terminal")
        parsed = json.loads(result)
        assert parsed == {"path": "/x", "command": "ls"}

    def test_empty_prefix_object_dropped(self):
        raw = '{}{"limit": "5"}'
        result = _repair_tool_call_arguments(raw, "terminal")
        parsed = json.loads(result)
        assert parsed == {"limit": "5"}

    def test_empty_suffix_object_dropped(self):
        raw = '{"command": "ls"}{}'
        result = _repair_tool_call_arguments(raw, "terminal")
        parsed = json.loads(result)
        assert parsed == {"command": "ls"}

    def test_later_duplicate_key_wins(self):
        raw = '{"timeout": 15}{"timeout": 60}'
        result = _repair_tool_call_arguments(raw, "terminal")
        parsed = json.loads(result)
        assert parsed == {"timeout": 60}

    def test_three_objects_concatenated(self):
        raw = '{"a": 1}{"b": 2}{"c": 3}'
        result = _repair_tool_call_arguments(raw, "t")
        assert json.loads(result) == {"a": 1, "b": 2, "c": 3}

    def test_real_world_skill_patrol_concat(self):
        """Exact shape from errors.log — two full objects concatenated."""
        raw = '{"path": "~/.hermes/loops/skill-health-patrol/STATE.md"}{"command": "python3 ~/.hermes/scripts/x.py"}'
        result = _repair_tool_call_arguments(raw, "terminal")
        parsed = json.loads(result)
        assert parsed["path"] == "~/.hermes/loops/skill-health-patrol/STATE.md"
        assert parsed["command"] == "python3 ~/.hermes/scripts/x.py"

    def test_concat_with_control_chars_in_strings(self):
        """A literal newline inside one object's string value must not
        confuse the brace-depth walker."""
        raw = '{"msg": "line\none"}{"x": 2}'
        result = _repair_tool_call_arguments(raw, "t")
        parsed = json.loads(result)
        assert parsed == {"msg": "line\none", "x": 2}

    def test_concat_objects_with_escaped_braces_in_strings(self):
        """Brace characters inside JSON string values must be ignored
        by the object-boundary walker."""
        raw = '{"cmd": "echo {\\"k\\":1}"}{"path": "/x"}'
        result = _repair_tool_call_arguments(raw, "t")
        parsed = json.loads(result)
        assert parsed["path"] == "/x"
        assert parsed["cmd"] == 'echo {"k":1}'

    def test_single_valid_object_not_treated_as_concat(self):
        """A lone valid object must pass through the strict=False path,
        NOT the concat path (otherwise we'd churn valid calls)."""
        raw = '{"path": "/tmp/foo"}'
        result = _repair_tool_call_arguments(raw, "read_file")
        assert json.loads(result) == {"path": "/tmp/foo"}

