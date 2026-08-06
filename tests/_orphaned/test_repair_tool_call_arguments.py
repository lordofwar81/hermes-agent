"""Tests for _repair_tool_call_arguments — malformed JSON repair pipeline."""

import json

from run_agent import _repair_tool_call_arguments


class TestRepairToolCallArguments:
    """Verify each repair stage in the pipeline."""

    # -- Stage 1: empty / whitespace-only --

    def test_empty_string_returns_empty_object(self):
        assert _repair_tool_call_arguments("", "t") == "{}"



    # -- Stage 2: Python None literal --



    # -- Stage 3: trailing comma repair --


    def test_trailing_comma_in_array(self):
        result = _repair_tool_call_arguments('{"a": [1, 2,]}', "t")
        parsed = json.loads(result)
        assert parsed == {"a": [1, 2]}


    # -- Stage 4: unclosed brackets --



    # -- Stage 5: excess closing delimiters --



    # -- Stage 6: last resort --


    def test_unrepairable_partial_returns_empty_object(self):
        # Truncated in the middle of a string key — bracket closing won't help
        assert _repair_tool_call_arguments('{"truncated": "val', "t") == "{}"

    # -- Valid JSON passthrough (this path is via except, but still works) --


    # -- Combined repairs --



    # -- Stage 0: strict=False (literal control chars in strings) --
    # llama.cpp backends sometimes emit literal tabs/newlines inside JSON
    # string values. strict=False accepts these; we re-serialise to the
    # canonical wire form (#12068).




    # -- Stage 4: control-char escape fallback --


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

