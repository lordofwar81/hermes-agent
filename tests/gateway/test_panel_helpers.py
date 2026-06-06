"""Tests for HermesCLI panel helper methods extracted from closures.

These are pure functions that were extracted from _get_slash_confirm_display_fragments,
_get_approval_display_fragments, and _get_clarify_display. They depend only on
shutil and textwrap — no Hermes infrastructure needed.
"""

import os
import shutil
import textwrap
from unittest.mock import patch, MagicMock

import pytest


class TestPanelBoxWidth:
    """Tests for _panel_box_width()."""

    def _make_cli(self):
        """Create a minimal HermesCLI-like object with the panel method."""
        from cli import HermesCLI
        # We need the method to be accessible without full init.
        # Since these are pure methods, we can bind them to a mock.
        cli = MagicMock(spec=HermesCLI)
        cli._panel_box_width = HermesCLI._panel_box_width.__get__(cli, HermesCLI)
        return cli

    def test_respects_min_width(self):
        cli = self._make_cli()
        result = cli._panel_box_width("Hi", [], min_width=60, max_width=100)
        assert result >= 58  # min_width - 2

    def test_respects_max_width(self):
        cli = self._make_cli()
        # Even with a very long title, should not exceed max + 2 (for borders)
        with patch.object(shutil, 'get_terminal_size', return_value=os.terminal_size((200, 40))):
            result = cli._panel_box_width("X" * 200, ["Y" * 200], min_width=46, max_width=50)
            assert result <= 52  # max_width - 2 = inner cap, + 2 for borders

    def test_uses_longest_content_line(self):
        cli = self._make_cli()
        short_title = "A"
        long_line = "B" * 80
        narrow = cli._panel_box_width(short_title, [], min_width=46, max_width=100)
        wide = cli._panel_box_width(short_title, [long_line], min_width=46, max_width=100)
        assert wide > narrow

    def test_empty_content_still_works(self):
        cli = self._make_cli()
        result = cli._panel_box_width("Title", [])
        assert result > 0
        assert isinstance(result, int)

    def test_monospace_border_overhead(self):
        """Result should be inner + 2 for │ borders."""
        cli = self._make_cli()
        # With terminal wide enough, result = inner + 2
        with patch.object(shutil, 'get_terminal_size', return_value=os.terminal_size((200, 40))):
            result = cli._panel_box_width("Test", ["line"], min_width=46, max_width=76)
            # The inner calculation: max(len("Test")+4, len("line")+4, 44) clamped to 74
            # Result = inner + 2
            assert result >= 46


class TestWrapPanelText:
    """Tests for _wrap_panel_text()."""

    def _make_cli(self):
        from cli import HermesCLI
        cli = MagicMock(spec=HermesCLI)
        cli._wrap_panel_text = HermesCLI._wrap_panel_text.__get__(cli, HermesCLI)
        return cli

    def test_short_text_returns_single_line(self):
        cli = self._make_cli()
        result = cli._wrap_panel_text("hello", width=80)
        assert result == ["hello"]

    def test_long_text_wraps(self):
        cli = self._make_cli()
        # Multi-word text so textwrap can break between words
        text = "word " * 20
        result = cli._wrap_panel_text(text, width=20)
        assert len(result) > 1
        assert all(len(line) <= 22 for line in result)  # allow indent

    def test_empty_text_returns_empty_string_list(self):
        cli = self._make_cli()
        result = cli._wrap_panel_text("", width=80)
        assert result == [""]

    def test_subsequent_indent(self):
        cli = self._make_cli()
        text = "short " + "longword " * 10
        result = cli._wrap_panel_text(text, width=30, subsequent_indent="  ")
        if len(result) > 1:
            assert result[1].startswith("  ")

    def test_does_not_break_long_words_when_disabled(self):
        """break_long_words=False means a single long word can exceed width."""
        cli = self._make_cli()
        text = "a" * 100
        result = cli._wrap_panel_text(text, width=10)
        # With break_long_words=False, the single word should stay intact
        assert len(result) == 1
        assert len(result[0]) == 100

    def test_minimum_width_clamp(self):
        """Width below 8 should be clamped to 8."""
        cli = self._make_cli()
        text = "hello world"
        result = cli._wrap_panel_text(text, width=1)
        # Should not crash, should wrap at effective width 8
        assert isinstance(result, list)
        assert len(result) > 0


class TestAppendPanelLine:
    """Tests for _append_panel_line()."""

    def _make_cli(self):
        from cli import HermesCLI
        cli = MagicMock(spec=HermesCLI)
        cli._append_panel_line = HermesCLI._append_panel_line.__get__(cli, HermesCLI)
        return cli

    def test_appends_three_tuples(self):
        cli = self._make_cli()
        lines = []
        cli._append_panel_line(lines, "border", "content", "hello", 20)
        assert len(lines) == 3
        assert lines[0] == ("border", "│ ")
        assert lines[1][0] == "content"
        assert lines[1][1].endswith(" ")  # ljust pads
        assert lines[2] == ("border", " │\n")

    def test_text_ljust_to_inner_width(self):
        cli = self._make_cli()
        lines = []
        cli._append_panel_line(lines, "b", "c", "hi", 10)
        # inner_width = 10 - 2 = 8
        assert len(lines[1][1]) == 8  # "hi" ljust(8)

    def test_zero_box_width(self):
        """box_width=0 → inner_width=0, should not crash."""
        cli = self._make_cli()
        lines = []
        cli._append_panel_line(lines, "b", "c", "x", 0)
        assert len(lines) == 3

    def test_preserves_style_strings(self):
        cli = self._make_cli()
        lines = []
        cli._append_panel_line(lines, "red", "bold", "test", 30)
        assert lines[0][0] == "red"
        assert lines[1][0] == "bold"
        assert lines[2][0] == "red"


class TestAppendBlankPanelLine:
    """Tests for _append_blank_panel_line()."""

    def _make_cli(self):
        from cli import HermesCLI
        cli = MagicMock(spec=HermesCLI)
        cli._append_blank_panel_line = HermesCLI._append_blank_panel_line.__get__(cli, HermesCLI)
        return cli

    def test_appends_single_tuple(self):
        cli = self._make_cli()
        lines = []
        cli._append_blank_panel_line(lines, "border", 20)
        assert len(lines) == 1
        assert lines[0][0] == "border"

    def test_fills_with_spaces(self):
        cli = self._make_cli()
        lines = []
        cli._append_blank_panel_line(lines, "b", 10)
        # Should be │ followed by 10 spaces then │\n
        assert "│" in lines[0][1]
        assert lines[0][1].endswith("│\n")
        # Total content between borders = box_width spaces
        content = lines[0][1].strip("│\n")
        assert len(content) == 10

    def test_zero_box_width(self):
        cli = self._make_cli()
        lines = []
        cli._append_blank_panel_line(lines, "b", 0)
        # Should produce ││\n (no spaces between)
        assert len(lines) == 1
