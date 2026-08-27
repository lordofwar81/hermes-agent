"""F7 regression (2026-08-23): checkpoint `git add -A` vs unreadable dirs.

Finding: the shadow-store checkpoint runs `git add -A` with the work tree at
the session's working directory. When that is $HOME on Ubuntu, root-owned
~/snap-private-tmp makes git exit rc=128 ("could not open directory") and the
checkpoint fails (21 occurrences in the gateway journal over 39h). The fix
excludes unreadable top-level dirs via pathspec inside _run_git.
"""

import os

import pytest

pytest.importorskip("tools.checkpoint_manager")
from tools import checkpoint_manager as cm  # noqa: E402


@pytest.fixture
def locked_dir(tmp_path):
    d = tmp_path / "snap-private-tmp"
    d.mkdir()
    os.chmod(d, 0)
    try:
        yield d
    finally:
        os.chmod(d, 0o755)


class TestUnreadableTopEntries:
    def test_locked_dir_is_listed(self, tmp_path, locked_dir):
        (tmp_path / "normal").mkdir()
        assert cm._unreadable_top_entries(tmp_path) == ["snap-private-tmp"]

    def test_clean_tree_lists_nothing(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "b.txt").write_text("x")
        assert cm._unreadable_top_entries(tmp_path) == []

    def test_missing_root_returns_empty(self, tmp_path):
        assert cm._unreadable_top_entries(tmp_path / "nope") == []


class TestRunGitAddExcludes:
    def test_add_a_appends_exclude_pathspecs(
        self, monkeypatch, tmp_path, locked_dir
    ):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd

            class R:
                returncode = 0
                stdout = ""
                stderr = ""

            return R()

        monkeypatch.setattr(cm.subprocess, "run", fake_run)
        ok, _, _ = cm._run_git(
            ["add", "-A"],
            store=tmp_path / "store.git",
            working_dir=str(tmp_path),
        )
        assert ok is True
        assert "--" in captured["cmd"]
        assert any(
            ":(exclude)snap-private-tmp" in part for part in captured["cmd"]
        )
        assert "." in captured["cmd"]  # positive pathspec keeps semantics of -A

    def test_add_a_on_clean_tree_is_unchanged(self, monkeypatch, tmp_path):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd

            class R:
                returncode = 0
                stdout = ""
                stderr = ""

            return R()

        monkeypatch.setattr(cm.subprocess, "run", fake_run)
        cm._run_git(
            ["add", "-A"], store=tmp_path / "store.git", working_dir=str(tmp_path)
        )
        assert captured["cmd"] == ["git", "add", "-A"]

    def test_other_git_commands_are_not_touched(self, monkeypatch, tmp_path, locked_dir):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd

            class R:
                returncode = 0
                stdout = ""
                stderr = ""

            return R()

        monkeypatch.setattr(cm.subprocess, "run", fake_run)
        cm._run_git(
            ["status", "--porcelain"],
            store=tmp_path / "store.git",
            working_dir=str(tmp_path),
        )
        assert captured["cmd"] == ["git", "status", "--porcelain"]
