"""Tests for scripts/fix_stale_skills.py."""

from pathlib import Path
import textwrap

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from skill_staleness_linter import scan_all_skills, scan_skill
from fix_stale_skills import fix_skill_file, add_ignore_directive, needs_ignore_already


def _make_skill(root: Path, name: str, content: str) -> Path:
    d = root / name
    d.mkdir(parents=True)
    p = d / "SKILL.md"
    p.write_text(textwrap.dedent(content))
    return p


class TestFixStaleSkills:
    def test_fixes_memory_store_db(self, tmp_path):
        p = _make_skill(tmp_path, "test-skill", """\
            # Test Skill
            Check ~/.hermes/memory_store.db for facts.
            """)
        findings = scan_skill(p.parent)
        assert len(findings) == 1
        assert findings[0].severity == "error"
        changed, msg = fix_skill_file(findings[0])
        assert changed, f"Expected fix to succeed: {msg}"
        content = p.read_text()
        assert "holographic_memory" in content
        assert "memory_store.db" not in content

    def test_fixes_session_jsonl(self, tmp_path):
        p = _make_skill(tmp_path, "test-skill", """\
            # Test Skill
            Uses session.jsonl for history.
            """)
        findings = scan_skill(p.parent)
        assert len(findings) >= 1
        changed, msg = fix_skill_file([f for f in findings if "jsonl" in f.pattern][0])
        assert changed, f"Expected fix to succeed: {msg}"
        content = p.read_text()
        assert "session.jsonl" not in content.lower() or "session.jsonl" not in content

    def test_adds_ignore_for_minimax(self, tmp_path):
        p = _make_skill(tmp_path, "test-skill", """\
            # Test Skill
            MiniMax-M2.7 runs on port 8199.
            """)
        findings = scan_skill(p.parent)
        minimax_findings = [f for f in findings if "minimax" in f.pattern.lower()]
        assert len(minimax_findings) >= 1
        changed, msg = fix_skill_file(minimax_findings[0])
        assert changed, f"Expected fix to succeed: {msg}"
        content = p.read_text()
        assert "staleness-linter: ignore minimax" in content
        assert "MiniMax-M2.7" in content  # original content preserved

    def test_skips_already_fixed(self, tmp_path):
        p = _make_skill(tmp_path, "test-skill", """\
            # Test Skill
            Check ~/.hermes/memory_store.db for facts.
            """)
        findings = scan_skill(p.parent)
        fix_skill_file(findings[0])
        # Same finding again — should be skipped (no match anymore)
        findings2 = scan_skill(p.parent)
        assert len(findings2) == 0

    def test_skips_already_ignored(self, tmp_path):
        p = _make_skill(tmp_path, "test-skill", """\
            # Test Skill
            <!-- staleness-linter: ignore minimax (intentional) -->
            MiniMax-M2.7 reference.
            """)
        findings = scan_skill(p.parent)
        minimax_findings = [f for f in findings if "minimax" in f.pattern.lower()]
        assert len(minimax_findings) == 0, "Linter should skip ignored patterns"

    def test_dry_run_does_not_modify(self, tmp_path):
        p = _make_skill(tmp_path, "test-skill", """\
            # Test Skill
            Check ~/.hermes/memory_store.db.
            """)
        original = p.read_text()
        findings = scan_skill(p.parent)
        changed, msg = fix_skill_file(findings[0], dry_run=True)
        assert changed, f"Expected dry-run to report change: {msg}"
        assert p.read_text() == original, "Dry run should not modify file"

    def test_backup_creates_bak(self, tmp_path):
        p = _make_skill(tmp_path, "test-skill", """\
            # Test Skill
            Check ~/.hermes/memory_store.db.
            """)
        findings = scan_skill(p.parent)
        fix_skill_file(findings[0], backup=True)
        bak = p.with_suffix(p.suffix + ".bak")
        assert bak.exists(), "Backup file should exist"
        assert "memory_store.db" in bak.read_text()

    def test_no_findings_returns_unchanged(self, tmp_path):
        p = _make_skill(tmp_path, "clean-skill", """\
            # Clean Skill
            Uses agent/routing.py for routing.
            """)
        findings = scan_skill(p.parent)
        assert len(findings) == 0

    def test_multiple_findings_in_one_file(self, tmp_path):
        """Multiple stale patterns in one file all get fixed."""
        p = _make_skill(tmp_path, "test-skill", """\
            # Test Skill
            Uses turn_router.py and memory_store.db.
            Also references model_router.py.
            """)
        findings = scan_skill(p.parent)
        assert len(findings) == 3
        for f in findings:
            fix_skill_file(f)
        content = p.read_text()
        assert "turn_router.py" not in content
        assert "model_router.py" not in content
        assert "memory_store.db" not in content

    def test_hermes_config_set_note_not_modified(self, tmp_path):
        """Info-level findings (hermes config set) are not touched."""
        p = _make_skill(tmp_path, "test-skill", """\
            # Test Skill
            Run hermes config set for settings.
            """)
        findings = scan_skill(p.parent)
        info_findings = [f for f in findings if f.severity == "info"]
        assert len(info_findings) >= 1
        # Should not be fixed (info severity not in ERROR_REPLACEMENTS or WARN_IGNORE_DIRECTIVES)
        for f in info_findings:
            changed, msg = fix_skill_file(f)
            assert not changed, f"Info finding should not be fixed: {msg}"
