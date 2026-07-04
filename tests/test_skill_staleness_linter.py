"""Tests for the skill staleness linter (scripts/skill_staleness_linter.py)."""

import pytest
from pathlib import Path
import tempfile
import textwrap

# Import the linter module
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from skill_staleness_linter import (
    scan_skill, scan_all_skills, check_cross_references,
    PATTERNS, Finding, format_report,
)


@pytest.fixture
def tmp_skills(tmp_path):
    """Create a temporary skills directory with test skills."""
    skills_root = tmp_path / "skills"
    skills_root.mkdir()

    # Skill 1: Has deprecated turn_router reference
    s1 = skills_root / "routing-old"
    s1.mkdir()
    (s1 / "SKILL.md").write_text(textwrap.dedent("""\
        # Routing Old Skill
        This skill references turn_router.py for routing decisions.
        Also model_router.py for provider selection.
        """))

    # Skill 2: Has dead memory_store.db reference
    s2 = skills_root / "memory-debug"
    s2.mkdir()
    (s2 / "SKILL.md").write_text(textwrap.dedent("""\
        # Memory Debug
        Check ~/.hermes/memory_store.db for facts.
        Also session.jsonl for history.
        """))

    # Skill 3: Has MiniMax reference (warning)
    s3 = skills_root / "llama-setup"
    s3.mkdir()
    (s3 / "SKILL.md").write_text(textwrap.dedent("""\
        # Llama Setup
        MiniMax-M2.7 loads on port 8199.
        Use hermes config set for config.
        """))

    # Skill 4: Clean skill (no findings)
    s4 = skills_root / "clean-skill"
    s4.mkdir()
    (s4 / "SKILL.md").write_text(textwrap.dedent("""\
        # Clean Skill
        No deprecated references here.
        Uses agent/routing.py for routing.
        """))

    # Skill 5: Has cross-reference to non-existent skill
    s5 = skills_root / "ref-checker"
    s5.mkdir()
    (s5 / "SKILL.md").write_text(textwrap.dedent("""\
        # Ref Checker
        Uses skill_view(name='nonexistent-skill') for something.
        """))

    return skills_root


class TestScanSkill:
    def test_finds_turn_router(self, tmp_skills):
        findings = scan_skill(tmp_skills / "routing-old")
        patterns = {f.pattern for f in findings}
        assert r'\bturn_router\.py\b' in patterns
        assert r'\bmodel_router\.py\b' in patterns

    def test_finds_memory_store_db(self, tmp_skills):
        findings = scan_skill(tmp_skills / "memory-debug")
        patterns = {f.pattern for f in findings}
        assert r'\bmemory_store\.db\b' in patterns
        assert r'\bsession\.jsonl\b' in patterns

    def test_finds_minimax_warning(self, tmp_skills):
        findings = scan_skill(tmp_skills / "llama-setup")
        patterns = {f.pattern for f in findings}
        assert r'\bminimax\b' in patterns
        # MiniMax is a warning, not an error
        minimax_finding = [f for f in findings if 'minimax' in f.pattern.lower()][0]
        assert minimax_finding.severity == "warn"

    def test_clean_skill_no_findings(self, tmp_skills):
        findings = scan_skill(tmp_skills / "clean-skill")
        # The skill references agent/routing.py which is valid — no findings
        assert len(findings) == 0

    def test_nonexistent_skill(self, tmp_skills):
        findings = scan_skill(tmp_skills / "does-not-exist")
        assert len(findings) == 0


class TestScanAllSkills:
    def test_scans_all(self, tmp_skills):
        findings = scan_all_skills(tmp_skills)
        skills_hit = {f.skill for f in findings}
        assert "routing-old" in skills_hit
        assert "memory-debug" in skills_hit
        assert "llama-setup" in skills_hit
        assert "clean-skill" not in skills_hit

    def test_correct_severity_counts(self, tmp_skills):
        findings = scan_all_skills(tmp_skills)
        errors = [f for f in findings if f.severity == "error"]
        warnings = [f for f in findings if f.severity == "warn"]
        info = [f for f in findings if f.severity == "info"]
        # turn_router, model_router, memory_store.db, session.jsonl = errors
        # minimax = warning, hermes config set = info
        assert len(errors) >= 4  # at least turn_router, model_router, memory_store.db, session.jsonl
        assert len(warnings) >= 1  # minimax
        assert len(info) >= 1    # hermes config set


class TestCrossReferences:
    def test_finds_broken_cross_ref(self, tmp_skills):
        findings = check_cross_references(tmp_skills)
        broken = [f for f in findings if "nonexistent-skill" in f.description]
        assert len(broken) == 1
        assert broken[0].severity == "warn"

    def test_valid_cross_ref_not_flagged(self, tmp_skills):
        # routing-old is a valid skill — reference to it should not be flagged
        s = tmp_skills / "self-ref"
        s.mkdir()
        (s / "SKILL.md").write_text("Uses skill_view(name='routing-old')")
        findings = check_cross_references(tmp_skills)
        ref_findings = [f for f in findings if "routing-old" in f.description]
        assert len(ref_findings) == 0

    def test_self_reference_not_flagged(self, tmp_skills):
        # A skill referencing its OWN slug (e.g. loading its own template) is a
        # legitimate self-reference, not a broken link. Regression test for the
        # false-positive that caused 3 working skills to be wrongly flagged.
        s = tmp_skills / "template-loader"
        s.mkdir()
        (s / "SKILL.md").write_text(
            "Load template: skill_view(name='template-loader', file_path='templates/t.html')"
        )
        findings = check_cross_references(tmp_skills)
        self_ref_findings = [f for f in findings if "template-loader" in f.description]
        assert len(self_ref_findings) == 0

    def test_bare_name_resolves_under_nested_dir(self, tmp_skills):
        # Skills live under category subdirs (e.g. creative/architecture-diagram)
        # but are referenced by bare slug (architecture-diagram). The checker
        # must resolve bare names against nested directory structures.
        cat = tmp_skills / "creative"
        nested = cat / "diagram-skill"
        nested.mkdir(parents=True)
        (nested / "SKILL.md").write_text("# diagram skill")
        # Another skill references it by bare name only
        other = tmp_skills / "caller"
        other.mkdir()
        (other / "SKILL.md").write_text("Uses skill_view(name='diagram-skill')")
        findings = check_cross_references(tmp_skills)
        broken = [f for f in findings if "diagram-skill" in f.description]
        assert len(broken) == 0


class TestFindings:
    def test_finding_to_dict(self, tmp_skills):
        findings = scan_skill(tmp_skills / "routing-old")
        assert len(findings) > 0
        d = findings[0].to_dict()
        assert "skill" in d
        assert "severity" in d
        assert "line" in d
        assert "description" in d


class TestFormatReport:
    def test_clean_report(self):
        report = format_report([])
        assert "No staleness findings" in report

    def test_report_with_findings(self, tmp_skills):
        findings = scan_all_skills(tmp_skills)
        report = format_report(findings, json_output=False)
        assert "ERRORS" in report
        assert "Total:" in report

    def test_json_report(self, tmp_skills):
        findings = scan_all_skills(tmp_skills)
        import json
        report = format_report(findings, json_output=True)
        parsed = json.loads(report)
        assert isinstance(parsed, list)
        assert len(parsed) > 0


class TestPatterns:
    def test_patterns_have_required_fields(self):
        """Every pattern tuple must have exactly 4 fields."""
        for p in PATTERNS:
            assert len(p) == 4, f"Pattern {p[0]} has {len(p)} fields, expected 4"
            regex, severity, desc, suggestion = p
            assert severity in ("error", "warn", "info"), f"Invalid severity: {severity}"
            assert isinstance(desc, str) and len(desc) > 0
            # suggestion can be None for info-only

    def test_patterns_compile(self):
        """Every regex pattern must compile without error."""
        import re
        for p in PATTERNS:
            re.compile(p[0], re.IGNORECASE)  # Must not raise
