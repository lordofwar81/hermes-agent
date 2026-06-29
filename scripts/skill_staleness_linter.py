#!/usr/bin/env python3
"""Skill Staleness Linter — scans SKILL.md files for deprecated patterns.

Identifies skills that reference:
1. Deprecated/removed APIs (turn_router, model_router, honcho, minimax, etc.)
2. Dead config commands or outdated CLI syntax
3. Stale file paths (memory_store.db, session.jsonl)
4. Dead provider references (deepinfra)
5. Broken internal references (non-existent skill slugs)

Usage:
    python scripts/skill_staleness_linter.py                    # scan all skills
    python scripts/skill_staleness_linter.py --skill <name>       # scan one skill
    python scripts/skill_staleness_linter.py --json               # JSON output
    python scripts/skill_staleness_linter.py --fix                # suggest fixes

Exit codes:
    0 — no findings
    1 — findings detected
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

SKILLS_ROOT = Path.home() / ".hermes" / "skills"

# ---------------------------------------------------------------------------
# Staleness patterns: (regex, severity, description, suggested_fix)
# severity: error (dead/broken), warn (deprecated), info (cosmetic)
# ---------------------------------------------------------------------------

PATTERNS: List[tuple] = [
    # --- Dead routers (deprecated & moved to _deprecated/) ---
    (r'\bturn_router\.py\b', 'error',
     'turn_router.py was deprecated and moved to _deprecated/',
     'Remove reference or update to agent/routing.py'),
    (r'\bmodel_router\.py\b', 'error',
     'model_router.py was deprecated and moved to _deprecated/',
     'Remove reference or update to agent/routing.py'),

    # --- Dead providers ---
    (r'\bminimax\b', 'warn',
     'MiniMax provider/model was removed from the fleet (May 2026)',
     'Remove MiniMax references'),
    (r'\bdeepinfra\b', 'warn',
     'DeepInfra provider was deprecated — may not be active',
     'Verify DeepInfra is still in use or remove reference'),

    # --- Dead memory config ---
    (r'\bmemory_store\.db\b', 'error',
     'memory_store.db is not the current memory backend filename',
     'Update to holographic memory references (fact_store, retrieval)'),
    (r'\bsession\.jsonl\b', 'error',
     'Session storage migrated to SQLite-only (JSONL purged May 2026)',
     'Remove JSONL references, use SQLite session storage'),
    (r'\bhoncho\b', 'error',
     'Honcho memory provider was replaced by holographic memory',
     'Update to holographic memory references'),

    # --- Deprecated CLI commands ---
    (r'hermes\s+config\s+set\b(?!\s+--)', 'info',
     'Consider hermes config set --help for current syntax',
     None),
    (r'\blitellm\b', 'warn',
     'LiteLLM internal references in skills are fragile — prefer public API',
     None),
    (r'openai\.ChatCompletion\b', 'warn',
     'openai.ChatCompletion is the legacy v0 syntax; v1 uses openai.chat.completions',
     'Update to openai.chat.completions.create()'),

    # --- Dead code references ---
    (r'\bamazon-bedrock\b.*\badapter\b', 'info',
     'Bedrock adapter exists but may have specific version requirements',
     None),

    # --- Stale embed model references ---
    (r'\btext-embedding-ada-002\b', 'warn',
     'text-embedding-ada-002 is deprecated — OpenAI recommends text-embedding-3-small',
     'Update to text-embedding-3-small or local Qwen3-Embed-8B'),

    # --- Stale port references ---
    (r':8201\b', 'warn',
     'Port 8201 was a dead model port (removed from cluster)',
     'Remove 8201 references or update to active ports'),
    (r':8102\b', 'warn',
     'Port 8102 (Qwen3-VL-30B-A3B vision) is deprecated — Gemma-4-26B-A4B '
     'on :8199 is natively multimodal (--mmproj) and serves vision directly',
     'Point vision calls at :8199/gemma-4-26b-a4b, or use the auxiliary.vision '
     'config block which already routes there'),
    (r'\bqwen3-vl-30b-a3b\b', 'warn',
     'Qwen3-VL-30B-A3B was replaced by Gemma-4-26B-A4B (natively multimodal, '
     'served on :8199 with --mmproj)',
     'Update to gemma-4-26b-a4b at http://127.0.0.1:8199/v1'),
]


@dataclass
class Finding:
    skill: str
    file: Path
    line: int
    severity: str
    pattern: str
    description: str
    suggestion: str | None
    context: str = ""

    def to_dict(self) -> dict:
        return {
            "skill": self.skill,
            "file": str(self.file),
            "line": self.line,
            "severity": self.severity,
            "pattern": self.pattern,
            "description": self.description,
            "suggestion": self.suggestion,
            "context": self.context.strip(),
        }


def parse_ignores(content: str) -> set:
    """Parse HTML comment ignore directives from SKILL.md frontmatter.

    Format: <!-- staleness-linter: ignore pattern1, pattern2 (reason) -->
    Pattern matching is substring-based (not regex) for simplicity.
    """
    ignores: set = set()
    for m in re.finditer(r'<!--\s*staleness-linter:\s*ignore\s+(.+?)\s*-->', content):
        raw = m.group(1)
        # Extract patterns before the parenthetical reason
        patterns_part = re.split(r'\s*\(', raw)[0].strip()
        for p in patterns_part.split(','):
            p = p.strip()
            if p:
                ignores.add(p)
    return ignores


def pattern_matches_ignore(pattern_regex: str, ignores: set) -> bool:
    """Check if a linter pattern regex matches any ignore directive.

    Ignore directives are simple strings like 'model_router.py', 'turn_router.py'.
    We normalize both sides (strip regex meta-chars) and check substring match.
    """
    # Normalize regex: remove word boundaries and regex escapes for comparison
    normalized = re.sub(r'\\[bB]', '', pattern_regex).replace('\\.', '.')
    for ignore in ignores:
        if ignore in normalized:
            return True
    return False


def scan_skill(skill_dir: Path) -> List[Finding]:
    """Scan a single skill directory for staleness patterns."""
    skill_name = skill_dir.name
    skill_file = skill_dir / "SKILL.md"
    findings: List[Finding] = []

    if not skill_file.exists():
        return findings

    try:
        content = skill_file.read_text(encoding="utf-8")
    except Exception as e:
        findings.append(Finding(
            skill=skill_name, file=skill_file, line=0,
            severity="error", pattern="<read_error>",
            description=f"Cannot read SKILL.md: {e}",
            suggestion=None,
        ))
        return findings

    ignores = parse_ignores(content)

    lines = content.splitlines()
    for line_num, line in enumerate(lines, start=1):
        for pattern, severity, description, suggestion in PATTERNS:
            if ignores and pattern_matches_ignore(pattern, ignores):
                continue
            if re.search(pattern, line, re.IGNORECASE):
                findings.append(Finding(
                    skill=skill_name, file=skill_file, line=line_num,
                    severity=severity, pattern=pattern,
                    description=description, suggestion=suggestion,
                    context=line.strip()[:120],
                ))

    return findings


def scan_all_skills(root: Path = SKILLS_ROOT) -> List[Finding]:
    """Scan all SKILL.md files under root."""
    findings: List[Finding] = []
    for skill_dir in sorted(root.rglob("*")):
        if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
            findings.extend(scan_skill(skill_dir))
    return findings


def check_cross_references(root: Path = SKILLS_ROOT) -> List[Finding]:
    """Check for skill references to non-existent skill slugs."""
    # Collect all valid skill slugs (directory names with SKILL.md)
    valid_slugs = set()
    for skill_dir in root.rglob("*"):
        if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
            # Slug is the relative path from root
            valid_slugs.add(str(skill_dir.relative_to(root)))

    # Also accept bare skill names (the final path segment) as valid — skills
    # legitimately reference each other by bare slug (e.g. name="architecture-diagram"),
    # which Hermes resolves at runtime, even though valid_slugs stores full paths
    # like "creative/architecture-diagram".
    bare_slugs = {slug.split("/")[-1] for slug in valid_slugs}

    findings: List[Finding] = []
    # Pattern: references like skill_view(name='other-skill') or skill_manage
    ref_pattern = re.compile(r"(?:skill_view|skill_manage|load.*skill).*?name\s*[=:]\s*['\"]([^'\"]+)['\"]")

    for skill_dir in sorted(root.rglob("*")):
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            continue

        try:
            content = skill_file.read_text(encoding="utf-8")
        except Exception:
            continue

        skill_name = str(skill_dir.relative_to(root))
        own_bare = skill_name.split("/")[-1]
        lines = content.splitlines()
        for line_num, line in enumerate(lines, start=1):
            for m in ref_pattern.finditer(line):
                ref_slug = m.group(1)
                # A skill referencing its own slug (e.g. loading its own
                # template) is a legitimate self-reference, not a broken link.
                if ref_slug == own_bare:
                    continue
                if ref_slug not in valid_slugs and ref_slug not in bare_slugs and not ref_slug.startswith("plugin:"):
                    findings.append(Finding(
                        skill=skill_name, file=skill_file, line=line_num,
                        severity="warn", pattern="cross_ref",
                        description=f"References skill '{ref_slug}' which doesn't exist",
                        suggestion=f"Verify the skill slug is correct or create '{ref_slug}'",
                        context=line.strip()[:120],
                    ))

    return findings


def format_report(findings: List[Finding], json_output: bool = False) -> str:
    if json_output:
        return json.dumps([f.to_dict() for f in findings], indent=2)

    if not findings:
        return "✅ No staleness findings. All skills are current."

    # Group by severity
    by_severity = {"error": [], "warn": [], "info": []}
    for f in findings:
        by_severity.setdefault(f.severity, []).append(f)

    lines = []
    for sev in ("error", "warn", "info"):
        items = by_severity.get(sev, [])
        if not items:
            continue
        icon = {"error": "🔴", "warn": "🟡", "info": "🔵"}[sev]
        label = {"error": "ERRORS", "warn": "WARNINGS", "info": "NOTES"}[sev]
        lines.append(f"\n{icon} {label} ({len(items)}):")
        lines.append("─" * 60)
        for f in items:
            lines.append(f"  [{f.severity.upper()}] {f.skill}")
            lines.append(f"  Line {f.line}: {f.description}")
            if f.suggestion:
                lines.append(f"  Fix: {f.suggestion}")
            if f.context:
                lines.append(f"  > {f.context}")
            lines.append("")

    # Summary
    n_skills = len({f.skill for f in findings})
    lines.append(f"\n{'='*60}")
    lines.append(f"Total: {len(findings)} findings across {n_skills} skills")
    lines.append(f"  🔴 {len(by_severity.get('error', []))} errors")
    lines.append(f"  🟡 {len(by_severity.get('warn', []))} warnings")
    lines.append(f"  🔵 {len(by_severity.get('info', []))} notes")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Skill Staleness Linter")
    parser.add_argument("--skill", type=str, help="Scan a specific skill by name")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--root", type=str, default=str(SKILLS_ROOT),
                        help="Skills root directory")
    parser.add_argument("--no-cross-refs", action="store_true",
                        help="Skip cross-reference checks")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: Skills root not found: {root}", file=sys.stderr)
        sys.exit(2)

    if args.skill:
        # Find skill dir by name
        matches = list(root.rglob(args.skill))
        if not matches:
            print(f"ERROR: Skill '{args.skill}' not found under {root}", file=sys.stderr)
            sys.exit(2)
        findings = []
        for m in matches:
            if m.is_dir() and (m / "SKILL.md").exists():
                findings.extend(scan_skill(m))
        if not findings:
            findings = scan_skill(root / args.skill)
    else:
        findings = scan_all_skills(root)
        if not args.no_cross_refs:
            findings.extend(check_cross_references(root))

    report = format_report(findings, json_output=args.json)
    print(report)

    sys.exit(1 if findings else 0)


if __name__ == "__main__":
    main()
