#!/usr/bin/env python3
"""Fix stale skill references automatically.

Scans SKILL.md files under ~/.hermes/skills/ for deprecated patterns
and fixes them:

  Error patterns (replaced in-place):
    memory_store.db  → holographic_memory
    session.jsonl    → SQLite session storage
    honcho           → holographic memory
    turn_router.py   → agent/routing.py
    model_router.py  → agent/routing.py

  Warning patterns (silenced with ignore directive):
    minimax, deepinfra, litellm, text-embedding-ada-002, :8201,
    openai.ChatCompletion

Usage:
    python scripts/fix_stale_skills.py              # fix all skills
    python scripts/fix_stale_skills.py --dry-run     # show what would change
    python scripts/fix_stale_skills.py --backup      # backup before modifying
    python scripts/fix_stale_skills.py --skill <name> # fix one skill
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Import the linter's patterns and scanner
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.skill_staleness_linter import (
    PATTERNS,
    SKILLS_ROOT,
    Finding,
    scan_all_skills,
    scan_skill,
)

# ── Lookup helpers ──────────────────────────────────────────────────────
# Normalize a linter pattern regex to a plain key for lookups.
def _pkey(raw_regex: str) -> str:
    return raw_regex.replace(r"\b", "").replace(r"\.", ".")

# For error-severity findings, the fix is simple text substitution.
# Keyed by normalized pattern (same _pkey used on Finding.pattern).
ERROR_REPLACEMENTS: Dict[str, str] = {
    _pkey(r"\bmemory_store\.db\b"): "holographic_memory",
    _pkey(r"\bsession\.jsonl\b"): "SQLite session storage",
    _pkey(r"\bhoncho\b"): "holographic memory",
    _pkey(r"\bturn_router\.py\b"): "agent/routing.py",
    _pkey(r"\bmodel_router\.py\b"): "agent/routing.py",
}

# For warn-severity findings, we add an ignore directive.
# Maps normalized pattern → ignore directive substring.
WARN_IGNORE_DIRECTIVES: Dict[str, str] = {
    _pkey(r"\bminimax\b"): "minimax",
    _pkey(r"\bdeepinfra\b"): "deepinfra",
    _pkey(r"\blitellm\b"): "litellm",
    _pkey(r"\btext-embedding-ada-002\b"): "text-embedding-ada-002",
    _pkey(r":8201\b"): ":8201",
    _pkey(r"openai\.ChatCompletion\b"): "openai.ChatCompletion",
}


def needs_ignore_already(content: str, ignore_pattern: str) -> bool:
    """Check if an ignore directive already exists for this pattern."""
    import re
    for m in re.finditer(r'<!--\s*staleness-linter:\s*ignore\s+(.+?)\s*-->', content):
        raw = m.group(1)
        patterns_part = re.split(r'\s*\(', raw)[0].strip()
        for p in patterns_part.split(','):
            if p.strip() == ignore_pattern:
                return True
    return False


def add_ignore_directive(content: str, ignore_pattern: str) -> str:
    """Add a staleness-linter ignore directive to SKILL.md content.

    Inserts after the frontmatter (---...--- block), or at the very top
    if no frontmatter is detected.
    """
    directive = f"<!-- staleness-linter: ignore {ignore_pattern} (intentional/archival reference) -->"
    lines = content.splitlines(keepends=True)

    # Find end of frontmatter (--- at start)
    insert_at = 0
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                insert_at = i + 1
                break

    # Check if directive already exists (belt-and-suspenders)
    if directive.rstrip("\n") in content:
        return content

    lines.insert(insert_at, directive + "\n")
    return "".join(lines)


def fix_skill_file(
    finding: Finding,
    *,
    dry_run: bool = False,
    backup: bool = False,
) -> Tuple[bool, str]:
    """Fix a single finding in a SKILL.md file.

    Returns (changed, description).
    """
    file_path = finding.file
    pattern = _pkey(finding.pattern)
    severity = finding.severity

    try:
        original = file_path.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"Cannot read {file_path}: {e}"

    content = original

    if severity == "error" and pattern in ERROR_REPLACEMENTS:
        # Apply the replacement
        import re
        replacement = ERROR_REPLACEMENTS[pattern]
        # Use the original regex from PATTERNS
        regex = finding.pattern
        new_content = re.sub(regex, replacement, content, flags=re.IGNORECASE)
        if new_content == content:
            return False, f"No match found for {pattern} in {file_path.name}"
        content = new_content

    elif severity == "warn" and pattern in WARN_IGNORE_DIRECTIVES:
        ignore_pattern = WARN_IGNORE_DIRECTIVES[pattern]
        if needs_ignore_already(content, ignore_pattern):
            return False, f"Ignore directive already present for {ignore_pattern}"
        content = add_ignore_directive(content, ignore_pattern)

    if content == original:
        return False, "No changes needed"

    if dry_run:
        return True, f"Would fix {severity}: {finding.description} in {file_path}"

    if backup:
        backup_path = file_path.with_suffix(file_path.suffix + ".bak")
        shutil.copy2(file_path, backup_path)

    try:
        file_path.write_text(content, encoding="utf-8")
    except Exception as e:
        return False, f"Cannot write {file_path}: {e}"

    return True, f"Fixed {severity}: {finding.description} in {file_path}"


def main():
    parser = argparse.ArgumentParser(
        description="Fix stale skill references automatically",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without modifying")
    parser.add_argument("--backup", action="store_true",
                        help="Create .bak copies before modifying")
    parser.add_argument("--skill", type=str, help="Fix a specific skill by name")
    parser.add_argument("--root", type=str, default=str(SKILLS_ROOT),
                        help="Skills root directory")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: Skills root not found: {root}", file=sys.stderr)
        sys.exit(2)

    # Collect findings
    if args.skill:
        matches = list(root.rglob(args.skill))
        findings = []
        for m in matches:
            if m.is_dir() and (m / "SKILL.md").exists():
                findings.extend(scan_skill(m))
        if not findings:
            findings = scan_skill(root / args.skill)
    else:
        findings = scan_all_skills(root)

    if not findings:
        print("✅ No stale findings to fix.")
        return

    # Fix each finding (deduplicating by file + pattern)
    seen: set = set()
    fixed = 0
    skipped = 0
    errors = 0

    for finding in findings:
        key = (str(finding.file), finding.pattern)
        if key in seen:
            continue
        seen.add(key)

        changed, msg = fix_skill_file(
            finding,
            dry_run=args.dry_run,
            backup=args.backup,
        )
        if changed:
            print(f"  ✓ {msg}")
            fixed += 1
        elif "Cannot" in msg or "Cannot write" in msg:
            print(f"  ✗ {msg}", file=sys.stderr)
            errors += 1
        else:
            skipped += 1

    # Summary
    label = " (DRY RUN)" if args.dry_run else ""
    print(f"\n{'='*60}{label}")
    print(f"  Fixed:  {fixed}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors:  {errors}")
    print(f"  Total findings: {len(findings)}")


if __name__ == "__main__":
    main()
