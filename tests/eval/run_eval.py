#!/usr/bin/env python3
"""Hermes TaskClassifier eval harness — the regression bar for Gulli-pattern work.

Runs the golden cases in ``evalset.yaml`` through ``TaskClassifier.classify``
and reports per-case pass/fail + aggregate accuracy. Exits non-zero on any
mismatch so it's CI-usable.

Phase 1 (this file): keyword classifier baseline only.
Phase 2 will add ``--semantic`` to exercise ``classify_semantic``.
Phase 3+ will extend to assert critic-fired / plan-generated.

Design: the harness is a thin comparator — it does NOT import agent internals
beyond ``routing.TaskClassifier`` and ``routing.Category``. This keeps it
decoupled from turn-loop state and lets it run standalone (no gateway, no
embedding server required for the baseline).

Usage:
    python tests/eval/run_eval.py            # baseline (keyword) — the bar
    python tests/eval/run_eval.py -v         # verbose: show every case
    python tests/eval/run_eval.py --semantic # Phase 2: semantic path (not yet)
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# Make the repo importable when run as a script (tests/eval/ → repo root).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent.routing import TaskClassifier, Category  # noqa: E402

try:
    import yaml
except ImportError:  # pragma: no cover — yaml is a Hermes core dep
    sys.stderr.write("error: PyYAML is required (pip install pyyaml)\n")
    sys.exit(2)


# ─── Eval case model ──────────────────────────────────────────────────────

@dataclass
class EvalCase:
    input: str
    expected_category: str
    note: str = ""
    # Filled at load time:
    expected: Optional[Category] = None


@dataclass
class EvalResult:
    case: EvalCase
    predicted: Category
    passed: bool
    is_adversarial: bool = False


@dataclass
class EvalReport:
    results: List[EvalResult] = field(default_factory=list)

    def add(self, r: EvalResult) -> None:
        self.results.append(r)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def accuracy(self) -> float:
        return self.passed_count / self.total if self.total else 0.0

    @property
    def adversarial_results(self) -> List[EvalResult]:
        return [r for r in self.results if r.is_adversarial]

    def failures_by_category(self) -> dict:
        """Which expected categories are being missed, and how often."""
        misses: dict = {}
        for r in self.results:
            if not r.passed:
                key = r.case.expected_category
                misses[key] = misses.get(key, 0) + 1
        return misses


# ─── Loader ───────────────────────────────────────────────────────────────

def load_evalset(path: Path) -> List[EvalCase]:
    """Load and validate the evalset YAML. Validates every expected_category."""
    raw = yaml.safe_load(path.read_text())
    cases_raw = raw.get("cases", []) if isinstance(raw, dict) else []
    if not cases_raw:
        sys.stderr.write(f"error: no cases found in {path}\n")
        sys.exit(2)

    valid_values = {c.value for c in Category}
    cases: List[EvalCase] = []
    for i, c in enumerate(cases_raw):
        cat_str = c.get("expected_category", "").strip().lower()
        if cat_str not in valid_values:
            sys.stderr.write(
                f"error: case {i} has invalid expected_category "
                f"'{cat_str}'. Valid: {sorted(valid_values)}\n"
            )
            sys.exit(2)
        note = c.get("note", "")
        cases.append(EvalCase(
            input=c["input"],
            expected_category=cat_str,
            expected=Category(cat_str),
            note=note,
        ))
    return cases


# ─── Runner ───────────────────────────────────────────────────────────────

def classify(message: str, *, semantic: bool = False) -> Category:
    """Dispatch to the requested classifier path.

    Phase 1: keyword only. Phase 2 wires ``semantic`` → ``classify_semantic``
    with keyword fallback (mirroring how ``Router.route`` will call it).
    """
    if semantic:
        # Phase 2 will implement classify_semantic. Until then, fall through
        # to keyword so the harness runs without the embedding server.
        method = getattr(TaskClassifier, "classify_semantic", None)
        if method is not None:
            result = method(message)
            if result is not None:
                return result
    return TaskClassifier.classify(message)


def run(cases: List[EvalCase], *, semantic: bool = False) -> EvalReport:
    report = EvalReport()
    for case in cases:
        predicted = classify(case.input, semantic=semantic)
        passed = predicted == case.expected
        is_adv = "AMBIGUOUS" in case.note.upper()
        report.add(EvalResult(
            case=case,
            predicted=predicted,
            passed=passed,
            is_adversarial=is_adv,
        ))
    return report


# ─── Reporting ────────────────────────────────────────────────────────────

_GREEN = "\033[32m" if sys.stdout.isatty() else ""
_RED = "\033[31m" if sys.stdout.isatty() else ""
_YELLOW = "\033[33m" if sys.stdout.isatty() else ""
_DIM = "\033[2m" if sys.stdout.isatty() else ""
_RESET = "\033[0m" if sys.stdout.isatty() else ""


def print_report(report: EvalReport, *, verbose: bool, label: str) -> None:
    mode = "semantic" if label == "semantic" else "keyword"
    print(f"\n{'='*70}")
    print(f"  Hermes TaskClassifier eval — {mode} path")
    print(f"{'='*70}")

    if verbose:
        for r in report.results:
            mark = f"{_GREEN}✓{_RESET}" if r.passed else f"{_RED}✗{_RESET}"
            adv = f" {_YELLOW}[adversarial]{_RESET}" if r.is_adversarial else ""
            exp = r.case.expected_category
            got = r.predicted.value
            mismatch = "" if r.passed else f" → got {got}"
            print(f"  {mark} [{exp:10s}]{mismatch}{adv}  {r.case.input!r}")
            if not r.passed and r.case.note:
                print(f"         {_DIM}note: {r.case.note}{_RESET}")

    # Aggregate
    acc = report.accuracy * 100
    color = _GREEN if acc == 100 else (_YELLOW if acc >= 80 else _RED)
    print(f"\n  {report.passed_count}/{report.total} passed  "
          f"({color}{acc:.1f}%{color})  [{mode}]")

    adv = report.adversarial_results
    if adv:
        adv_pass = sum(1 for r in adv if r.passed)
        print(f"  adversarial: {adv_pass}/{len(adv)} passed "
              f"(these are where the semantic classifier must add value)")

    misses = report.failures_by_category()
    if misses:
        detail = ", ".join(f"{k}: {v}" for k, v in sorted(misses.items()))
        print(f"  misses by expected category: {detail}")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    verbose = "-v" in argv or "--verbose" in argv
    semantic = "--semantic" in argv
    quiet = "-q" in argv or "--quiet" in argv

    evalset_path = Path(__file__).parent / "evalset.yaml"
    cases = load_evalset(evalset_path)

    report = run(cases, semantic=semantic)
    print_report(report, verbose=verbose or not quiet,
                 label="semantic" if semantic else "keyword")

    # Exit non-zero on any failure → CI-usable.
    return 0 if report.passed_count == report.total else 1


if __name__ == "__main__":
    sys.exit(main())
