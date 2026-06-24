"""Role-specialized delegate personas (Gulli Ch7 Expert Teams, Phase 5).

A persona is a fixed system-prompt prefix injected into a delegate child's
prompt, giving it a specialized identity (coder, researcher, critic) with a
defined approach and toolset expectation. Personas are orthogonal to the
existing ``role`` (leaf/orchestrator) — a persona shapes *how* the child
works, while role controls *whether* it can re-delegate.

Minimal scope (per locked decision): schema field + prompt prefix only.
No per-child model override (deferred follow-up). The persona's default
toolsets are a suggestion, not enforced — the caller can still override
via the existing ``toolsets`` parameter.

Personas are gated behind HERMES_PERSONAS_ENABLED (default off). When off,
the ``persona`` field is accepted by the schema but ignored (no prefix
injected) — forward-compatible, zero breaking change.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Persona:
    """A delegate persona definition."""
    name: str
    prompt_prefix: str
    default_toolsets: List[str] = field(default_factory=list)
    description: str = ""


# ─── Registry ─────────────────────────────────────────────────────────────
# Seeded with three personas per the locked decision. Resist adding more
# until a concrete need exists (Iron Law: no speculative engineering).

_PERSONAS: Dict[str, Persona] = {
    "coder": Persona(
        name="coder",
        prompt_prefix=(
            "You are a senior software engineer. Your job is to write, fix, "
            "and test code with precision.\n\n"
            "Principles:\n"
            "- Read the relevant code before making changes.\n"
            "- Make minimal, surgical changes. Match existing style.\n"
            "- Verify your changes compile/pass before reporting done.\n"
            "- If you encounter an error, fix it — don't hand it back.\n"
        ),
        default_toolsets=["terminal", "file"],
        description="Senior software engineer — writes, fixes, and tests code.",
    ),
    "researcher": Persona(
        name="researcher",
        prompt_prefix=(
            "You are a research analyst. Your job is to gather, synthesize, "
            "and report information accurately.\n\n"
            "Principles:\n"
            "- Search broadly first, then drill into the most relevant sources.\n"
            "- Distinguish facts from opinions. Cite sources.\n"
            "- Report findings in a structured format (summary → details → sources).\n"
            "- If you can't verify a claim, say so explicitly.\n"
        ),
        default_toolsets=["web"],
        description="Research analyst — gathers, synthesizes, and reports information.",
    ),
    "critic": Persona(
        name="critic",
        prompt_prefix=(
            "You are a quality critic. Your job is to evaluate a deliverable "
            "against criteria and provide actionable feedback.\n\n"
            "Principles:\n"
            "- Evaluate against: correctness, completeness, clarity, edge cases.\n"
            "- Be specific: point to exact issues, not vague concerns.\n"
            "- Prioritize: flag blocking issues separately from nice-to-haves.\n"
            "- If the work is solid, say so — don't manufacture criticism.\n"
        ),
        default_toolsets=["file"],
        description="Quality critic — evaluates deliverables against criteria.",
    ),
}


def get_persona(name: str) -> Optional[Persona]:
    """Look up a persona by name (case-insensitive). None if not found."""
    if not name:
        return None
    return _PERSONAS.get(name.strip().lower())


def list_personas() -> List[str]:
    """Return all registered persona names."""
    return sorted(_PERSONAS.keys())


def resolve_persona_toolsets(name: str, caller_toolsets: Optional[List[str]] = None) -> Optional[List[str]]:
    """Resolve the effective toolsets for a persona-aware delegate.

    If the caller explicitly passed toolsets, those win (caller knows best).
    If not, fall back to the persona's default_toolsets.
    Returns None if no persona is set and no caller toolsets given (caller
    will inherit the parent's toolsets — the existing behavior).
    """
    if caller_toolsets is not None:
        return caller_toolsets
    persona = get_persona(name)
    if persona and persona.default_toolsets:
        return list(persona.default_toolsets)
    return None
