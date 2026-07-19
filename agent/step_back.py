"""Step-back prompting (Gulli Appendix A, Tier 2 pattern).

Before answering a complex question, ask a cheap auxiliary model to derive
the underlying principle or abstraction ("step back" from the specific
instance to the general rule). Inject that principle as context so the main
model's answer is grounded in first principles rather than surface pattern
matching.

This mirrors the structure of ``reflection.py`` (Phase 3 critic) and
``planning_gate.py`` (Phase 4 plan):

  - Cheap model (glm-4.7): the step-back pass is a separate, simple call —
    no need to spend the strong model on it.
  - Category-gated: fires only on REASONING/ANALYSIS/EXPERT turns. GREETING/
    SIMPLE turns don't justify the latency, and CODE turns are tool-heavy
    where principle-retrieval rarely changes the patch.
  - Non-blocking: on any failure (network, parse, exception), the original
    turn proceeds unchanged — no empty principle is ever injected.

Cost: ~1 cheap-model call per qualifying turn. Bounded by the category gate +
the step_back flag (default off until proven on test).

Wiring: called from ``turn_finalizer.finalize_turn`` (alongside the critic),
NOT from the main tool loop — the principle is derived from the question
alone and injected into the response context. See ``turn_finalizer.py`` for
the single call site.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Categories where stepping back adds value. Mirrors the critic/planner gates.
_STEP_BACK_CATEGORIES = {"reasoning", "analysis", "expert"}

# System prompt: ask for the governing principle, not the answer.
_STEP_BACK_SYSTEM_PROMPT = """\
You are a reasoning coach. Given a specific question, identify the ONE \
underlying principle, concept, or framework that governs it. The goal is to \
"step back" from the specific instance to the general rule that an expert \
would apply.

Respond in EXACTLY this format:
PRINCIPLE:
<one or two sentences stating the governing principle or framework — no \
preamble, no restating the question>

Rules:
- Name the principle/concept/framework, not a procedure or answer.
- Be concise: 1-2 sentences maximum.
- If the question is purely factual or definitional with no deeper principle \
(e.g. "what year did X happen"), respond with exactly: PRINCIPLE:\nN/A
"""


def _parse_step_back_response(content: str) -> Optional[str]:
    """Parse the step-back response.

    Returns:
      The principle string, or None if the response is unparseable or N/A.
    """
    if not content:
        return None
    text = content.strip()

    # Extract everything after PRINCIPLE:
    lines = text.split("\n")
    principle_lines = []
    found_marker = False
    for line in lines:
        if found_marker:
            principle_lines.append(line)
        elif line.strip().lower().startswith("principle:"):
            rest = line.split(":", 1)[1] if ":" in line else ""
            if rest.strip():
                principle_lines.append(rest)
            found_marker = True

    if not principle_lines:
        logger.debug("Step-back response unparseable: %s", content[:120])
        return None

    principle = "\n".join(principle_lines).strip()
    if not principle or principle.lower() == "n/a":
        return None
    return principle


def run_step_back(agent, user_message: str) -> Optional[str]:
    """Run a step-back pass on the user's question. Returns the governing
    principle string, or None on any skip/failure (never raises).

    This is the single entry point. It handles all gating (flag, category),
    failure, and cost control internally.

    Args:
        agent: The AIAgent instance (for runtime credentials + category lookup).
        user_message: The original user message.

    Returns:
        The principle string to inject as context, or None when the step-back
        is skipped (flag off, wrong category, no principle found, or any
        failure). Callers MUST treat None as "no-op" and proceed unchanged.
    """
    if not user_message or not user_message.strip():
        return None

    # Gate 1: flag must be on.
    try:
        from agent.feature_flags import step_back_enabled
        if not step_back_enabled():
            return None
    except Exception:
        return None  # flag module unavailable → skip

    # Gate 2: category must be in the step-back-eligible set.
    # Same semantic-first resolution as reflection.py / planning_gate.py:
    # try classify_semantic (96.7%) first, fall back to keyword (66.7%) on None.
    try:
        from agent.routing import TaskClassifier
        category = None
        try:
            from agent.feature_flags import semantic_classifier_enabled
            if semantic_classifier_enabled():
                category = TaskClassifier.classify_semantic(user_message)
        except Exception:
            pass
        if category is None:
            category = TaskClassifier.classify(user_message)
        if category.value not in _STEP_BACK_CATEGORIES:
            return None
    except Exception:
        return None  # classifier unavailable → skip

    # Make the step-back call. Never raises — any failure returns None.
    try:
        from agent.auxiliary_client import call_llm

        provider = "zai"
        model = "glm-4.7"
        # Allow config override: auxiliary.step_back.{provider,model}
        try:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
            aux = cfg.get("auxiliary", {})
            if isinstance(aux, dict):
                sb_cfg = aux.get("step_back", {})
                if isinstance(sb_cfg, dict):
                    provider = sb_cfg.get("provider", provider)
                    model = sb_cfg.get("model", model)
        except Exception:
            pass

        messages = [
            {"role": "system", "content": _STEP_BACK_SYSTEM_PROMPT},
            {"role": "user", "content": user_message[:2000]},
        ]

        response = call_llm(
            provider=provider,
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=120,  # one or two sentences
            timeout=15.0,
            main_runtime=agent._current_main_runtime(),
        )
        content = response.choices[0].message.content
        principle = _parse_step_back_response(content)

        if principle is not None:
            logger.info(
                "Step-back principle for %s turn: %s",
                category.value, principle[:100],
            )
        else:
            logger.debug("Step-back returned no principle for %s turn", category.value)
        return principle

    except Exception as exc:
        logger.warning("Step-back failed (no principle injected): %s", exc)
        return None
