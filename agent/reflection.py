"""Pre-delivery reflection / critic loop (Gulli Ch4 Producer-Critic).

A separate critic pass evaluates the agent's final answer against quality
criteria before it's delivered to the user. If the critic finds a substantive
problem, it returns a revised answer (revise-once, no recursion).

This is SYNCHRONOUS — it runs inline in ``finalize_turn`` before the result
dict is assembled, unlike ``background_review.py`` which is a fire-and-forget
daemon thread for memory/skill review. The critic must be able to mutate the
answer the user receives; a background thread cannot.

Design:
  - Cheap model (glm-4.7): the critic is a separate LLM from the author
    (Gulli Ch11: never use the same LLM as both author and judge).
  - Category-gated: fires only on REASONING/ANALYSIS/EXPERT turns (where
    quality matters most; greetings/simple don't justify the latency).
  - Revise-once: one critic pass. If revised, the revision is final — no
    re-critique loop (bounded cost and latency).
  - Never blocks delivery: on any failure (network, parse, exception), the
    original ``final_response`` is returned unchanged.

Cost: ~1 cheap-model call per qualifying turn. Bounded by category gate +
the reflection flag (default off until proven on test).
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Categories where the critic adds value. GREETING/SIMPLE turns are short and
# low-stakes — the latency isn't justified. CODE turns are often tool-heavy
# and the answer's correctness depends on execution, not prose quality.
_CRITIC_CATEGORIES = {"reasoning", "analysis", "expert"}

# Critic system prompt. Asks for a structured verdict so parsing is reliable.
_CRITIC_SYSTEM_PROMPT = """\
You are a quality critic for an AI assistant's answer. Given the user's \
question and the assistant's answer, evaluate the answer against these criteria:

1. Correctness: Are the facts and reasoning accurate?
2. Completeness: Does it address the full question, not just part?
3. Groundedness: Does it avoid unsupported claims or hallucination?
4. Clarity: Is it clear and well-structured?

Respond in EXACTLY this format:
VERDICT: ACCEPT
or
VERDICT: REVISE
REVISED_ANSWER:
<your improved version of the answer, complete and self-contained>

Rules:
- Use ACCEPT if the answer is good enough to deliver as-is.
- Use REVISE only if there's a substantive problem (wrong fact, missing key \
point, hallucination, serious clarity issue). Don't revise for style alone.
- The revised answer must be a COMPLETE answer, not a diff or correction note.
"""


def _parse_critic_response(content: str) -> Optional[str]:
    """Parse the critic's response.

    Returns:
      None if the critic says ACCEPT (or the response is unparseable).
      The revised answer string if the critic says REVISE.
    """
    if not content:
        return None
    text = content.strip()

    # Check for VERDICT line (case-insensitive).
    lines = text.split("\n")
    verdict_line = ""
    for line in lines:
        if line.strip().lower().startswith("verdict:"):
            verdict_line = line.strip().lower()
            break

    if "accept" in verdict_line:
        return None  # Accept — no revision needed.

    if "revise" in verdict_line:
        # Extract everything after REVISED_ANSWER:
        revised_lines = []
        found_marker = False
        for line in lines:
            if found_marker:
                revised_lines.append(line)
            elif line.strip().lower().startswith("revised_answer:"):
                # Content on same line after the colon.
                rest = line.split(":", 1)[1] if ":" in line else ""
                if rest.strip():
                    revised_lines.append(rest)
                found_marker = True
        if revised_lines:
            revised = "\n".join(revised_lines).strip()
            if revised:
                return revised

    # Unparseable → treat as accept (don't block delivery with a bad parse).
    logger.debug("Critic response unparseable, treating as ACCEPT: %s",
                 content[:120])
    return None


def run_critic(agent, final_response: str, user_message: str) -> str:
    """Run the critic on the agent's final answer. Returns the (possibly revised) response.

    This is the single entry point, called from ``finalize_turn``. It handles
    all gating (flag, category), failure, and cost control internally.

    Args:
        agent: The AIAgent instance (for runtime credentials + category lookup).
        final_response: The agent's final answer text.
        user_message: The original user message (for context).

    Returns:
        The response to deliver — either the original (ACCEPT / failure / skipped)
        or the revised version (REVISE).
    """
    if not final_response or not final_response.strip():
        return final_response

    # Gate 1: flag must be on.
    try:
        from agent.feature_flags import reflection_enabled
        if not reflection_enabled():
            return final_response
    except Exception:
        return final_response  # flag module unavailable → skip critic

    # Gate 2: category must be in the critic-eligible set.
    # Uses the same semantic-first resolution as Router.route: when the
    # semantic flag is on, try classify_semantic (96.7% accurate) first,
    # fall back to keyword classify (66.7%) on None. This ensures the critic
    # fires on the right turns — without it, "Design a rate limiter" hits a
    # code keyword and skips the critic even though it's an EXPERT turn.
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
        if category.value not in _CRITIC_CATEGORIES:
            return final_response
    except Exception:
        return final_response  # classifier unavailable → skip critic

    # Make the critic call. Never raises — any failure returns the original.
    try:
        from agent.auxiliary_client import call_llm

        provider = "zai"
        model = "glm-4.7"
        # Allow config override: auxiliary.critic.{provider,model}
        try:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
            aux = cfg.get("auxiliary", {})
            if isinstance(aux, dict):
                critic_cfg = aux.get("critic", {})
                if isinstance(critic_cfg, dict):
                    provider = critic_cfg.get("provider", provider)
                    model = critic_cfg.get("model", model)
        except Exception:
            pass

        messages = [
            {"role": "system", "content": _CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"User question:\n{user_message[:2000]}\n\n"
                f"Assistant's answer:\n{final_response[:4000]}"
            )},
        ]

        response = call_llm(
            provider=provider,
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=max(len(final_response) * 2, 2000),  # room for a full revision
            timeout=30.0,
            main_runtime=agent._current_main_runtime(),
        )
        content = response.choices[0].message.content
        revised = _parse_critic_response(content)

        if revised is not None:
            logger.info(
                "Critic REVISED answer for %s turn "
                "(orig %d chars → revised %d chars)",
                category.value, len(final_response), len(revised),
            )
            # Emit a subtle status so the user knows a revision happened.
            try:
                agent._buffer_status("✏️ Answer reviewed and revised by critic")
            except Exception:
                pass
            return revised
        else:
            logger.debug("Critic ACCEPTED answer for %s turn", category.value)
            return final_response

    except Exception as exc:
        logger.warning("Critic failed (returning original answer): %s", exc)
        return final_response
