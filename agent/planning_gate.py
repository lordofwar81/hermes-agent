"""Planning gate — mandatory plan for complex turns (Gulli Ch6 plan-then-execute).

For EXPERT/CODE/ANALYSIS turns, generates an explicit plan (numbered steps +
done-criteria) before tool dispatch begins. The plan is surfaced to the user
via the status buffer so they can see and steer the approach.

This is NOT a blocking gate — it doesn't pause execution or require user
approval before proceeding. The plan shapes the agent's approach (it's
injected as a system-level hint into the conversation) but doesn't block the
tool-calling loop. A blocking approval gate was considered and rejected: it
would add latency and break conversational flow for most turns. Users who
want explicit approval can use the existing /steer or interrupt mechanism.

Design:
  - Cheap model (glm-4.7): the plan is a quick structural pass, not a
    deep reasoning task. The main model still does the actual execution.
  - Category-gated: fires only on EXPERT/CODE/ANALYSIS. GREETING/SIMPLE/
    REASONING turns don't benefit from a plan (too short or explanatory).
  - Non-blocking: the plan is generated and surfaced, then execution proceeds.
    On any failure, execution proceeds without a plan (same as pre-Phase-4).
  - Uses semantic-first classifier resolution (same as reflection.py) so the
    gate fires on the right turns.

Cost: ~1 cheap-model call per qualifying turn. Bounded by category gate +
the planning_gate flag (default off until proven on test).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Categories where a plan adds value. REASONING turns are explanatory (no
# multi-step execution to plan). GREETING/SIMPLE are too short.
_PLAN_CATEGORIES = {"expert", "code", "analysis"}

# The planning prompt — asks for structured output so the plan is parseable
# and presentable. Kept concise so the cheap model can handle it reliably.
_PLAN_SYSTEM_PROMPT = """\
You are a planning assistant. Given a user's request, produce a concise \
execution plan that an AI agent will follow to complete it.

Respond in this format:
PLAN:
1. <step description>
2. <step description>
...
DONE_CRITERIA:
- <criterion for knowing the task is complete>
- <another criterion>

Rules:
- Keep the plan to 3-7 steps. Don't over-decompose.
- Each step should be a concrete action (research, write, test, deploy), not \
a vague goal.
- DONE_CRITERIA should be verifiable (e.g. "tests pass", "API returns 200", \
not "it works").
- If the request is simple enough to do in one step, say so: \
"PLAN:\n1. <the single step>"
"""


@dataclass
class Plan:
    """A generated execution plan."""
    steps: list  # list of step strings
    done_criteria: list  # list of criterion strings

    def render(self) -> str:
        """Render the plan as a user-presentable string."""
        lines = ["📋 **Execution Plan:**"]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"   {i}. {step}")
        if self.done_criteria:
            lines.append("   **Done when:**")
            for c in self.done_criteria:
                lines.append(f"   ✓ {c}")
        return "\n".join(lines)


def _parse_plan(content: str) -> Optional[Plan]:
    """Parse the LLM's plan response into a Plan object.

    Returns None if the response doesn't contain a valid plan (unparseable,
    empty, or missing the PLAN: marker).
    """
    if not content:
        return None
    text = content.strip()

    # Find the PLAN: and DONE_CRITERIA: sections.
    steps: list = []
    criteria: list = []

    in_plan = False
    in_criteria = False
    for line in text.split("\n"):
        stripped = line.strip()
        lower = stripped.lower()

        if lower.startswith("plan:"):
            in_plan = True
            in_criteria = False
            # Check if content is on the same line after the colon.
            rest = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
            if rest:
                steps.append(rest)
            continue
        if lower.startswith("done_criteria:") or lower.startswith("done criteria:"):
            in_plan = False
            in_criteria = True
            rest = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
            if rest:
                criteria.append(rest)
            continue

        if in_plan or in_criteria:
            # Parse numbered or bulleted lines.
            # Strip leading "1." "2." "-" "✓" etc.
            cleaned = stripped.lstrip("0123456789.-*✓ ").strip()
            if cleaned:
                if in_plan:
                    steps.append(cleaned)
                else:
                    criteria.append(cleaned)

    if not steps:
        return None
    return Plan(steps=steps, done_criteria=criteria)


def build_plan(agent, user_message: str) -> Optional[Plan]:
    """Generate an execution plan for a complex turn.

    Returns the Plan, or None if the planning gate is off, the category isn't
    eligible, or any failure occurs. Never raises.

    This is the single entry point, called from ``run_conversation`` before
    the main tool-calling loop. When a plan is generated, it's surfaced to
    the user via the status buffer.

    Args:
        agent: The AIAgent instance.
        user_message: The user's message text.

    Returns:
        A Plan object, or None.
    """
    if not user_message or not user_message.strip():
        return None

    # Gate 1: flag must be on.
    try:
        from agent.feature_flags import planning_gate_enabled
        if not planning_gate_enabled():
            return None
    except Exception:
        return None  # flag module unavailable → skip planning

    # Gate 2: category must be plan-eligible. Semantic-first resolution
    # (same as reflection.py) so the gate fires on the right turns.
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
        if category.value not in _PLAN_CATEGORIES:
            return None
    except Exception:
        return None  # classifier unavailable → skip planning

    # Generate the plan. Never raises — any failure returns None.
    try:
        from agent.auxiliary_client import call_llm

        provider = "zai"
        model = "glm-4.7"
        try:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
            aux = cfg.get("auxiliary", {})
            if isinstance(aux, dict):
                planner_cfg = aux.get("planner", {})
                if isinstance(planner_cfg, dict):
                    provider = planner_cfg.get("provider", provider)
                    model = planner_cfg.get("model", model)
        except Exception:
            pass

        messages = [
            {"role": "system", "content": _PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": user_message[:2000]},
        ]

        response = call_llm(
            provider=provider,
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=800,  # enough for 3-7 steps + criteria
            timeout=15.0,
            main_runtime=agent._current_main_runtime(),
        )
        content = response.choices[0].message.content
        plan = _parse_plan(content)

        if plan is not None:
            logger.info(
                "Plan generated for %s turn: %d steps, %d criteria",
                category.value, len(plan.steps), len(plan.done_criteria),
            )
            # Surface the plan to the user.
            try:
                agent._buffer_status(plan.render())
            except Exception:
                pass
            return plan
        else:
            logger.debug("Plan parsing failed for %s turn", category.value)
            return None

    except Exception as exc:
        logger.warning("Planning gate failed: %s", exc)
        return None
