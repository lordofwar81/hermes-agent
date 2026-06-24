"""Feature flags for the Gulli design-pattern work.

Each flag follows the established env-var-overrides-config idiom (see
``AIAgent._file_mutation_verifier_enabled`` in ``run_agent.py:2568``), with one
key difference: these default to **off**.  The Gulli-pattern features ship
behind flags so they can be developed and proven on a test box before being
flipped live on Strix.

Resolution order per flag:
  1. ``HERMES_<NAME>_ENABLED`` env var — truthy unless in ``{0,false,no,off}``.
  2. ``feature_flags.<name>`` in ``config.yaml`` (bool).
  3. ``False`` (safe default — feature stays off until explicitly enabled).

Exposed as module-level functions (not ``AIAgent`` methods) because the call
sites span ``routing.py``, ``turn_finalizer.py``, ``conversation_loop.py`` and
``delegate_tool.py`` — several of which have no agent instance in scope.
"""

from __future__ import annotations

import os
from typing import Dict, Any

__all__ = [
    "eval_enabled",
    "semantic_classifier_enabled",
    "reflection_enabled",
    "planning_gate_enabled",
    "personas_enabled",
]

# Strings that count as "off" for a truthy env-var check.  Mirrors the
# ``_file_mutation_verifier_enabled`` set exactly (run_agent.py:2580).
_FALSEY = {"0", "false", "no", "off"}


def _load_cfg() -> Dict[str, Any]:
    """Best-effort config load.  Never raises — returns ``{}`` on any failure.

    Imported lazily inside the function so importing this module never triggers
    a config read (and the startup-time import cycle that ``_file_mutation_verifier_enabled``
    explicitly avoids at run_agent.py:2583-2587).
    """
    try:
        from hermes_cli.config import load_config as _load_config
        return _load_config() or {}
    except Exception:
        return {}


def _flag(env_var: str, config_key: str, default: bool = False) -> bool:
    """Resolve a single boolean flag.

    Args:
        env_var: e.g. ``"HERMES_REFLECTION_ENABLED"``.
        config_key: dotted path under ``feature_flags`` in config.yaml,
            e.g. ``"reflection"`` → ``feature_flags.reflection``.
        default: returned when neither env nor config speaks.  ``False`` for
            all Gulli-pattern flags (develop-behind-flags).
    """
    try:
        env = os.environ.get(env_var)
        if env is not None:
            return env.strip().lower() not in _FALSEY
        cfg = _load_cfg()
        section = cfg.get("feature_flags") if isinstance(cfg, dict) else None
        if isinstance(section, dict) and config_key in section:
            return bool(section.get(config_key))
    except Exception:
        pass
    return default


def eval_enabled() -> bool:
    """Eval harness active (Phase 1).

    When on, the eval harness is the regression bar for every later phase.
    The flag gates *running* evals as part of the turn loop is NOT intended —
    evals are an offline tool.  This flag exists so the harness code path is
    wired but inert until you opt in.
    """
    return _flag("HERMES_EVAL_ENABLED", "eval")


def semantic_classifier_enabled() -> bool:
    """Embedding-based semantic classifier (Phase 2, Gulli Ch2/Ch16).

    When on, ``Router.route`` tries ``TaskClassifier.classify_semantic`` first,
    falling back to the keyword ``classify`` on ``None``.  Off → keyword only
    (the current, pre-Phase-2 behavior, byte-identical).
    """
    return _flag("HERMES_SEMANTIC_CLASSIFIER_ENABLED", "semantic_classifier")


def reflection_enabled() -> bool:
    """Pre-delivery critic / reflection loop (Phase 3, Gulli Ch4 Producer-Critic).

    When on, ``finalize_turn`` runs a synchronous critic pass (cheap aux model)
    on REASONING/ANALYSIS/EXPERT turns and may revise the answer once before
    the result dict is assembled.  Off → no critic, current behavior.
    """
    return _flag("HERMES_REFLECTION_ENABLED", "reflection")


def planning_gate_enabled() -> bool:
    """Mandatory planning phase for complex turns (Phase 4, Gulli Ch6).

    When on, EXPERT/CODE/ANALYSIS turns generate an explicit plan (steps +
    done-criteria) before tool dispatch.  Off → current behavior (kanban
    decompose is an optional CLI tool, not a turn-loop gate).
    """
    return _flag("HERMES_PLANNING_GATE_ENABLED", "planning_gate")


def personas_enabled() -> bool:
    """Role-specialized delegate personas (Phase 5, Gulli Ch7 Expert Teams).

    When on, the delegate schema accepts an optional ``persona`` field that
    injects a fixed system-prompt prefix from the persona registry.  Off → the
    ``persona`` field is ignored (schema still accepts it for forward-compat,
    but no prefix is injected).
    """
    return _flag("HERMES_PERSONAS_ENABLED", "personas")
