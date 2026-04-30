"""Priority routing integration - activates Local-First + Budget Enforcement.

This module integrates the new priority router components into the existing
routing system. It provides a unified entry point that:

1. Detects trivial tasks and routes to local/Z.ai (NEVER Venice)
2. Enforces Venice budget limits before any routing decision
3. Uses the fallback chain for graceful degradation
4. Prioritizes free/fast models (local → Z.ai) before budgeted Venice

Usage:
    >>> from agent.priority_routing_integration import route_with_priority
    >>> decision = route_with_priority(message, primary_config, smart_routing_config)
    >>> if decision.never_use_venice:
    ...     # Route to local/Z.ai
    ...     return decision.route
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

import yaml

from agent.trivial_task_classifier import TrivialTaskClassifier, ComplexityDetector
from agent.budget_enforcer import VeniceBudgetEnforcer
from agent.fallback_chain import ModelFallbackChain
from agent.priority_router import PriorityRouter

logger = logging.getLogger(__name__)


@dataclass
class PriorityRoutingDecision:
    """Routing decision with priority-based analysis."""

    route: dict[str, Any]
    never_use_venice: bool
    reason: str
    complexity: Optional[str] = None
    trivial_type: Optional[str] = None
    budget_exhausted: bool = False


def route_with_priority(
    message: str,
    primary: dict[str, Any],
    smart_routing_config: dict[str, Any],
    config_path: str = None,
) -> PriorityRoutingDecision:
    """Route a message with priority-based Local-First strategy.

    This is the main entry point that integrates all priority routing components:
    1. Trivial task detection (greetings, chitchat, acknowledgments)
    2. Complexity detection (simple/medium/complex/expert)
    3. Budget enforcement (Venice budget status)
    4. Fallback chain (local → Z.ai → Venice)

    Priority order:
    - Trivial tasks: NEVER use Venice, route to local/Z.ai
    - Budget exhausted: NEVER use Venice, route to local/Z.ai
    - Complex tasks: Use intelligent routing with fallback chain

    Args:
        message: The user message to route
        primary: Primary route config (model, api_key, provider, etc.)
        smart_routing_config: Smart model routing config from YAML
        config_path: Path to config.yaml file

    Returns:
        PriorityRoutingDecision with route and analysis
    """
    # Initialize components
    router = PriorityRouter(config_path)
    budget_enforcer = VeniceBudgetEnforcer(config_path)

    # Step 1: Check budget FIRST (before any routing)
    budget_status = budget_enforcer.get_budget_status()
    budget_exhausted = budget_status["remaining_budget"] <= 0

    if budget_exhausted:
        # Budget exhausted - use fallback chain (local → Z.ai)
        logger.info(
            f"Venice budget exhausted (${budget_status['spent_usd']:.2f}/${budget_status['daily_limit_usd']:.2f}). "
            f"Routing to local/Z.ai fallback chain."
        )

        # Get fallback chain
        fallback_chain = ModelFallbackChain(config_path)
        fallback_route = fallback_chain.get_fallback_route(message)

        if fallback_route:
            # Ensure fallback_route has all required fields
            if "signature" not in fallback_route:
                fallback_route["signature"] = (
                    fallback_route.get("model"),
                    fallback_route.get("provider", "unknown"),
                    fallback_route.get("base_url", ""),
                    fallback_route.get("api_mode", ""),
                    fallback_route.get("command", ""),
                    tuple(fallback_route.get("args") or ()),
                )
            if "label" not in fallback_route:
                fallback_route["label"] = None
            if "request_overrides" not in fallback_route:
                fallback_route["request_overrides"] = None
            return PriorityRoutingDecision(
                route=fallback_route,
                never_use_venice=True,
                reason=f"Budget exhausted - using fallback chain",
                budget_exhausted=True,
            )

        # If no fallback available, return primary (will fail, but better than Venice)
        return PriorityRoutingDecision(
            route={
                "model": primary.get("model"),
                "runtime": {
                    "api_key": primary.get("api_key"),
                    "base_url": primary.get("base_url"),
                    "provider": primary.get("provider"),
                    "api_mode": primary.get("api_mode"),
                    "command": primary.get("command"),
                    "args": list(primary.get("args") or []),
                    "credential_pool": primary.get("credential_pool"),
                },
                "label": None,
                "signature": (
                    primary.get("model"),
                    primary.get("provider"),
                    primary.get("base_url"),
                    primary.get("api_mode"),
                    primary.get("command"),
                    tuple(primary.get("args") or ()),
                ),
                "request_overrides": None,
            },
            never_use_venice=True,
            reason="Budget exhausted - no fallback available",
            budget_exhausted=True,
        )

    # Step 2: Detect triviality
    trivial_classifier = TrivialTaskClassifier(config_path)
    trivial_result = asyncio.run(trivial_classifier.classify(message))
    is_trivial = trivial_result["trivial"]

    if is_trivial:
        # Trivial task - NEVER use Venice
        logger.info(
            f"Trivial task detected ({trivial_result['type']}): routing to local/Z.ai, skipping Venice"
        )

        # Use fallback chain for trivial tasks
        fallback_chain = ModelFallbackChain(config_path)
        fallback_route = fallback_chain.get_fallback_route(message)

        if fallback_route:
            # Ensure fallback_route has all required fields
            if "signature" not in fallback_route:
                fallback_route["signature"] = (
                    fallback_route.get("model"),
                    fallback_route.get("provider", "unknown"),
                    fallback_route.get("base_url", ""),
                    fallback_route.get("api_mode", ""),
                    fallback_route.get("command", ""),
                    tuple(fallback_route.get("args") or ()),
                )
            if "label" not in fallback_route:
                fallback_route["label"] = None
            if "request_overrides" not in fallback_route:
                fallback_route["request_overrides"] = None
            return PriorityRoutingDecision(
                route=fallback_route,
                never_use_venice=True,
                reason=f"Trivial task ({trivial_result['type']}) - using local/Z.ai",
                trivial_type=trivial_result["type"],
            )

        # Fallback to primary (local model)
        return PriorityRoutingDecision(
            route={
                "model": primary.get("model"),
                "runtime": {
                    "api_key": primary.get("api_key"),
                    "base_url": primary.get("base_url"),
                    "provider": primary.get("provider"),
                    "api_mode": primary.get("api_mode"),
                    "command": primary.get("command"),
                    "args": list(primary.get("args") or []),
                    "credential_pool": primary.get("credential_pool"),
                },
                "label": None,
                "signature": (
                    primary.get("model"),
                    primary.get("provider"),
                    primary.get("base_url"),
                    primary.get("api_mode"),
                    primary.get("command"),
                    tuple(primary.get("args") or ()),
                ),
                "request_overrides": None,
            },
            never_use_venice=True,
            reason=f"Trivial task ({trivial_result['type']}) - using primary",
            trivial_type=trivial_result["type"],
        )

    # Step 3: Detect complexity
    complexity_detector = ComplexityDetector(config_path)
    complexity = complexity_detector.detect(message)

    # Step 4: Use intelligent model selector if enabled
    if smart_routing_config.get("use_model_selector", False):
        from agent.model_selector import smart_select_route

        route = smart_select_route(message, smart_routing_config, primary)

        if route:
            logger.info(
                f"Model selector chose: {route['model']} ({route['runtime'].get('provider')})"
            )
            return PriorityRoutingDecision(
                route=route,
                never_use_venice=False,
                reason="Model selector routing",
                complexity=complexity,
            )

    # Step 5: Use existing binary classifier as fallback
    from agent.smart_model_routing import resolve_turn_route

    route = resolve_turn_route(message, smart_routing_config, primary)

    return PriorityRoutingDecision(
        route=route,
        never_use_venice=False,
        reason="Binary classifier routing",
        complexity=complexity,
    )


def check_venice_budget_and_block(
    config_path: str = None,
) -> tuple[bool, str]:
    """Check Venice budget and return (should_allow, reason).

    This is a standalone function to check budget before any routing decision.
    Use this when you want to block Venice usage entirely (e.g., before starting
    a long conversation).

    Args:
        config_path: Path to config.yaml file (defaults to ~/.hermes/config.yaml)

    Returns:
        Tuple of (should_allow_venice, reason_string)
    """
    enforcer = VeniceBudgetEnforcer(config_path)
    budget_status = enforcer.get_budget_status()

    remaining = budget_status["remaining_budget"]
    spent = budget_status["spent_usd"]
    limit = budget_status["daily_limit_usd"]

    if remaining <= 0:
        return (
            False,
            f"Venice budget exhausted (${spent:.2f}/${limit:.2f}). "
            f"Resetting at {budget_status['reset_time']}.",
        )

    # Check warning thresholds
    usage_pct = (spent / limit) * 100

    if usage_pct >= 90:
        return (
            False,
            f"Venice budget at {usage_pct:.1f}% ({spent:.2f}/${limit:.2f}). "
            f"Using fallback chain instead.",
        )
    elif usage_pct >= 75:
        logger.warning(
            f"Venice budget at {usage_pct:.1f}% ({spent:.2f}/${limit:.2f}). "
            f"Consider using local/Z.ai models."
        )

    return (True, f"Venice budget OK (${remaining:.2f} remaining)")


def get_priority_routing_summary(
    message: str,
    config_path: str = None,
) -> dict[str, Any]:
    """Get a summary of priority routing analysis.

    This provides a detailed breakdown of:
    - Triviality detection
    - Complexity analysis
    - Budget status
    - Recommended routing

    Args:
        message: The message to analyze
        config_path: Path to config.yaml file

    Returns:
        Dict with routing analysis summary
    """
    router = PriorityRouter(config_path)
    budget_enforcer = VeniceBudgetEnforcer(config_path)

    # Get routing decision
    decision = route_with_priority(message, {}, {}, config_path)

    return {
        "message": message,
        "never_use_venice": decision.never_use_venice,
        "reason": decision.reason,
        "complexity": decision.complexity,
        "trivial_type": decision.trivial_type,
        "budget_exhausted": decision.budget_exhausted,
        "route_model": decision.route.get("model"),
    }
