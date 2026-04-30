"""Priority router for model selection with Local-First strategy.

Implements intelligent routing that prioritizes free, fast models (local → Z.ai)
before using budgeted Venice models. Uses ML-based detection for accuracy.

Classes:
    PriorityRouter: Priority-based model router with Local-First strategy
"""

from __future__ import annotations

import asyncio
from typing import Any

import yaml

from agent.trivial_task_classifier import TrivialTaskClassifier, ComplexityDetector
from agent.budget_enforcer import VeniceBudgetEnforcer
from agent.fallback_chain import ModelFallbackChain


class PriorityRouter:
    """Priority-based model router with Local-First strategy."""

    def __init__(self, config_path: str = None):
        """Initialize router with config loading.

        Args:
            config_path: Path to config.yaml file (defaults to ~/.hermes/config.yaml)
        """
        if config_path is None:
            config_path = "/home/lordofwarai/.hermes/config.yaml"
        self._config_path = config_path
        self._load_config()

        # Initialize components
        self._trivial_classifier = TrivialTaskClassifier(config_path)
        self._complexity_detector = ComplexityDetector(config_path)
        self._budget_enforcer = VeniceBudgetEnforcer(config_path)
        self._fallback_chain = ModelFallbackChain(config_path)

        # Routing strategy
        self._local_first = self._config.get("routing", {}).get("local_first", True)
        self._zai_priority = self._config.get("routing", {}).get("zai_priority", True)

    def _load_config(self) -> None:
        """Load configuration from YAML file.

        Parses routing settings from config.yaml:
        - routing.local_first: true (prioritize free models)
        - routing.zai_priority: true (prefer Z.ai over Venice)
        - complexity_detection.method: "ml" (ML-based detection)
        """
        with open(self._config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

        # Get complexity detection settings
        complexity_config = self._config.get("complexity_detection", {})
        self._complexity_method = complexity_config.get("method", "ml")

        # Get fallback settings
        fallback_config = self._config.get("fallback", {})
        self._fallback_to_zai = fallback_config.get("use_zai_for_complexity", True)

    def get_routing_decision(self, message: str) -> dict[str, Any]:
        """Get routing decision for a message.

        Priority:
        1. Trivial task detection (greetings, chitchat, acknowledgments)
        2. Complexity detection (simple/medium/complex/expert)
        3. Budget check (Venice budget status)
        4. Fallback chain (local → Z.ai → Venice)

        Args:
            message: The message to route

        Returns:
            Dict with routing decision
        """
        # Step 1: Detect triviality
        trivial_result = asyncio.run(self._trivial_classifier.classify(message))
        is_trivial = trivial_result["trivial"]

        # Step 2: Detect complexity
        complexity = asyncio.run(self._complexity_detector.detect(message))

        # Step 3: Check budget
        budget_status = self._budget_enforcer.get_budget_status()
        budget_exhausted = budget_status["remaining_budget"] <= 0

        # Step 4: Determine routing
        routing = self._determine_routing(
            message, is_trivial, complexity, budget_exhausted
        )

        return {
            "message": message,
            "is_trivial": is_trivial,
            "trivial_type": trivial_result.get("type", "unknown"),
            "complexity": complexity,
            "budget_status": budget_status,
            "budget_exhausted": budget_exhausted,
            "routing": routing,
        }

    def _determine_routing(
        self, message: str, is_trivial: bool, complexity: str, budget_exhausted: bool
    ) -> dict[str, Any]:
        """Determine routing based on triviality, complexity, and budget.

        Logic:
        1. Trivial tasks: NEVER use Venice
        2. Simple/medium: prefer local/Z.ai (never Venice)
        3. Complex: use local/Z.ai first, Venice only if failed
        4. Expert: use Venice if budget available and fallback failed

        Args:
            message: The message to route
            is_trivial: Whether task is trivial
            complexity: Task complexity level
            budget_exhausted: Whether budget is exhausted

        Returns:
            Dict with routing decision
        """
        # Trivial tasks: route to local/Z.ai (never Venice)
        if is_trivial:
            return {
                "should_use_venice": False,
                "reason": "trivial task - route to local/Z.ai",
                "priority_model": "local" if self._local_first else "zai",
                "fallback_chain": self._get_fallback_chain(
                    is_trivial, complexity, budget_exhausted
                ),
            }

        # Check budget
        if budget_exhausted:
            return {
                "should_use_venice": False,
                "reason": "budget exhausted",
                "priority_model": "local" if self._local_first else "zai",
                "fallback_chain": self._get_fallback_chain(
                    is_trivial, complexity, budget_exhausted
                ),
            }

        # Simple/medium: prefer free models (never Venice)
        if complexity in ("simple", "medium"):
            return {
                "should_use_venice": False,
                "reason": f"{complexity} task - prefer local/Z.ai",
                "priority_model": "local" if self._local_first else "zai",
                "fallback_chain": self._get_fallback_chain(
                    is_trivial, complexity, budget_exhausted
                ),
            }

        # Complex: use local/Z.ai first, Venice only if failed
        if complexity == "complex":
            return {
                "should_use_venice": True,  # Only if fallback chain fails
                "reason": "complex task - try local/Z.ai first",
                "priority_model": "local" if self._local_first else "zai",
                "fallback_chain": self._get_fallback_chain(
                    is_trivial, complexity, budget_exhausted
                ),
            }

        # Expert: use Venice if budget available
        if complexity == "expert":
            return {
                "should_use_venice": True,  # Always allowed if budget available
                "reason": "expert task - Venice allowed",
                "priority_model": "local" if self._local_first else "zai",
                "fallback_chain": self._get_fallback_chain(
                    is_trivial, complexity, budget_exhausted
                ),
            }

        # Unknown complexity: default to local/Z.ai
        return {
            "should_use_venice": False,
            "reason": "unknown complexity - prefer local/Z.ai",
            "priority_model": "local" if self._local_first else "zai",
            "fallback_chain": self._get_fallback_chain(
                is_trivial, complexity, budget_exhausted
            ),
        }

    def _get_fallback_chain(
        self, is_trivial: bool, complexity: str, budget_exhausted: bool
    ) -> list[dict[str, Any]]:
        """Get fallback chain based on conditions.

        Returns:
            List of model configurations in priority order
        """
        # Trivial tasks: local only
        if is_trivial:
            return [
                {"model": "local", "priority": 1},
                {"model": "zai", "priority": 2},
            ]

        # Budget exhausted: local/Z.ai only
        if budget_exhausted:
            return [
                {"model": "local", "priority": 1},
                {"model": "zai", "priority": 2},
            ]

        # Simple/medium: local/Z.ai only
        if complexity in ("simple", "medium"):
            return [
                {"model": "local", "priority": 1},
                {"model": "zai", "priority": 2},
            ]

        # Complex: local/Z.ai → Venice
        if complexity == "complex":
            return [
                {"model": "local", "priority": 1},
                {"model": "zai", "priority": 2},
                {"model": "venice", "priority": 3},
            ]

        # Expert: local/Z.ai → Venice
        if complexity == "expert":
            return [
                {"model": "local", "priority": 1},
                {"model": "zai", "priority": 2},
                {"model": "venice", "priority": 3},
            ]

        # Unknown: local/Z.ai only
        return [
            {"model": "local", "priority": 1},
            {"model": "zai", "priority": 2},
        ]

    async def route_message(self, message: str) -> dict[str, Any]:
        """Route message with comprehensive analysis.

        Args:
            message: The message to route

        Returns:
            Dict with routing decision and execution plan
        """
        # Get routing decision
        routing_decision = self.get_routing_decision(message)

        # Get priority model
        priority_model = routing_decision["routing"]["priority_model"]

        # Build execution plan
        execution_plan = {
            "primary_model": priority_model,
            "fallback_models": [],
            "should_use_venice": routing_decision["routing"]["should_use_venice"],
            "reason": routing_decision["routing"]["reason"],
        }

        # Add fallback models
        for model_info in routing_decision["routing"]["fallback_chain"]:
            if model_info["model"] != priority_model:
                execution_plan["fallback_models"].append(model_info["model"])

        return {
            **routing_decision,
            "execution_plan": execution_plan,
        }
