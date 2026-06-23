#!/usr/bin/env python3
"""
agent.routing — Single-source model routing for Hermes Agent.

Replaces all prior routing attempts (model_router.py, turn_router.py,
smart_model_routing.py, model_selector.py, priority_router.py, etc.).

Design principles:
  1. Config-driven: provider list comes from config.yaml, not hardcoded.
  2. Per-turn classification: classify message → route to optimal model.
  3. Circuit breaker: consecutive failures block provider temporarily.
  4. Venice budget gate: daily spend cap with auto-reset.
  5. Local health checks: pre-flight ping with 30s cache.
  6. Merge-safe: self-contained, no fragile cross-file wiring.

Entry point:
  route_turn(message, primary_config, routing_config) -> dict

Author: Hermes Routing v1 — 2026-05-29
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────

FAILURE_THRESHOLD = 3          # consecutive failures before circuit break
RECOVERY_TIMEOUT_SECONDS = 180  # 3 minutes per blocked provider
HEALTH_CHECK_TIMEOUT = 5        # seconds for local model pre-flight ping
HEALTH_CACHE_TTL = 30           # seconds between health checks
LOCAL_CONTEXT_RATIO = 0.6       # use 60% of local model context for safety


# ─── Task Categories ─────────────────────────────────────────────────────

class Category(Enum):
    """Message classification for routing decisions."""
    GREETING = "greeting"      # hi, thanks, ok — short social
    SIMPLE = "simple"          # short factual question, no tools needed
    CODE = "code"              # debugging, implementation, code blocks
    REASONING = "reasoning"    # explain why, how, design questions
    ANALYSIS = "analysis"      # compare, evaluate, audit, metrics
    EXPERT = "expert"          # full system design, end-to-end, multi-step


# ─── Data Classes ────────────────────────────────────────────────────────

@dataclass
class ProviderSlot:
    """A single model endpoint available for routing."""
    id: str                    # unique key (e.g., "zai-glm-5.1")
    provider: str              # group name (e.g., "zai", "strix", "mac_studio", "venice")
    model: str                 # model ID sent to API
    base_url: str              # API endpoint
    api_key: str               # resolved at init, not per-turn
    context_length: int = 0    # 0 = unknown/unlimited
    request_timeout: int = 0  # 0 = use default
    is_local: bool = False
    cost_per_1k_input: float = 0.0   # USD per 1K input tokens (0 = free)
    cost_per_1k_output: float = 0.0


@dataclass
class RouteResult:
    """Output of the routing decision."""
    model: str
    base_url: str
    api_key: str
    provider: str
    category: Category
    is_local: bool
    suppress_tools: bool       # True for greeting/simple to save tokens
    label: str                 # human-readable description
    fallback_count: int = 0    # how many providers were skipped


# ─── Task Classifier ──────────────────────────────────────────────────────

class TaskClassifier:
    """Deterministic, zero-dependency message classifier."""

    # ── Single source of truth for CODE keywords ──────────────────────────
    # Define each word/phrase ONCE; derive the boundary regex, the single-word
    # set, and the multi-word phrase set from these. Previously the same words
    # were duplicated across _CODE_KW, _CODE_BOUNDARY_KW, and the _CODE_BOUNDARY_RE
    # regex literal — easy to drift apart (3 prior "edge case fix" commits).
    _CODE_SINGLE_WORDS = frozenset({
        "debug", "debugging", "implement", "implementation", "patch",
        "traceback", "stacktrace", "exception", "fix", "bug", "module",
        "script", "server", "test", "security", "deploy", "compile",
        "refactor", "rewrite", "rework", "design",
        "cron", "docker", "compose",
        "container", "pipeline", "workflow",
    })
    _CODE_MULTIWORD_PHRASES = frozenset({
        "next step",
        "do it", "go on",
    })
    # Derived (do not edit — regenerate from the two sets above if changed)

    _CODE_BOUNDARY_KW = _CODE_SINGLE_WORDS
    _CODE_PHRASES = _CODE_MULTIWORD_PHRASES
    _CODE_KW = _CODE_SINGLE_WORDS | _CODE_MULTIWORD_PHRASES

    # Standalone continuation phrases — CODE only with a code keyword present.
    # These do NOT go into _CODE_KW; handled in classify() via _has_real_code_keyword.
    _CONTINUATION_PHRASES = frozenset({
        "continue", "proceed", "go ahead", "keep going",
        "move on", "move forward",
    })

    # ANALYSIS keywords strong enough to override CODE keywords.
    # "compare test results" → ANALYSIS (compare overrides test),
    # but "security audit needed" → CODE (audit is NOT in this set).
    _ANALYSIS_OVERRIDE_KW = frozenset({
        "analyze", "analysis", "compare", "contrast", "evaluate",
        "performance", "metrics", "benchmark",
        "teardown", "cost-benefit", "roi",
    })

    _ANALYSIS_KW = frozenset({
        "analyze", "analysis", "compare", "contrast", "evaluate",
        "performance", "metrics", "report", "audit", "benchmark",
        "teardown", "cost-benefit", "roi",
    })

    _REASONING_KW = frozenset({
        "explain why", "explain how", "why does", "how does",
        "what causes", "redesign", "improve", "tradeoff", "trade-off",
        "advantage", "disadvantage",
    })

    _GREETING_KW = frozenset({
        "hello", "hi", "hey", "thanks", "thank you", "ty",
        "cheers", "ok", "okay", "got it", "understood", "bye", "goodbye",
    })

    _EXPERT_PHRASES = [
        "design a system", "design a complete system", "design an architecture",
        "implement a complete", "end-to-end", "full system",
        "architect a", "architect the",
        "build a complete", "build an entire", "comprehensive system",
        "from scratch", "production-ready", "production ready", "multi-region",
        "system design", "system architecture",
    ]

    # Pre-compiled regex: single-word CODE keywords joined with \| for
    # a single word-boundary match. Derived from _CODE_SINGLE_WORDS so it
    # cannot drift from the frozenset above.
    _CODE_BOUNDARY_RE = re.compile(
        r'\b(?:' + '|'.join(re.escape(kw) for kw in sorted(_CODE_SINGLE_WORDS)) + r')\b'
    )

    @classmethod
    def _has_code_keyword(cls, text_lower: str) -> bool:
        """Check for code keywords using word boundaries for single words.

        Single-word keywords use a pre-compiled regex with \\b boundaries
        so 'fix' doesn't match 'fixed'. Multi-word phrases use substring match.
        Does NOT include continuation phrases (those are checked separately).
        """
        # Fast path: single compiled regex for all single-word keywords
        if cls._CODE_BOUNDARY_RE.search(text_lower):
            return True
        # Slow path: multi-word phrases (only 3 items)
        for kw in cls._CODE_PHRASES:
            if kw in text_lower:
                return True
        return False


    @classmethod
    def classify(cls, message: str) -> Category:
        text = (message or "").strip()
        text_lower = text.lower()

        # Code: code blocks (checked BEFORE greeting)
        if "```" in text or "`" in text:
            return Category.CODE

        # Greeting: short message with greeting keyword, but NOT if it contains
        # code intent (e.g., "ok, deploy it" is CODE, not GREETING).
        # Uses word-boundary matching so 'fix' doesn't block 'fixed' greetings.
        if len(text) <= 25:
            if not cls._has_code_keyword(text_lower):
                for g in cls._GREETING_KW:
                    if len(g.split()) == 1:
                        if re.search(r'\b' + re.escape(g) + r'\b', text_lower):
                            return Category.GREETING
                    elif g in text_lower:
                        return Category.GREETING

        # Expert: system design phrases (before code — expert tasks contain "implement")
        if any(phrase in text_lower for phrase in cls._EXPERT_PHRASES):
            return Category.EXPERT

        # Reasoning: explanation/design questions (BEFORE code — multi-word
        # reasoning phrases like "explain why" should override single-word
        # code vocabulary like "test" or "deploy" in explanation-seeking queries).
        if any(kw in text_lower for kw in cls._REASONING_KW):
            return Category.REASONING

        # Analysis override: when analysis keywords co-occur with code keywords,
        # analysis wins (e.g., "compare test results" → ANALYSIS, not CODE).
        # Only checked if at least one analysis keyword is present.
        if any(kw in text_lower for kw in cls._ANALYSIS_OVERRIDE_KW):
            return Category.ANALYSIS

        # Code: code keywords
        if cls._has_code_keyword(text_lower):
            return Category.CODE

        # Continuation phrases: "continue", "go ahead", "keep going", etc.
        # These become CODE only when a real code keyword is also present.
        # Standing alone (e.g., "continue the story", "proceed with the plan"),
        # they fall through to SIMPLE.
        if any(kw in text_lower for kw in cls._CONTINUATION_PHRASES):
            # Check if there's a real code keyword co-occurring (not another
            # continuation phrase). Use the word-boundary regex for speed.
            if cls._CODE_BOUNDARY_RE.search(text_lower):
                return Category.CODE
            # No code keyword → SIMPLE (fall through)

        # Analysis: comparison/evaluation (non-override path, no code kw overlap)
        if any(kw in text_lower for kw in cls._ANALYSIS_KW):
            return Category.ANALYSIS

        # Default: simple
        return Category.SIMPLE


# ─── Circuit Breaker ─────────────────────────────────────────────────────

class CircuitBreaker:
    """Per-provider circuit breaker with automatic recovery."""

    def __init__(self):
        self._failures: Dict[str, int] = {}
        self._tripped_until: Dict[str, float] = {}

    def record_failure(self, provider: str) -> None:
        self._failures[provider] = self._failures.get(provider, 0) + 1
        if self._failures[provider] >= FAILURE_THRESHOLD:
            self._tripped_until[provider] = time.time() + RECOVERY_TIMEOUT_SECONDS
            logger.warning(
                "Circuit breaker TRIPPED: %s blocked for %ds",
                provider, RECOVERY_TIMEOUT_SECONDS,
            )

    def record_success(self, provider: str) -> None:
        self._failures[provider] = 0
        self._tripped_until.pop(provider, None)

    def is_available(self, provider: str) -> bool:
        if provider not in self._tripped_until:
            return True
        if time.time() > self._tripped_until[provider]:
            del self._tripped_until[provider]
            self._failures[provider] = 0
            logger.info("Circuit breaker RECOVERED: %s", provider)
            return True
        return False

    def blocked_providers(self) -> List[str]:
        now = time.time()
        return [p for p, t in self._tripped_until.items() if t >= now]


# ─── Venice Budget Tracker ───────────────────────────────────────────────

class BudgetTracker:
    """Daily budget tracker for Venice API spend."""

    def __init__(self, daily_limit: float = 7.40):
        self._daily_limit = daily_limit
        self._file = Path.home() / ".hermes" / "venice_budget.json"
        self._cache: Optional[dict] = None
        self._cache_time: float = 0

    def _load(self) -> dict:
        now = time.time()
        if self._cache and (now - self._cache_time) < 10:
            return self._cache

        now_pst = self._now_pst()
        date_str = now_pst.strftime("%Y-%m-%d")
        data = {"last_reset": date_str, "spent": 0.0}

        if self._file.exists():
            try:
                with open(self._file) as f:
                    data = json.load(f)
                if data.get("last_reset") != date_str:
                    data = {"last_reset": date_str, "spent": 0.0}
            except Exception:
                data = {"last_reset": date_str, "spent": 0.0}

        self._cache = data
        self._cache_time = now
        return data

    def is_available(self) -> bool:
        data = self._load()
        return data["spent"] < self._daily_limit

    def spent_ratio(self) -> float:
        data = self._load()
        return data["spent"] / self._daily_limit if self._daily_limit > 0 else 1.0

    def record(self, amount: float) -> None:
        data = self._load()
        data["spent"] = min(data["spent"] + amount, self._daily_limit * 2)
        try:
            with open(self._file, "w") as f:
                json.dump(data, f)
        except Exception as exc:
            logger.warning("Failed to write budget file: %s", exc)
        self._cache = None  # invalidate cache

    @staticmethod
    def _now_pst() -> datetime:
        # Use America/Los_Angeles for correct DST handling (PST/PDT). The old
        # fixed UTC-8 offset was wrong during summer (off by an hour, straddling
        # the budget day boundary). Name kept for backward compat.
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Los_Angeles"))


# ─── Provider Registry ───────────────────────────────────────────────────

class ProviderRegistry:
    """Loads providers from config.yaml. No hardcoded credentials."""

    def __init__(self, config: dict, env: Optional[Dict[str, str]] = None):
        self._providers: Dict[str, ProviderSlot] = {}
        self._category_chains: Dict[Category, List[str]] = {}
        self._env = env or os.environ
        self._load(config)

    def _resolve_key(self, raw: str) -> str:
        """Resolve ${ENV_VAR} patterns in config values."""
        if raw and raw.startswith("${") and raw.endswith("}"):
            var_name = raw[2:-1]
            return self._env.get(var_name, "")
        return raw or ""

    def _load(self, config: dict):
        routing = config.get("routing", {})

        # Load providers from routing.providers list
        for entry in routing.get("providers", []):
            try:
                slot = ProviderSlot(
                    id=entry["id"],
                    provider=entry["provider"],
                    model=entry["model"],
                    base_url=entry["base_url"],
                    api_key=self._resolve_key(entry.get("api_key", "")),
                    context_length=entry.get("context_length", 0),
                    request_timeout=entry.get("timeout", 0),
                    is_local=entry.get("local", False),
                    cost_per_1k_input=entry.get("cost_input", 0.0),
                    cost_per_1k_output=entry.get("cost_output", 0.0),
                )
                if slot.api_key or slot.is_local:
                    # Local/no-auth servers (e.g. vLLM without --api-key) are
                    # allowed with an empty key. Remote providers require a key.
                    self._providers[slot.id] = slot
                else:
                    logger.warning("No API key resolved for provider %s — skipping", slot.id)
            except (KeyError, TypeError) as exc:
                logger.warning("Invalid routing provider entry: %s", exc)

        # Load category chains from routing.chains
        for cat_name, chain_ids in routing.get("chains", {}).items():
            try:
                cat = Category(cat_name)
                self._category_chains[cat] = chain_ids
            except ValueError:
                logger.warning("Unknown routing category: %s", cat_name)

        # Fallback chain for categories not explicitly mapped
        default_chain = routing.get("default_chain", [])
        for cat in Category:
            if cat not in self._category_chains and default_chain:
                self._category_chains[cat] = default_chain

        logger.info(
            "Routing loaded: %d providers, %d category chains",
            len(self._providers), len(self._category_chains),
        )

    def get(self, slot_id: str) -> Optional[ProviderSlot]:
        return self._providers.get(slot_id)

    def chain(self, category) -> List[ProviderSlot]:
        """Get ordered provider chain for a category. Accepts Category or str."""
        if isinstance(category, str):
            try:
                category = Category(category)
            except ValueError:
                return []
        ids = self._category_chains.get(category, [])
        result = []
        seen = set()
        for slot_id in ids:
            if slot_id in seen:
                continue
            seen.add(slot_id)
            slot = self._providers.get(slot_id)
            if slot:
                result.append(slot)
        return result

    def all_providers(self) -> List[ProviderSlot]:
        return list(self._providers.values())


# ─── Local Health Check ─────────────────────────────────────────────────

class HealthChecker:
    """Pre-flight health pings for local endpoints."""

    def __init__(self):
        self._cache: Dict[str, tuple] = {}  # provider -> (ok, timestamp)

    def check(self, slot: ProviderSlot) -> bool:
        if not slot.is_local:
            return True

        now = time.time()
        cached = self._cache.get(slot.provider)
        if cached and (now - cached[1]) < HEALTH_CACHE_TTL:
            return cached[0]

        try:
            import urllib.request
            url = f"{slot.base_url.rstrip('/')}/models"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=HEALTH_CHECK_TIMEOUT) as resp:
                ok = resp.status == 200
        except Exception:
            ok = False
            logger.debug("Health check FAILED: %s at %s", slot.provider, slot.base_url)

        self._cache[slot.provider] = (ok, now)
        return ok


# ─── Router ──────────────────────────────────────────────────────────────

class Router:
    """Main routing engine. Single entry point for all routing decisions."""

    def __init__(self, config: dict):
        self._registry = ProviderRegistry(config)
        self._breaker = CircuitBreaker()
        self._health = HealthChecker()
        self._budget = BudgetTracker(
            daily_limit=config.get("routing", {}).get("venice_daily_budget", 7.40),
        )

    def route(
        self,
        message: str,
        primary_config: dict,
        suppress_tools_override: Optional[bool] = None,
    ) -> RouteResult:
        """Route a message to the optimal model.

        Args:
            message: User message text.
            primary_config: Session's primary model config (fallback).
            suppress_tools_override: Force tool suppression on/off.

        Returns:
            RouteResult with model, credentials, and routing metadata.
        """
        category = TaskClassifier.classify(message)
        blocked = self._breaker.blocked_providers()

        # Get the ordered provider chain for this category
        chain = self._registry.chain(category)
        if not chain:
            logger.debug("No chain for %s — using primary", category.value)
            return self._primary_route(primary_config, category)

        # Walk the chain, skip blocked/unhealthy/budget-exceeded
        fallback_count = 0
        for slot in chain:
            # Circuit breaker
            if slot.provider in blocked:
                fallback_count += 1
                continue

            # Local health check
            if slot.is_local and not self._health.check(slot):
                fallback_count += 1
                continue

            # Venice budget gate
            if slot.provider == "venice" and not self._budget.is_available():
                ratio = self._budget.spent_ratio()
                logger.info(
                    "Venice budget %.0f%% used (%.2f) — skipping %s",
                    ratio * 100, self._budget._load()["spent"], slot.model,
                )
                fallback_count += 1
                continue

            # Found a valid provider
            suppress = suppress_tools_override
            if suppress is None:
                # Suppress tools for GREETING and SIMPLE — saves ~2K tokens per
                # turn on short queries (time checks, yes/no, etc.) routed to
                # local models that don't need tool definitions.
                suppress = category in (Category.GREETING, Category.SIMPLE)

            logger.info(
                "Route: %s -> %s (%s, local=%s, tools=%s, fallbacks=%d)",
                category.value, slot.model, slot.provider,
                slot.is_local, not suppress, fallback_count,
            )

            return RouteResult(
                model=slot.model,
                base_url=slot.base_url,
                api_key=slot.api_key,
                provider=slot.provider,
                category=category,
                is_local=slot.is_local,
                suppress_tools=suppress,
                label=f"{category.value} -> {slot.model}",
                fallback_count=fallback_count,
            )

        # All chain providers exhausted — fall to primary
        logger.warning(
            "Chain exhausted for %s (%d skipped) — falling to primary",
            category.value, fallback_count,
        )
        return self._primary_route(primary_config, category, fallback_count)

    def _primary_route(
        self, primary_config: dict, category: Category, fallback_count: int = 0,
    ) -> RouteResult:
        """Fall back to the session's primary model config."""
        return RouteResult(
            model=primary_config.get("model", ""),
            base_url=primary_config.get("base_url", ""),
            api_key=primary_config.get("api_key", ""),
            provider=primary_config.get("provider", "primary"),
            category=category,
            is_local=False,
            suppress_tools=False,
            label=f"primary -> {primary_config.get('model', 'unknown')}",
            fallback_count=fallback_count,
        )

    def is_provider_blocked(self, provider: str) -> bool:
        """True if the provider is currently circuit-broken."""
        return not self._breaker.is_available(provider)

    def record_failure(self, provider: str) -> None:
        self._breaker.record_failure(provider)

    def record_success(self, provider: str) -> None:
        self._breaker.record_success(provider)

    def record_venice_spend(self, amount: float) -> None:
        self._budget.record(amount)

    def status(self) -> dict:
        budget_data = self._budget._load()
        return {
            "providers": {p.id: {"model": p.model, "provider": p.provider, "local": p.is_local}
                          for p in self._registry.all_providers()},
            "chains": {cat.value: chain for cat, chain in self._registry._category_chains.items()},
            "blocked": self._breaker.blocked_providers(),
            "budget": {
                "limit": self._budget._daily_limit,
                "spent": budget_data.get("spent", 0),
                "available": self._budget.is_available(),
            },
        }


# ─── Singleton & Gateway Interface ─────────────────────────────────────

_instance: Optional[Router] = None
_config_path: Optional[str] = None


def init_router(config: dict) -> Router:
    """Initialize (or re-initialize) the global router singleton."""
    global _instance
    _instance = Router(config)
    logger.info("Router initialized with %d providers", len(_instance._registry.all_providers()))
    return _instance


def get_router() -> Optional[Router]:
    """Get the global router instance. None if not initialized."""
    return _instance


def route_turn(
    message: str,
    primary_config: dict,
    suppress_tools_override: Optional[bool] = None,
) -> Optional[RouteResult]:
    """Gateway entry point. Returns None if router not initialized."""
    if _instance is None:
        return None
    return _instance.route(message, primary_config, suppress_tools_override)


def record_routing_failure(provider: str) -> None:
    if _instance:
        _instance.record_failure(provider)


def record_routing_success(provider: str) -> None:
    if _instance:
        _instance.record_success(provider)


def is_provider_blocked(provider: str) -> bool:
    """Check if a provider is circuit-broken (3 consecutive failures, 180s cooldown).

    Returns False (not blocked) when the router isn't initialized, so
    the fallback chain degrades gracefully if routing isn't active.
    """
    if _instance is None:
        return False
    return _instance.is_provider_blocked(provider)


def record_venice_spend(amount: float) -> None:
    """Record Venice spend to the budget tracker."""
    if _instance:
        _instance.record_venice_spend(amount)


def routing_status() -> Optional[dict]:
    if _instance:
        return _instance.status()
    return None
