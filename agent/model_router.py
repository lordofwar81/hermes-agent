#!/usr/bin/env python3
"""
agent.model_router — Deterministic, single-source routing for Hermes.

Canonical implementation (replaces: smart_model_routing.py, model_selector.py,
trivial_task_classifier.py, complexity_detector logic, priority_router.py).

Design:
  - All credentials pre-resolved at gateway startup (no per-turn API calls)
  - Deterministic task classification (greeting/simple/code/analysis/reasoning/expert)
  - Circuit breaker for failing providers (Venice 503s, timeouts)
  - Daily Venice budget tracking with 5PM PST reset
  - Fail-open to primary model with explicit logging (no silent reverts)

Entry point:
  resolve_turn_route(message, primary_config, routing_config, config_path) -> route_dict

Usage (gateway):
  from agent.model_router import resolve_turn_route, record_routing_failure, record_routing_success

Author: Hermes Routing Rebuild — 2026-04-28
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────

FAILURE_THRESHOLD = 2
RECOVERY_TIMEOUT_SECONDS = 180  # 3 minutes per blocked provider
HEALTH_CHECK_TIMEOUT = 5  # seconds for pre-flight health ping to local models

# ─── Enums ──────────────────────────────────────────────────────────────

class TaskCategory(Enum):
    GREETING = "greeting"           # hello, hi, thanks (<=20 chars)
    SIMPLE = "simple"               # short questions, no code
    CODE = "code"                   # contains code blocks or debug keywords
    ANALYSIS = "analysis"           # compare, evaluate, architecture
    REASONING = "reasoning"         # explain why/how, design questions
    EXPERT = "expert"               # full system design, end-to-end impl

# ─── Data Classes ───────────────────────────────────────────────────────

@dataclass
class ProviderCredential:
    provider: str
    model: str
    base_url: str
    api_key: str  # resolved from env var at startup, NOT per-turn
    context_length: int = 0  # max context for this model (0 = unknown/unlimited)
    request_timeout: int = 0  # seconds (0 = use default)
    is_local: bool = False  # local models get extra safeguards

LOCAL_MODEL_MAX_CONTEXT_RATIO = 0.6  # only use 60% of local model context for safety
LOCAL_MODEL_TURN_TIMEOUT = 120  # seconds before considering local model too slow

@dataclass
class BudgetState:
    daily_limit_usd: float = 7.40
    spent_usd: float = 0.0
    last_reset_date: str = ""  # YYYY-MM-DD (PST)

# ─── Task Classifier (Deterministic, No Async) ──────────────────────────

class TaskClassifier:
    """Deterministic task classification. No async calls, no regex guessing."""

    CODE_KEYWORDS = frozenset({
        "debug", "debugging", "implement", "implementation",
        "patch", "traceback", "stacktrace", "exception", "error", "fix",
        "bug", "module", "script", "server", "test", "security",
    })

    REFACTOR_KEYWORDS = frozenset({
        "refactor", "rewrite", "rework",
    })

    ANALYSIS_KEYWORDS = frozenset({
        "analyze", "analysis", "compare", "contrast", "evaluate",
        "performance", "metrics", "report"
    })

    REASONING_KEYWORDS = frozenset({
        "explain why", "explain how", "why does", "how does", "what causes",
        "redesign", "improve"
    })

    GREETING_KEYWORDS = frozenset({
        "hello", "hi", "hey", "thanks", "thank you", "ty",
        "cheers", "ok", "okay", "got it", "understood"
    })

    EXPERT_PHRASES = [
        "design a system", "design an architecture", "implement a complete",
        "end-to-end", "full system", "architect a", "architect the",
        "build a complete", "build an entire", "comprehensive system",
        "from scratch", "production-ready", "multi-region", "event sourcing",
        "service mesh", "microservice",
        "system design", "system architecture",
    ]

    @classmethod
    def classify(cls, message: str) -> TaskCategory:
        import re

        text = (message or "").strip()

        # Greeting: short message with greeting keywords (word-boundary match)
        if len(text) <= 20:
            text_lower = text.lower()
            for g in cls.GREETING_KEYWORDS:
                if len(g.split()) == 1:
                    # Single-word greetings: use word boundary to avoid false matches
                    # e.g., "hi" should not match inside "this"
                    if re.search(r'\b' + re.escape(g) + r'\b', text_lower):
                        return TaskCategory.GREETING
                else:
                    # Multi-word greetings: substring match is fine
                    if g in text_lower:
                        return TaskCategory.GREETING

        # Expert: system design, end-to-end implementation (checked BEFORE code —
        # expert tasks often include "implement" which would otherwise be caught
        # by code keywords)
        text_lower = text.lower()
        if any(phrase in text_lower for phrase in cls.EXPERT_PHRASES):
            return TaskCategory.EXPERT

        # Refactor: treat as reasoning, not code — MiniMax is too slow for this
        if any(kw in text_lower for kw in cls.REFACTOR_KEYWORDS):
            return TaskCategory.REASONING

        # Code: code blocks or code keywords present
        if "```" in text or "`" in text:
            return TaskCategory.CODE
        if any(kw in text_lower for kw in cls.CODE_KEYWORDS):
            return TaskCategory.CODE

        # Reasoning: explanation/design questions (checked BEFORE analysis)
        if any(kw in text_lower for kw in cls.REASONING_KEYWORDS):
            return TaskCategory.REASONING

        # Analysis: comparison/evaluation keywords (architecture removed — caught by expert)
        if any(kw in text_lower for kw in cls.ANALYSIS_KEYWORDS):
            return TaskCategory.ANALYSIS

        # Default: simple
        return TaskCategory.SIMPLE

# ─── Budget Tracker (Daily Reset at 5PM PST) ─────────────────────────────

class DailyBudgetTracker:
    """Tracks Venice spending with daily reset at 5PM PST."""

    def __init__(self, config_path: str):
        self._config_path = Path(config_path)
        self._budget_file = self._config_path.parent / "venice_budget.json"
        self._daily_limit = 7.40

        # Load from config if available
        try:
            with open(self._config_path, "r") as f:
                import yaml
                cfg = yaml.safe_load(f)
                venice_budget = cfg.get("smart_model_routing", {}).get(
                    "venice_budget", {}
                )
                self._daily_limit = venice_budget.get("daily_limit_usd", 7.40)
        except Exception:
            pass

    def get_status(self) -> BudgetState:
        now_pst = self._now_pst()
        date_str = now_pst.strftime("%Y-%m-%d")

        # Load existing state
        spent = 0.0
        last_reset = ""

        if self._budget_file.exists():
            try:
                with open(self._budget_file, "r") as f:
                    data = json.load(f)
                last_reset = data.get("last_reset_date", "")
                spent = data.get("spent_usd", 0.0)
            except Exception:
                spent = 0.0

        # Reset if new day (PST)
        if last_reset != date_str:
            spent = 0.0
            last_reset = date_str

        return BudgetState(
            daily_limit_usd=self._daily_limit,
            spent_usd=spent,
            last_reset_date=date_str,
        )

    def record_spending(self, amount_usd: float) -> None:
        now_pst = self._now_pst()
        date_str = now_pst.strftime("%Y-%m-%d")

        data: Dict[str, Any] = {"last_reset_date": date_str, "spent_usd": 0.0}

        if self._budget_file.exists():
            try:
                with open(self._budget_file, "r") as f:
                    data = json.load(f)
            except Exception:
                pass

        data["last_reset_date"] = date_str
        data["spent_usd"] = min(
            data.get("spent_usd", 0.0) + amount_usd, self._daily_limit * 2
        )

        try:
            with open(self._budget_file, "w") as f:
                json.dump(data, f)
        except Exception as exc:
            logger.warning("Failed to write budget file: %s", exc)

    def is_budget_available(self) -> bool:
        status = self.get_status()
        return status.spent_usd < status.daily_limit_usd

    @staticmethod
    def _now_pst() -> datetime:
        # PST = UTC-8 (standard) or UTC-7 (DST)
        return datetime.now(timezone(timedelta(hours=-8)))

# ─── Circuit Breaker ─────────────────────────────────────────────────────

class RouterCircuitBreaker:
    """Blocks providers that are failing consecutively."""

    def __init__(self):
        self._failures: Dict[str, int] = {}
        self._tripped_until: Dict[str, float] = {}

    def record_failure(self, provider: str) -> None:
        self._failures[provider] = self._failures.get(provider, 0) + 1
        if self._failures[provider] >= FAILURE_THRESHOLD:
            self._tripped_until[provider] = time.time() + RECOVERY_TIMEOUT_SECONDS
            logger.warning(
                "Circuit breaker TRIPPED for provider '%s' — blocking for %d seconds",
                provider,
                RECOVERY_TIMEOUT_SECONDS,
            )

    def record_success(self, provider: str) -> None:
        self._failures[provider] = 0
        if provider in self._tripped_until:
            del self._tripped_until[provider]

    def is_available(self, provider: str) -> bool:
        if provider not in self._tripped_until:
            return True

        if time.time() > self._tripped_until[provider]:
            # Recovery period over — reset and allow
            del self._tripped_until[provider]
            self._failures[provider] = 0
            logger.info(
                "Circuit breaker UNTRIPPED for provider '%s' — restoring", provider
            )
            return True

        return False

    def get_blocked_providers(self) -> list[str]:
        now = time.time()
        return [p for p in self._tripped_until if self._tripped_until[p] >= now]

# ─── Provider Registry (Pre-Resolved Credentials) ────────────────────────

class ProviderRegistry:
    """All provider credentials resolved ONCE at gateway startup."""

    def __init__(self, config_path: str):
        self._providers: Dict[str, ProviderCredential] = {}
        self._load(config_path)

    def _load(self, config_path: str):
        try:
            with open(config_path, "r") as f:
                import yaml

                cfg = yaml.safe_load(f)
        except Exception:
            logger.warning("Failed to load config for provider registry")
            cfg = {}

        # ZAI providers (from config.yaml)
        zai_cfg = cfg.get("z-ai", {}) or cfg.get("providers", {}).get("z-ai", {})
        zai_key = self._resolve_api_key(zai_cfg.get("api_key"), "ZAI_API_KEY")
        zai_base = zai_cfg.get(
            "base_url", "https://api.z.ai/api/coding/paas/v4"
        )

        if zai_key:
            self._providers["zai-glm-5.1"] = ProviderCredential(
                provider="zai", model="glm-5.1", base_url=zai_base, api_key=zai_key,
                context_length=198000, request_timeout=300, is_local=False,
            )
            self._providers["zai-glm-5-turbo"] = ProviderCredential(
                provider="zai", model="glm-5-turbo", base_url=zai_base, api_key=zai_key,
                context_length=200000, request_timeout=300, is_local=False,
            )
            self._providers["zai-glm-5"] = ProviderCredential(
                provider="zai", model="glm-5", base_url=zai_base, api_key=zai_key,
                context_length=198000, request_timeout=300, is_local=False,
            )
        else:
            logger.warning("ZAI API key not found — ZAI providers unavailable")

        # MiniMax (privacy-focused)
        minimax_cfg = cfg.get("minimax_worker", {}) or cfg.get(
            "minimax_specialist", {}
        )
        minimax_key = self._resolve_api_key(
            minimax_cfg.get("api_key"), "MINIMAX_API_KEY"
        )
        if minimax_key:
            self._providers["minimax-m27"] = ProviderCredential(
                provider="minimax",
                model=minimax_cfg.get("model", "MiniMax-M2.7-APEX-I-Mini"),
                base_url=minimax_cfg.get(
                    "base_url", "http://192.168.1.229:8199/v1"
                ),
                api_key=minimax_key,
                context_length=minimax_cfg.get("context_length", 65536),
                request_timeout=LOCAL_MODEL_TURN_TIMEOUT,
                is_local=True,
            )
        else:
            logger.warning("MiniMax API key not found — minimax unavailable")

        # Mac Studio Orchestrator (Qwen 3.6)
        mac_cfg = cfg.get("mac_studio_orchestrator", {})
        mac_key = self._resolve_api_key(mac_cfg.get("api_key"), "MAC_STUDIO_KEY")
        if mac_key:
            self._providers["mac-studio-qwen36"] = ProviderCredential(
                provider="mac_studio",
                model=mac_cfg.get(
                    "model", "qwen3.6-35b-a3b-abliterated-heretic-mlx"
                ),
                base_url=mac_cfg.get(
                    "base_url", "http://192.168.1.149:1234/v1"
                ),
                api_key=mac_key,
                context_length=mac_cfg.get("context_length", 32768),
                request_timeout=LOCAL_MODEL_TURN_TIMEOUT,
                is_local=True,
            )
        else:
            logger.warning(
                "Mac Studio key not found — orchestrator unavailable"
            )

        # Venice providers (with circuit breaker awareness)
        venice_cfg = cfg.get("providers", {}).get("venice", {}) or {}
        venice_fb = cfg.get("venice_fallback", {}) or {}
        venice_key = (
            self._resolve_api_key(venice_cfg.get("api_key"), "VENICE_INFERENCE_KEY")
            or self._resolve_api_key(venice_fb.get("api_key"), "VENICE_INFERENCE_KEY")
        )
        venice_base = (
            venice_cfg.get("api")
            or venice_cfg.get("base_url")
            or venice_fb.get("base_url")
            or "https://api.venice.ai/api/v1"
        )

        if venice_key:
            self._providers["venice-claude-sonnet-4-6"] = ProviderCredential(
                provider="venice",
                model="claude-sonnet-4-6",
                base_url=venice_base,
                api_key=venice_key,
                context_length=200000,
                request_timeout=300,
                is_local=False,
            )
            self._providers["venice-qwen-3-6-plus"] = ProviderCredential(
                provider="venice",
                model="qwen-3-6-plus",
                base_url=venice_base,
                api_key=venice_key,
                context_length=1000000,
                request_timeout=300,
                is_local=False,
            )
        else:
            logger.warning("Venice API key not found — Venice providers unavailable")

    @staticmethod
    def _resolve_api_key(
        config_value: Optional[str], env_var_name: str
    ) -> Optional[str]:
        """Resolve API key from config or environment variable."""
        if config_value and not str(config_value).startswith("${"):
            return str(config_value)
        return os.getenv(env_var_name)

    def get(self, key: str) -> Optional[ProviderCredential]:
        return self._providers.get(key)

    def get_by_provider(self, provider: str) -> list[ProviderCredential]:
        return [p for p in self._providers.values() if p.provider == provider]

    def list_available(self) -> list[ProviderCredential]:
        return list(self._providers.values())

# ─── Main Router ─────────────────────────────────────────────────────────

class ModelRouter:
    """Deterministic model router — single source of truth for all routing."""

    def __init__(self, config_path: str):
        self._config_path = config_path
        self._registry = ProviderRegistry(config_path)
        self._budget = DailyBudgetTracker(config_path)
        self._breaker = RouterCircuitBreaker()
        self._health_cache: Dict[str, tuple] = {}  # provider -> (ok, timestamp)

    def _check_local_health(self, cred: ProviderCredential) -> bool:
        """Pre-flight health check for local models. Returns True if responsive.
        Results are cached for 30 seconds to avoid hammering on every message.
        """
        if not cred.is_local:
            return True

        now = time.time()
        cached = self._health_cache.get(cred.provider)
        if cached and (now - cached[1]) < 30:
            return cached[0]

        try:
            import urllib.request
            url = f"{cred.base_url.rstrip('/')}/models"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=HEALTH_CHECK_TIMEOUT) as resp:
                ok = resp.status == 200
        except Exception:
            ok = False
            logger.info(
                "Health check FAILED for local provider '%s' at %s — skipping",
                cred.provider, cred.base_url,
            )

        self._health_cache[cred.provider] = (ok, now)
        return ok

    def route(self, message: str, primary_config: dict, estimated_tokens: int = 0) -> Dict[str, Any]:
        """Route a message to the appropriate model.

        Returns a route dict compatible with gateway/run.py expectations:
        {model, runtime:{api_key, base_url, provider}, label, signature}

        Args:
            message: The user's message text.
            primary_config: The primary model config dict for fallback.
            estimated_tokens: Estimated total context tokens for this turn.
                              Used to skip local models with insufficient context.
        """
        category = TaskClassifier.classify(message)

        blocked = self._breaker.get_blocked_providers()

        credential = self._select_credential(category, blocked, estimated_tokens)

        if not credential:
            logger.warning(
                "No routing credential available for category '%s' — using primary",
                category.value,
            )
            return self._fallback_to_primary(primary_config, category)

        # Step 3: Check Venice budget if routing to Venice
        if credential.provider == "venice" and not self._budget.is_budget_available():
            logger.info(
                "Venice budget exhausted (%.2f/%.2f) — rerouting to ZAI",
                self._budget.get_status().spent_usd,
                self._budget.get_status().daily_limit_usd,
            )
            credential = self._get_best_zai()

        # Step 4: Build route dict
        return {
            "model": credential.model,
            "runtime": {
                "api_key": credential.api_key,
                "base_url": credential.base_url,
                "provider": credential.provider,
                "api_mode": "chat_completions",
                "command": None,
                "args": [],
                "credential_pool": None,
            },
            "label": f"router -> {credential.model} ({credential.provider})",
            "signature": (
                credential.model,
                credential.provider,
                credential.base_url,
                "chat_completions",
                None,
                (),
            ),
            "request_overrides": None,
            "context_length": credential.context_length,
            "is_local": credential.is_local,
        }

    def record_api_failure(self, provider: str) -> None:
        """Call this when an API call to a provider fails (503, timeout, etc.)."""
        self._breaker.record_failure(provider)

    def record_api_success(self, provider: str) -> None:
        """Call this when an API call succeeds."""
        self._breaker.record_success(provider)

    def _select_credential(
        self, category: TaskCategory, blocked: list[str], estimated_tokens: int = 0,
    ) -> Optional[ProviderCredential]:
        """Select the best credential for a task category.

        ZAI is the default for all interactive tasks — local models (Mac Studio,
        MiniMax) have context detection issues, compression failures, and speed
        problems.  Local models are only used when ZAI is circuit-broken.
        """

        if category == TaskCategory.EXPERT:
            if self._budget.is_budget_available():
                venice = self._get_best_venice(blocked)
                if venice:
                    return venice

        zai = self._get_best_zai()
        if zai:
            return zai

        if category in (TaskCategory.CODE, TaskCategory.ANALYSIS, TaskCategory.REASONING):
            mac = self._registry.get("mac-studio-qwen36")
            if mac and self._breaker.is_available("mac_studio") and self._fits_context(mac, estimated_tokens) and self._check_local_health(mac):
                return mac

        return self._get_fastest_available(blocked, estimated_tokens)

    def _fits_context(self, cred: ProviderCredential, estimated_tokens: int) -> bool:
        """Check if a provider has enough context for the estimated token count.

        For local models, we only use 60% of the stated context to leave room
        for the response and avoid edge-case failures.
        """
        if not cred.context_length or estimated_tokens <= 0:
            return True
        limit = cred.context_length
        if cred.is_local:
            limit = int(limit * LOCAL_MODEL_MAX_CONTEXT_RATIO)
        return estimated_tokens < limit

    def _get_best_zai(self) -> Optional[ProviderCredential]:
        """Get the best ZAI model for the task."""
        for key in ["zai-glm-5.1", "zai-glm-5-turbo", "zai-glm-5"]:
            cred = self._registry.get(key)
            if cred and self._breaker.is_available("zai"):
                return cred
        return None

    def _get_best_venice(self, blocked: list[str]) -> Optional[ProviderCredential]:
        """Get best available Venice model. Claude Sonnet for quality, Qwen-3-6-plus for long context."""
        for key in ["venice-claude-sonnet-4-6", "venice-qwen-3-6-plus"]:
            cred = self._registry.get(key)
            if cred and "venice" not in blocked:
                return cred
        return None

    def _get_fastest_available(self, blocked: list[str], estimated_tokens: int = 0) -> Optional[ProviderCredential]:
        """Get fastest available model (Mac Studio or ZAI turbo)."""
        mac = self._registry.get("mac-studio-qwen36")
        if mac and "mac_studio" not in blocked and self._fits_context(mac, estimated_tokens):
            return mac

        zai = self._get_best_zai()
        if zai:
            return zai

        return None

    def _fallback_to_primary(
        self, primary_config: dict, category: TaskCategory
    ) -> Dict[str, Any]:
        """Fallback to primary model when no routing is possible."""
        return {
            "model": primary_config.get("model"),
            "runtime": {
                "api_key": primary_config.get("api_key"),
                "base_url": primary_config.get("base_url"),
                "provider": primary_config.get("provider"),
                "api_mode": primary_config.get("api_mode", "chat_completions"),
                "command": None,
                "args": [],
                "credential_pool": primary_config.get("credential_pool"),
            },
            "label": f"fallback (no routing available) -> {category.value}",
            "signature": (
                primary_config.get("model"),
                primary_config.get("provider"),
                primary_config.get("base_url"),
                primary_config.get("api_mode", "chat_completions"),
                None,
                (),
            ),
            "request_overrides": None,
        }

    def status_report(self) -> Dict[str, Any]:
        """Return a diagnostic snapshot of the router state."""
        return {
            "providers_loaded": len(self._registry.list_available()),
            "provider_names": [p.model for p in self._registry.list_available()],
            "blocked_providers": self._breaker.get_blocked_providers(),
            "budget_status": {
                "daily_limit_usd": self._budget.get_status().daily_limit_usd,
                "spent_usd": self._budget.get_status().spent_usd,
                "available": self._budget.is_budget_available(),
            },
        }

# ─── Gateway Entry Point (Drop-in Replacement) ──────────────────────────

_router_instance: Optional[ModelRouter] = None


def get_router(config_path: str) -> ModelRouter:
    """Get or create the singleton router instance."""
    global _router_instance
    if (
        _router_instance is None
        or _router_instance._config_path != config_path
    ):
        _router_instance = ModelRouter(config_path)
    return _router_instance


def resolve_turn_route(
    user_message: str,
    routing_config: Optional[Dict[str, Any]],
    primary: Dict[str, Any],
    config_path: str = None,
    estimated_tokens: int = 0,
) -> Dict[str, Any]:
    """Gateway entry point — drop-in replacement for agent.smart_model_routing.resolve_turn_route.

    This is the ONLY function gateway/run.py should import for routing.

    Args:
        estimated_tokens: Estimated total context tokens for this turn.
                          When > 0, local models with insufficient context are skipped.
    """
    if config_path is None:
        config_path = os.path.expanduser("~/.hermes/config.yaml")

    router = get_router(config_path)
    return router.route(user_message, primary, estimated_tokens=estimated_tokens)


def record_routing_failure(provider: str) -> None:
    """Call from gateway when an API call to a provider fails."""
    router = _router_instance
    if router:
        router.record_api_failure(provider)


def record_routing_success(provider: str) -> None:
    """Call from gateway when an API call to a provider succeeds."""
    router = _router_instance
    if router:
        router.record_api_success(provider)
