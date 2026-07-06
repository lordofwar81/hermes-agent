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
from functools import lru_cache
from typing import Dict, List, Optional
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
BUDGET_RESET_HOUR_PST = 17      # Venice account quota resets 5pm Pacific


# ─── Task Categories ─────────────────────────────────────────────────────

class Category(Enum):
    """Message classification for routing decisions."""
    GREETING = "greeting"      # hi, thanks, ok — short social
    SIMPLE = "simple"          # short factual question, no tools needed
    CODE = "code"              # debugging, implementation, code blocks
    REASONING = "reasoning"    # explain why, how, design questions
    ANALYSIS = "analysis"      # compare, evaluate, audit, metrics
    EXPERT = "expert"          # full system design, end-to-end, multi-step
    HEALTH = "health"          # peptide/dosing/reconstitution (Gemma refuses; route to abliterated)


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

    # HEALTH keywords — peptide/dosing/reconstitution turns that Gemma refuses.
    # Checked FIRST in classify() so health math isn't hijacked by code/greeting
    # branches. Single-word compounds use word boundaries; multiword + unit
    # patterns use substring. Route to the abliterated Qwen on mac-studio.
    _HEALTH_SINGLE_WORDS = frozenset({
        # peptide compounds (operator's documented protocol)
        "tesamorelin", "tb-500", "tb500", "bpc-157", "bpc157",
        "ipamorelin", "cjc", "cjc-1295", "semaglutide", "retatrutide",
        "ghk-cu", "ghkcu", "pregnyl", "melanotan", "pt-141",
        # dosing/reconstitution vocabulary
        "peptide", "peptides", "reconstitute", "reconstitution",
        "bacteriostatic", "reconstituted", "subcutaneous", "subq",
        "mcg", "syringe", "intramuscular",
    })
    _HEALTH_MULTIWORD_PHRASES = frozenset({
        "bac water", "bpc-157 blend", "glow blend", "dose me", "injection site",
        "dose volume", "weekly dose",
    })
    _HEALTH_KW = _HEALTH_SINGLE_WORDS | _HEALTH_MULTIWORD_PHRASES
    # Word-boundary regex for single-word compounds (so "dose" won't match
    # "proposal" — but "dose" alone is intentionally NOT in the single-word
    # set; only unambiguous compounds are).
    _HEALTH_BOUNDARY_RE = re.compile(
        r'\b(?:' + '|'.join(re.escape(kw) for kw in sorted(_HEALTH_SINGLE_WORDS)) + r')\b',
        re.IGNORECASE,
    )

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
    def _has_health_keyword(cls, text_lower: str) -> bool:
        """Check for HEALTH keywords: peptide/dosing/reconstitution vocabulary.

        Single-word compounds (tesamorelin, bpc-157, ipamorelin, etc.) use
        the compiled word-boundary regex so 'glow' won't match 'glowering'.
        Multi-word phrases (bac water, injection site) use substring match.
        """
        if cls._HEALTH_BOUNDARY_RE.search(text_lower):
            return True
        for kw in cls._HEALTH_MULTIWORD_PHRASES:
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

        # HEALTH: peptide/dosing/reconstitution. Checked before everything
        # else (except code blocks) because Gemma refuses these and the
        # classifier otherwise lands them in SIMPLE → local Gemma. Routes
        # to the abliterated Qwen on mac-studio via the health chain.
        if cls._has_health_keyword(text_lower):
            return Category.HEALTH

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

    # ───────────────────────────────────────────────────────────────────────
    # Semantic classifier (Phase 2 — Gulli Ch2/Ch16, LLM-based)
    # ───────────────────────────────────────────────────────────────────────
    # Uses the cheap aux model (glm-4.7) to classify message intent.
    # Returns None on any failure (network error, unparseable response, flag
    # off) so the caller falls back to the keyword ``classify``. Never raises.
    #
    # Design note: an embedding-centroid approach was tried first and rejected
    # — Qwen3-Embedding-8B does not discriminate between the 6 intent
    # categories (same-category pairs scored ~0.948, different-category pairs
    # ~0.962; the signal wasn't there). LLM classification is the tool for
    # intent understanding. See evalset.yaml + tests/eval/diagnose.py for the
    # proof. The keyword classifier remains the deterministic source of truth.

    # Category descriptions for the LLM prompt — these are what the model
    # actually matches against. Written to be self-explanatory and distinct.
    _CATEGORY_DESCRIPTIONS: Dict["Category", str] = {
        Category.GREETING: "greeting — a social opener, thanks, acknowledgment, or farewell with no task (e.g. 'hello', 'thanks that worked', 'got it')",
        Category.SIMPLE: "simple — a short factual question or trivia that needs no tools or code (e.g. 'capital of France', 'what does 404 mean')",
        Category.CODE: "code — debugging, implementation, bug fixing, testing, deployment, or any technical task involving writing or fixing code (e.g. 'the function throws KeyError', 'add validation', 'tests failing in CI', 'refactor to async')",
        Category.REASONING: "reasoning — asking for an explanation of why/how something works, understanding a concept, or walking through logic (e.g. 'why does it re-render', 'explain how JWT works', 'help me understand why')",
        Category.ANALYSIS: "analysis — comparing, evaluating, benchmarking, or assessing tradeoffs between options (e.g. 'compare Postgres vs MongoDB', 'evaluate the tradeoffs', 'cost-benefit analysis')",
        Category.EXPERT: "expert — designing a complete system, architecture, or end-to-end solution from scratch (e.g. 'design a distributed system', 'architect a multi-region API', 'build an end-to-end pipeline')",
    }

    @classmethod
    def _build_classify_prompt(cls, message: str) -> List[dict]:
        """Build the chat messages for the LLM classifier."""
        descriptions = "\n".join(
            f"- {desc}" for desc in cls._CATEGORY_DESCRIPTIONS.values()
        )
        system = (
            "You are a message intent classifier. Given a user message, "
            "respond with EXACTLY ONE word from this list: "
            "greeting, simple, code, reasoning, analysis, expert.\n\n"
            "Categories:\n" + descriptions + "\n\n"
            "Rules:\n"
            "- Respond with only the category word, nothing else.\n"
            "- If a message involves writing, fixing, debugging, or deploying code, it is 'code'.\n"
            "- If it asks to explain why/how or understand a concept, it is 'reasoning'.\n"
            "- If it asks to design or architect a full system, it is 'expert'.\n"
            "- If it compares or evaluates options, it is 'analysis'.\n"
            "- If it is a short factual question with no code, it is 'simple'.\n"
            "- If it is a social message with no task, it is 'greeting'."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": message[:800]},  # truncate like get_embedding does
        ]

    @classmethod
    def _parse_llm_category(cls, response_text: str) -> Optional["Category"]:
        """Extract a Category from the LLM's response. None if unparseable."""
        if not response_text:
            return None
        text = response_text.strip().lower()
        # The prompt asks for exactly one word, but be defensive: take the
        # first word and strip punctuation. Also handle the model wrapping
        # the word in markdown or adding a period.
        first_word = text.split()[0].strip(".,!?:;`*#-\"'()[]") if text else ""
        for cat in Category:
            if cat.value == first_word:
                return cat
        # Fallback: substring match (handles "The category is: code")
        for cat in Category:
            if cat.value in text:
                return cat
        return None

    @classmethod
    @lru_cache(maxsize=512)
    def classify_semantic(cls, message: str) -> Optional["Category"]:
        """Classify a message via the cheap aux LLM (glm-4.7).

        Returns the predicted Category, or None on any failure (network error,
        unparseable response, exception). Never raises — the caller falls back
        to keyword ``classify`` on None.

        Cached via lru_cache(512) keyed by message. The LLM call is ~200-500ms
        and would otherwise hit every turn.

        Provider/model resolution: reads ``auxiliary.classifier.{provider,model}``
        from config if present; defaults to ``zai``/``glm-4.7`` (the
        cheap model per auxiliary_client.py:340).
        """
        if not message or not message.strip():
            return None
        try:
            from agent.auxiliary_client import call_llm
            messages = cls._build_classify_prompt(message)

            # Resolve provider/model: config override first, then the cheap
            # default. This keeps the classifier on the fast/cheap model
            # regardless of the main turn's provider.
            provider = "zai"
            model = "glm-4.7"
            try:
                from hermes_cli.config import load_config
                cfg = load_config() or {}
                aux = cfg.get("auxiliary", {})
                if isinstance(aux, dict):
                    clf_cfg = aux.get("classifier", {})
                    if isinstance(clf_cfg, dict):
                        provider = clf_cfg.get("provider", provider)
                        model = clf_cfg.get("model", model)
            except Exception:
                pass  # config read failure → use defaults

            response = call_llm(
                provider=provider,
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=10,  # one word
                timeout=10.0,
            )
            content = response.choices[0].message.content
            result = cls._parse_llm_category(content)
            if result is not None:
                logger.debug(
                    "Semantic classify (LLM): %r -> %s",
                    message[:60], result.value,
                )
            else:
                logger.debug(
                    "Semantic classify (LLM): unparseable response %r for %r",
                    content[:80], message[:60],
                )
            return result
        except Exception as exc:
            logger.debug("Semantic classify (LLM) failed: %s", exc)
            return None


# ─── Circuit Breaker ─────────────────────────────────────────────────────


# ─── Circuit Breaker ─────────────────────────────────────────────────────

class CircuitBreaker:
    """Per-provider circuit breaker with automatic recovery.

    State is persisted to ``~/.hermes/circuit_breaker.json`` so that a gateway
    restart no longer wipes failure counts — a provider that was failing before
    the restart stays blocked until it either recovers or the cooldown elapses.
    Mirrors the BudgetTracker persistence pattern (see below).
    """

    def __init__(self, state_file: Optional[Path] = None):
        self._failures: Dict[str, int] = {}
        self._tripped_until: Dict[str, float] = {}
        if state_file is not None:
            self._state_file = state_file
        else:
            # Resolve via get_hermes_home() so HERMES_HOME overrides (tests,
            # alternate profiles) are respected. Using Path.home()/".hermes"
            # directly is a known bug pattern (see tests/conftest.py:372-373).
            from hermes_constants import get_hermes_home
            self._state_file = get_hermes_home() / "circuit_breaker.json"
        self._load()

    def _load(self) -> None:
        """Restore failure counts and active trip windows from disk.

        Expired trips are dropped on load (a restart shouldn't re-block a
        provider whose cooldown already elapsed while the gateway was down).
        Unexpired trips keep their absolute deadline.
        """
        if not self._state_file.exists():
            return
        try:
            with open(self._state_file) as f:
                data = json.load(f)
            now = time.time()
            for provider, count in data.get("failures", {}).items():
                self._failures[provider] = int(count)
            for provider, deadline in data.get("tripped_until", {}).items():
                if float(deadline) > now:
                    self._tripped_until[provider] = float(deadline)
                else:
                    # Cooldown elapsed while down — clear the failure count too.
                    self._failures.pop(provider, None)
        except Exception as exc:
            logger.warning("Failed to load circuit breaker state: %s", exc)

    def _save(self) -> None:
        """Persist current state. Best-effort; never blocks routing."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_file, "w") as f:
                json.dump({
                    "failures": self._failures,
                    "tripped_until": self._tripped_until,
                }, f)
        except Exception as exc:
            logger.warning("Failed to persist circuit breaker state: %s", exc)

    def record_failure(self, provider: str) -> None:
        self._failures[provider] = self._failures.get(provider, 0) + 1
        if self._failures[provider] >= FAILURE_THRESHOLD:
            self._tripped_until[provider] = time.time() + RECOVERY_TIMEOUT_SECONDS
            logger.warning(
                "Circuit breaker TRIPPED: %s blocked for %ds",
                provider, RECOVERY_TIMEOUT_SECONDS,
            )
        self._save()

    def record_success(self, provider: str) -> None:
        self._failures[provider] = 0
        self._tripped_until.pop(provider, None)
        self._save()

    def is_available(self, provider: str) -> bool:
        if provider not in self._tripped_until:
            return True
        if time.time() > self._tripped_until[provider]:
            del self._tripped_until[provider]
            self._failures[provider] = 0
            logger.info("Circuit breaker RECOVERED: %s", provider)
            self._save()
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
        date_str = self._budget_cycle_label(now_pst)
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

    @classmethod
    def _budget_cycle_label(cls, now_pst: Optional[datetime] = None) -> str:
        """Label the current Venice budget cycle.

        Venice's account quota resets at 5pm Pacific (BUDGET_RESET_HOUR_PST),
        not local midnight. So the cycle "2026-06-25" runs from
        2026-06-25 17:00 PT through 2026-06-26 17:00 PT. A spend recorded at
        4pm on the 26th still belongs to the 25th's cycle; one at 5:01pm on
        the 25th belongs to the 26th's.

        Returns a YYYY-MM-DD string: the calendar date of the cycle start.
        """
        from zoneinfo import ZoneInfo
        if now_pst is None:
            now_pst = cls._now_pst()
        if now_pst.hour < BUDGET_RESET_HOUR_PST:
            # Before 5pm: still in the cycle that started yesterday at 5pm.
            cycle_start = now_pst - timedelta(days=1)
        else:
            # 5pm or later: in the cycle that started today at 5pm.
            cycle_start = now_pst
        return cycle_start.strftime("%Y-%m-%d")


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
        self._cache: Dict[str, tuple] = {}  # base_url -> (ok, timestamp)

    def check(self, slot: ProviderSlot) -> bool:
        if not slot.is_local:
            return True

        now = time.time()
        # Cache by base_url, not provider name: multiple endpoints can share
        # a provider (e.g. strix serves :8199 and :8200) and have independent
        # health. Keying by provider would mark all endpoints healthy if any
        # one responds, masking a dead endpoint in the same provider group.
        cached = self._cache.get(slot.base_url)
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

        self._cache[slot.base_url] = (ok, now)
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
        from hermes_constants import get_hermes_home
        self._history_file = get_hermes_home() / "routing_history.jsonl"

    def _record_decision(self, result: "RouteResult", message: str) -> None:
        """Append a routing decision to routing_history.jsonl for observability.

        Best-effort: never lets a logging failure break routing. Each line is
        a JSON object so the file can be tailed, grepped, or analyzed offline
        to verify routing is actually firing in production.
        """
        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "category": result.category.value,
                "provider": result.provider,
                "model": result.model,
                "is_local": result.is_local,
                "fallback_count": result.fallback_count,
                "suppress_tools": result.suppress_tools,
                "message_preview": (message or "")[:120],
            }
            with open(self._history_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.debug("Failed to record routing decision: %s", exc)

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
        # Phase 2: try semantic classifier first (when enabled), fall back to
        # keyword on None (server down, low confidence, flag off). The keyword
        # path remains the source of truth for the 146 deterministic unit tests.
        category = None
        try:
            from agent.feature_flags import semantic_classifier_enabled
            if semantic_classifier_enabled():
                category = TaskClassifier.classify_semantic(message)
        except Exception:
            pass  # flag module unavailable → keyword path (never block routing)
        if category is None:
            category = TaskClassifier.classify(message)
        blocked = self._breaker.blocked_providers()

        # Get the ordered provider chain for this category
        chain = self._registry.chain(category)
        if not chain:
            logger.debug("No chain for %s — using primary", category.value)
            result = self._primary_route(primary_config, category)
            self._record_decision(result, message)
            return result

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

            result = RouteResult(
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
            self._record_decision(result, message)
            return result

        # All chain providers exhausted — fall to primary
        logger.warning(
            "Chain exhausted for %s (%d skipped) — falling to primary",
            category.value, fallback_count,
        )
        result = self._primary_route(primary_config, category, fallback_count)
        self._record_decision(result, message)
        return result

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
