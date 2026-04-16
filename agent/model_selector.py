"""
Model Selector v1.0 — Intelligent multi-model routing for Hermes.
Sits ABOVE smart_model_routing.py. Feature-flagged via routing.use_model_selector.
Returns None to fall through to the existing binary classifier.
"""

from __future__ import annotations

import re
import os
import threading
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Per-model concurrency tracker
# ---------------------------------------------------------------------------
# Z.ai enforces per-model concurrency limits (e.g. glm-5.1=1, glm-4.7=2).
# This tracker prevents the selector from routing to a model that's already
# at its provider-defined limit. In-flight requests increment; on completion
# they decrement. When a model is at capacity, the selector skips it and
# picks the next-best candidate.
# ---------------------------------------------------------------------------


class _ConcurrencyTracker:
    """Thread-safe per-model in-flight request counter."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def acquire(self, model: str, limit: int) -> bool:
        """Try to increment in-flight count. Returns True if under limit."""
        if limit <= 0:
            return True  # no limit
        with self._lock:
            current = self._counts.get(model, 0)
            if current >= limit:
                return False
            self._counts[model] = current + 1
            return True

    def release(self, model: str) -> None:
        with self._lock:
            current = self._counts.get(model, 0)
            self._counts[model] = max(0, current - 1)


# Module-level singleton — shared across all selector calls
concurrency_tracker = _ConcurrencyTracker()


# ---------------------------------------------------------------------------
# Model capabilities database
# ---------------------------------------------------------------------------
# Each entry maps model name to its capabilities and estimated costs.
# This is the single source of truth for routing decisions.
# ---------------------------------------------------------------------------


@dataclass
class ModelProfile:
    """Static profile for a model — capabilities, costs, characteristics."""

    name: str
    provider: str  # "zai", "venice", "local"
    # Capability scores (0.0–1.0, relative within pool)
    code_quality: float = 0.5
    reasoning: float = 0.5
    writing: float = 0.5
    analysis: float = 0.5
    creative: float = 0.5
    general: float = 0.5
    # Performance
    speed: float = 0.5  # tokens/sec relative (1.0 = fastest in pool)
    # Cost
    cost_per_request: float = 0.0  # 0.0 = free (local)
    # Concurrency limit from provider (0 = unlimited)
    max_concurrent: int = 0


# Build the model profiles from the known pool
def _build_model_profiles() -> dict[str, ModelProfile]:
    profiles: dict[str, ModelProfile] = {}

    def _add(name, provider, **kwargs):
        profiles[name] = ModelProfile(name=name, provider=provider, **kwargs)

    # === z-ai models (unlimited pre-paid, per-model concurrency limits) ===
    _add(
        "glm-5.1",
        "zai",
        code_quality=0.92,
        reasoning=0.93,
        writing=0.88,
        analysis=0.90,
        creative=0.80,
        general=0.88,
        speed=0.35,
        cost_per_request=0.0,
        max_concurrent=1,
    )
    _add(
        "glm-5-turbo",
        "zai",
        code_quality=0.88,
        reasoning=0.88,
        writing=0.85,
        analysis=0.86,
        creative=0.78,
        general=0.85,
        speed=0.65,
        cost_per_request=0.0,
        max_concurrent=1,
    )
    _add(
        "glm-5",
        "zai",
        code_quality=0.85,
        reasoning=0.87,
        writing=0.84,
        analysis=0.85,
        creative=0.76,
        general=0.82,
        speed=0.55,
        cost_per_request=0.0,
        max_concurrent=2,
    )
    _add(
        "glm-4.7",
        "zai",
        code_quality=0.80,
        reasoning=0.82,
        writing=0.86,
        analysis=0.83,
        creative=0.78,
        general=0.82,
        speed=0.65,
        cost_per_request=0.0,
        max_concurrent=2,
    )
    _add(
        "glm-4.6",
        "zai",
        code_quality=0.75,
        reasoning=0.76,
        writing=0.80,
        analysis=0.77,
        creative=0.74,
        general=0.78,
        speed=0.65,
        cost_per_request=0.0,
        max_concurrent=3,
    )
    _add(
        "glm-4.5",
        "zai",
        code_quality=0.72,
        reasoning=0.73,
        writing=0.78,
        analysis=0.74,
        creative=0.72,
        general=0.75,
        speed=0.65,
        cost_per_request=0.0,
        max_concurrent=10,
    )
    _add(
        "glm-4.5-air",
        "zai",
        code_quality=0.60,
        reasoning=0.62,
        writing=0.68,
        analysis=0.64,
        creative=0.65,
        general=0.66,
        speed=0.90,
        cost_per_request=0.0,
        max_concurrent=5,
    )

    # === Venice models ($7.40/day budget) ===
    # Venice-wide concurrency: 3 max concurrent requests across all Venice models.
    # Prevents budget drain from parallel requests ($7.40/day budget).
    # Premium models (Sonnet, qwen-3-6-plus) get 1 concurrent to limit cost spikes.
    _add(
        "qwen-3-6-plus",
        "venice",
        code_quality=0.90,
        reasoning=0.91,
        writing=0.85,
        analysis=0.88,
        creative=0.82,
        general=0.87,
        speed=0.50,
        cost_per_request=0.05,
        max_concurrent=2,
    )
    _add(
        "claude-sonnet-4-6",
        "venice",
        code_quality=0.94,
        reasoning=0.95,
        writing=0.92,
        analysis=0.93,
        creative=0.88,
        general=0.91,
        speed=0.45,
        cost_per_request=0.22,
        max_concurrent=1,  # Most expensive — limit to 1
    )
    _add(
        "zai-org-glm-5",
        "venice",
        code_quality=0.88,
        reasoning=0.89,
        writing=0.84,
        analysis=0.86,
        creative=0.78,
        general=0.84,
        speed=0.40,
        cost_per_request=0.04,
        max_concurrent=3,
    )
    _add(
        "zai-org-glm-4.7",
        "venice",
        code_quality=0.78,
        reasoning=0.80,
        writing=0.83,
        analysis=0.81,
        creative=0.76,
        general=0.80,
        speed=0.55,
        cost_per_request=0.03,
        max_concurrent=3,
    )
    _add(
        "zai-org-glm-4.7-flash",
        "venice",
        code_quality=0.65,
        reasoning=0.66,
        writing=0.70,
        analysis=0.67,
        creative=0.68,
        general=0.68,
        speed=0.80,
        cost_per_request=0.007,
        max_concurrent=5,  # Cheapest Venice — allow more parallel
    )
    _add(
        "deepseek-v3.2",
        "venice",
        code_quality=0.82,
        reasoning=0.84,
        writing=0.78,
        analysis=0.82,
        creative=0.72,
        general=0.80,
        speed=0.70,
        cost_per_request=0.008,
        max_concurrent=5,  # Budget workhorse — allow more parallel
    )
    _add(
        "grok-4-20-beta",
        "venice",
        code_quality=0.85,
        reasoning=0.88,
        writing=0.82,
        analysis=0.86,
        creative=0.80,
        general=0.84,
        speed=0.30,
        cost_per_request=0.10,
        max_concurrent=2,
    )
    _add(
        "qwen3-coder-480b-a35b-instruct",
        "venice",
        code_quality=0.93,
        reasoning=0.82,
        writing=0.72,
        analysis=0.80,
        creative=0.60,
        general=0.75,
        speed=0.35,
        cost_per_request=0.04,
        max_concurrent=2,
    )
    _add(
        "qwen3-5-35b-a3b",
        "venice",
        code_quality=0.76,
        reasoning=0.78,
        writing=0.74,
        analysis=0.76,
        creative=0.72,
        general=0.76,
        speed=0.65,
        cost_per_request=0.02,
        max_concurrent=3,
    )
    _add(
        "venice-uncensored",
        "venice",
        code_quality=0.60,
        reasoning=0.62,
        writing=0.72,
        analysis=0.58,
        creative=0.85,
        general=0.65,
        speed=0.75,
        cost_per_request=0.01,
        max_concurrent=3,
    )

    # === Local APEX models (free, private, Vulkan) ===
    _add(
        "Qwen3-Coder-30B-APEX-I-Compact",
        "local",
        code_quality=0.78,
        reasoning=0.74,
        writing=0.68,
        analysis=0.72,
        creative=0.60,
        general=0.70,
        speed=0.55,
        cost_per_request=0.0,
    )
    _add(
        "LFM2-24B-A2B-APEX-I-Compact",
        "local",
        code_quality=0.72,
        reasoning=0.70,
        writing=0.65,
        analysis=0.68,
        creative=0.58,
        general=0.68,
        speed=0.70,
        cost_per_request=0.0,
    )
    _add(
        "Qwopus-MoE-35B-A3B-APEX-I-Compact",
        "local",
        code_quality=0.74,
        reasoning=0.76,
        writing=0.66,
        analysis=0.72,
        creative=0.56,
        general=0.68,
        speed=0.40,
        cost_per_request=0.0,
    )
    _add(
        "Huihui3.5-67B-A3B-APEX-I-Compact",
        "local",
        code_quality=0.76,
        reasoning=0.80,
        writing=0.70,
        analysis=0.76,
        creative=0.60,
        general=0.72,
        speed=0.38,
        cost_per_request=0.0,
    )

    return profiles


# Singleton — built once, never modified
MODEL_PROFILES = _build_model_profiles()


# ---------------------------------------------------------------------------
# Task classification — heuristic (no LLM call needed)
# ---------------------------------------------------------------------------
# Maps user message patterns to (task_type, complexity, urgency, quality_level)
# ---------------------------------------------------------------------------

# Unified keyword-to-category map — single lookup per word.
_KEYWORD_MAP: dict[str, str] = {
    # code
    **{w: "code" for w in (
        "debug", "implement", "refactor", "traceback", "error", "function",
        "module", "api", "endpoint", "build", "test", "database", "query",
        "schema", "kubernetes", "container", "fix", "bug", "crash", "python",
        "script", "algorithm", "audit", "vulnerabilities", "security",
        "authentication", "middleware", "migration", "deployment",
        "microservices", "distributed", "server", "incident",
    )},
    # reasoning
    **{w: "reasoning" for w in (
        "evaluate", "architecture", "optimize", "performance", "slow",
        "research", "cause", "explain", "redesign", "causes",
    )},
    # writing
    **{w: "writing" for w in (
        "write", "draft", "compose", "summarize", "rewrite", "edit",
        "email", "blog", "post", "readme", "documentation",
    )},
    # creative
    **{w: "creative" for w in (
        "creative", "story", "poem", "design", "ideas", "funny",
    )},
    # analysis
    **{w: "analysis" for w in (
        "analyze", "data", "dashboard", "metrics", "correlation",
        "distribution", "coverage", "report", "percentage", "rate",
    )},
}

_WORD_RE = re.compile(r"\w+")

# Question-starting patterns → reasoning intent boost
_QUESTION_RE = re.compile(
    r"^(why|how|is\s+the|what'?s?\s+the|where|when|who)\b", re.IGNORECASE
)

# Phrase-level intent overrides — signal stronger than individual keywords (+2 bonus)
_PHRASE_OVERRIDES: list[tuple[re.Pattern, str]] = [
    # Writing intent — "write/draft/compose/rewrite/edit X about Y"
    (re.compile(r"\b(write|draft|compose|rewrite|edit)\b.*\b(documentation|readme|blog|email|post|proposal|announcement|error|text|content|letter)\b", re.I), "writing"),
    (re.compile(r"\bsummarize\b", re.I), "writing"),
    # Creative intent — "write a poem/funny/story"
    (re.compile(r"\b(write|compose)\b.*\b(poem|funny|story|creative|joke|song|haiku)\b", re.I), "creative"),
    # Analysis intent — quantitative queries
    (re.compile(r"\b(how\s+many|what\s+percentage|show\s+me\s+the|error\s+rate|execution\s+time)\b", re.I), "analysis"),
    # Reasoning intent — "explain X"
    (re.compile(r"\bexplain\b", re.I), "reasoning"),
]


def _classify_heuristic(message: str) -> dict[str, str]:
    """Keyword + phrase heuristic classifier. Returns task_type, complexity, urgency, quality_level."""
    msg_lower = message.lower()
    words = set(_WORD_RE.findall(msg_lower))

    # Count keyword hits via single-pass lookup
    hits = {"code": 0, "reasoning": 0, "writing": 0, "analysis": 0, "creative": 0}
    for w in words:
        cat = _KEYWORD_MAP.get(w)
        if cat:
            hits[cat] += 1

    # Question pattern → reasoning boost
    if _QUESTION_RE.search(msg_lower):
        hits["reasoning"] += 1

    # Phrase overrides — strong intent signals (+2 bonus)
    for pattern, cat in _PHRASE_OVERRIDES:
        if pattern.search(message):
            hits[cat] += 2

    total_hits = sum(hits.values())

    if total_hits > 0:
        # Tie-breaking priority: reasoning > code > analysis > writing > creative
        best = max(hits.values())
        for cat in ("reasoning", "code", "analysis", "writing", "creative"):
            if hits[cat] == best:
                task_type = cat
                break
    else:
        task_type = "general"

    # Complexity — composite score from keyword density and length
    complexity_score = total_hits
    if len(message) > 50:
        complexity_score += (len(message) - 50) / 25.0

    if complexity_score >= 5.0:
        complexity = "expert"
    elif complexity_score >= 3.0:
        complexity = "complex"
    elif complexity_score >= 1.0:
        complexity = "moderate"
    else:
        complexity = "simple"

    # Urgency — quick keyword for realtime, expert/complex for deep
    if "quick" in words:
        urgency = "realtime"
    elif complexity in ("expert", "complex"):
        urgency = "deep"
    else:
        urgency = "normal"

    # Quality level — code/reasoning or complex tasks get elevated quality
    if task_type in ("code", "reasoning") and complexity in ("expert", "complex"):
        quality_level = "maximum"
    elif task_type in ("code", "reasoning") or complexity == "complex":
        quality_level = "high"
    else:
        quality_level = "standard"

    return {
        "task_type": task_type,
        "complexity": complexity,
        "urgency": urgency,
        "quality_level": quality_level,
    }


# ---------------------------------------------------------------------------
# LLM-based classification — uses glm-4.5-air for high-accuracy routing
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM_PROMPT = """You are a message classifier for an AI model routing system. Classify the user message with maximum precision.

Return ONLY valid JSON with exactly these keys:
{"task_type": "...", "complexity": "...", "urgency": "...", "quality_level": "..."}

CLASSIFICATION RULES:

task_type — classify by PRIMARY INTENT:
- "code": implementation, debugging, refactoring, deployment, testing, git ops, infrastructure changes. User wants code produced or modified.
- "reasoning": explaining why/how, comparing approaches, evaluating trade-offs, root cause analysis, architecture decisions, research into causes/mechanisms. User wants understanding or a decision.
- "writing": composing prose: docs, READMEs, emails, reports, articles, blog posts, documentation. User wants text authored.
- "analysis": examining data, metrics, patterns, coverage, trends, telemetry, dashboards. User wants insights extracted from data.
- "creative": fiction, art, brainstorming, humor, roleplay, design exploration. User wants creative output.
- "general": greetings, acknowledgments, simple questions, chitchat.

KEY DISTINCTIONS (memorize these):
- "write a Python script" → code. "write a README" → writing. "write documentation" → writing.
- "analyze performance" → analysis. "explain why it's slow" → reasoning.
- "research the root cause" → reasoning. "evaluate the test report" → analysis.
- "review the code/PR" → code. "review the dashboards/metrics" → analysis.
- "deploy to production" → code. "explain deployment strategy" → reasoning.
- "implement X with Y" → code. "compare X vs Y" → reasoning.
- "debug the error" → code. "what causes the error" → reasoning.
- "audit the codebase" → code. "audit the architecture" → reasoning.

complexity — err on the side of HIGHER complexity for technical work:
- simple: trivial, < 30 seconds, factual lookup, greeting, single-word answer
- moderate: single clear task, one file/component, routine operation, basic script
- complex: multi-step, multi-component, requires domain expertise, integration between systems, optimization, root cause investigation, any task involving Kubernetes/Docker/distributed systems/microservices
- expert: architecture decisions, system-wide scope, security audits, production deployments, "from scratch" builds, anything involving "entire codebase/system", deep domain research requiring specialist knowledge

urgency:
- realtime: user explicitly wants speed (quick, fast, ASAP, simple, brief)
- normal: standard processing
- deep: complex work benefiting from the most capable model

quality_level — correlates with complexity and task importance:
- standard: casual, routine, greetings, trivial tasks
- high: any non-trivial coding, reasoning, analysis, or professional writing task
- maximum: production-critical, architecture decisions, comprehensive reviews, system-wide scope, security-related tasks"""


# LLM classification validation constants (hoisted from _classify_with_llm)
_LLM_VALID = {
    "task_type": {"code", "reasoning", "writing", "analysis", "creative", "general"},
    "complexity": {"simple", "moderate", "complex", "expert"},
    "urgency": {"realtime", "normal", "deep"},
    "quality_level": {"standard", "high", "maximum"},
}
_LLM_DEFAULTS = {"urgency": "normal", "quality_level": "standard"}


def _classify_with_llm(message: str) -> dict | None:
    """Use glm-4.5-air for high-accuracy classification. Returns None on failure."""
    try:
        import json
        from openai import OpenAI

        # Discover Z.ai API credentials
        api_key = None
        for env_var in ("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"):
            api_key = os.environ.get(env_var)
            if api_key:
                break
        if not api_key:
            base = os.environ.get("OPENAI_BASE_URL", "")
            if "z.ai" in base:
                api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            return None

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.z.ai/api/coding/paas/v4",
            timeout=5.0,
        )

        response = client.chat.completions.create(
            model="glm-4.5-air",
            messages=[
                {"role": "system", "content": _CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            temperature=0.0,
            extra_body={"thinking": {"type": "disabled"}},
        )

        content = response.choices[0].message.content
        if not content:
            return None
        content = content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.removeprefix("```json\n").removeprefix("```\n").removeprefix("```")
            content = content.removesuffix("```").strip()

        result = json.loads(content)

        # Validate all fields (see module-level _LLM_VALID / _LLM_DEFAULTS)

        for field, valid in _LLM_VALID.items():
            val = result.get(field)
            if val not in valid:
                if field in _LLM_DEFAULTS:
                    result[field] = _LLM_DEFAULTS[field]
                else:
                    return None

        return {k: result[k] for k in ("task_type", "complexity", "urgency", "quality_level")}
    except Exception:
        return None


def classify_message(
    message: str, routing_config: dict[str, Any] = None
) -> dict[str, str]:
    """Classify a user message. Tries LLM first, then heuristic fallback."""
    if routing_config is not None:
        llm_result = _classify_with_llm(message)
        if llm_result is not None:
            return llm_result
    return _classify_heuristic(message)


# ---------------------------------------------------------------------------
# Model selection engine
# ---------------------------------------------------------------------------


# Weight parser — extracts percentage from config strings like "Quality (40%)"
_WEIGHT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_DEFAULT_WEIGHTS = {"quality": 0.40, "speed": 0.30, "cost": 0.30}


def _parse_weights(priorities_cfg: dict) -> tuple[float, float, float]:
    """Parse priority weights from config, returning (quality, speed, cost)."""
    def _pw(val, default):
        if isinstance(val, (int, float)):
            return float(val)
        m = _WEIGHT_RE.search(str(val))
        return float(m.group(1)) / 100.0 if m else default

    return tuple(
        _pw(priorities_cfg.get(k, f"{int(v*100)}%"), v)
        for k, v in _DEFAULT_WEIGHTS.items()
    )


def select_model(
    message: str,
    routing_config: dict[str, Any],
    primary_provider: str,
    primary_model: str,
) -> tuple[str, str, str] | None:
    """Select best (provider, model, reason) for a message. Returns None to skip."""
    # Check feature flag
    if not routing_config.get("use_model_selector", False):
        return None

    # Classify the message
    classification = classify_message(message, routing_config)
    task_type = classification["task_type"]
    complexity = classification["complexity"]
    quality_level = classification["quality_level"]

    # If simple complexity, let the existing binary classifier handle it
    if complexity == "simple":
        return None

    # Dynamic reweighting: for important tasks, quality dominates so that
    # specialist models can overcome the primary's speed/cost advantages.
    if quality_level != "standard" or complexity == "expert":
        w_quality, w_speed, w_cost = 0.70, 0.08, 0.10
    else:
        try:
            w_quality, w_speed, w_cost = _parse_weights(
                routing_config.get("priorities", {})
            )
        except Exception:
            w_quality, w_speed, w_cost = 0.40, 0.30, 0.30

    # Build candidate list from config's model pool
    candidates = [
        MODEL_PROFILES[m]
        for model_names in routing_config.get("models", {}).values()
        for m in model_names
        if m in MODEL_PROFILES
    ]

    if not candidates:
        return None

    min_quality = 0.82 if quality_level == "maximum" else 0.0
    scored: list[tuple[float, ModelProfile]] = []

    for profile in candidates:
        # Quality score: task-specific capability
        quality_score = getattr(profile, "code_quality" if task_type == "code" else task_type, profile.general)

        # Apply quality level filter (before acquiring concurrency slot)
        if quality_score < min_quality:
            continue

        # Concurrency gate: skip model if at provider-defined limit
        if profile.max_concurrent > 0 and not concurrency_tracker.acquire(profile.name, profile.max_concurrent):
            continue

        # Weighted composite score
        composite = (
            w_quality * quality_score
            + w_speed * profile.speed
            + w_cost * max(0.2, 0.7 - profile.cost_per_request * 15.0)
        )

        scored.append((composite, profile))

    if not scored:
        return None  # No candidates passed filters

    # Sort by composite score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Release all candidates except the winner — concurrency slot reserved
    # only for the model we actually return.
    for _, profile in scored[1:]:
        if profile.max_concurrent > 0:
            concurrency_tracker.release(profile.name)

    _, best_profile = scored[0]
    best_reason = f"{task_type}/{complexity}"

    # Don't route away from primary if the selector picks the same model
    # (prevents unnecessary agent rebuilds regardless of provider)
    if best_profile.name == primary_model:
        if best_profile.max_concurrent > 0:
            concurrency_tracker.release(best_profile.name)
        return None

    return (best_profile.provider, best_profile.name, best_reason)


# ---------------------------------------------------------------------------
# Main entry point — drop-in replacement for resolve_turn_route calls
# ---------------------------------------------------------------------------

_RUNTIME_KEYS = ("api_key", "base_url", "provider", "api_mode", "command", "args", "credential_pool")


def smart_select_route(
    user_message: str,
    routing_config: dict[str, Any],
    primary: dict[str, Any],
) -> dict[str, Any] | None:
    """Try intelligent model selection. Returns None to fall through to smart_model_routing."""
    selection = select_model(
        message=user_message,
        routing_config=routing_config,
        primary_provider=str(primary.get("provider", "") or "").strip().lower(),
        primary_model=str(primary.get("model", "") or "").strip(),
    )

    if selection is None:
        return None

    selected_provider, selected_model, reason = selection

    # Resolve runtime credentials for the selected model.
    # Only extract keys that AIAgent.__init__ accepts — resolve_runtime_provider()
    # returns extras like "requested_provider", "source", etc. that would cause
    # TypeError on unpack.
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        raw_runtime = resolve_runtime_provider(requested=selected_provider)
        if not isinstance(raw_runtime, dict):
            raise ValueError

        runtime = {k: raw_runtime.get(k, "") if k == "base_url" else raw_runtime[k] for k in _RUNTIME_KEYS if k in raw_runtime}
        runtime["provider"] = selected_provider

    except Exception:
        # Fallback: construct runtime from primary's credentials pattern
        runtime = {
            "api_key": primary.get("api_key") or "",
            "base_url": primary.get("base_url") or "",
            "provider": selected_provider,
            "api_mode": primary.get("api_mode") or "chat_completions",
            "command": primary.get("command"),
            "args": list(primary.get("args") or []),
            "credential_pool": primary.get("credential_pool"),
        }

    return {
        "model": selected_model,
        "runtime": runtime,
        "label": f"selector → {selected_model} ({selected_provider}) [{reason}]",
        "signature": (
            selected_model,
            selected_provider,
            runtime.get("base_url") or "",
            runtime.get("api_mode") or "",
            runtime.get("command"),
            tuple(runtime.get("args") or []),
        ),
    }
