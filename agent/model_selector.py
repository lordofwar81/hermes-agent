"""
Model Selector v1.0 — Intelligent multi-model routing for Hermes.
Sits ABOVE smart_model_routing.py. Feature-flagged via routing.use_model_selector.
Returns None to fall through to the existing binary classifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


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


# Build the model profiles from the known pool
def _build_model_profiles() -> dict[str, ModelProfile]:
    profiles: dict[str, ModelProfile] = {}

    def _add(name, provider, **kwargs):
        profiles[name] = ModelProfile(name=name, provider=provider, **kwargs)

    # === z-ai models (unlimited pre-paid) ===
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
    )

    # === Venice models ($7.40/day budget) ===
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
    r"^(why|how|is\s+the|what'?s?\s+the)\b", re.IGNORECASE
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
# LLM-based classification — stub (kept for test mock compatibility)
# ---------------------------------------------------------------------------


def _classify_with_llm(message: str) -> dict | None:
    """Stub — LLM classifier removed; heuristic handles all classification."""
    return None


def classify_message(
    message: str, routing_config: dict[str, Any] = None
) -> dict[str, str]:
    """Classify a user message. Heuristic-only classification."""
    return _classify_heuristic(message)


# ---------------------------------------------------------------------------
# Model selection engine
# ---------------------------------------------------------------------------


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

    # Dynamic reweighting: quality dominates for important tasks
    if quality_level != "standard" or complexity == "expert":
        w_quality, w_speed, w_cost = 0.70, 0.08, 0.10
    else:
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

        # Apply quality level filter
        if quality_score < min_quality:
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

    _, best_profile = scored[0]
    best_reason = f"{task_type}/{complexity}"

    # Don't route away from primary if the selector picks the same model
    # (prevents unnecessary agent rebuilds regardless of provider)
    if best_profile.name == primary_model:
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
