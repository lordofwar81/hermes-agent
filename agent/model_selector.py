"""
Model Selector v1.0 — Intelligent multi-model routing for Hermes.
Sits ABOVE smart_model_routing.py. Feature-flagged via routing.use_model_selector.
Returns None to fall through to the existing binary classifier.
"""

from __future__ import annotations

import re
from typing import NamedTuple


class ModelProfile(NamedTuple):
    """Static profile for a model — capabilities, costs, characteristics."""
    name: str
    provider: str  # "zai", "venice", "local"
    code_quality: float = 0.5
    reasoning: float = 0.5
    writing: float = 0.5
    analysis: float = 0.5
    creative: float = 0.5
    general: float = 0.5
    speed: float = 0.5
    cost_per_request: float = 0.0


# Build the model profiles from the known pool
# Each entry: (name, provider, code_quality, reasoning, writing, analysis, creative, general, speed, cost_per_request)
_MODELS = (
    # z-ai (unlimited pre-paid)
    ("glm-5.1", "zai", 0.92, 0.93, 0.88, 0.90, 0.80, 0.88, 0.35, 0.0),
    ("glm-5-turbo", "zai", 0.88, 0.88, 0.85, 0.86, 0.78, 0.85, 0.65, 0.0),
    ("glm-5", "zai", 0.85, 0.87, 0.84, 0.85, 0.76, 0.82, 0.55, 0.0),
    ("glm-4.7", "zai", 0.80, 0.82, 0.86, 0.83, 0.78, 0.82, 0.65, 0.0),
    ("glm-4.6", "zai", 0.75, 0.76, 0.80, 0.77, 0.74, 0.78, 0.65, 0.0),
    ("glm-4.5", "zai", 0.72, 0.73, 0.78, 0.74, 0.72, 0.75, 0.65, 0.0),
    ("glm-4.5-air", "zai", 0.60, 0.62, 0.68, 0.64, 0.65, 0.66, 0.90, 0.0),
    # venice ($7.40/day budget)
    ("qwen-3-6-plus", "venice", 0.90, 0.91, 0.85, 0.88, 0.82, 0.87, 0.50, 0.05),
    ("claude-sonnet-4-6", "venice", 0.94, 0.95, 0.92, 0.93, 0.88, 0.91, 0.45, 0.22),
    ("zai-org-glm-5", "venice", 0.88, 0.89, 0.84, 0.86, 0.78, 0.84, 0.40, 0.04),
    ("zai-org-glm-4.7", "venice", 0.78, 0.80, 0.83, 0.81, 0.76, 0.80, 0.55, 0.03),
    ("zai-org-glm-4.7-flash", "venice", 0.65, 0.66, 0.70, 0.67, 0.68, 0.68, 0.80, 0.007),
    ("deepseek-v3.2", "venice", 0.82, 0.84, 0.78, 0.82, 0.72, 0.80, 0.70, 0.008),
    ("grok-4-20-beta", "venice", 0.85, 0.88, 0.82, 0.86, 0.80, 0.84, 0.30, 0.10),
    ("qwen3-coder-480b-a35b-instruct", "venice", 0.93, 0.82, 0.72, 0.80, 0.60, 0.75, 0.35, 0.04),
    ("qwen3-5-35b-a3b", "venice", 0.76, 0.78, 0.74, 0.76, 0.72, 0.76, 0.65, 0.02),
    ("venice-uncensored", "venice", 0.60, 0.62, 0.72, 0.58, 0.85, 0.65, 0.75, 0.01),
    # local (free, private, Vulkan)
    ("Qwen3-Coder-30B-APEX-I-Compact", "local", 0.78, 0.74, 0.68, 0.72, 0.60, 0.70, 0.55, 0.0),
    ("LFM2-24B-A2B-APEX-I-Compact", "local", 0.72, 0.70, 0.65, 0.68, 0.58, 0.68, 0.70, 0.0),
    ("Qwopus-MoE-35B-A3B-APEX-I-Compact", "local", 0.74, 0.76, 0.66, 0.72, 0.56, 0.68, 0.40, 0.0),
    ("Huihui3.5-67B-A3B-APEX-I-Compact", "local", 0.76, 0.80, 0.70, 0.76, 0.60, 0.72, 0.38, 0.0),
)

MODEL_PROFILES = {m[0]: ModelProfile(*m) for m in _MODELS}


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
        "research", "cause", "explain", "redesign", "causes", "difference",
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

# Question-starting prefixes → reasoning intent boost
_Q_PREFIXES = ("why ", "how ", "is the ", "what's ")

# Single combined phrase regex — one search instead of four (+2 bonus)
_PHRASE_RE = re.compile(
    r"\b(?P<writing>summarize|(?:write|draft|compose|rewrite|edit)\b.*\b(?:documentation|readme|blog|email|post|proposal|announcement|error|text|content|letter))"
    r"|\b(?P<creative>(?:write|compose)\b.*\b(?:poem|funny|story|creative|joke|song|haiku))"
    r"|\b(?P<analysis>how\s+many|what\s+percentage|show\s+me\s+the|error\s+rate|execution\s+time)"
    r"|\b(?P<reasoning>explain)\b",
    re.I,
)


def classify_message(message: str) -> dict[str, str]:
    """Classify a user message. Keyword + phrase heuristic classifier."""
    msg_lower = message.lower()
    words = set(_WORD_RE.findall(msg_lower))

    # Count keyword hits via single-pass lookup
    hits = {"code": 0, "reasoning": 0, "writing": 0, "analysis": 0, "creative": 0}
    for w in words:
        cat = _KEYWORD_MAP.get(w)
        if cat:
            hits[cat] += 1

    # Question pattern → reasoning boost
    if any(msg_lower.startswith(p) for p in _Q_PREFIXES):
        hits["reasoning"] += 1

    # Phrase override — single regex search (+2 bonus)
    m = _PHRASE_RE.search(message)
    if m:
        hits[m.lastgroup] += 2

    total_hits = sum(hits.values())

    if total_hits > 0:
        # Tie-breaking priority: reasoning > code > analysis > writing > creative
        task_type = max(("reasoning", "code", "analysis", "writing", "creative"), key=hits.__getitem__)
    else:
        task_type = "general"

    # Complexity — keyword density and length
    complexity_score = total_hits + max(0.0, (len(message) - 50) / 25.0)
    complexity = ("simple", "moderate", "complex", "expert")[min(int(complexity_score), 3)]

    # Urgency — quick keyword for realtime, expert/complex for deep
    urgency = "realtime" if "quick" in words else ("deep" if complexity in ("expert", "complex") else "normal")

    return {
        "task_type": task_type, "complexity": complexity,
        "urgency": urgency,
        "quality_level": "maximum" if task_type in ("code", "reasoning") and complexity in ("expert", "complex")
        else "high" if task_type in ("code", "reasoning") or complexity == "complex"
        else "standard",
    }


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
    classification = classify_message(message)
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
