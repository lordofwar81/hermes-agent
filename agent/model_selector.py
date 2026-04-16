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
# Minimal set: phrase regex handles most classification; keywords break ties
# and provide signal for cases without phrase matches.
_KEYWORD_MAP: dict[str, str] = {
    # code
    **{w: "code" for w in (
        "debug", "implement", "module", "fix", "bug", "script",
        "audit", "security", "server",
    )},
    # reasoning
    **{w: "reasoning" for w in (
        "evaluate", "architecture", "performance", "research",
        "cause", "redesign", "causes",
    )},
    # writing (most handled by phrase regex; readme is keyword-only)
    "readme": "writing",
    # analysis
    **{w: "analysis" for w in (
        "analyze", "data", "metrics", "report", "rate",
    )},
    # creative (most handled by phrase regex)
    "design": "creative",
    "ideas": "creative",
}

# Question-starting prefixes → reasoning intent boost (tuple for startswith)
_Q_PREFIXES = ("why ", "how ", "is the ", "what's ")

# Punctuation to strip from split words
_PUNCT = ".,!?;:'\"()[]{}"

# Single combined phrase regex — one search instead of four (+2 bonus)
_PHRASE_RE = re.compile(
    r"\b(?P<writing>summarize|(?:write|draft|compose|rewrite|edit)\b.*\b(?:documentation|readme|blog|email|post|proposal|announcement|error|text|content|letter))"
    r"|\b(?P<creative>(?:write|compose)\b.*\b(?:poem|funny|story|creative|joke|song|haiku))"
    r"|\b(?P<analysis>how\s+many|what\s+percentage|show\s+me\s+the|error\s+rate|execution\s+time)"
    r"|\b(?P<reasoning>explain)\b",
)


def classify_message(message: str) -> dict[str, str]:
    """Classify a user message. Keyword + phrase heuristic classifier."""
    msg_lower = message.lower()

    # Count keyword hits via single-pass lookup (split + strip, no regex)
    hits = {"code": 0, "reasoning": 0, "writing": 0, "analysis": 0, "creative": 0}
    for w in msg_lower.split():
        w = w.strip(_PUNCT)
        cat = _KEYWORD_MAP.get(w)
        if cat:
            hits[cat] += 1

    # Question pattern → reasoning boost (startswith, no regex)
    if msg_lower.startswith(_Q_PREFIXES):
        hits["reasoning"] += 1

    # Code signal: backtick-enclosed code snippets
    if "`" in message:
        hits["code"] += 1

    # Phrase override — single regex search (+2 bonus)
    m = _PHRASE_RE.search(msg_lower)
    if m:
        hits[m.lastgroup] += 2

    total_hits = sum(hits.values())

    if total_hits > 0:
        # Tie-breaking priority: reasoning > code > analysis > writing > creative
        task_type = max(("reasoning", "code", "analysis", "writing", "creative"), key=hits.__getitem__)
    else:
        task_type = "general"

    # Complexity — keyword hit count only
    complexity = ("simple", "moderate", "complex", "expert")[min(total_hits, 3)]

    # Urgency — quick keyword for realtime, expert/complex for deep
    urgency = "realtime" if "quick" in msg_lower else ("deep" if complexity in ("expert", "complex") else "normal")

    return {
        "task_type": task_type, "complexity": complexity,
        "urgency": urgency,
    }


def select_model(
    message: str,
    routing_config: dict,
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
    # Quality-dominated weighting (code/reasoning get stronger quality preference)
    q_weight = 0.85 if task_type in ("code", "reasoning") else 0.60

    # If simple complexity, let the existing binary classifier handle it
    if complexity == "simple":
        return None

    # Build candidate list from config's model pool
    candidates = [
        MODEL_PROFILES[m]
        for model_names in routing_config.get("models", {}).values()
        for m in model_names
        if m in MODEL_PROFILES
    ]

    if not candidates:
        return None

    # Find best model by weighted composite score
    attr = "code_quality" if task_type == "code" else task_type
    best_profile = max(
        candidates,
        key=lambda p: q_weight * getattr(p, attr, p.general) + (1 - q_weight) * p.speed,
    )
    best_reason = f"{task_type}/{complexity}"

    # Don't route away from primary if the selector picks the same model
    # (prevents unnecessary agent rebuilds regardless of provider)
    if best_profile.name == primary_model:
        return None

    return (best_profile.provider, best_profile.name, best_reason)


_RUNTIME_KEYS = ("api_key", "base_url", "provider", "api_mode", "command", "args", "credential_pool")


def smart_select_route(
    user_message: str,
    routing_config: dict,
    primary: dict,
) -> dict | None:
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
