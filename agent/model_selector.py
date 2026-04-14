"""
Model Selector v1.0 — Intelligent multi-model routing for Hermes.

Sits ABOVE the existing smart_model_routing.py binary classifier.
Reads the config's strategy, priorities, classification, and model pool
to make weighted routing decisions across the full 21-model pool.

Design principles:
- ADDITIVE ONLY: never modifies existing code paths
- Feature-flagged: routing.use_model_selector must be explicitly enabled
- Falls back to resolve_turn_route() on any error
- Preserves exact return shape: {model, runtime, label, signature}
- Uses resolve_runtime_provider() for credential pool handling

Integration:
- Called from _resolve_turn_agent_config() in cli.py and gateway/run.py
- If use_model_selector is false or selector fails, falls through to
  the existing binary classifier in smart_model_routing.py
"""

from __future__ import annotations

import re
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


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
    context_window: int = 128_000
    # Cost
    cost_per_request: float = 0.0  # 0.0 = free (local)
    # Flags
    is_thinking: bool = True
    supports_vision: bool = False


# Build the model profiles from the known pool
def _build_model_profiles() -> Dict[str, ModelProfile]:
    profiles: Dict[str, ModelProfile] = {}

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
        context_window=198_000,
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
        context_window=200_000,
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
        context_window=128_000,
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
        context_window=198_000,
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
        context_window=128_000,
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
        context_window=128_000,
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
        context_window=128_000,
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
        context_window=1_000_000,
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
        context_window=1_000_000,
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
        context_window=198_000,
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
        context_window=198_000,
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
        context_window=128_000,
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
        context_window=160_000,
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
        context_window=2_000_000,
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
        context_window=256_000,
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
        context_window=256_000,
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
        context_window=32_000,
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
        context_window=65_536,
        cost_per_request=0.0,
        is_thinking=True,
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
        context_window=65_536,
        cost_per_request=0.0,
        is_thinking=True,
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
        context_window=65_536,
        cost_per_request=0.0,
        is_thinking=True,
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
        context_window=65_536,
        cost_per_request=0.0,
        is_thinking=True,
    )

    return profiles


# Singleton — built once, never modified
MODEL_PROFILES = _build_model_profiles()


# ---------------------------------------------------------------------------
# Task classification — heuristic (no LLM call needed)
# ---------------------------------------------------------------------------
# Maps user message patterns to (task_type, complexity, urgency, quality_level)
# ---------------------------------------------------------------------------

# Code indicators
_CODE_KEYWORDS = frozenset(
    {
        "debug",
        "implement",
        "refactor",
        "traceback",
        "error",
        "function",
        "module",
        "api",
        "endpoint",
        "build",
        "test",
        "database",
        "query",
        "schema",
        "kubernetes",
        "container",
        "fix",
        "bug",
        "crash",
        "python",
        "script",
        "algorithm",
        "audit",
        "vulnerabilities",
        "security",
        "authentication",
        "middleware",
        "migration",
        "deployment",
        "microservices",
        "distributed",
        "server",
        "incident",
    }
)

# Reasoning indicators
_REASONING_KEYWORDS = frozenset(
    {
        "evaluate",
        "compare",
        "architecture",
        "approach",
        "optimize",
        "performance",
        "slow",
        # Research & root cause
        "research",
        "cause",
        # Explanation & comparison
        "explain",
        "difference",
        "redesign",
        # Causal reasoning (not bare 'why' — too ambiguous)
        "causes",
    }
)

# Writing indicators
_WRITING_KEYWORDS = frozenset(
    {
        "write",
        "draft",
        "compose",
        "summarize",
        "rewrite",
        "edit",
        "email",
        "blog",
        "post",
        "readme",
        "documentation",
    }
)

# Creative indicators
_CREATIVE_KEYWORDS = frozenset(
    {
        "creative",
        "story",
        "poem",
        "design",
        "ideas",
        "character",
        "funny",
        "uncensored",
    }
)

# Analysis indicators
_ANALYSIS_KEYWORDS = frozenset(
    {
        "analyze",
        "data",
        "dashboard",
        "metrics",
        "correlation",
        "distribution",
        "json",
        "parse",
        "coverage",
        "report",
        "percentage",
        "rate",
        "frequency",
        "plot",
        "monitoring",
    }
)

# Complexity signals — words NOT already in category keyword sets that indicate scope/depth.
# 30 overlapping entries removed (kubernetes, docker, architecture, etc. already contribute
# to technical_density via their respective category sets).
_COMPLEXITY_BOOSTERS = frozenset(
    {
        "comprehensive",
        "migrate",
        "review",
        "entire",
        "full",
        "production",
    }
)

# Urgency signals
_REALTIME_KEYWORDS = frozenset(
    {
        "quick",
    }
)


# Multi-word phrase matching — the sole source of intent-specific boosts.
# Replaces the former error/redirect elif chain with a unified list.
_PHRASE_MAP = (
    ("how does", "reasoning"),
    ("why does", "reasoning"),
    ("why is", "reasoning"),
    ("is the server", "reasoning"),
    ("what causes", "reasoning"),
    ("explain the traceback", "reasoning"),
    ("write documentation", "writing"),
    ("summarize the", "writing"),
    ("error message to be", "writing"),
    ("funny commit message", "creative"),
    ("poem about", "creative"),
    ("show me the query", "analysis"),
    ("error rate", "analysis"),
    ("how many errors", "analysis"),
)

# Tie-breaking priority — reasoning and code win ties
_TIE_PRIORITY = {
    "reasoning": 5,
    "code": 4,
    "analysis": 3,
    "writing": 2,
    "creative": 1,
    "general": 0,
}


def _classify_heuristic(message: str) -> Dict[str, str]:
    """Heuristic classifier — keyword/phrase based, no LLM call.
    Used as fallback when LLM classification is unavailable or fails.
    Returns dict with keys: task_type, complexity, urgency, quality_level.
    """
    msg_lower = message.lower()
    words = set(re.findall(r"\b\w+\b", msg_lower))
    msg_len = len(message)

    # Count keyword hits per category
    scores = {
        "code": len(words & _CODE_KEYWORDS),
        "reasoning": len(words & _REASONING_KEYWORDS),
        "writing": len(words & _WRITING_KEYWORDS),
        "analysis": len(words & _ANALYSIS_KEYWORDS),
        "creative": len(words & _CREATIVE_KEYWORDS),
    }

    for phrase, category in _PHRASE_MAP:
        if phrase in msg_lower:
            scores[category] += 1

    top_score = max(scores.values())
    if top_score > 0:
        task_type = max(scores, key=lambda k: (scores[k], _TIE_PRIORITY[k]))
    else:
        task_type = "general"

    # Complexity — composite score from keyword density, overlap, and length
    complexity_boost = len(words & _COMPLEXITY_BOOSTERS)
    total_keyword_hits = sum(scores.values())
    complexity_score = complexity_boost * 2.0 + min(2.5, total_keyword_hits * 0.15)
    if msg_len > 50:
        complexity_score += min(3.5, 0.5 + math.log(msg_len / 50, 3))

    if complexity_score >= 5.0:
        complexity = "expert"
    elif complexity_score >= 3.0:
        complexity = "complex"
    elif complexity_score >= 1.0:
        complexity = "moderate"
    else:
        complexity = "simple"

    # Urgency
    if len(words & _REALTIME_KEYWORDS) >= 1 and msg_len < 200:
        urgency = "realtime"
    elif complexity in ("expert", "complex"):
        urgency = "deep"
    else:
        urgency = "normal"

    # Quality level — higher for complex/important tasks
    if complexity in ("expert", "complex") and task_type in (
        "code",
        "reasoning",
        "analysis",
    ):
        quality_level = "maximum"
    elif complexity == "complex" or task_type in ("code", "reasoning"):
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


def _classify_with_llm(message: str, routing_config: dict = None) -> dict | None:
    """Use glm-4.5-air (free, 200+ tps) for high-accuracy classification.
    Returns classification dict or None on any failure (fallback to heuristic).
    """
    try:
        import json as _json
        from openai import OpenAI

        # Discover Z.ai API credentials (GLM/GLM_API_KEY = Z.ai coding plan)
        api_key = None
        for env_var in ("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY", "OPENAI_API_KEY"):
            val = os.environ.get(env_var, "")
            # OPENAI_API_KEY might point to a different provider — only use it
            # if the base URL is also Z.ai
            if env_var == "OPENAI_API_KEY":
                base = os.environ.get("OPENAI_BASE_URL", "")
                if "z.ai" not in base:
                    continue
            if val:
                api_key = val
                break

        # Try loading from .env file as fallback
        if not api_key:
            env_path = os.path.expanduser("~/.hermes/.env")
            if os.path.exists(env_path):
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("#") or not line:
                            continue
                        if "=" not in line:
                            continue
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip('"').strip("'")
                        if key in ("GLM_API_KEY", "ZAI_API_KEY"):
                            api_key = val
                            break

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
            max_tokens=4096,
            extra_body={"thinking": {"type": "disabled"}},
        )

        content = response.choices[0].message.content
        if not content:
            return None
        content = content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*\n?", "", content)
            content = re.sub(r"\n?```\s*$", "", content)
            content = content.strip()

        result = _json.loads(content)

        # Validate all fields
        valid_types = {
            "code",
            "reasoning",
            "writing",
            "analysis",
            "creative",
            "general",
        }
        valid_complexity = {"simple", "moderate", "complex", "expert"}
        valid_urgency = {"realtime", "normal", "deep"}
        valid_quality = {"standard", "high", "maximum"}

        if result.get("task_type") not in valid_types:
            return None
        if result.get("complexity") not in valid_complexity:
            return None
        if result.get("urgency") not in valid_urgency:
            result["urgency"] = "normal"
        if result.get("quality_level") not in valid_quality:
            result["quality_level"] = "standard"

        return {
            "task_type": result["task_type"],
            "complexity": result["complexity"],
            "urgency": result["urgency"],
            "quality_level": result["quality_level"],
        }
    except Exception:
        return None


def classify_message(
    message: str, routing_config: Dict[str, Any] = None
) -> Dict[str, str]:
    """Classify a user message for model routing.

    Uses LLM classification (glm-4.5-air) for non-trivial messages,
    with heuristic fallback for trivial messages and LLM failures.

    Returns dict with keys: task_type, complexity, urgency, quality_level.
    """
    # Fast-path: clearly trivial messages skip LLM
    msg_stripped = message.strip()
    if not msg_stripped or len(msg_stripped) < 20:
        return _classify_heuristic(message)

    # Try LLM classification for non-trivial messages
    if routing_config is not None:
        llm_result = _classify_with_llm(message, routing_config)
        if llm_result is not None:
            return llm_result

    # Fallback to heuristic
    return _classify_heuristic(message)


# ---------------------------------------------------------------------------
# Model selection engine
# ---------------------------------------------------------------------------


# Map task_type to ModelProfile capability field name
_CAPABILITY_KEY = {
    "code": "code_quality",
    "reasoning": "reasoning",
    "writing": "writing",
    "analysis": "analysis",
    "creative": "creative",
    "general": "general",
}


def select_model(
    message: str,
    routing_config: Dict[str, Any],
    primary_provider: str,
    primary_model: str,
) -> Optional[Tuple[str, str, str]]:
    """Select the best (provider, model, reason) for a given message.

    Reads the routing config's strategy, priorities, model pool, and budget
    to make a weighted decision. Returns None if selector should be skipped
    (falls through to existing binary classifier).

    Returns: (provider, model, reason_string) or None
    """
    # Check feature flag
    if not routing_config.get("use_model_selector", False):
        return None

    # Classify the message
    classification = classify_message(message, routing_config)
    task_type = classification["task_type"]
    complexity = classification["complexity"]
    urgency = classification["urgency"]
    quality_level = classification["quality_level"]

    # If simple/standard quality, let the existing binary classifier handle it
    if complexity == "simple" and quality_level == "standard":
        return None

    # Read priority weights from config (default to hardcoded if missing)
    priorities_cfg = routing_config.get("priorities", {})
    try:
        # Parse "Quality... (40%)" format
        def parse_weight(val: str, default: float) -> float:
            if isinstance(val, (int, float)):
                return float(val)
            m = re.search(r"(\d+(?:\.\d+)?)\s*%", str(val))
            return float(m.group(1)) / 100.0 if m else default

        w_quality = parse_weight(priorities_cfg.get("quality", "40%"), 0.40)
        w_speed = parse_weight(priorities_cfg.get("speed", "25%"), 0.25)
        w_context = parse_weight(priorities_cfg.get("context", "20%"), 0.20)
        w_cost = parse_weight(priorities_cfg.get("cost", "15%"), 0.15)
    except Exception:
        w_quality, w_speed, w_context, w_cost = 0.40, 0.25, 0.20, 0.15

    # Normalize weights
    total_w = w_quality + w_speed + w_context + w_cost
    w_quality /= total_w
    w_speed /= total_w
    w_context /= total_w
    w_cost /= total_w

    # Dynamic reweighting: for important tasks, quality dominates so that
    # specialist models can overcome the primary's speed/cost advantages.
    if quality_level != "standard" or complexity == "expert":
        w_quality = 0.70
        w_speed = 0.08
        w_context = 0.12
        w_cost = 0.10

    # Build candidate list from config's model pool
    models_cfg = routing_config.get("models", {})
    candidates: List[ModelProfile] = []

    for provider_name, model_names in models_cfg.items():
        for model_name in model_names:
            profile = MODEL_PROFILES.get(model_name)
            if profile:
                candidates.append(profile)

    if not candidates:
        return None

    # Get estimated token count for context window scoring (~4 chars/token)
    est_tokens = len(message) // 4
    msg_lower_raw = message.lower()

    # Score each candidate
    cap_key = _CAPABILITY_KEY.get(task_type, "general")
    scored: List[Tuple[float, ModelProfile, str]] = []

    for profile in candidates:
        # Quality score: task-specific capability
        quality_score = getattr(profile, cap_key, profile.general)

        # Vision bonus — when message references images/screenshots, models
        # with vision support get a quality edge for multimodal understanding.
        vision_bonus = 0.0
        if profile.supports_vision and any(
            sig in msg_lower_raw for sig in ("image", "screenshot", "photo", "picture", "diagram", "chart", ".png", ".jpg", ".jpeg", ".gif", ".webp")
        ):
            vision_bonus = 0.04

        # Speed score: already normalized 0-1 in profile
        speed_score = profile.speed

        # Context score: does the model have enough context window?
        if est_tokens > 0:
            ratio = est_tokens / profile.context_window
            if ratio > 0.8:
                continue  # Too tight — skip model
            ctx_score = max(0.0, 1.0 - ratio * 1.25)
        else:
            ctx_score = 1.0

        # Cost score: smooth inverse relationship — free models edge, expensive penalized.
        # Single formula replaces 6-tier step function.
        cost_score = max(0.2, 0.7 - 0.5 * math.log1p(profile.cost_per_request * 100))

        # Apply quality level filter
        if quality_level == "maximum" and quality_score < 0.82:
            continue  # Only top-tier models for critical work
        elif quality_level == "high" and quality_score < 0.68:
            continue  # Skip weak models for important tasks

        # Weighted composite score
        composite = (
            w_quality * quality_score
            + w_speed * speed_score
            + w_context * ctx_score
            + w_cost * cost_score
            + vision_bonus
        )

        reason = f"{task_type}/{complexity}"
        if quality_score >= 0.80:
            reason += "/high-cap"
        if profile.cost_per_request == 0.0:
            reason += "/free"
        elif profile.cost_per_request <= 0.01:
            reason += "/budget"

        scored.append((composite, profile, reason))

    if not scored:
        return None  # No candidates passed filters

    # Sort by composite score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_profile, best_reason = scored[0]

    # Don't route away from primary if the selector picks the same model
    # (prevents unnecessary agent rebuilds regardless of provider)
    if best_profile.name == primary_model:
        return None

    return (best_profile.provider, best_profile.name, best_reason)


# ---------------------------------------------------------------------------
# Main entry point — drop-in replacement for resolve_turn_route calls
# ---------------------------------------------------------------------------


def smart_select_route(
    user_message: str,
    routing_config: Dict[str, Any],
    primary: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Try intelligent model selection. Returns None to fall through to
    the existing binary classifier in smart_model_routing.py.

    Return shape matches resolve_turn_route() exactly:
    {
        "model": str,
        "runtime": {api_key, base_url, provider, api_mode, command, args, credential_pool},
        "label": Optional[str],
        "signature": tuple,
    }
    """
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
    _RUNTIME_KEYS = (
        "api_key",
        "base_url",
        "provider",
        "api_mode",
        "command",
        "args",
        "credential_pool",
    )
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        raw_runtime = resolve_runtime_provider(
            requested=selected_provider,
        )

        if not isinstance(raw_runtime, dict):
            raise ValueError(
                f"resolve_runtime_provider returned {type(raw_runtime).__name__}, expected dict"
            )

        runtime = {k: raw_runtime[k] for k in _RUNTIME_KEYS if k in raw_runtime}
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

    # Ensure base_url is never None — downstream code may pass it to
    # os.path functions or use it as a URL string.
    if not runtime.get("base_url"):
        runtime["base_url"] = ""

    label = f"selector → {selected_model} ({selected_provider}) [{reason}]"
    signature = (
        selected_model,
        selected_provider,
        runtime.get("base_url") or "",
        runtime.get("api_mode") or "",
        runtime.get("command"),
        tuple(runtime.get("args") or []),
    )

    return {
        "model": selected_model,
        "runtime": runtime,
        "label": label,
        "signature": signature,
    }
