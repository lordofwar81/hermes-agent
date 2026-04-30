"""Model Selector — intelligent multi-model routing for Hermes."""

import re
from typing import NamedTuple


class ModelProfile(NamedTuple):
    name: str
    provider: str
    code_quality: float = 0.5
    reasoning: float = 0.5
    writing: float = 0.5
    analysis: float = 0.5
    creative: float = 0.5
    general: float = 0.5
    speed: float = 0.5


_MODELS = (
    ("glm-5.1", "zai", 0.92, 0.93, 0.88, 0.90, 0.80, 0.88, 0.35),
    ("glm-5-turbo", "zai", 0.88, 0.88, 0.85, 0.86, 0.78, 0.85, 0.65),
    ("glm-5", "zai", 0.85, 0.87, 0.84, 0.85, 0.76, 0.82, 0.55),
    ("glm-4.7", "zai", 0.80, 0.82, 0.86, 0.83, 0.78, 0.82, 0.65),
    ("glm-4.6", "zai", 0.75, 0.76, 0.80, 0.77, 0.74, 0.78, 0.65),
    ("glm-4.5", "zai", 0.72, 0.73, 0.78, 0.74, 0.72, 0.75, 0.65),
    ("glm-4.5-air", "zai", 0.60, 0.62, 0.68, 0.64, 0.65, 0.66, 0.90),
    ("qwen-3-6-plus", "venice", 0.90, 0.91, 0.85, 0.88, 0.82, 0.87, 0.50),
    ("claude-sonnet-4-6", "venice", 0.94, 0.95, 0.92, 0.93, 0.88, 0.91, 0.45),
    ("zai-org-glm-5", "venice", 0.88, 0.89, 0.84, 0.86, 0.78, 0.84, 0.40),
    ("zai-org-glm-4.7", "venice", 0.78, 0.80, 0.83, 0.81, 0.76, 0.80, 0.55),
    ("zai-org-glm-4.7-flash", "venice", 0.65, 0.66, 0.70, 0.67, 0.68, 0.68, 0.80),
    ("deepseek-v3.2", "venice", 0.82, 0.84, 0.78, 0.82, 0.72, 0.80, 0.70),
    ("grok-4-20-beta", "venice", 0.85, 0.88, 0.82, 0.86, 0.80, 0.84, 0.30),
    ("qwen3-coder-480b-a35b-instruct", "venice", 0.93, 0.82, 0.72, 0.80, 0.60, 0.75, 0.35),
    ("qwen3-5-35b-a3b", "venice", 0.76, 0.78, 0.74, 0.76, 0.72, 0.76, 0.65),
    ("venice-uncensored", "venice", 0.60, 0.62, 0.72, 0.58, 0.85, 0.65, 0.75),
    ("Carnice-Qwen3.6-MoE-35B-A3B-APEX-I-Compact", "local", 0.82, 0.84, 0.77, 0.80, 0.67, 0.80, 0.67),
    ("Qwen3.5-122B-A10B-APEX-I-Compact", "local", 0.85, 0.86, 0.78, 0.82, 0.70, 0.80, 0.35),
)

MODEL_PROFILES = {m[0]: ModelProfile(*m) for m in _MODELS}

_CLASSIFY_RE = re.compile(
    r"(?P<writing>summarize|(?:write|draft|compose|rewrite|edit)\b.*\b(?:documentation|email|proposal|announcement|error|blog)|\breadme\b)"
    r"|(?P<creative>(?:write|compose)\b.*\b(?:poem|funny|creative)|\b(?:design|ideas)\b)"
    r"|(?P<analysis>how\s+many|what\s+percentage|show\s+me\s+the|error\s+rate|test\s+coverage|\b(?:analyze|data|metrics|report|coverage)\b)"
    r"|(?P<reasoning>explain|why\s+does|what\s+causes|how\s+does|is\s+the|\b(?:evaluate|architecture|performance|research|redesign|improve)\b)"
    r"|(?P<code>\b(?:debug|implement|module|fix|bug|script|security|server|test)\b)"
, re.IGNORECASE)
# reordered groups for performance
# experimental change

def classify_message(message: str) -> str:
    m = _CLASSIFY_RE.search(message)
    return m.lastgroup if m else "general"


def select_model(message: str, routing_config: dict, primary_model: str) -> tuple[str, str, str] | None:
    if not routing_config.get("use_model_selector", False):
        return None
    task_type = classify_message(message)
    if task_type == "general":
        return None
    if task_type in ("code", "reasoning"):
        qw, sw = (0.80, 0.20)
    elif task_type == "analysis":
        qw, sw = (0.70, 0.30)
    else:
        qw, sw = (0.70, 0.30)
    attr = "code_quality" if task_type == "code" else task_type
    pool = {m: MODEL_PROFILES[m] for names in routing_config.get("models", {}).values() for m in names if m in MODEL_PROFILES}
    best = max(pool.values(), key=lambda p: qw * getattr(p, attr, p.general) + sw * p.speed, default=None)
    cap = getattr(best, attr, best.general) if best else 0
    if not best or cap < 0.70 or best.name == primary_model:
        return None
    return (best.provider, best.name, task_type)

def smart_select_route(user_message: str, routing_config: dict, primary: dict) -> dict | None:
    _RK = ("api_key", "base_url", "provider", "api_mode", "command", "args", "credential_pool")
    selection = select_model(user_message, routing_config, (primary.get("model") or "").strip())
    if selection is None:
        return None
    selected_provider, selected_model, reason = selection

    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        raw = resolve_runtime_provider(requested=selected_provider)
        runtime = {k: (raw.get(k, "") if k == "base_url" else raw[k]) for k in _RK if k in raw}
    except Exception:
        _def = {"api_mode": "chat_completions", "command": None, "args": [], "credential_pool": None}
        runtime = {k: (primary.get(k) or _def.get(k, "")) for k in _RK}
    runtime["provider"] = selected_provider
    return {
        "model": selected_model,
        "runtime": runtime,
        "label": f"selector → {selected_model} ({selected_provider}) [{reason}]",
        "signature": (selected_model, selected_provider, runtime.get("base_url") or "", runtime.get("api_mode") or "", runtime.get("command"), tuple(runtime.get("args") or [])),
    }


def _run_adversarial_benchmark():
    import importlib.util
    import time

    spec = importlib.util.spec_from_file_location("bench", "autoresearch-routing/benchmark_adversarial.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    examples = mod.LABELED_EXAMPLES

    correct = 0
    total_latency = 0.0
    mismatches = []
    for msg, expected in examples:
        start = time.perf_counter()
        predicted = classify_message(msg)
        total_latency += time.perf_counter() - start
        if predicted == expected:
            correct += 1
        else:
            mismatches.append((msg, predicted, expected))

    n = len(examples)
    print(f"---")
    print(f"pass_count:          {correct}/{n}")
    print(f"latency_ms:     {total_latency/n*1000:.3f}")
    if mismatches:
        print("MISMATCHES:")
        for msg, pred, exp in mismatches:
            print(f"  '{msg}' -> predicted '{pred}', expected '{exp}'")


if __name__ == "__main__":
    _run_adversarial_benchmark()
