"""
Agent cache management for the gateway.

Extracted from gateway/run.py to reduce the God file size.
Manages the LRU cache of AIAgent instances, computes config signatures
for cache invalidation, and enforces cache size/idle-TTL limits.
"""

import hashlib
import json as _json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Cache limits are defined in gateway/run.py (_AGENT_CACHE_MAX_SIZE = 128,
# _AGENT_CACHE_IDLE_TTL_SECS = 3600.0).  Tests monkeypatch those module-level
# values, so each function below lazily imports from gateway.run at call time.

# --- Config keys that bust the cache ---------------------------------------

CACHE_BUSTING_CONFIG_KEYS: Tuple[Tuple[str, str], ...] = (
    ("model", "context_length"),
    ("model", "max_tokens"),
    ("compression", "enabled"),
    ("compression", "threshold"),
    ("compression", "target_ratio"),
    ("compression", "protect_last_n"),
    ("agent", "disabled_toolsets"),
)


def extract_cache_busting_config(user_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Pull values that must bust the cached agent.

    Returns a flat dict keyed by 'section.key'.  Missing config keys and
    non-dict sections yield None values, which still contribute to the
    signature (so 'absent' vs 'present-and-null' differ).

    The live tool registry generation is included too.  MCP reloads and
    dynamic MCP tool-list changes mutate the registry without necessarily
    changing config.yaml.  Cached AIAgent instances freeze their tool
    schemas at construction time, so a registry generation change must
    rebuild the agent before the next turn.
    """
    out: Dict[str, Any] = {}
    cfg = user_config if isinstance(user_config, dict) else {}
    for section, key in CACHE_BUSTING_CONFIG_KEYS:
        section_val = cfg.get(section)
        if isinstance(section_val, dict):
            out[f"{section}.{key}"] = section_val.get(key)
        else:
            out[f"{section}.{key}"] = None
    try:
        from tools.registry import registry

        out["tools.registry_generation"] = getattr(registry, "_generation", None)
    except Exception:
        out["tools.registry_generation"] = None
    return out


def agent_config_signature(
    model: str,
    runtime: dict,
    enabled_toolsets: list,
    ephemeral_prompt: str,
    cache_keys: Optional[Dict[str, Any]] = None,
) -> str:
    """Compute a stable string key from agent config values.

    When this signature changes between messages, the cached AIAgent is
    discarded and rebuilt.  When it stays the same, the cached agent is
    reused — preserving the frozen system prompt and tool schemas for
    prompt cache hits.

    ``cache_keys`` is an optional flat dict of additional config values
    that should invalidate the cache when they change.  Callers pass
    the output of ``extract_cache_busting_config(user_config)`` so
    edits to model.context_length / compression.* in config.yaml are
    picked up on the next gateway message without a manual restart.
    """
    _api_key = str(runtime.get("api_key", "") or "")
    _api_key_fingerprint = hashlib.sha256(_api_key.encode()).hexdigest() if _api_key else ""
    _cache_keys_sorted = sorted((cache_keys or {}).items())

    blob = _json.dumps(
        [
            model,
            _api_key_fingerprint,
            runtime.get("base_url", ""),
            runtime.get("provider", ""),
            runtime.get("api_mode", ""),
            sorted(enabled_toolsets) if enabled_toolsets else [],
            # reasoning_config excluded — it's set per-message on the
            # cached agent and doesn't affect system prompt or tools.
            ephemeral_prompt or "",
            _cache_keys_sorted,
        ],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def init_cached_agent_for_turn(agent: Any, interrupt_depth: int) -> None:
    """Reset per-turn state on a cached agent before a new turn starts.

    Both _last_activity_ts and _last_activity_desc are only reset for
    fresh external turns (depth 0); they are semantically paired —
    desc describes the activity *at* ts, so updating one without the
    other would make get_activity_summary() misleading.
    For interrupt-recursive turns both are preserved so the inactivity
    watchdog can accumulate stuck-turn idle time and fire the 30-min
    timeout (#15654).  The depth-0 reset is still needed: a session
    idle for 29 min would otherwise trip the watchdog before the new
    turn makes its first API call (#9051).
    """
    if interrupt_depth == 0:
        agent._last_activity_ts = time.time()
        agent._last_activity_desc = "starting new turn (cached)"
    agent._api_call_count = 0


def evict_cached_agent(
    agent_cache: Dict[str, Any],
    agent_cache_lock: Optional[threading.Lock],
    session_key: str,
) -> None:
    """Remove a cached agent for a session (called on /new, /model, etc)."""
    if agent_cache_lock:
        with agent_cache_lock:
            agent_cache.pop(session_key, None)


def release_evicted_agent_soft(
    agent: Any,
    cleanup_fn: Callable[[Any], None],
) -> None:
    """Soft cleanup for cache-evicted agents — preserves session tool state.

    Called from enforce_agent_cache_cap and sweep_idle_cached_agents.
    Distinct from a full cleanup (which would tear down terminal sandbox,
    browser daemon, and tracked bg processes) because a cache-evicted
    session may resume at any time.
    """
    if agent is None:
        return
    try:
        if hasattr(agent, "release_clients"):
            agent.release_clients()
        else:
            cleanup_fn(agent)
    except Exception:
        pass


def enforce_agent_cache_cap(
    agent_cache: Dict[str, Any],
    agent_cache_lock: Optional[threading.Lock],
    running_agents: Dict[str, Any],
    agent_pending_sentinel: object,
    release_fn: Callable[[Any], None],
) -> None:
    """Evict oldest cached agents when cache exceeds the cap.

    Must be called with _agent_cache_lock held.  Resource cleanup
    (memory provider shutdown, tool resource close) is scheduled
    on a daemon thread so the caller doesn't block on slow teardown
    while holding the cache lock.

    Agents currently in _running_agents are SKIPPED — their clients,
    terminal sandboxes, background processes, and child subagents
    are all in active use by the running turn.  Evicting them would
    tear down those resources mid-turn and crash the request.  If
    every candidate in the LRU order is active, we simply leave the
    cache over the cap; it will be re-checked on the next insert.
    """
    from gateway.run import _AGENT_CACHE_MAX_SIZE

    if agent_cache is None:
        return
    if not hasattr(agent_cache, "move_to_end"):
        return

    running_ids = {
        id(a)
        for a in running_agents.values()
        if a is not None and a is not agent_pending_sentinel
    }

    excess = max(0, len(agent_cache) - _AGENT_CACHE_MAX_SIZE)
    evict_plan: List[Tuple[str, Any]] = []
    if excess > 0:
        ordered_keys = list(agent_cache.keys())
        for key in ordered_keys[:excess]:
            entry = agent_cache.get(key)
            agent = entry[0] if isinstance(entry, tuple) and entry else None
            if agent is not None and id(agent) in running_ids:
                continue
            evict_plan.append((key, agent))

    for key, _ in evict_plan:
        agent_cache.pop(key, None)

    remaining_over_cap = len(agent_cache) - _AGENT_CACHE_MAX_SIZE
    if remaining_over_cap > 0:
        logger.warning(
            "Agent cache over cap (%d > %d); %d excess slot(s) held by "
            "mid-turn agents — will re-check on next insert.",
            len(agent_cache),
            _AGENT_CACHE_MAX_SIZE,
            remaining_over_cap,
        )

    for key, agent in evict_plan:
        logger.info(
            "Agent cache at cap; evicting LRU session=%s (cache_size=%d)",
            key,
            len(agent_cache),
        )
        if agent is not None:
            threading.Thread(
                target=release_fn,
                args=(agent,),
                daemon=True,
                name=f"agent-cache-evict-{key[:24]}",
            ).start()


def sweep_idle_cached_agents(
    agent_cache: Dict[str, Any],
    agent_cache_lock: Optional[threading.Lock],
    running_agents: Dict[str, Any],
    agent_pending_sentinel: object,
    release_fn: Callable[[Any], None],
) -> int:
    """Evict cached agents whose AIAgent has been idle past the idle TTL.

    Safe to call without holding the cache lock — acquires it internally.
    Returns the number of entries evicted.  Resource cleanup is scheduled
    on daemon threads.

    Agents currently in _running_agents are SKIPPED for the same reason
    as enforce_agent_cache_cap: tearing down an active turn's clients
    mid-flight would crash the request.
    """
    from gateway.run import _AGENT_CACHE_IDLE_TTL_SECS

    if agent_cache is None or agent_cache_lock is None:
        return 0
    now = time.time()
    to_evict: List[Tuple[str, Any]] = []
    running_ids = {
        id(a)
        for a in running_agents.values()
        if a is not None and a is not agent_pending_sentinel
    }
    with agent_cache_lock:
        for key, entry in list(agent_cache.items()):
            agent = entry[0] if isinstance(entry, tuple) and entry else None
            if agent is None:
                continue
            if id(agent) in running_ids:
                continue
            last_activity = getattr(agent, "_last_activity_ts", None)
            if last_activity is None:
                continue
            if (now - last_activity) > _AGENT_CACHE_IDLE_TTL_SECS:
                to_evict.append((key, agent))
        for key, _ in to_evict:
            agent_cache.pop(key, None)
    for key, agent in to_evict:
        logger.info(
            "Agent cache idle-TTL evict: session=%s (idle=%.0fs)",
            key,
            now - getattr(agent, "_last_activity_ts", now),
        )
        threading.Thread(
            target=release_fn,
            args=(agent,),
            daemon=True,
            name=f"agent-cache-idle-{key[:24]}",
        ).start()
    return len(to_evict)


def release_evicted_agent_soft(
    agent: Any,
    cleanup_fn: Optional[Callable[[Any], None]] = None,
) -> None:
    """Soft cleanup for cache-evicted agents — preserves session tool state.

    Called from enforce_agent_cache_cap and sweep_idle_cached_agents.
    Distinct from full cleanup because a cache-evicted session may resume
    at any time — its terminal sandbox, browser daemon, and tracked bg
    processes must outlive the Python AIAgent instance so the next agent
    built for the same task_id inherits them.
    """
    if agent is None:
        return
    try:
        if hasattr(agent, "release_clients"):
            agent.release_clients()
        else:
            # Older agent instance (shouldn't happen in practice) —
            # fall back to the legacy full-close path.
            if cleanup_fn:
                cleanup_fn(agent)
    except Exception:
        pass
