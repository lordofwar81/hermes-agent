"""
Agent configuration utilities for GatewayRunner.

Extracted from gateway/run.py to reduce the God file size.
Provides functions for resolving agent configuration per turn.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gateway.run import GatewayRunner

logger = logging.getLogger(__name__)


def resolve_turn_agent_config(
    runner,  # GatewayRunner instance
    user_message: str,
    model: str,
    runtime_kwargs: dict,
) -> dict:
    """Build the effective model/runtime config for a single turn.

    Always uses the session's primary model/provider.  If `/fast` is
    enabled and the model supports Priority Processing / Anthropic fast
    mode, attach `request_overrides` so the API call is marked
    accordingly.
    """
    from hermes_cli.models import resolve_fast_mode_overrides

    runtime = {
        "api_key": runtime_kwargs.get("api_key"),
        "base_url": runtime_kwargs.get("base_url"),
        "provider": runtime_kwargs.get("provider"),
        "api_mode": runtime_kwargs.get("api_mode"),
        "command": runtime_kwargs.get("command"),
        "args": list(runtime_kwargs.get("args") or []),
        "credential_pool": runtime_kwargs.get("credential_pool"),
    }
    route = {
        "model": model,
        "runtime": runtime,
        "signature": (
            model,
            runtime["provider"],
            runtime["base_url"],
            runtime["api_mode"],
            runtime["command"],
            tuple(runtime["args"]),
        ),
    }

    service_tier = getattr(runner, "_service_tier", None)
    if not service_tier:
        route["request_overrides"] = {}
        return route

    try:
        overrides = resolve_fast_mode_overrides(route["model"])
    except Exception:
        overrides = None
    route["request_overrides"] = overrides or {}
    return route


def extract_cache_busting_config(user_config: dict | None) -> dict:
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
    from typing import Any, Dict

    # Cache-busting config keys from GatewayRunner
    _CACHE_BUSTING_CONFIG_KEYS = [
        ("agents", "timeout"),
        ("agents", "max_depth"),
        ("agents", "max_consecutive_auto_tool_calls"),
        ("agents", "max_total_tool_calls"),
        ("agents", "max_turns"),
        ("agents", "parallel_tool_calls"),
        ("agents", "reasoning_effort"),
        ("reasoning", "effort"),
    ]

    out: Dict[str, Any] = {}
    cfg = user_config if isinstance(user_config, dict) else {}
    for section, key in _CACHE_BUSTING_CONFIG_KEYS:
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

    # Honcho identity-mapping keys live in honcho.json, not user_config.
    # Only read that file when Honcho is the active memory provider.
    provider = cfg.get("memory", {}).get("provider") if isinstance(cfg.get("memory"), dict) else None
    if isinstance(provider, str) and provider.lower() == "honcho":
        out.update(_extract_honcho_cache_busting_config())
    else:
        out.update(_empty_honcho_cache_busting_config())

    return out


def _empty_honcho_cache_busting_config() -> dict:
    """Return empty Honcho cache-busting config."""
    return {
        "honcho.peer_name": None,
        "honcho.ai_peer": None,
        "honcho.pin_peer_name": None,
        "honcho.runtime_peer_prefix": None,
        "honcho.user_peer_aliases": None,
    }


def _extract_honcho_cache_busting_config() -> dict:
    """Extract Honcho identity keys for cache busting."""
    try:
        from plugins.memory.honcho.client import HonchoClientConfig, resolve_config_path

        path = resolve_config_path()
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            mtime_ns = None

        hcfg = HonchoClientConfig.from_global_config(config_path=path)
        aliases = hcfg.user_peer_aliases or {}
        return {
            "honcho.peer_name": hcfg.peer_name,
            "honcho.ai_peer": hcfg.ai_peer,
            "honcho.pin_peer_name": bool(hcfg.pin_peer_name),
            "honcho.runtime_peer_prefix": hcfg.runtime_peer_prefix or "",
            "honcho.user_peer_aliases": sorted(aliases.items()) if isinstance(aliases, dict) else [],
        }
    except Exception:
        return _empty_honcho_cache_busting_config()
