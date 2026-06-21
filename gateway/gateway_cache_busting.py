"""Agent cache-busting config extraction — migrated from GatewayRunner classmethods.

Round 15 of gateway decomposition. These were @classmethod carrying class-level
state: _CACHE_BUSTING_CONFIG_KEYS + _HONCHO_CACHE_BUSTING_KEYS (immutable tuples)
and _HONCHO_CACHE_BUSTING_MEMO (a mutable memo dict). Migrated here as module
globals. Behaviour is identical — the memo now lives at module scope instead
of class scope; same lifetime in practice (module is loaded once for the
gateway daemon's whole run).

Depends on cfg_get from hermes_cli.config.
"""

from typing import Any, Dict

from hermes_cli.config import cfg_get

import logging
logger = logging.getLogger("gateway.run")


# Sections/keys in config.yaml whose change must invalidate the cached AIAgent.
_CACHE_BUSTING_CONFIG_KEYS: tuple = (
    ("model", "context_length"),
    ("model", "max_tokens"),
    ("compression", "enabled"),
    ("compression", "threshold"),
    ("compression", "target_ratio"),
    ("compression", "protect_last_n"),
    ("agent", "disabled_toolsets"),
    ("memory", "provider"),
)

_HONCHO_CACHE_BUSTING_KEYS = (
    "honcho.peer_name",
    "honcho.ai_peer",
    "honcho.pin_peer_name",
    "honcho.runtime_peer_prefix",
    "honcho.user_peer_aliases",
)

_HONCHO_CACHE_BUSTING_MEMO: dict[tuple[str, int | None], dict[str, Any]] = {}


def _empty_honcho_cache_busting_config() -> dict[str, Any]:
    return {key: None for key in _HONCHO_CACHE_BUSTING_KEYS}


def _extract_honcho_cache_busting_config() -> dict[str, Any]:
    """Extract Honcho identity keys, memoized by honcho.json mtime."""
    try:
        from plugins.memory.honcho.client import HonchoClientConfig, resolve_config_path

        path = resolve_config_path()
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            mtime_ns = None
        memo_key = (str(path), mtime_ns)
        cached = _HONCHO_CACHE_BUSTING_MEMO.get(memo_key)
        if cached is not None:
            return dict(cached)

        hcfg = HonchoClientConfig.from_global_config(config_path=path)
        aliases = hcfg.user_peer_aliases or {}
        values = {
            "honcho.peer_name": hcfg.peer_name,
            "honcho.ai_peer": hcfg.ai_peer,
            "honcho.pin_peer_name": bool(hcfg.pin_peer_name),
            "honcho.runtime_peer_prefix": hcfg.runtime_peer_prefix or "",
            "honcho.user_peer_aliases": sorted(aliases.items()) if isinstance(aliases, dict) else [],
        }
        _HONCHO_CACHE_BUSTING_MEMO[memo_key] = values
        return dict(values)
    except Exception:
        return _empty_honcho_cache_busting_config()


def _extract_cache_busting_config(user_config: dict | None) -> dict:
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
    provider = cfg_get(cfg, "memory", "provider")
    if isinstance(provider, str) and provider.lower() == "honcho":
        out.update(_extract_honcho_cache_busting_config())
    else:
        out.update(_empty_honcho_cache_busting_config())

    return out
