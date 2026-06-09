"""
Gateway configuration resolution utilities.

This module contains functions for loading and resolving gateway configuration
values from various sources (config files, environment variables, defaults).
"""

import os
from typing import Any, Optional


def _float_env(name: str, default: float) -> float:
    """Read an env var as float, falling back to ``default`` on typos/empty.

    A misconfigured env var (e.g. ``HERMES_AGENT_TIMEOUT=abc``) must not
    crash the gateway or an agent turn.  Unset/empty also falls back.
    """
    raw = os.environ.get(name, "")
    if not raw or not raw.strip():
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _resolve_hermes_bin(default: str = "hermes") -> str:
    """Resolve the HERMES_BIN env var to a binary path.

    Falls back to the provided default (typically ``hermes`` or ``hermes-agent``)
    when unset or empty.  This is used by the gateway's shell command execution
    and subprocess spawning logic.
    """
    raw = os.environ.get("HERMES_BIN", "")
    if not raw or not raw.strip():
        return default
    return raw.strip()


def _auto_continue_freshness_window(default_seconds: float = 3600.0) -> float:
    """Return the configured auto-continue freshness window in seconds.

    Reads ``HERMES_AUTO_CONTINUE_FRESHNESS`` (bridged from
    ``config.yaml`` ``agent.gateway_auto_continue_freshness`` at gateway
    startup).  Falls back to the provided default when unset or malformed.
    Non-positive values disable the freshness gate.
    """
    raw = os.environ.get("HERMES_AUTO_CONTINUE_FRESHNESS")
    if raw is None or raw == "":
        return default_seconds
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default_seconds


def _gateway_agent_timeout(default_seconds: float = 1800.0) -> float:
    """Return the configured gateway agent timeout in seconds.

    Reads ``HERMES_AGENT_TIMEOUT`` (bridged from
    ``config.yaml`` ``agent.gateway_timeout`` at gateway startup).
    Falls back to the provided default when unset or malformed.
    """
    return _float_env("HERMES_AGENT_TIMEOUT", default_seconds)


def _gateway_model_provider() -> Optional[str]:
    """Return the configured model provider from environment.

    Reads ``HERMES_PROVIDER`` for explicit provider override.
    Returns None when unset, allowing config.yaml to take precedence.
    """
    raw = os.environ.get("HERMES_PROVIDER", "")
    if not raw or not raw.strip():
        return None
    return raw.strip()


def _gateway_fallback_model() -> Optional[str]:
    """Return the configured fallback model from environment.

    Reads ``HERMES_FALLBACK_MODEL`` for fallback model override.
    Returns None when unset.
    """
    raw = os.environ.get("HERMES_FALLBACK_MODEL", "")
    if not raw or not raw.strip():
        return None
    return raw.strip()


def _get_gateway_platform_value(platform: Any) -> str:
    """Return a normalized gateway platform value for enums or raw strings.

    This is a convenience wrapper for gateway.utils.gateway_helpers._gateway_platform_value
    to avoid circular imports in config resolution code.
    """
    from gateway.utils.gateway_helpers import _gateway_platform_value
    return _gateway_platform_value(platform)


def _resolve_gateway_model(config: dict | None = None) -> str:
    """Read model from config.yaml — single source of truth.

    Without this, temporary AIAgent instances (e.g. /compress) fall
    back to the hardcoded default which fails when the active provider is
    openai-codex.
    """
    from gateway.adapter_factory import _load_gateway_config

    cfg = config if config is not None else _load_gateway_config()
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, str):
        return model_cfg
    elif isinstance(model_cfg, dict):
        return model_cfg.get("default") or model_cfg.get("model") or ""
    return ""
