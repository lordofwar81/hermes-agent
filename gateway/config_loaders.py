"""
Gateway configuration loader functions.

This module contains helper functions for loading configuration values from
the gateway config, environment variables, and session overrides.

These functions are used by GatewayRunner to load ephemeral configuration
that is injected at runtime without persistence.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from gateway.utils.config_resolvers import _float_env, _resolve_hermes_bin

logger = logging.getLogger(__name__)


def load_prefill_messages(cfg) -> List[Dict[str, Any]]:
    """Load prefill messages from config.

    Returns a list of message dictionaries that should be prepended
    to every agent session.
    """
    prefill = []
    for msg in cfg.get("prefill_messages", []):
        if isinstance(msg, dict) and msg.get("enabled", True):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                prefill.append({"role": role, "content": content})
    return prefill


def load_ephemeral_system_prompt(cfg) -> Optional[str]:
    """Load ephemeral system prompt from config.

    Returns the system prompt string if configured, None otherwise.
    """
    prompt = cfg.get("ephemeral_system_prompt", "")
    return prompt if prompt else None


def load_reasoning_config(cfg) -> Dict[str, Any]:
    """Load reasoning configuration from config.

    Returns a dict with reasoning mode settings.
    """
    reasoning_cfg = cfg.get("reasoning", {})
    return {
        "enabled": reasoning_cfg.get("enabled", False),
        "max_tokens": reasoning_cfg.get("max_tokens", 8192),
        "budget_seconds": reasoning_cfg.get("budget_seconds", 30.0),
    }


def parse_reasoning_command_args(args: str) -> Dict[str, Any]:
    """Parse reasoning command arguments.

    Args:
        args: Command arguments string

    Returns:
        Dict with parsed settings
    """
    parts = args.split()
    result = {}
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            if key == "max_tokens":
                result["max_tokens"] = int(value)
            elif key == "budget":
                result["budget_seconds"] = float(value)
    return result


def resolve_session_reasoning_config(
    session_overrides: Dict[str, Dict],
    session_key: str,
    default_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve reasoning config for a session.

    Args:
        session_overrides: Dict of session-specific overrides
        session_key: Current session key
        default_config: Default reasoning config

    Returns:
        Resolved reasoning config for the session
    """
    if session_key in session_overrides:
        override = session_overrides[session_key].get("reasoning")
        if override is not None:
            return override
    return default_config


def set_session_reasoning_override(
    session_overrides: Dict[str, Dict],
    session_key: str,
    enabled: bool,
) -> None:
    """Set reasoning override for a session.

    Args:
        session_overrides: Dict of session-specific overrides
        session_key: Current session key
        enabled: Whether reasoning is enabled
    """
    if session_key not in session_overrides:
        session_overrides[session_key] = {}
    session_overrides[session_key]["reasoning"] = enabled


def load_service_tier(cfg) -> str:
    """Load service tier from config.

    Returns the service tier string (e.g., 'free', 'paid').
    """
    return cfg.get("service_tier", "free")


def load_show_reasoning(cfg) -> bool:
    """Load show_reasoning setting from config.

    Returns True if reasoning output should be displayed.
    """
    return cfg.get("show_reasoning", False)


def load_busy_input_mode(cfg) -> str:
    """Load busy input mode from config.

    Returns the mode: 'interrupt', 'queue', or 'ignore'.
    """
    return cfg.get("busy_input_mode", "interrupt")


def load_busy_text_mode(cfg) -> str:
    """Load busy text mode from config.

    Returns the mode: 'interrupt', 'queue', or 'ignore'.
    """
    return cfg.get("busy_text_mode", "interrupt")


def load_restart_drain_timeout(cfg) -> float:
    """Load restart drain timeout from config.

    Returns the timeout in seconds for draining active sessions
    before restart.
    """
    timeout = cfg.get("restart_drain_timeout", 1800.0)
    if isinstance(timeout, (int, float)):
        return float(timeout)
    return 1800.0


def load_background_notifications_mode(cfg) -> str:
    """Load background notifications mode from config.

    Returns the mode for handling notifications during
    background task execution.
    """
    return cfg.get("background_notifications_mode", "silent")


def load_provider_routing(cfg) -> Dict[str, str]:
    """Load provider routing configuration.

    Returns a dict mapping task types to provider names.
    """
    routing = cfg.get("provider_routing", {})
    if isinstance(routing, dict):
        return routing
    return {}


def load_fallback_model(cfg) -> Optional[str]:
    """Load fallback model from config.

    Returns the fallback model identifier, or None if not configured.
    """
    fallback = cfg.get("fallback_model")
    return fallback if fallback else None
