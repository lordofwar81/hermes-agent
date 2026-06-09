"""
Configuration utility functions for GatewayRunner.

Extracted from gateway/run.py to reduce the God file size.
Provides functions for loading configuration values.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_hermes_home = Path.home() / ".hermes"


def load_gateway_runtime_config() -> dict:
    """Load and parse ~/.hermes/config.yaml, returning {} on any error."""
    from hermes_cli.config import load_config
    try:
        return load_config()
    except Exception:
        return {}


def load_prefill_messages() -> List[Dict[str, Any]]:
    """Load ephemeral prefill messages from config or env var.
    
    Checks HERMES_PREFILL_MESSAGES_FILE env var first, then falls back to
    the prefill_messages_file key in ~/.hermes/config.yaml.
    Relative paths are resolved from ~/.hermes/.
    """
    file_path = os.getenv("HERMES_PREFILL_MESSAGES_FILE", "")
    if not file_path:
        cfg = load_gateway_runtime_config()
        file_path = str(cfg.get("prefill_messages_file", "") or "")
    if not file_path:
        return []
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = _hermes_home / path
    if not path.exists():
        logger.warning("Prefill messages file not found: %s", path)
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning("Prefill messages file must contain a JSON array: %s", path)
            return []
        return data
    except Exception as e:
        logger.warning("Failed to load prefill messages from %s: %s", path, e)
        return []


def load_ephemeral_system_prompt() -> str:
    """Load ephemeral system prompt from config or env var.

    Checks HERMES_EPHEMERAL_SYSTEM_PROMPT env var first, then falls back to
    agent.system_prompt in ~/.hermes/config.yaml.
    """
    prompt = os.getenv("HERMES_EPHEMERAL_SYSTEM_PROMPT", "")
    if prompt:
        return prompt
    cfg = load_gateway_runtime_config()
    return str(cfg_get(cfg, "agent", "system_prompt", default="") or "").strip()


def cfg_get(cfg: dict, *keys, default=None):
    """Get a nested config value safely."""
    for key in keys:
        if isinstance(cfg, dict):
            cfg = cfg.get(key)
        else:
            return default
    return cfg if cfg is not None else default


def load_background_notifications_mode() -> str:
    """Load background process notification mode from config or env var.

    Modes:
      - ``all``    — push running-output updates *and* the final message (default)
      - ``result`` — only the final completion message (regardless of exit code)
      - ``error``  — only the final message when exit code is non-zero
      - ``off``    — no watcher messages at all
    """
    mode = os.getenv("HERMES_BACKGROUND_NOTIFICATIONS", "")
    if not mode:
        cfg = load_gateway_runtime_config()
        raw = cfg_get(cfg, "display", "background_process_notifications")
        if raw is False:
            mode = "off"
        elif raw not in {None, ""}:
            mode = str(raw)
    mode = (mode or "all").strip().lower()
    valid = {"all", "result", "error", "off"}
    if mode not in valid:
        logger.warning(
            "Unknown background_process_notifications '%s', defaulting to 'all'",
            mode,
        )
        return "all"
    return mode


def load_restart_drain_timeout(
    parse_restart_drain_timeout,
    default_gateway_restart_drain_timeout: float,
) -> float:
    """Load graceful gateway restart/stop drain timeout in seconds.

    Args:
        parse_restart_drain_timeout: Function to parse timeout string to float
        default_gateway_restart_drain_timeout: Default timeout in seconds

    Returns:
        The drain timeout in seconds
    """
    raw = os.getenv("HERMES_RESTART_DRAIN_TIMEOUT", "").strip()
    if not raw:
        cfg = load_gateway_runtime_config()
        raw = str(cfg_get(cfg, "agent", "restart_drain_timeout", default="") or "").strip()
    value = parse_restart_drain_timeout(raw)
    if raw and value == default_gateway_restart_drain_timeout:
        try:
            float(raw)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid restart_drain_timeout '%s', using default %.0fs",
                raw,
                default_gateway_restart_drain_timeout,
            )
    return value


def resolve_session_reasoning_config(
    runner,  # GatewayRunner instance
    *,
    source=None,
    session_key=None,
) -> dict | None:
    """Resolve reasoning effort for a session, honoring session overrides."""
    resolved_session_key = session_key
    if not resolved_session_key and source is not None:
        try:
            resolved_session_key = runner._session_key_for_source(source)
        except Exception:
            resolved_session_key = None

    overrides = getattr(runner, "_session_reasoning_overrides", {}) or {}
    if resolved_session_key and resolved_session_key in overrides:
        return overrides[resolved_session_key]
    return runner._load_reasoning_config()


def set_session_reasoning_override(
    runner,  # GatewayRunner instance
    session_key: str,
    reasoning_config: dict | None,
) -> None:
    """Set or clear the session-scoped reasoning override."""
    if not session_key:
        return
    if not hasattr(runner, "_session_reasoning_overrides"):
        runner._session_reasoning_overrides = {}
    if reasoning_config is None:
        runner._session_reasoning_overrides.pop(session_key, None)
    else:
        runner._session_reasoning_overrides[session_key] = dict(reasoning_config)


def load_busy_input_mode(
    load_gateway_runtime_config_fn,
    cfg_get_fn,
) -> str:
    """Load gateway drain-time busy-input behavior from config/env."""
    mode = os.getenv("HERMES_GATEWAY_BUSY_INPUT_MODE", "").strip().lower()
    if not mode:
        cfg = load_gateway_runtime_config_fn()
        mode = str(cfg_get_fn(cfg, "display", "busy_input_mode", default="") or "").strip().lower()
    if mode == "queue":
        return "queue"
    if mode == "steer":
        return "steer"
    return "interrupt"


def load_busy_text_mode(
    load_gateway_runtime_config_fn,
    cfg_get_fn,
) -> str:
    """Load normal busy TEXT follow-up behavior from config/env."""
    mode = os.getenv("HERMES_GATEWAY_BUSY_TEXT_MODE", "").strip().lower()
    if not mode:
        cfg = load_gateway_runtime_config_fn()
        mode = str(cfg_get_fn(cfg, "display", "busy_text_mode", default="") or "").strip().lower()
    if mode == "interrupt":
        return "interrupt"
    return "queue"


def load_provider_routing() -> dict:
    """Load OpenRouter provider routing preferences from config.yaml."""
    try:
        import yaml

        cfg_path = _hermes_home / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            return cfg.get("provider_routing", {}) or {}
    except Exception:
        pass
    return {}
