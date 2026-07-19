"""Gateway reasoning & config loading helpers.

Extracted from gateway/run.py to reduce the God file size.
Provides functions for loading reasoning config, service tier,
prefill messages, ephemeral system prompts, busy modes, and
other per-session configuration values.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_cli.config import cfg_get
from hermes_cli.fallback_config import get_fallback_chain
from utils import is_truthy_value

logger = logging.getLogger(__name__)


def load_prefill_messages() -> List[Dict[str, Any]]:
    """Load ephemeral prefill messages from config or env var.

    Checks HERMES_PREFILL_MESSAGES_FILE env var first, then falls back to
    the prefill_messages_file key in ~/.hermes/config.yaml.
    Relative paths are resolved from get_hermes_home().
    """
    from gateway.run import _load_gateway_runtime_config

    file_path = os.getenv("HERMES_PREFILL_MESSAGES_FILE", "")
    if not file_path:
        cfg = _load_gateway_runtime_config()
        file_path = str(cfg.get("prefill_messages_file", "") or "")
    if not file_path:
        return []
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        from gateway.run import _hermes_home
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
    from gateway.run import _load_gateway_runtime_config

    prompt = os.getenv("HERMES_EPHEMERAL_SYSTEM_PROMPT", "")
    if prompt:
        return prompt
    cfg = _load_gateway_runtime_config()
    return str(cfg_get(cfg, "agent", "system_prompt", default="") or "").strip()


def load_reasoning_config() -> Optional[dict]:
    """Load reasoning effort from config.yaml.

    Reads agent.reasoning_effort from config.yaml. Valid: "none",
    "minimal", "low", "medium", "high", "xhigh". Returns None to use
    default (medium).
    """
    from gateway.run import _load_gateway_runtime_config
    from hermes_constants import parse_reasoning_effort

    cfg = _load_gateway_runtime_config()
    effort = str(cfg_get(cfg, "agent", "reasoning_effort", default="") or "").strip()
    result = parse_reasoning_effort(effort)
    if effort and effort.strip() and result is None:
        logger.warning("Unknown reasoning_effort '%s', using default (medium)", effort)
    return result


def parse_reasoning_command_args(raw_args: str) -> tuple[str, bool]:
    """Parse `/reasoning` args into `(value, persist_global)`."""
    import shlex

    text = str(raw_args or "").strip().replace("\u2014", "--")
    if not text:
        return "", False
    try:
        tokens = shlex.split(text)
    except ValueError:
        tokens = text.split()

    persist_global = False
    value_tokens = []
    for token in tokens:
        if token == "--global":
            persist_global = True
        else:
            value_tokens.append(token)
    return " ".join(value_tokens).strip().lower(), persist_global


def resolve_session_reasoning_config(
    session_reasoning_overrides: dict,
    session_key: Optional[str],
    fallback: Optional[dict],
) -> Optional[dict]:
    """Resolve reasoning effort for a session, honoring session overrides."""
    if session_key and session_key in session_reasoning_overrides:
        return session_reasoning_overrides[session_key]
    return fallback


def set_session_reasoning_override(
    session_reasoning_overrides: dict,
    session_key: str,
    reasoning_config: Optional[dict],
) -> None:
    """Set or clear the session-scoped reasoning override."""
    if not session_key:
        return
    if reasoning_config is None:
        session_reasoning_overrides.pop(session_key, None)
    else:
        session_reasoning_overrides[session_key] = dict(reasoning_config)


def load_service_tier() -> Optional[str]:
    """Load Priority Processing setting from config.yaml."""
    from gateway.run import _load_gateway_runtime_config

    cfg = _load_gateway_runtime_config()
    raw = str(cfg_get(cfg, "agent", "service_tier", default="") or "").strip()
    value = raw.lower()
    if not value or value in {"normal", "default", "standard", "off", "none"}:
        return None
    if value in {"fast", "priority", "on"}:
        return "priority"
    logger.warning("Unknown service_tier '%s', ignoring", raw)
    return None


def load_show_reasoning() -> bool:
    """Load show_reasoning toggle from config.yaml display section."""
    from gateway.run import _load_gateway_runtime_config
    from hermes_cli.config import cfg_get as _cfg_get

    cfg = _load_gateway_runtime_config()
    return is_truthy_value(
        _cfg_get(cfg, "display", "show_reasoning"),
        default=False,
    )


def load_busy_input_mode() -> str:
    """Load gateway drain-time busy-input behavior from config/env."""
    from gateway.run import _load_gateway_runtime_config

    mode = os.getenv("HERMES_GATEWAY_BUSY_INPUT_MODE", "").strip().lower()
    if not mode:
        cfg = _load_gateway_runtime_config()
        mode = str(cfg_get(cfg, "display", "busy_input_mode", default="") or "").strip().lower()
    if mode == "queue":
        return "queue"
    if mode == "steer":
        return "steer"
    return "interrupt"


def load_busy_text_mode() -> str:
    """Load normal busy TEXT follow-up behavior from config/env."""
    from gateway.run import _load_gateway_runtime_config

    mode = os.getenv("HERMES_GATEWAY_BUSY_TEXT_MODE", "").strip().lower()
    if not mode:
        cfg = _load_gateway_runtime_config()
        mode = str(cfg_get(cfg, "display", "busy_text_mode", default="") or "").strip().lower()
    if mode == "interrupt":
        return "interrupt"
    return "queue"


def load_restart_drain_timeout() -> float:
    """Load graceful gateway restart/stop drain timeout in seconds."""
    from gateway.restart import (
        DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
        parse_restart_drain_timeout,
    )

    raw = os.getenv("HERMES_RESTART_DRAIN_TIMEOUT", "").strip()
    if not raw:
        from gateway.run import _load_gateway_runtime_config

        cfg = _load_gateway_runtime_config()
        raw = str(cfg_get(cfg, "agent", "restart_drain_timeout", default="") or "").strip()
    value = parse_restart_drain_timeout(raw)
    if raw and value == DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT:
        try:
            float(raw)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid restart_drain_timeout '%s', using default %.0fs",
                raw,
                DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
            )
    return value


def load_background_notifications_mode() -> str:
    """Load background process notification mode from config or env var."""
    from gateway.run import _load_gateway_runtime_config

    mode = os.getenv("HERMES_BACKGROUND_NOTIFICATIONS", "")
    if not mode:
        cfg = _load_gateway_runtime_config()
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


def load_provider_routing() -> dict:
    """Load OpenRouter provider routing preferences from config.yaml."""
    from gateway.run import _hermes_home

    try:
        import yaml as _y

        cfg_path = _hermes_home / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as _f:
                cfg = _y.safe_load(_f) or {}
            return cfg.get("provider_routing", {}) or {}
    except Exception:
        pass
    return {}


def load_fallback_model() -> Optional[list]:
    """Load fallback provider chain from config.yaml."""
    from gateway.run import _hermes_home

    try:
        import yaml as _y

        cfg_path = _hermes_home / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as _f:
                cfg = _y.safe_load(_f) or {}
            fb = get_fallback_chain(cfg)
            if fb:
                return fb
    except Exception:
        pass
    return None
