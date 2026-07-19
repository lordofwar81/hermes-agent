"""Config-loader functions extracted from gateway/run.py GatewayRunner.

Round 6 of gateway decomposition — first god-class extraction. These were
defined as @staticmethod / no-self methods inside GatewayRunner but touch no
instance state: they read HermesHome config/env vars and return typed values.
Moved here as plain module functions.

Names kept identical to originals so call sites change minimally
(self._load_X() -> _load_X()). The one exception is _load_voice_modes,
which took self only to read self._VOICE_MODE_PATH; it now takes that path
as an explicit argument.

Deps on gateway.run internals (_hermes_home, _load_gateway_runtime_config)
are imported lazily inside function bodies to avoid a circular import at
module load time (gateway.run imports this module at its top level).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_cli.config import cfg_get
from hermes_cli.fallback_config import get_fallback_chain
from gateway.restart import (
    DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
    parse_restart_drain_timeout,
)
from utils import is_truthy_value

import logging

logger = logging.getLogger("gateway.run")


def _runtime() -> dict:
    """Lazy accessor for gateway.run._load_gateway_runtime_config()."""
    from gateway.run import _load_gateway_runtime_config
    return _load_gateway_runtime_config()


def _home() -> Path:
    """Lazy accessor for gateway.run._hermes_home."""
    from gateway.run import _hermes_home
    return _hermes_home


def _load_prefill_messages() -> List[Dict[str, Any]]:
    """Load ephemeral prefill messages from config or env var.

    Checks HERMES_PREFILL_MESSAGES_FILE env var first, then falls back to
    the top-level prefill_messages_file key in ~/.hermes/config.yaml.
    agent.prefill_messages_file is accepted as a legacy fallback.
    Relative paths are resolved from ~/.hermes/.
    """
    file_path = os.getenv("HERMES_PREFILL_MESSAGES_FILE", "")
    if not file_path:
        cfg = _runtime()
        file_path = str(cfg.get("prefill_messages_file", "") or "")
        if not file_path:
            file_path = str(cfg_get(cfg, "agent", "prefill_messages_file", default="") or "")
    if not file_path:
        return []
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = _home() / path
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


def _load_ephemeral_system_prompt() -> str:
    """Load ephemeral system prompt from config or env var.

    Checks HERMES_EPHEMERAL_SYSTEM_PROMPT env var first, then falls back to
    agent.system_prompt in ~/.hermes/config.yaml.
    """
    prompt = os.getenv("HERMES_EPHEMERAL_SYSTEM_PROMPT", "")
    if prompt:
        return prompt
    cfg = _runtime()
    return str(cfg_get(cfg, "agent", "system_prompt", default="") or "").strip()


def _load_reasoning_config() -> dict | None:
    """Load reasoning effort from config.yaml.

    Reads agent.reasoning_effort from config.yaml. Valid: "none",
    "minimal", "low", "medium", "high", "xhigh". Returns None to use
    default (medium).
    """
    from hermes_constants import parse_reasoning_effort
    cfg = _runtime()
    effort = str(cfg_get(cfg, "agent", "reasoning_effort", default="") or "").strip()
    result = parse_reasoning_effort(effort)
    if effort and effort.strip() and result is None:
        logger.warning("Unknown reasoning_effort '%s', using default (medium)", effort)
    return result


def _load_service_tier() -> str | None:
    """Load Priority Processing setting from config.yaml.

    Reads agent.service_tier from config.yaml. Accepted values mirror the CLI:
    "fast"/"priority"/"on" => "priority", while "normal"/"off" disables it.
    Returns None when unset or unsupported.
    """
    cfg = _runtime()
    raw = str(cfg_get(cfg, "agent", "service_tier", default="") or "").strip()

    value = raw.lower()
    if not value or value in {"normal", "default", "standard", "off", "none"}:
        return None
    if value in {"fast", "priority", "on"}:
        return "priority"
    logger.warning("Unknown service_tier '%s', ignoring", raw)
    return None


def _load_show_reasoning() -> bool:
    """Load show_reasoning toggle from config.yaml display section."""
    cfg = _runtime()
    return is_truthy_value(
        cfg_get(cfg, "display", "show_reasoning"),
        default=False,
    )


def _load_busy_input_mode() -> str:
    """Load gateway drain-time busy-input behavior from config/env."""
    mode = os.getenv("HERMES_GATEWAY_BUSY_INPUT_MODE", "").strip().lower()
    if not mode:
        cfg = _runtime()
        mode = str(cfg_get(cfg, "display", "busy_input_mode", default="") or "").strip().lower()
    if mode == "queue":
        return "queue"
    if mode == "steer":
        return "steer"
    return "interrupt"


def _load_busy_text_mode() -> str:
    """Resolve normal busy TEXT follow-up behavior.

    ``busy_input_mode`` is the single source of truth (default
    ``interrupt``). The legacy ``busy_text_mode`` knob is honored only
    when a user explicitly set it, so existing queue setups keep
    working; new installs follow ``busy_input_mode``. Returns one of
    ``interrupt`` | ``queue`` (``steer`` is handled upstream by
    ``busy_input_mode`` and maps to non-queue text handling here).
    """
    # Legacy explicit override wins for backward compat.
    legacy = os.getenv("HERMES_GATEWAY_BUSY_TEXT_MODE", "").strip().lower()
    if not legacy:
        cfg = _runtime()
        legacy = str(cfg_get(cfg, "display", "busy_text_mode", default="") or "").strip().lower()
    if legacy == "interrupt":
        return "interrupt"
    if legacy == "queue":
        return "queue"
    # No explicit legacy knob → follow busy_input_mode.
    input_mode = _load_busy_input_mode()
    return "queue" if input_mode == "queue" else "interrupt"


def _load_restart_drain_timeout() -> float:
    """Load graceful gateway restart/stop drain timeout in seconds."""
    raw = os.getenv("HERMES_RESTART_DRAIN_TIMEOUT", "").strip()
    if not raw:
        cfg = _runtime()
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


def _load_background_notifications_mode() -> str:
    """Load background process notification mode from config or env var.

    Modes:
      - ``all``    — push running-output updates *and* the final message (default)
      - ``result`` — only the final completion message (regardless of exit code)
      - ``error``  — only the final message when exit code is non-zero
      - ``off``    — no watcher messages at all
    """
    mode = os.getenv("HERMES_BACKGROUND_NOTIFICATIONS", "")
    if not mode:
        cfg = _runtime()
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


def _load_provider_routing() -> dict:
    """Load OpenRouter provider routing preferences from config.yaml."""
    try:
        import yaml as _y
        cfg_path = _home() / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as _f:
                cfg = _y.safe_load(_f) or {}
            return cfg.get("provider_routing", {}) or {}
    except Exception:
        pass
    return {}


def _load_fallback_model() -> list | None:
    """Load fallback provider chain from config.yaml.

    Returns the merged effective chain from ``fallback_providers`` plus any
    legacy ``fallback_model`` entries. ``fallback_providers`` stays first
    when both keys are present.
    """
    try:
        import yaml as _y
        cfg_path = _home() / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as _f:
                cfg = _y.safe_load(_f) or {}
            fb = get_fallback_chain(cfg)
            if fb:
                return fb
    except Exception:
        pass
    return None


def _load_voice_modes(voice_mode_path: Path) -> Dict[str, str]:
    """Load per-chat voice-mode overrides from gateway_voice_mode.json.

    Extracted from GatewayRunner._load_voice_modes; the path that was
    ``self._VOICE_MODE_PATH`` is now an explicit argument so the function
    carries no instance state.
    """
    try:
        data = json.loads(voice_mode_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}

    if not isinstance(data, dict):
        return {}

    valid_modes = {"off", "voice_only", "all"}
    result = {}
    for chat_id, mode in data.items():
        if mode not in valid_modes:
            continue
        key = str(chat_id)
        # Skip legacy unprefixed keys (warn and skip)
        if ":" not in key:
            logger.warning(
                "Skipping legacy unprefixed voice mode key %r during migration. "
                "Re-enable voice mode on that chat to rebuild the prefixed key.",
                key,
            )
            continue
        result[key] = mode
    return result
