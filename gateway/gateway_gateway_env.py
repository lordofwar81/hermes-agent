"""Gateway environment / runtime-config readers extracted from GatewayRunner.

Round 9 of gateway decomposition. These methods read env vars, config.yaml,
or the active profile — none touch instance state. Moved as plain functions.
Deps on run.py module globals (_load_gateway_config, _ADAPTER_*_DEFAULT,
logger) are imported lazily to avoid circular imports.
"""

import os
from typing import Optional

import logging
logger = logging.getLogger("gateway.run")


def _adapter_disconnect_timeout_secs() -> float:
    """Return the per-adapter disconnect timeout used during shutdown."""
    from gateway.run import _ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT
    raw = os.getenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "").strip()
    if raw:
        try:
            timeout = float(raw)
        except ValueError:
            logger.warning(
                "Ignoring invalid HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT=%r",
                raw,
            )
        else:
            return max(0.0, timeout)
    return _ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT


def _platform_connect_timeout_secs() -> float:
    """Return the per-platform connect timeout used during startup/retry."""
    from gateway.run import _PLATFORM_CONNECT_TIMEOUT_SECS_DEFAULT
    raw = os.getenv("HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT", "").strip()
    if raw:
        try:
            timeout = float(raw)
        except ValueError:
            logger.warning(
                "Ignoring invalid HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT=%r",
                raw,
            )
        else:
            return max(0.0, timeout)
    return _PLATFORM_CONNECT_TIMEOUT_SECS_DEFAULT


def _get_proxy_url() -> Optional[str]:
    """Return the proxy URL if proxy mode is configured, else None.

    Checks GATEWAY_PROXY_URL env var first (convenient for Docker),
    then ``gateway.proxy_url`` in config.yaml.
    """
    from gateway.run import _load_gateway_config
    url = os.getenv("GATEWAY_PROXY_URL", "").strip()
    if url:
        return url.rstrip("/")
    cfg = _load_gateway_config()
    url = (cfg.get("gateway") or {}).get("proxy_url", "").strip()
    if url:
        return url.rstrip("/")
    return None


def _active_profile_name() -> str:
    """Return the profile name this gateway represents."""
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name() or "default"
    except Exception:
        return "default"


def _has_setup_skill() -> bool:
    """Check if the hermes-agent-setup skill is installed."""
    try:
        from tools.skill_manager_tool import _find_skill
        return _find_skill("hermes-agent-setup") is not None
    except Exception:
        return False


def _clear_session_env(tokens: list) -> None:
    """Restore session context variables to their pre-handler values."""
    from gateway.session_context import clear_session_vars
    clear_session_vars(tokens)
