"""
Gateway runner utility checks and warnings.

Extracted from gateway/run.py to reduce the God file size.
Provides functions for Docker media delivery warnings, setup skill checks,
and timeout configuration.
"""

import json
import logging
import os
import re
from typing import List

from hermes_cli.enums import Platform

logger = logging.getLogger(__name__)

_DOCKER_VOLUME_SPEC_RE = re.compile(
    r"^(?P<host>[^:]+):(?P<container>[^:]+)(?::(?P<mode>[^:]+))?$"
)
_DOCKER_MEDIA_OUTPUT_CONTAINER_PATHS = {"/output", "/documents", "/workspace"}

_ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT = 5.0
_PLATFORM_CONNECT_TIMEOUT_SECS_DEFAULT = 30.0


def warn_if_docker_media_delivery_is_risky(config) -> None:
    """Warn when Docker-backed gateways lack an explicit export mount.

    MEDIA delivery happens in the gateway process, so paths emitted by the model
    must be readable from the host. A plain container-local path like
    `/workspace/report.txt` or `/output/report.txt` often exists only inside
    Docker, so users commonly need a dedicated export mount such as
    `host-dir:/output`.
    """
    if os.getenv("TERMINAL_ENV", "").strip().lower() != "docker":
        return

    connected = config.get_connected_platforms()
    messaging_platforms = [
        p for p in connected
        if p not in {Platform.LOCAL, Platform.API_SERVER, Platform.WEBHOOK}
    ]
    if not messaging_platforms:
        return

    raw_volumes = os.getenv("TERMINAL_DOCKER_VOLUMES", "").strip()
    volumes: List[str] = []
    if raw_volumes:
        try:
            parsed = json.loads(raw_volumes)
            if isinstance(parsed, list):
                volumes = [str(v) for v in parsed if isinstance(v, str)]
        except Exception:
            logger.debug(
                "Could not parse TERMINAL_DOCKER_VOLUMES for gateway media warning",
                exc_info=True,
            )

    has_explicit_output_mount = False
    for spec in volumes:
        match = _DOCKER_VOLUME_SPEC_RE.match(spec)
        if not match:
            continue
        container_path = match.group("container")
        if container_path in _DOCKER_MEDIA_OUTPUT_CONTAINER_PATHS:
            has_explicit_output_mount = True
            break

    if has_explicit_output_mount:
        return

    logger.warning(
        "Docker backend is enabled for the messaging gateway but no explicit host-visible "
        "output mount (for example '/home/user/.hermes/cache/documents:/output') is configured. "
        "This is fine if the model already emits host-visible paths, but MEDIA file delivery can fail "
        "for container-local paths like '/workspace/...' or '/output/...'."
    )


def has_setup_skill() -> bool:
    """Check if the hermes-agent-setup skill is installed."""
    try:
        from tools.skill_manager_tool import _find_skill
        return _find_skill("hermes-agent-setup") is not None
    except Exception:
        return False


def adapter_disconnect_timeout_secs() -> float:
    """Return the per-adapter disconnect timeout used during shutdown."""
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


def platform_connect_timeout_secs() -> float:
    """Return the per-platform connect timeout used during startup/retry."""
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
