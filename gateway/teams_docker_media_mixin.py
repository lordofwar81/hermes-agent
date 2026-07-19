"""Teams pipeline + Docker media warnings for ``GatewayRunner``.

Round 38 of the god-file decomposition. Lifted verbatim from
gateway/run.py into gateway/teams_docker_media_mixin.py.

Two small SYNC helpers cluster naturally — both touch runtime wiring /
sanity-checks that run during startup, neither depends on the other:

- ``_wire_teams_pipeline_runtime``: binds the Teams meeting pipeline
  runtime to Graph webhook ingress. No-op when the msgraph_webhook
  adapter isn't running or the teams_pipeline plugin isn't enabled.
- ``_warn_if_docker_media_delivery_is_risky``: warns when a Docker-backed
  gateway lacks an explicit host-visible output mount, since MEDIA file
  delivery happens in the gateway process and paths emitted by the model
  must be readable from the host.

``self.*`` references resolve unchanged via the MRO. Behavior-neutral
lift matching the existing mixin pattern.

``gateway.run`` module-level runtime globals (``logger``, ``_DOCKER_*``
compiled regexes / container paths, and the run.py-defined free function
``_teams_pipeline_plugin_enabled``) are lazy-imported at the top of each
method body to avoid the circular import (``gateway.run`` imports this
mixin at module top). Stdlib (``json``, ``os``) and the non-circular
module symbols (``Platform``, ``List``) are imported at module top.
``bind_gateway_runtime`` is imported in-body within a try/except (already
lazy in source) and kept verbatim.
"""

from __future__ import annotations

import json
import os

from typing import List

from gateway.config import Platform


class TeamsDockerMediaMixin:
    def _wire_teams_pipeline_runtime(self) -> None:
        """Bind the Teams meeting pipeline runtime to Graph webhook ingress.

        No-op when the msgraph_webhook adapter isn't running or the
        teams_pipeline plugin isn't enabled — lets the gateway start cleanly
        whether or not the user has opted into the pipeline.
        """
        from gateway.run import _teams_pipeline_plugin_enabled, logger

        if Platform.MSGRAPH_WEBHOOK not in self.adapters:
            return
        if not _teams_pipeline_plugin_enabled():
            logger.debug("Teams pipeline plugin is disabled; skipping runtime wiring")
            return
        try:
            from plugins.teams_pipeline.runtime import bind_gateway_runtime
        except Exception as exc:
            logger.warning("Teams pipeline runtime import failed: %s", exc)
            return
        try:
            bound = bind_gateway_runtime(self)
        except Exception as exc:
            logger.warning("Teams pipeline runtime wiring failed: %s", exc)
            return
        if bound:
            logger.info("Teams pipeline runtime bound to msgraph webhook ingress")
        elif self._teams_pipeline_runtime_error:
            logger.warning(
                "Teams pipeline runtime unavailable: %s",
                self._teams_pipeline_runtime_error,
            )

    def _warn_if_docker_media_delivery_is_risky(self) -> None:
        """Warn when Docker-backed gateways lack an explicit export mount.

        MEDIA delivery happens in the gateway process, so paths emitted by the model
        must be readable from the host. A plain container-local path like
        `/workspace/report.txt` or `/output/report.txt` often exists only inside
        Docker, so users commonly need a dedicated export mount such as
        `host-dir:/output`.
        """
        from gateway.run import (
            _DOCKER_MEDIA_OUTPUT_CONTAINER_PATHS,
            _DOCKER_VOLUME_SPEC_RE,
            logger,
        )

        if os.getenv("TERMINAL_ENV", "").strip().lower() != "docker":
            return

        connected = self.config.get_connected_platforms()
        messaging_platforms = [p for p in connected if p not in {Platform.LOCAL, Platform.API_SERVER, Platform.WEBHOOK}]
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
                logger.debug("Could not parse TERMINAL_DOCKER_VOLUMES for gateway media warning", exc_info=True)

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
