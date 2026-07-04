"""Multiplex-profile adapter management for ``GatewayRunner``.

Restored from the secondary-profile adapter registry feature (commit d5d02eabb,
"multiplex phase 3"). The gateway decomposition refactor that split
``gateway/run.py`` into per-concern mixins dropped these four methods along
with the startup call site, silently disabling ``gateway.multiplex_profiles``
(the config flag still parsed, but nothing acted on it). This mixin restores
the methods; the call site lives in ``start_mixin_r54.py``.

Mixed into ``GatewayRunner``; all state lives on the runner and is touched via
``self.*``. Module-level helpers (``_profile_runtime_scope``,
``_PORT_BINDING_PLATFORM_VALUES``, ``MultiplexConfigError``) are lazy-imported
from ``gateway.run`` to avoid the circular import (run.py imports this mixin).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class MultiplexAdaptersMixin:
    """Bring up + manage adapters for every non-active profile this gateway serves."""

    async def _start_secondary_profile_adapters(self) -> int:
        """Bring up adapters for every non-active profile this gateway serves.

        Returns the number of secondary adapters that connected. No-op (returns
        0) unless ``gateway.multiplex_profiles`` is on.

        Each profile's adapters are created and connected under that profile's
        HERMES_HOME + secret scope (``_profile_runtime_scope``), stored in
        ``self._profile_adapters[profile]``, and given a message handler that
        stamps ``source.profile`` before delegating to the shared
        ``_handle_message`` — so the agent turn resolves that profile's config,
        skills, and credentials. Same-platform credential collisions (two
        profiles polling the same bot token) are detected and refused here, the
        only point that sees every profile's resolved credentials together.
        """
        if not getattr(self.config, "multiplex_profiles", False):
            return 0

        try:
            from hermes_cli.profiles import profiles_to_serve, get_active_profile_name
        except Exception:
            return 0

        # Lazy import to avoid the circular: gateway.run imports this mixin.
        from gateway.run import MultiplexConfigError

        active = get_active_profile_name() or "default"
        connected = 0
        # (platform, token-fingerprint) -> profile that claimed it. Detects two
        # profiles trying to poll the same bot credential (impossible to do
        # concurrently). Seed with the active profile's adapters.
        claimed: Dict[tuple, str] = {}
        for _plat, _ad in self.adapters.items():
            fp = self._adapter_credential_fingerprint(_ad)
            if fp is not None:
                claimed[(_plat, fp)] = active

        for profile_name, profile_home in profiles_to_serve(multiplex=True):
            if profile_name == active:
                continue  # handled by the primary startup loop
            try:
                connected += await self._start_one_profile_adapters(
                    profile_name, profile_home, claimed
                )
            except MultiplexConfigError:
                # Config error (e.g. a secondary profile binding a port) is not
                # transient — propagate so startup aborts cleanly instead of
                # limping along with a half-configured multiplexer.
                raise
            except Exception as e:
                from gateway.run import logger
                logger.error(
                    "Failed to start adapters for profile '%s': %s",
                    profile_name, e, exc_info=True,
                )
        return connected

    async def _start_one_profile_adapters(
        self, profile_name: str, profile_home: "Path", claimed: Dict[tuple, str]
    ) -> int:
        """Create+connect one profile's adapters under its runtime scope."""
        from gateway.config import load_gateway_config
        from gateway.run import (
            MultiplexConfigError,
            _PORT_BINDING_PLATFORM_VALUES,
            _profile_runtime_scope,
            logger,
        )

        with _profile_runtime_scope(profile_home):
            profile_cfg = load_gateway_config()

        profile_map = self._profile_adapters.setdefault(profile_name, {})
        connected = 0
        for platform, platform_config in profile_cfg.platforms.items():
            if not platform_config.enabled:
                continue
            # A secondary profile must NOT enable a port-binding platform: the
            # default profile's listener already serves every profile via the
            # /p/<profile>/ prefix, so a second bind can only collide. This is a
            # config error, not a transient failure — fail fast and loud.
            if platform.value in _PORT_BINDING_PLATFORM_VALUES:
                raise MultiplexConfigError(
                    f"Profile '{profile_name}' enables the port-binding platform "
                    f"'{platform.value}', but gateway.multiplex_profiles is on. The "
                    f"default profile owns the single shared HTTP listener and "
                    f"serves every profile through the /p/{profile_name}/ URL "
                    f"prefix — a secondary profile cannot bind its own port. "
                    f"Remove platforms.{platform.value} from profile "
                    f"'{profile_name}'s config.yaml (configure it only on the "
                    f"default profile)."
                )
            with _profile_runtime_scope(profile_home):
                adapter = self._create_adapter(platform, platform_config)
            if not adapter:
                continue

            # Same-token conflict detection — refuse a duplicate poll.
            fp = self._adapter_credential_fingerprint(adapter)
            if fp is not None:
                owner = claimed.get((platform, fp))
                if owner is not None:
                    logger.error(
                        "Profile '%s' and '%s' both configure %s with the same "
                        "credential — refusing to start the duplicate (a single "
                        "bot token cannot be polled twice). Give each profile its "
                        "own %s credential.",
                        owner, profile_name, platform.value, platform.value,
                    )
                    await self._safe_adapter_disconnect(adapter, platform)
                    continue
                claimed[(platform, fp)] = profile_name

            # Stamp every inbound event from this adapter with its profile so
            # the agent turn (and session key) resolve to the right home.
            adapter.set_message_handler(
                self._make_profile_message_handler(profile_name)
            )
            adapter.set_fatal_error_handler(self._handle_adapter_fatal_error)
            adapter.set_session_store(self.session_store)
            adapter.set_busy_session_handler(self._handle_active_session_busy_message)
            adapter.set_topic_recovery_fn(self._recover_telegram_topic_thread_id)
            adapter._busy_text_mode = self._busy_text_mode

            try:
                with _profile_runtime_scope(profile_home):
                    success = await self._connect_adapter_with_timeout(adapter, platform)
                if success:
                    profile_map[platform] = adapter
                    connected += 1
                    logger.info("✓ %s connected (profile: %s)", platform.value, profile_name)
                else:
                    logger.warning("✗ %s failed to connect (profile: %s)", platform.value, profile_name)
                    await self._safe_adapter_disconnect(adapter, platform)
            except Exception as e:
                logger.error("✗ %s error (profile: %s): %s", platform.value, profile_name, e)
                await self._safe_adapter_disconnect(adapter, platform)
        return connected

    def _make_profile_message_handler(self, profile_name: str):
        """Return a message handler that stamps source.profile then delegates."""
        async def _handler(event):
            try:
                if getattr(event, "source", None) is not None and not event.source.profile:
                    event.source.profile = profile_name
            except Exception:
                pass
            return await self._handle_message(event)
        return _handler

    @staticmethod
    def _adapter_credential_fingerprint(adapter: Any) -> Optional[str]:
        """Return a stable, log-safe fingerprint of an adapter's credential.

        Used only to detect two profiles claiming the same bot token. Returns a
        salted hash (never the token itself) of the adapter's primary
        credential, or None when no credential is discoverable (in which case
        we don't attempt conflict detection for it).
        """
        token = None
        for attr in ("token", "bot_token", "_token", "api_token", "_bot_token"):
            val = getattr(adapter, attr, None)
            if isinstance(val, str) and val.strip():
                token = val.strip()
                break
        if not token:
            return None
        import hashlib
        return hashlib.sha256(("hermes-mux:" + token).encode("utf-8")).hexdigest()[:16]
