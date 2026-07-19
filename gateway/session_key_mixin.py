"""Session-key derivation + runtime-resolution methods for ``GatewayRunner``.

Round 25 of the god-file decomposition. Lifted verbatim into a mixin.
Three cohesive session-key/runtime methods moved together:

1. **Session-key derivation** — ``_session_key_for_source``
   (``self.session_store``, ``self.config``; falls back to
   ``build_session_key`` honoring ``group_sessions_per_user`` /
   ``thread_sessions_per_user`` config).
2. **Source normalization** — ``_normalize_source_for_session_key``
   (applies Telegram DM topic recovery via
   ``self._recover_telegram_topic_thread_id`` so override storage and
   read use an identical key; #30479).
3. **Runtime resolution** — ``_resolve_session_agent_runtime``
   (honors session-scoped /model overrides, env-based runtime fallback,
   provider default-model recovery, and the last-known-good model
   safety net #35314 when config-cache misses return empty).

``self.*`` references resolve unchanged via the MRO. Behavior-neutral
lift matching the existing mixin pattern. NOTE: the interleaved class
attributes between these methods in run.py
(``_TELEGRAM_GENERAL_TOPIC_IDS``, ``_TELEGRAM_LOBBY_REMINDER_COOLDOWN_S``)
are NOT moved — they remain in ``GatewayRunner`` and stay reachable
via the MRO.

``gateway.run`` module-level globals (``build_session_key``,
``dataclasses``, ``logger``, ``_resolve_gateway_model``,
``_resolve_runtime_agent_kwargs``) are lazy-imported inside each
method body to avoid a circular import (``gateway.run`` imports this
mixin at module top). Type-only imports (``Optional``,
``SessionSource``) come from modules with no circular dependency on
``gateway.run`` and are imported at module top.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

from gateway.session import SessionSource


class GatewaySessionKeyMixin:
    """Session-key derivation + runtime-resolution methods for ``GatewayRunner``."""

    def _session_key_for_source(self, source: SessionSource) -> str:
        """Resolve the current session key for a source, honoring gateway config when available."""
        from gateway.run import build_session_key
        if hasattr(self, "session_store") and self.session_store is not None:
            try:
                session_key = self.session_store._generate_session_key(source)
                if isinstance(session_key, str) and session_key:
                    return session_key
            except Exception:
                pass
        config = getattr(self, "config", None)
        return build_session_key(
            source,
            group_sessions_per_user=getattr(config, "group_sessions_per_user", True),
            thread_sessions_per_user=getattr(config, "thread_sessions_per_user", False),
        )

    def _normalize_source_for_session_key(
        self,
        source: SessionSource,
    ) -> SessionSource:
        """Apply Telegram DM topic recovery to a source for session-key purposes.

        ``_handle_message_with_agent`` rewrites ``source.thread_id`` via
        ``_recover_telegram_topic_thread_id`` *before* deriving the session
        key for a normal message turn (a lobby/stripped reply gets pinned to
        the user's last-active topic).  Session-scoped command handlers like
        ``/model`` and ``/reasoning`` derive their override key from the raw
        inbound ``event.source``, which skips that recovery — so the override
        is stored under a different key than the next message turn reads,
        and the override is silently dropped on Telegram forum topics and
        after compression session splits (#30479).

        Returns a recovery-normalized copy when a rewrite applies, otherwise
        the original source unchanged.  Always derive the override storage key
        from the result so storage and read use an identical key.
        """
        try:
            recovered = self._recover_telegram_topic_thread_id(source)
        except Exception:
            return source
        if recovered is None:
            return source
        return dataclasses.replace(source, thread_id=recovered)

    def _resolve_session_agent_runtime(
        self,
        *,
        source: Optional[SessionSource] = None,
        session_key: Optional[str] = None,
        user_config: Optional[dict] = None,
    ) -> tuple[str, dict]:
        """Resolve model/runtime for a session, honoring session-scoped /model overrides.

        If the session override already contains a complete provider bundle
        (provider/api_key/base_url/api_mode), prefer it directly instead of
        resolving fresh global runtime state first.
        """
        from gateway.run import logger, _resolve_gateway_model, _resolve_runtime_agent_kwargs
        resolved_session_key = session_key
        if not resolved_session_key and source is not None:
            try:
                resolved_session_key = self._session_key_for_source(source)
            except Exception:
                resolved_session_key = None

        model = _resolve_gateway_model(user_config)
        override = self._session_model_overrides.get(resolved_session_key) if resolved_session_key else None
        if override:
            override_model = override.get("model", model)
            override_runtime = {
                "provider": override.get("provider"),
                "api_key": override.get("api_key"),
                "base_url": override.get("base_url"),
                "api_mode": override.get("api_mode"),
                "max_tokens": override.get("max_tokens"),
            }
            if override_runtime.get("api_key"):
                logger.debug(
                    "Session model override (fast): session=%s config_model=%s -> override_model=%s provider=%s",
                    resolved_session_key or "", model, override_model,
                    override_runtime.get("provider"),
                )
                return override_model, override_runtime
            # Override exists but has no api_key — fall through to env-based
            # resolution and apply model/provider from the override on top.
            logger.debug(
                "Session model override (no api_key, fallback): session=%s config_model=%s override_model=%s",
                resolved_session_key or "", model, override_model,
            )
        else:
            logger.debug(
                "No session model override: session=%s config_model=%s override_keys=%s",
                resolved_session_key or "", model,
                list(self._session_model_overrides.keys())[:5] if self._session_model_overrides else "[]",
            )

        runtime_kwargs = _resolve_runtime_agent_kwargs()
        runtime_model = runtime_kwargs.pop("model", None)
        if runtime_model:
            logger.info(
                "Runtime provider supplied explicit model override: %s -> %s",
                model,
                runtime_model,
            )
            model = runtime_model
        if override and resolved_session_key:
            model, runtime_kwargs = self._apply_session_model_override(
                resolved_session_key, model, runtime_kwargs
            )

        # When the config has no model.default but a provider was resolved
        # (e.g. user ran `hermes auth add openai-codex` without `hermes model`),
        # fall back to the provider's first catalog model so the API call
        # doesn't fail with "model must be a non-empty string".
        if not model and runtime_kwargs.get("provider"):
            try:
                from hermes_cli.models import get_default_model_for_provider
                model = get_default_model_for_provider(runtime_kwargs["provider"])
                if model:
                    logger.info(
                        "No model configured — defaulting to %s for provider %s",
                        model, runtime_kwargs["provider"],
                    )
            except Exception:
                pass

        # Final safety net (#35314): if resolution still produced an empty
        # model — e.g. a transient config-cache miss during a post-interrupt
        # recovery turn returned an empty user_config — reuse the last model we
        # successfully resolved for this session (or, failing that, the most
        # recent one resolved process-wide). Building an agent with model=""
        # makes every API call fail HTTP 400 "No models provided" and the
        # session goes silent until the user manually re-sends. ``getattr``
        # guards against bare test runners built via ``object.__new__``.
        _last_good = getattr(self, "_last_resolved_model", None)
        if _last_good is not None:
            if not model:
                _recovered = _last_good.get(resolved_session_key or "") or _last_good.get("*")
                if _recovered:
                    logger.warning(
                        "Empty model resolved for session=%s — recovering "
                        "last-known-good model %s (config read likely returned "
                        "empty; see #35314)",
                        resolved_session_key or "", _recovered,
                    )
                    model = _recovered
            elif model:
                # Cache the good resolution for future recovery turns.
                if resolved_session_key:
                    _last_good[resolved_session_key] = model
                _last_good["*"] = model

        return model, runtime_kwargs
