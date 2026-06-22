"""Per-turn agent routing for ``GatewayRunner``.

Round 39 of the god-file decomposition. Lifted verbatim from
gateway/run.py into gateway/turn_agent_config_mixin.py.

``_resolve_turn_agent_config`` builds the effective model/runtime config
for a single turn: it consults the custom per-turn router (agent.routing)
to pick an optimal provider per message category, falls back to the
session's primary model/provider when routing is unavailable or returns
no match, and — if ``/fast`` is enabled and the model supports
Priority Processing / Anthropic fast mode — attaches ``request_overrides``
so the API call is marked accordingly.

``self.*`` references resolve unchanged via the MRO. Behavior-neutral
lift matching the existing mixin pattern.

``gateway.run`` module-level runtime global (``logger``) is lazy-imported
at the top of the method body to avoid the circular import (``gateway.run``
imports this mixin at module top). The router helpers
``resolve_fast_mode_overrides`` and ``route_turn`` are imported in-body
within try/except (already lazy in source) and kept verbatim. The method
takes no module-level imports beyond the lazy one.
"""

from __future__ import annotations


class TurnAgentConfigMixin:
    def _resolve_turn_agent_config(self, user_message: str, model: str, runtime_kwargs: dict) -> dict:
        """Build the effective model/runtime config for a single turn.

        Consults the custom per-turn router (agent.routing) to pick an optimal
        provider per message category; falls back to the session's primary
        model/provider when routing is unavailable or returns no match. If
        `/fast` is enabled and the model supports Priority Processing /
        Anthropic fast mode, attach `request_overrides` so the API call is
        marked accordingly.
        """
        from gateway.run import logger

        from hermes_cli.models import resolve_fast_mode_overrides

        runtime = {
            "api_key": runtime_kwargs.get("api_key"),
            "base_url": runtime_kwargs.get("base_url"),
            "provider": runtime_kwargs.get("provider"),
            "api_mode": runtime_kwargs.get("api_mode"),
            "command": runtime_kwargs.get("command"),
            "args": list(runtime_kwargs.get("args") or []),
            "credential_pool": runtime_kwargs.get("credential_pool"),
            "max_tokens": runtime_kwargs.get("max_tokens"),
        }

        # Custom per-turn routing: classify the message and pick the optimal
        # provider from the configured category chain. Returns None when the
        # router isn't initialized or no chain matches — in that case we fall
        # through to the session primary (the values already in model/runtime).
        # Best-effort: any error degrades to primary-only, never blocks a turn.
        try:
            from agent.routing import route_turn
            _primary_cfg = {
                "model": model,
                "base_url": runtime["base_url"],
                "api_key": runtime["api_key"],
                "provider": runtime["provider"],
            }
            _route = route_turn(user_message, _primary_cfg)
            if _route is not None:
                model = _route.model
                runtime["base_url"] = _route.base_url
                runtime["api_key"] = _route.api_key
                runtime["provider"] = _route.provider
        except Exception as _re:
            logger.debug("Per-turn routing unavailable, using primary: %s", _re)

        route = {
            "model": model,
            "runtime": runtime,
            "signature": (
                model,
                runtime["provider"],
                runtime["base_url"],
                runtime["api_mode"],
                runtime["command"],
                tuple(runtime["args"]),
            ),
        }

        service_tier = getattr(self, "_service_tier", None)
        if not service_tier:
            route["request_overrides"] = {}
            return route

        try:
            overrides = resolve_fast_mode_overrides(route["model"])
        except Exception:
            overrides = None
        route["request_overrides"] = overrides or {}
        return route
