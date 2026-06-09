"""Agent runtime configuration resolution.

Handles model/provider resolution for sessions and turns, including session
overrides and credential fallback.

Extracted from gateway/run.py for modular organization.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from gateway.session import SessionSource

logger = logging.getLogger(__name__)

# Module-level cache for hermes home directory - matches run.py pattern
_hermes_home = Path.home() / ".hermes"


def load_hermes_dotenv() -> None:
    """Load .env from ~/.hermes if present.

    Mirrors the function in run.py to avoid circular imports.
    """
    dotenv_path = _hermes_home / ".env"
    if dotenv_path.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(dotenv_path)
        except Exception:
            pass


def _load_gateway_config() -> dict:
    """Load and parse ~/.hermes/config.yaml, returning {} on any error.

    Uses the module-level ``_hermes_home`` (so tests that monkeypatch it
    still see their fixture) and shares the mtime-keyed raw-yaml cache
    from ``hermes_cli.config.read_raw_config`` when the paths match.
    """
    config_path = _hermes_home / "config.yaml"
    try:
        from hermes_cli.config import get_config_path, read_raw_config

        # Fast path: if _hermes_home agrees with the canonical config
        # location, reuse the shared cache. Otherwise fall through to a
        # direct read (keeps test fixtures with a monkeypatched
        # _hermes_home working).
        if config_path == get_config_path():
            return read_raw_config()
    except Exception:
        pass

    try:
        if config_path.exists():
            import yaml

            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        logger.debug("Could not load gateway config from %s", config_path)
    return {}


def _try_resolve_fallback_provider() -> dict | None:
    """Attempt to resolve credentials from the fallback_model/fallback_providers config."""
    from hermes_cli.fallback_config import get_fallback_chain
    from hermes_cli.runtime_provider import resolve_runtime_provider

    try:
        import yaml

        cfg_path = _hermes_home / "config.yaml"
        if not cfg_path.exists():
            return None
        with open(cfg_path, encoding="utf-8") as _f:
            cfg = yaml.safe_load(_f) or {}
        fb_list = get_fallback_chain(cfg)
        if not fb_list:
            return None
        for entry in fb_list:
            try:
                explicit_api_key = entry.get("api_key")
                if not explicit_api_key:
                    key_env = str(
                        entry.get("key_env") or entry.get("api_key_env") or ""
                    ).strip()
                    if key_env:
                        explicit_api_key = os.getenv(key_env, "").strip() or None
                runtime = resolve_runtime_provider(
                    requested=entry.get("provider"),
                    explicit_base_url=entry.get("base_url"),
                    explicit_api_key=explicit_api_key,
                )
                # Log the literal `provider` key from config, not the resolved
                # runtime category — an Ollama fallback resolves through the
                # OpenAI-compatible path and would otherwise be logged as
                # "openrouter", contradicting the operator's config (#32790).
                logger.info(
                    "Fallback provider resolved: %s model=%s",
                    entry.get("provider") or runtime.get("provider"),
                    entry.get("model"),
                )
                return {
                    "api_key": runtime.get("api_key"),
                    "base_url": runtime.get("base_url"),
                    "provider": runtime.get("provider"),
                    "api_mode": runtime.get("api_mode"),
                    "command": runtime.get("command"),
                    "args": list(runtime.get("args") or []),
                    "credential_pool": runtime.get("credential_pool"),
                    "model": entry.get("model"),
                }
            except Exception as fb_exc:
                logger.debug("Fallback entry %s failed: %s", entry.get("provider"), fb_exc)
    except Exception as exc:
        logger.debug("Fallback provider resolution failed: %s", exc)
    return None


def _resolve_runtime_agent_kwargs() -> dict:
    """Resolve provider credentials for gateway-created AIAgent instances.

    Provider is read from ``config.yaml`` ``model.provider`` (the single
    source of truth). ``resolve_runtime_provider()`` falls through to env
    var lookups internally for legacy compatibility, but the gateway does
    not consult environment variables for behavioral config — config.yaml
    is authoritative.

    If the primary provider fails with an authentication error, attempt to
    resolve credentials using the fallback provider chain from config.yaml
    before giving up.
    """
    from hermes_cli.auth import AuthError, is_rate_limited_auth_error
    from hermes_cli.runtime_provider import (
        format_runtime_provider_error,
        resolve_runtime_provider,
    )

    try:
        runtime = resolve_runtime_provider()
    except AuthError as auth_exc:
        # Distinguish a transient rate-limit/quota cap (credentials are fine,
        # re-auth cannot help) from a genuine auth failure (expired/revoked
        # token). Both fall through to the fallback chain, but the log message
        # must not mislabel a quota exhaustion as an auth failure (#32790).
        if is_rate_limited_auth_error(auth_exc):
            logger.warning("Primary provider rate-limited (429): %s — trying fallback", auth_exc)
        else:
            logger.warning("Primary provider auth failed: %s — trying fallback", auth_exc)
        fb_config = _try_resolve_fallback_provider()
        if fb_config is not None:
            return fb_config
        raise RuntimeError(format_runtime_provider_error(auth_exc)) from auth_exc
    except Exception as exc:
        raise RuntimeError(format_runtime_provider_error(exc)) from exc

    return {
        "api_key": runtime.get("api_key"),
        "base_url": runtime.get("base_url"),
        "provider": runtime.get("provider"),
        "api_mode": runtime.get("api_mode"),
        "command": runtime.get("command"),
        "args": list(runtime.get("args") or []),
        "credential_pool": runtime.get("credential_pool"),
    }


def _resolve_gateway_model(config: dict | None = None) -> str:
    """Read model from config.yaml — single source of truth.

    Without this, temporary AIAgent instances (e.g. /compress) fall
    back to the hardcoded default which fails when the active provider is
    openai-codex.
    """
    cfg = config if config is not None else _load_gateway_config()
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, str):
        return model_cfg
    elif isinstance(model_cfg, dict):
        return model_cfg.get("default") or model_cfg.get("model") or ""
    return ""


def resolve_session_agent_runtime(
    *,
    source: Optional[SessionSource] = None,
    session_key: Optional[str] = None,
    user_config: Optional[dict] = None,
    session_model_overrides: dict,
    session_key_for_source_func: callable,
    last_resolved_model: Optional[dict] = None,
) -> tuple[str, dict]:
    """Resolve model/runtime for a session, honoring session-scoped /model overrides.

    If the session override already contains a complete provider bundle
    (provider/api_key/base_url/api_mode), prefer it directly instead of
    resolving fresh global runtime state first.

    Args:
        source: The session source (platform/chat info).
        session_key: The session key if already known.
        user_config: User configuration dict (optional).
        session_model_overrides: Dict of per-session model overrides from /model command.
        session_key_for_source_func: Function to resolve session key from source.
        last_resolved_model: Optional dict of last-good model cache for recovery.

    Returns:
        Tuple of (model_name, runtime_kwargs_dict).
    """
    resolved_session_key = session_key
    if not resolved_session_key and source is not None:
        try:
            resolved_session_key = session_key_for_source_func(source)
        except Exception:
            resolved_session_key = None

    model = _resolve_gateway_model(user_config)
    override = session_model_overrides.get(resolved_session_key) if resolved_session_key else None
    if override:
        override_model = override.get("model", model)
        override_runtime = {
            "provider": override.get("provider"),
            "api_key": override.get("api_key"),
            "base_url": override.get("base_url"),
            "api_mode": override.get("api_mode"),
        }
        if override_runtime.get("api_key"):
            logger.debug(
                "Session model override (fast): session=%s config_model=%s -> override_model=%s provider=%s",
                resolved_session_key or "",
                model,
                override_model,
                override_runtime.get("provider"),
            )
            return override_model, override_runtime
        # Override exists but has no api_key — fall through to env-based
        # resolution and apply model/provider from the override on top.
        logger.debug(
            "Session model override (no api_key, fallback): session=%s config_model=%s override_model=%s",
            resolved_session_key or "",
            model,
            override_model,
        )
    else:
        logger.debug(
            "No session model override: session=%s config_model=%s override_keys=%s",
            resolved_session_key or "",
            model,
            list(session_model_overrides.keys())[:5] if session_model_overrides else "[]",
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
        model, runtime_kwargs = apply_session_model_override(
            session_model_overrides,
            resolved_session_key,
            model,
            runtime_kwargs,
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
                    model,
                    runtime_kwargs["provider"],
                )
        except Exception:
            pass

    # Final safety net (#35314): if resolution still produced an empty
    # model — e.g. a transient config-cache miss during a post-interrupt
    # recovery turn returned an empty user_config — reuse the last model we
    # successfully resolved for this session (or, failing that, the most
    # recent one resolved process-wide). Building an agent with model=""
    # makes every API call fail HTTP 400 "No models provided" and the
    # session goes silent until the user manually re-sends.
    _last_good = last_resolved_model
    if _last_good is not None:
        if not model:
            _recovered = _last_good.get(resolved_session_key or "") or _last_good.get("*")
            if _recovered:
                logger.warning(
                    "Empty model resolved for session=%s — recovering "
                    "last-known-good model %s (config read likely returned "
                    "empty; see #35314)",
                    resolved_session_key or "",
                    _recovered,
                )
                model = _recovered
        elif model:
            # Cache the good resolution for future recovery turns.
            if resolved_session_key:
                _last_good[resolved_session_key] = model
            _last_good["*"] = model

    return model, runtime_kwargs


def resolve_turn_agent_config(
    user_message: str,
    model: str,
    runtime_kwargs: dict,
    service_tier: Optional[str],
) -> dict:
    """Build the effective model/runtime config for a single turn.

    Always uses the session's primary model/provider.  If `/fast` is
    enabled and the model supports Priority Processing / Anthropic fast
    mode, attach `request_overrides` so the API call is marked
    accordingly.

    Args:
        user_message: The user's message text.
        model: The resolved model name.
        runtime_kwargs: Runtime keyword arguments from session resolution.
        service_tier: The service tier (e.g. "fast") or None.

    Returns:
        Dict with model, runtime, signature, and optional request_overrides.
    """
    runtime = {
        "api_key": runtime_kwargs.get("api_key"),
        "base_url": runtime_kwargs.get("base_url"),
        "provider": runtime_kwargs.get("provider"),
        "api_mode": runtime_kwargs.get("api_mode"),
        "command": runtime_kwargs.get("command"),
        "args": list(runtime_kwargs.get("args") or []),
        "credential_pool": runtime_kwargs.get("credential_pool"),
    }
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

    if not service_tier:
        route["request_overrides"] = {}
        return route

    try:
        from hermes_cli.models import resolve_fast_mode_overrides

        overrides = resolve_fast_mode_overrides(route["model"])
    except Exception:
        overrides = None
    route["request_overrides"] = overrides or {}
    return route


def compute_agent_config_signature(
    model: str,
    runtime: dict,
    enabled_toolsets: list,
    ephemeral_prompt: str,
    cache_keys: dict | None = None,
    user_id: str | None = None,
    user_id_alt: str | None = None,
) -> str:
    """Compute a stable string key from agent config values.

    When this signature changes between messages, the cached AIAgent is
    discarded and rebuilt.  When it stays the same, the cached agent is
    reused — preserving the frozen system prompt and tool schemas for
    prompt cache hits.

    ``cache_keys`` is an optional flat dict of additional config values
    that should invalidate the cache when they change.  Callers pass
    the output of ``_extract_cache_busting_config(user_config)`` so
    edits to model.context_length / compression.* in config.yaml are
    picked up on the next gateway message without a manual restart.

    ``user_id`` and ``user_id_alt`` are the runtime user identities
    carried by the current message's gateway source.  They participate
    in the cache key because the Honcho memory provider freezes them
    into ``HonchoSessionManager`` at first-message init (see
    ``plugins/memory/honcho/__init__.py::_do_session_init``).  Without
    them in the signature, a shared-thread session_key (one in which
    ``build_session_key`` intentionally omits the participant ID,
    e.g. ``thread_sessions_per_user=False``) would reuse the cached
    AIAgent across distinct users, causing the second user's messages
    to be attributed to the first user's resolved Honcho peer.  This
    broke #27371's per-user-peer contract in multi-user gateways.
    Per-user agent rebuilds in shared threads trade prompt-cache
    warmth for correct memory attribution.
    """
    # Fingerprint the FULL credential string instead of using a short
    # prefix. OAuth/JWT-style tokens frequently share a common prefix
    # (e.g. "eyJhbGci"), which can cause false cache hits across auth
    # switches if only the first few characters are considered.
    _api_key = str(runtime.get("api_key", "") or "")
    _api_key_fingerprint = hashlib.sha256(_api_key.encode()).hexdigest() if _api_key else ""

    _cache_keys_sorted = sorted((cache_keys or {}).items())

    blob = json.dumps(
        [
            model,
            _api_key_fingerprint,
            runtime.get("base_url", ""),
            runtime.get("provider", ""),
            runtime.get("api_mode", ""),
            sorted(enabled_toolsets) if enabled_toolsets else [],
            # reasoning_config excluded — it's set per-message on the
            # cached agent and doesn't affect system prompt or tools.
            ephemeral_prompt or "",
            _cache_keys_sorted,
            str(user_id or ""),
            str(user_id_alt or ""),
        ],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def apply_session_model_override(
    session_model_overrides: dict,
    session_key: str,
    model: str,
    runtime_kwargs: dict,
) -> tuple[str, dict]:
    """Apply /model session overrides if present, returning (model, runtime_kwargs).

    The gateway /model command stores per-session overrides in
    ``_session_model_overrides``.  These must take precedence over
    config.yaml defaults so the switched model is actually used for
    subsequent messages.  Fields with ``None`` values are skipped so
    partial overrides don't clobber valid config defaults.
    """
    override = session_model_overrides.get(session_key)
    if not override:
        return model, runtime_kwargs
    model = override.get("model", model)
    for key in ("provider", "api_key", "base_url", "api_mode"):
        val = override.get(key)
        if val is not None:
            runtime_kwargs[key] = val
    return model, runtime_kwargs


def is_intentional_model_switch(session_model_overrides: dict, session_key: str, agent_model: str) -> bool:
    """Return True if *agent_model* matches an active /model session override."""
    override = session_model_overrides.get(session_key)
    return override is not None and override.get("model") == agent_model
