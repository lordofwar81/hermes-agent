"""
Session management utilities extracted from GatewayRunner.

Provides methods for:
- Session key resolution for message sources
- Session source caching (LRU cache with cap)
- Active profile name detection
- User config reading
- Session context environment variable management
- Session info formatting (model, provider, context)
"""

import dataclasses
import logging
from collections import OrderedDict
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def session_key_for_source(
    source: "SessionSource",
    session_store=None,
    config: Optional["GatewayConfig"] = None,
) -> str:
    """Resolve the current session key for a source, honoring gateway config when available.

    Args:
        source: The SessionSource to generate a key for.
        session_store: Optional SessionStore for key generation.
        config: Optional GatewayConfig for session policy settings.

    Returns:
        The session key string.
    """
    from gateway.session import build_session_key

    if session_store is not None:
        try:
            session_key = session_store._generate_session_key(source)
            if isinstance(session_key, str) and session_key:
                return session_key
        except Exception:
            pass
    return build_session_key(
        source,
        group_sessions_per_user=getattr(config, "group_sessions_per_user", True),
        thread_sessions_per_user=getattr(config, "thread_sessions_per_user", False),
    )


def cache_session_source(
    session_key: str,
    source,
    session_sources: Optional[OrderedDict],
    max_size: int = 512,
) -> None:
    """Cache a session source for later synthetic event routing.

    Uses LRU eviction when the cache exceeds max_size.

    Args:
        session_key: The session key to cache under.
        source: The SessionSource to cache.
        session_sources: The OrderedDict cache (may be None).
        max_size: Maximum number of entries to keep.
    """
    if not session_key or source is None:
        return
    if session_sources is None:
        session_sources = OrderedDict()
    try:
        session_sources[session_key] = dataclasses.replace(source)
    except Exception:
        logger.debug("Failed to cache live session source for %s", session_key, exc_info=True)
        return
    # LRU: mark as most-recently-used and trim to max size.
    try:
        session_sources.move_to_end(session_key)
        while len(session_sources) > max_size:
            session_sources.popitem(last=False)
    except Exception:
        pass


def get_cached_session_source(
    session_key: str,
    session_sources: Optional[OrderedDict],
):
    """Retrieve a cached session source, updating its LRU position.

    Args:
        session_key: The session key to look up.
        session_sources: The OrderedDict cache.

    Returns:
        The cached SessionSource or None.
    """
    if not session_key:
        return None
    if not session_sources:
        return None
    source = session_sources.get(session_key)
    if source is not None:
        try:
            session_sources.move_to_end(session_key)
        except Exception:
            pass
    return source


def active_profile_name() -> str:
    """Return the active Hermes profile name.

    Returns:
        The profile name, or "default" if unavailable.
    """
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name() or "default"
    except Exception:
        return "default"


def read_user_config() -> Dict[str, Any]:
    """Read the user's raw config.yaml (cached) for gate lookups.

    Used by slash-confirm gates that must reflect on-disk state changes
    (e.g. a prior "Always Approve" click) without a gateway restart.

    Returns:
        The loaded config dict, or an empty dict on failure.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def set_session_env(context: "SessionContext") -> list:
    """Set session context variables for the current async task.

    Uses contextvars instead of os.environ so that concurrent
    gateway messages cannot overwrite each other's session state.

    Args:
        context: The SessionContext with source information.

    Returns:
        A list of reset tokens; pass them to clear_session_env()
        in a finally block.
    """
    from gateway.session_context import set_session_vars
    return set_session_vars(
        platform=context.source.platform.value,
        chat_id=context.source.chat_id,
        chat_name=context.source.chat_name or "",
        thread_id=str(context.source.thread_id) if context.source.thread_id else "",
        user_id=str(context.source.user_id) if context.source.user_id else "",
        user_name=str(context.source.user_name) if context.source.user_name else "",
        session_key=context.session_key,
        message_id=str(context.source.message_id) if context.source.message_id else "",
    )


def clear_session_env(tokens: list) -> None:
    """Restore session context variables to their pre-handler values.

    Args:
        tokens: The list of tokens returned by set_session_env().
    """
    from gateway.session_context import clear_session_vars
    clear_session_vars(tokens)


def format_session_info(
    resolve_gateway_model=None,
    load_gateway_config=None,
    resolve_runtime_agent_kwargs=None,
) -> str:
    """Resolve current model config and return a formatted info block.

    Surfaces model, provider, context length, and endpoint so gateway
    users can immediately see if context detection went wrong (e.g.
    local models falling to the 128K default).

    Args:
        resolve_gateway_model: Callable that returns the current model string.
        load_gateway_config: Callable that returns the gateway config dict.
        resolve_runtime_agent_kwargs: Callable that returns runtime kwargs.

    Returns:
        A formatted multi-line string with model info.
    """
    from agent.model_metadata import get_model_context_length, DEFAULT_FALLBACK_CONTEXT

    # Use provided callables or import defaults
    if resolve_gateway_model is None:
        from gateway.run import _resolve_gateway_model
        resolve_gateway_model = _resolve_gateway_model
    if load_gateway_config is None:
        from gateway.run import _load_gateway_config
        load_gateway_config = _load_gateway_config
    if resolve_runtime_agent_kwargs is None:
        from gateway.run import _resolve_runtime_agent_kwargs
        resolve_runtime_agent_kwargs = _resolve_runtime_agent_kwargs

    model = resolve_gateway_model()
    config_context_length = None
    provider = None
    base_url = None
    api_key = None
    custom_provs = None
    data = None

    try:
        data = load_gateway_config()
        if data:
            model_cfg = data.get("model", {})
            if isinstance(model_cfg, dict):
                raw_ctx = model_cfg.get("context_length")
                if raw_ctx is not None:
                    try:
                        config_context_length = int(raw_ctx)
                    except (TypeError, ValueError):
                        pass
                provider = model_cfg.get("provider") or None
                base_url = model_cfg.get("base_url") or None
            try:
                from hermes_cli.config import get_compatible_custom_providers
                custom_provs = get_compatible_custom_providers(data)
            except Exception:
                custom_provs = data.get("custom_providers")
    except Exception:
        pass

    # Also check custom_providers for context_length when top-level model.context_length is not set
    if config_context_length is None and data:
        try:
            custom_providers = data.get("custom_providers", [])
            if custom_providers:
                for cp in custom_providers:
                    if not isinstance(cp, dict):
                        continue
                    cp_model = cp.get("model") or ""
                    cp_models = cp.get("models") or {}
                    # Match provider model to current model
                    if cp_model and cp_model == model:
                        raw_cp_ctx = cp.get("context_length")
                        if raw_cp_ctx is not None:
                            try:
                                config_context_length = int(raw_cp_ctx)
                                break
                            except (TypeError, ValueError):
                                pass
                    # Also check per-model context_length
                    if isinstance(cp_models, dict):
                        model_entry = cp_models.get(model)
                        if isinstance(model_entry, dict):
                            model_ctx = model_entry.get("context_length")
                        else:
                            model_ctx = model_entry
                        if model_ctx is not None and isinstance(model_ctx, (int, float)):
                            try:
                                config_context_length = int(model_ctx)
                                break
                            except (TypeError, ValueError):
                                pass
        except Exception:
            pass

    # Resolve runtime credentials for probing
    try:
        runtime = resolve_runtime_agent_kwargs()
        provider = provider or runtime.get("provider")
        base_url = base_url or runtime.get("base_url")
        api_key = runtime.get("api_key")
    except Exception:
        pass

    context_length = get_model_context_length(
        model,
        base_url=base_url or "",
        api_key=api_key or "",
        config_context_length=config_context_length,
        provider=provider or "",
        custom_providers=custom_provs,
    )

    # Format context source hint
    if config_context_length is not None:
        ctx_source = "config"
    elif context_length == DEFAULT_FALLBACK_CONTEXT:
        ctx_source = "default — set model.context_length in config to override"
    else:
        ctx_source = "detected"

    # Format context length for display
    if context_length >= 1_000_000:
        ctx_display = f"{context_length / 1_000_000:.1f}M"
    elif context_length >= 1_000:
        ctx_display = f"{context_length // 1_000}K"
    else:
        ctx_display = str(context_length)

    lines = [
        f"◆ Model: `{model}`",
        f"◆ Provider: {provider or 'openrouter'}",
        f"◆ Context: {ctx_display} tokens ({ctx_source})",
    ]

    # Show endpoint for local/custom setups
    if base_url and ("localhost" in base_url or "127.0.0.1" in base_url or "0.0.0.0" in base_url):
        lines.append(f"◆ Endpoint: {base_url}")

    return "\n".join(lines)


def begin_session_run_generation(
    runner,  # GatewayRunner instance
    session_key: str,
) -> int:
    """Begin a new run generation for a session.

    Returns the new generation number.
    """
    generations = runner.__dict__.get("_session_run_generation") or {}
    current = generations.get(session_key, 0)
    next_generation = current + 1
    generations[session_key] = next_generation
    runner._session_run_generation = generations
    return next_generation


def invalidate_session_run_generation(
    runner,  # GatewayRunner instance
    session_key: str,
    *,
    reason: str = "",
) -> int:
    """Invalidate any in-flight run token for ``session_key``."""
    generation = begin_session_run_generation(runner, session_key)
    if reason:
        logger.info(
            "Invalidated run generation for %s → %d (%s)",
            session_key,
            generation,
            reason,
        )
    return generation


def is_session_run_current(
    runner,  # GatewayRunner instance
    session_key: str,
    generation: int,
) -> bool:
    """Return True when ``generation`` is still current for ``session_key``."""
    if not session_key or generation is None:
        return True
    generations = runner.__dict__.get("_session_run_generation") or {}
    return int(generations.get(session_key, 0)) == int(generation)
