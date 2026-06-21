"""Gateway /status session-info formatting extracted from GatewayRunner.

Round 13 of gateway decomposition. _format_session_info resolves the active
model/provider/context and renders the info block shown by /status. Pure of
instance state — reads config via run.py module helpers (lazy-imported).
"""

import logging
logger = logging.getLogger("gateway.run")


def _format_session_info() -> str:
    """Resolve current model config and return a formatted info block.

    Surfaces model, provider, context length, and endpoint so gateway
    users can immediately see if context detection went wrong (e.g.
    local models falling to the 128K default).
    """
    from gateway.run import _resolve_gateway_model, _load_gateway_config, _resolve_runtime_agent_kwargs
    from agent.model_metadata import get_model_context_length, DEFAULT_FALLBACK_CONTEXT

    model = _resolve_gateway_model()
    config_context_length = None
    provider = None
    base_url = None
    api_key = None
    custom_provs = None
    data = None

    try:
        data = _load_gateway_config()
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
        runtime = _resolve_runtime_agent_kwargs()
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
