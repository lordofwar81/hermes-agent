"""
Gateway command handlers.

Extracted command handlers from GatewayRunner to reduce module size.
Each handler is a module-level function that takes the runner instance as first parameter.
"""

import json
import logging
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from agent.i18n import t
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.platforms.base import MessageType
from hermes_cli.config import cfg_get, is_managed, format_managed_message
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Sentinel for pending agent state (mirrors gateway/run.py)
_AGENT_PENDING_SENTINEL = object()


def _load_gateway_config():
    """Load gateway config from hermes home."""
    from hermes_constants import hermes_home
    from hermes_cli.config import load_config

    return load_config(hermes_home())


async def handle_model_command(runner, event: MessageEvent) -> Optional[str]:
    """Handle /model command — switch model for this session.

    Supports:
      /model                              — interactive picker (Telegram/Discord) or text list
      /model <name>                       — switch for this session only
      /model <name> --global              — switch and persist to config.yaml
      /model <name> --provider <provider> — switch provider + model
      /model --provider <provider>        — switch to provider, auto-detect model
    """
    import yaml

    from hermes_cli.model_switch import (
        switch_model as _switch_model,
        parse_model_flags,
        list_authenticated_providers,
        list_picker_providers,
    )
    from hermes_cli.providers import get_label

    raw_args = event.get_command_args().strip()

    # Parse --provider, --global, and --refresh flags
    model_input, explicit_provider, persist_global, force_refresh = parse_model_flags(
        raw_args
    )

    # --refresh: bust the disk cache so the picker shows live data.
    if force_refresh:
        try:
            from hermes_cli.models import clear_provider_models_cache

            clear_provider_models_cache()
        except Exception:
            pass

    # Read current model/provider from config
    current_model = ""
    current_provider = "openrouter"
    current_base_url = ""
    current_api_key = ""
    user_provs = None
    custom_provs = None

    from hermes_constants import hermes_home

    _hermes_home = hermes_home()
    config_path = _hermes_home / "config.yaml"

    try:
        cfg = _load_gateway_config()
        if cfg:
            model_cfg = cfg.get("model", {})
            if isinstance(model_cfg, dict):
                current_model = model_cfg.get("default", "")
                current_provider = model_cfg.get("provider", current_provider)
                current_base_url = model_cfg.get("base_url", "")
            user_provs = cfg.get("providers")
            try:
                from hermes_cli.config import get_compatible_custom_providers

                custom_provs = get_compatible_custom_providers(cfg)
            except Exception:
                custom_provs = cfg.get("custom_providers")
    except Exception:
        pass

    # Check for session override
    source = event.source
    session_key = runner._session_key_for_source(source)
    override = runner._session_model_overrides.get(session_key, {})
    if override:
        current_model = override.get("model", current_model)
        current_provider = override.get("provider", current_provider)
        current_base_url = override.get("base_url", current_base_url)
        current_api_key = override.get("api_key", current_api_key)

    # No args: show interactive picker (Telegram/Discord) or text list
    if not model_input and not explicit_provider:
        # Try interactive picker if the platform supports it
        adapter = runner.adapters.get(source.platform)
        has_picker = (
            adapter is not None
            and getattr(type(adapter), "send_model_picker", None) is not None
        )

        if has_picker:
            try:
                providers = list_picker_providers(
                    current_provider=current_provider,
                    current_base_url=current_base_url,
                    current_model=current_model,
                    user_providers=user_provs,
                    custom_providers=custom_provs,
                    max_models=50,
                )
            except Exception:
                providers = []

            if providers:
                # Build a callback closure for when the user picks a model.
                # Captures runner + locals needed for the switch logic.
                _session_key = session_key
                _cur_model = current_model
                _cur_provider = current_provider
                _cur_base_url = current_base_url
                _cur_api_key = current_api_key

                async def _on_model_selected(
                    _chat_id: str, model_id: str, provider_slug: str
                ) -> str:
                    """Perform the model switch and return confirmation text."""
                    result = _switch_model(
                        raw_input=model_id,
                        current_provider=_cur_provider,
                        current_model=_cur_model,
                        current_base_url=_cur_base_url,
                        current_api_key=_cur_api_key,
                        is_global=False,
                        explicit_provider=provider_slug,
                        user_providers=user_provs,
                        custom_providers=custom_provs,
                    )
                    if not result.success:
                        return t("gateway.model.error_prefix", error=result.error_message)

                    # Update cached agent in-place
                    cached_entry = None
                    _cache_lock = getattr(runner, "_agent_cache_lock", None)
                    _cache = getattr(runner, "_agent_cache", None)
                    if _cache_lock and _cache is not None:
                        with _cache_lock:
                            cached_entry = _cache.get(_session_key)
                    if cached_entry and cached_entry[0] is not None:
                        try:
                            cached_entry[0].switch_model(
                                new_model=result.new_model,
                                new_provider=result.target_provider,
                                api_key=result.api_key,
                                base_url=result.base_url,
                                api_mode=result.api_mode,
                            )
                        except Exception as exc:
                            logger.warning(
                                "Picker model switch failed for cached agent: %s", exc
                            )

                    # Persist the new model to the session DB so the
                    # dashboard shows the updated model (#34850).
                    _sess_db = getattr(runner, "_session_db", None)
                    if _sess_db is not None:
                        try:
                            _sess_entry = runner.session_store.get_or_create_session(
                                event.source
                            )
                            _sess_db.update_session_model(
                                _sess_entry.session_id, result.new_model
                            )
                        except Exception as exc:
                            logger.debug("Failed to persist model switch to DB: %s", exc)

                    # Store model note + session override
                    if not hasattr(runner, "_pending_model_notes"):
                        runner._pending_model_notes = {}
                    runner._pending_model_notes[_session_key] = (
                        f"[Note: model was just switched from {_cur_model} to {result.new_model} "
                        f"via {result.provider_label or result.target_provider}. "
                        f"Adjust your self-identification accordingly.]"
                    )
                    runner._session_model_overrides[_session_key] = {
                        "model": result.new_model,
                        "provider": result.target_provider,
                        "api_key": result.api_key,
                        "base_url": result.base_url,
                        "api_mode": result.api_mode,
                    }

                    # Evict cached agent so the next turn creates a fresh
                    # agent from the override rather than relying on the
                    # stale cache signature to trigger a rebuild.
                    runner._evict_cached_agent(_session_key)

                    # Build confirmation text
                    plabel = result.provider_label or result.target_provider
                    lines = [t("gateway.model.switched", model=result.new_model)]
                    lines.append(t("gateway.model.provider_label", provider=plabel))
                    mi = result.model_info
                    from hermes_cli.model_switch import resolve_display_context_length

                    _sw_config_ctx = None
                    try:
                        _sw_cfg = _load_gateway_config()
                        _sw_model_cfg = _sw_cfg.get("model", {})
                        if isinstance(_sw_model_cfg, dict):
                            _sw_raw = _sw_model_cfg.get("context_length")
                            if _sw_raw is not None:
                                _sw_config_ctx = int(_sw_raw)
                    except Exception:
                        pass
                    ctx = resolve_display_context_length(
                        result.new_model,
                        result.target_provider,
                        base_url=result.base_url or current_base_url or "",
                        api_key=result.api_key or current_api_key or "",
                        model_info=mi,
                        custom_providers=custom_provs,
                        config_context_length=_sw_config_ctx,
                    )
                    if ctx:
                        lines.append(t("gateway.model.context_label", tokens=f"{ctx:,}"))
                    if mi:
                        if mi.max_output:
                            lines.append(
                                t(
                                    "gateway.model.max_output_label",
                                    tokens=f"{mi.max_output:,}",
                                )
                            )
                        if mi.has_cost_data():
                            lines.append(
                                t("gateway.model.cost_label", cost=mi.format_cost())
                            )
                        lines.append(
                            t(
                                "gateway.model.capabilities_label",
                                capabilities=mi.format_capabilities(),
                            )
                        )
                    lines.append(t("gateway.model.session_only_hint"))
                    return "\n".join(lines)

                metadata = runner._thread_metadata_for_source(
                    source, runner._reply_anchor_for_event(event)
                )
                result = await adapter.send_model_picker(
                    chat_id=source.chat_id,
                    providers=providers,
                    current_model=current_model,
                    current_provider=current_provider,
                    session_key=session_key,
                    on_model_selected=_on_model_selected,
                    metadata=metadata,
                )
                if result.success:
                    return None  # Picker sent — adapter handles the response

        # Fallback: text list (for platforms without picker or if picker failed)
        provider_label = get_label(current_provider)
        lines = [
            t(
                "gateway.model.current_label",
                model=current_model or "unknown",
                provider=provider_label,
            ),
            "",
        ]

        try:
            providers = list_authenticated_providers(
                current_provider=current_provider,
                current_base_url=current_base_url,
                current_model=current_model,
                user_providers=user_provs,
                custom_providers=custom_provs,
                max_models=5,
            )
            for p in providers:
                tag = t("gateway.model.current_tag") if p["is_current"] else ""
                lines.append(f"**{p['name']}** `--provider {p['slug']}`{tag}:")
                if p["models"]:
                    model_strs = ", ".join(f"`{m}`" for m in p["models"])
                    extra = (
                        t(
                            "gateway.model.more_models_suffix",
                            count=p["total_models"] - len(p["models"]),
                        )
                        if p["total_models"] > len(p["models"])
                        else ""
                    )
                    lines.append(f"  {model_strs}{extra}")
                elif p.get("api_url"):
                    lines.append(f"  `{p['api_url']}`")
                lines.append("")
        except Exception:
            pass

        lines.append(t("gateway.model.usage_switch_model"))
        lines.append(t("gateway.model.usage_switch_provider"))
        lines.append(t("gateway.model.usage_persist"))
        return "\n".join(lines)

    # Perform the switch
    result = _switch_model(
        raw_input=model_input,
        current_provider=current_provider,
        current_model=current_model,
        current_base_url=current_base_url,
        current_api_key=current_api_key,
        is_global=persist_global,
        explicit_provider=explicit_provider,
        user_providers=user_provs,
        custom_providers=custom_provs,
    )

    if not result.success:
        return t("gateway.model.error_prefix", error=result.error_message)

    # If there's a cached agent, update it in-place
    cached_entry = None
    _cache_lock = getattr(runner, "_agent_cache_lock", None)
    _cache = getattr(runner, "_agent_cache", None)
    if _cache_lock and _cache is not None:
        with _cache_lock:
            cached_entry = _cache.get(session_key)

    if cached_entry and cached_entry[0] is not None:
        try:
            cached_entry[0].switch_model(
                new_model=result.new_model,
                new_provider=result.target_provider,
                api_key=result.api_key,
                base_url=result.base_url,
                api_mode=result.api_mode,
            )
        except Exception as exc:
            logger.warning("In-place model switch failed for cached agent: %s", exc)

    # Persist the new model to the session DB so the dashboard
    # shows the updated model (#34850).
    _sess_db = getattr(runner, "_session_db", None)
    if _sess_db is not None:
        try:
            _sess_entry = runner.session_store.get_or_create_session(source)
            _sess_db.update_session_model(_sess_entry.session_id, result.new_model)
        except Exception as exc:
            logger.debug("Failed to persist model switch to DB: %s", exc)

    # Store a note to prepend to the next user message so the model
    # knows about the switch (avoids system messages mid-history).
    if not hasattr(runner, "_pending_model_notes"):
        runner._pending_model_notes = {}
    runner._pending_model_notes[session_key] = (
        f"[Note: model was just switched from {current_model} to {result.new_model} "
        f"via {result.provider_label or result.target_provider}. "
        f"Adjust your self-identification accordingly.]"
    )

    # Store session override so next agent creation uses the new model
    runner._session_model_overrides[session_key] = {
        "model": result.new_model,
        "provider": result.target_provider,
        "api_key": result.api_key,
        "base_url": result.base_url,
        "api_mode": result.api_mode,
    }

    # Evict cached agent so the next turn creates a fresh agent from the
    # override rather than relying on cache signature mismatch detection.
    runner._evict_cached_agent(session_key)

    # Persist to config if --global
    if persist_global:
        try:
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
            else:
                cfg = {}
            # Coerce scalar/None ``model:`` into a dict before mutation —
            # otherwise ``cfg.setdefault("model", {})`` returns the existing
            # scalar and the next assignment raises
            # ``TypeError: 'str' object does not support item assignment``.
            # Reproduces when ``config.yaml`` has ``model: <name>`` (flat
            # string) instead of the proper nested ``model: {default: ...}``.
            raw_model = cfg.get("model")
            if isinstance(raw_model, dict):
                model_cfg = raw_model
            elif isinstance(raw_model, str) and raw_model.strip():
                model_cfg = {"default": raw_model.strip()}
                cfg["model"] = model_cfg
            else:
                model_cfg = {}
                cfg["model"] = model_cfg
            model_cfg["default"] = result.new_model
            model_cfg["provider"] = result.target_provider
            if result.base_url:
                model_cfg["base_url"] = result.base_url
            from hermes_cli.config import save_config

            save_config(cfg)
        except Exception as e:
            logger.warning("Failed to persist model switch: %s", e)

    # Build confirmation message with full metadata
    provider_label = result.provider_label or result.target_provider
    lines = [t("gateway.model.switched", model=result.new_model)]
    lines.append(t("gateway.model.provider_label", provider=provider_label))

    # Context: always resolve via the provider-aware chain so Codex OAuth,
    # Copilot, and Nous-enforced caps win over the raw models.dev entry.
    mi = result.model_info
    from hermes_cli.model_switch import resolve_display_context_length

    _sw2_config_ctx = None
    try:
        _sw2_cfg = _load_gateway_config()
        _sw2_model_cfg = _sw2_cfg.get("model", {})
        if isinstance(_sw2_model_cfg, dict):
            _sw2_raw = _sw2_model_cfg.get("context_length")
            if _sw2_raw is not None:
                _sw2_config_ctx = int(_sw2_raw)
    except Exception:
        pass
    ctx = resolve_display_context_length(
        result.new_model,
        result.target_provider,
        base_url=result.base_url or current_base_url or "",
        api_key=result.api_key or current_api_key or "",
        model_info=mi,
        custom_providers=custom_provs,
        config_context_length=_sw2_config_ctx,
    )
    if ctx:
        lines.append(t("gateway.model.context_label", tokens=f"{ctx:,}"))
    if mi:
        if mi.max_output:
            lines.append(
                t("gateway.model.max_output_label", tokens=f"{mi.max_output:,}")
            )
        if mi.has_cost_data():
            lines.append(t("gateway.model.cost_label", cost=mi.format_cost()))
        lines.append(
            t("gateway.model.capabilities_label", capabilities=mi.format_capabilities())
        )

    # Cache notice
    from utils import base_url_host_matches

    cache_enabled = (
        (base_url_host_matches(result.base_url or "", "openrouter.ai")
        and "claude" in result.new_model.lower())
        or result.api_mode == "anthropic_messages"
    )
    if cache_enabled:
        lines.append(t("gateway.model.prompt_caching_enabled"))

    if result.warning_message:
        lines.append(t("gateway.model.warning_prefix", warning=result.warning_message))

    if persist_global:
        lines.append(t("gateway.model.saved_global"))
    else:
        lines.append(t("gateway.model.session_only_hint"))

    return "\n".join(lines)


async def handle_compress_command(runner, event: MessageEvent) -> str:
    """Handle /compress command -- manually compress conversation context.

    Accepts an optional focus topic: ``/compress <focus>`` guides the
    summariser to preserve information related to *focus* while being
    more aggressive about discarding everything else.

    Also accepts the boundary-aware form ``/compress here [N]``:
    summarize everything except the most recent ``N`` exchanges
    (default 2), kept verbatim. Inspired by Claude Code's Rewind
    "Summarize up to here" action (v2.1.139, May 2026,
    https://code.claude.com/docs/en/whats-new/2026-w20).
    """
    import asyncio
    from logging import getLogger

    from hermes_cli.partial_compress import (
        parse_partial_compress_args,
        rejoin_compressed_head_and_tail,
        split_history_for_partial_compress,
    )

    logger = getLogger(__name__)

    source = event.source
    session_entry = runner.session_store.get_or_create_session(source)
    history = runner.session_store.load_transcript(session_entry.session_id)

    if not history or len(history) < 4:
        from hermes_cli.i18n import t
        return t("gateway.compress.not_enough")

    # Parse args: either a focus topic (full compress) or the
    # boundary-aware "here [N]" form (partial compress).
    _raw_args = (event.get_command_args() or "").strip()
    partial, keep_last, focus_topic = parse_partial_compress_args(_raw_args)

    try:
        from run_agent import AIAgent
        from agent.manual_compression_feedback import summarize_manual_compression
        from agent.model_metadata import estimate_request_tokens_rough
        from hermes_cli.i18n import t

        session_key = runner._session_key_for_source(source)
        model, runtime_kwargs = runner._resolve_session_agent_runtime(
            source=source,
            session_key=session_key,
        )
        if not runtime_kwargs.get("api_key"):
            return t("gateway.compress.no_provider")

        msgs = [
            {"role": m.get("role"), "content": m.get("content")}
            for m in history
            if m.get("role") in {"user", "assistant"} and m.get("content")
        ]

        # Boundary-aware split: only the head is summarized; the most
        # recent `keep_last` exchanges are preserved verbatim. The
        # split snaps the tail to a user-turn start so the rejoined
        # transcript keeps role alternation valid.
        tail: list = []
        head = msgs
        if partial:
            head, tail = split_history_for_partial_compress(msgs, keep_last)
            if not tail:
                # Degenerate split — fall back to full compression.
                partial = False
                head = msgs

        tmp_agent = AIAgent(
            **runtime_kwargs,
            model=model,
            max_iterations=4,
            quiet_mode=True,
            skip_memory=True,
            enabled_toolsets=["memory"],
            session_id=session_entry.session_id,
        )
        try:
            tmp_agent._print_fn = lambda *a, **kw: None

            # Estimate with system prompt + tool schemas included so the
            # figure reflects real request pressure, not a transcript-only
            # underestimate (#6217). Must be computed after tmp_agent is
            # built so _cached_system_prompt/tools are populated.
            _sys_prompt = getattr(tmp_agent, "_cached_system_prompt", "") or ""
            _tools = getattr(tmp_agent, "tools", None) or None
            approx_tokens = estimate_request_tokens_rough(
                msgs, system_prompt=_sys_prompt, tools=_tools
            )

            compressor = tmp_agent.context_compressor
            if not compressor.has_content_to_compress(head):
                return t("gateway.compress.nothing_to_do")

            loop = asyncio.get_running_loop()
            compressed, _ = await loop.run_in_executor(
                None,
                lambda: tmp_agent._compress_context(head, "", approx_tokens=approx_tokens, focus_topic=focus_topic, force=True)
            )

            # Re-append the verbatim tail after the compressed head,
            # guarding the seam against illegal role adjacency.
            if partial and tail:
                compressed = rejoin_compressed_head_and_tail(compressed, tail)

            # _compress_context already calls end_session() on the old session
            # (preserving its full transcript in SQLite) and creates a new
            # session_id for the continuation.  Write the compressed messages
            # into the NEW session so the original history stays searchable.
            new_session_id = tmp_agent.session_id
            if new_session_id != session_entry.session_id:
                session_entry.session_id = new_session_id
                runner.session_store._save()
                runner._sync_telegram_topic_binding(
                    source, session_entry, reason="compress-command",
                )

            runner.session_store.rewrite_transcript(new_session_id, compressed)
            # Reset stored token count — transcript changed, old value is stale
            runner.session_store.update_session(
                session_entry.session_key, last_prompt_tokens=0
            )
            new_tokens = estimate_request_tokens_rough(
                compressed, system_prompt=_sys_prompt, tools=_tools
            )
            summary = summarize_manual_compression(
                msgs,
                compressed,
                approx_tokens,
                new_tokens,
            )
            # Detect summary-generation failure so we can surface a
            # visible warning to the user even on the manual /compress
            # path (otherwise the failure is silently logged).
            # _last_compress_aborted means the aux LLM returned no
            # usable summary and the compressor preserved messages
            # unchanged (no drop, no placeholder).  force=True was
            # passed above so any active cooldown is bypassed.
            _summary_aborted = bool(getattr(compressor, "_last_compress_aborted", False))
            _summary_err = getattr(compressor, "_last_summary_error", None)
            # Separately: did the user's CONFIGURED aux model fail
            # and we recovered via main?  Surface that as an info
            # note so they can fix their config.
            _aux_fail_model = getattr(compressor, "_last_aux_model_failure_model", None)
            _aux_fail_err = getattr(compressor, "_last_aux_model_failure_error", None)
        finally:
            # Evict cached agent so next turn rebuilds system prompt
            # from current files (SOUL.md, memory, etc.).
            runner._evict_cached_agent(session_key)
            runner._cleanup_agent_resources(tmp_agent)
        lines = [f"🗜️ {summary['headline']}"]
        if focus_topic:
            lines.append(t("gateway.compress.focus_line", topic=focus_topic))
        lines.append(summary["token_line"])
        if summary["note"]:
            lines.append(summary["note"])
        if _summary_aborted:
            lines.append(
                t(
                    "gateway.compress.aborted",
                    error=(_summary_err or "unknown error"),
                )
            )
        elif _aux_fail_model:
            lines.append(
                t(
                    "gateway.compress.aux_failed",
                    model=_aux_fail_model,
                    error=(_aux_fail_err or "unknown error"),
                )
            )
        return "\n".join(lines)
    except Exception as e:
        from hermes_cli.i18n import t
        logger.warning("Manual compress failed: %s", e)
        return t("gateway.compress.failed", error=e)


async def handle_reset_command(runner, event) -> str:
    """Handle /new or /reset command.

    Args:
        runner: GatewayRunner instance
        event: MessageEvent containing the command and context

    Returns:
        EphemeralReply: Response message
    """
    from gateway.platforms.base import EphemeralReply
    from hermes_cli.i18n import t

    source = event.source

    # Get existing session key
    session_key = runner._session_key_for_source(source)
    runner._invalidate_session_run_generation(session_key, reason="session_reset")

    # Snapshot the old entry so on_session_finalize can report the
    # expiring session id before reset_session() rotates it.
    old_entry = runner.session_store._entries.get(session_key)

    # Close tool resources on the old agent (terminal sandboxes, browser
    # daemons, background processes) before evicting from cache.
    # Guard with getattr because test fixtures may skip __init__.
    _cache_lock = getattr(runner, "_agent_cache_lock", None)
    if _cache_lock is not None:
        with _cache_lock:
            _cached = runner._agent_cache.get(session_key)
            _old_agent = _cached[0] if isinstance(_cached, tuple) else _cached if _cached else None
        if _old_agent is not None:
            runner._cleanup_agent_resources(_old_agent)
    runner._evict_cached_agent(session_key)

    # Discard any /queue overflow for this session — /new is a
    # conversation-boundary operation, queued follow-ups from the
    # previous conversation must not bleed into the new one.
    _qe = getattr(runner, "_queued_events", None)
    if _qe is not None:
        _qe.pop(session_key, None)

    try:
        from tools.env_passthrough import clear_env_passthrough
        clear_env_passthrough()
    except Exception:
        pass

    try:
        from tools.credential_files import clear_credential_files
        clear_credential_files()
    except Exception:
        pass

    # Reset the session
    new_entry = runner.session_store.reset_session(session_key)

    # Clear any session-scoped model/reasoning overrides so the next agent
    # picks up configured defaults instead of previous session switches.
    runner._session_model_overrides.pop(session_key, None)
    runner._set_session_reasoning_override(session_key, None)
    if hasattr(runner, "_pending_model_notes"):
        runner._pending_model_notes.pop(session_key, None)

    # Clear session-scoped dangerous-command approvals and /yolo state.
    # /new is a conversation-boundary operation — approval state from the
    # previous conversation must not survive the reset.
    runner._clear_session_boundary_security_state(session_key)

    _old_sid = old_entry.session_id if old_entry else None

    # Fire plugin on_session_finalize hook (session boundary)
    try:
        from hermes_cli.plugins import invoke_hook as _invoke_hook
        _invoke_hook(
            "on_session_finalize",
            session_id=_old_sid,
            platform=source.platform.value if source.platform else "",
            reason="new_session",
            old_session_id=_old_sid,
            new_session_id=new_entry.session_id if new_entry else None,
        )
    except Exception:
        pass

    # Emit session:end hook (session is ending)
    await runner.hooks.emit("session:end", {
        "platform": source.platform.value if source.platform else "",
        "user_id": source.user_id,
        "session_key": session_key,
    })

    # Emit session:reset hook
    await runner.hooks.emit("session:reset", {
        "platform": source.platform.value if source.platform else "",
        "user_id": source.user_id,
        "session_key": session_key,
    })

    # Resolve session config info to surface to the user
    try:
        session_info = runner._format_session_info()
    except Exception:
        session_info = ""

    if new_entry:
        header = runner._telegram_topic_new_header(source) or t("gateway.reset.header_default")
    else:
        # No existing session, just create one
        new_entry = runner.session_store.get_or_create_session(source, force_new=True)
        header = runner._telegram_topic_new_header(source) or t("gateway.reset.header_new")

    # Set session title if provided with /new <title>
    _title_arg = event.get_command_args().strip()
    _title_note = ""
    if _title_arg and runner._session_db and new_entry:
        from hermes_state import SessionDB
        try:
            sanitized = SessionDB.sanitize_title(_title_arg)
        except ValueError as e:
            sanitized = None
            _title_note = t("gateway.reset.title_rejected", error=str(e))
        if sanitized:
            try:
                runner._session_db.set_session_title(new_entry.session_id, sanitized)
                header = t("gateway.reset.header_titled", title=sanitized)
            except ValueError as e:
                _title_note = t("gateway.reset.title_error_untitled", error=str(e))
            except Exception:
                pass
        elif not _title_note:
            # sanitize_title returned empty (whitespace-only / unprintable)
            _title_note = t("gateway.reset.title_empty_untitled")
    header = header + _title_note

    # When /new runs inside a Telegram DM topic lane, rewrite the
    # (chat_id, thread_id) → session_id binding so the next message
    # uses the freshly-created session. Without this, the binding
    # still points at the old session and the binding-lookup at the
    # top of _handle_message_with_agent would switch right back.
    if runner._is_telegram_topic_lane(source) and new_entry is not None:
        try:
            runner._record_telegram_topic_binding(source, new_entry)
        except Exception:
            from gateway.logging import logger
            logger.debug("Failed to rebind Telegram topic after /new", exc_info=True)

    # Fire plugin on_session_reset hook (new session guaranteed to exist)
    try:
        from hermes_cli.plugins import invoke_hook as _invoke_hook
        _new_sid = new_entry.session_id if new_entry else None
        _invoke_hook(
            "on_session_reset",
            session_id=_new_sid,
            platform=source.platform.value if source.platform else "",
            reason="new_session",
            old_session_id=_old_sid,
            new_session_id=_new_sid,
        )
    except Exception:
        pass

    # Append a random tip to the reset message
    try:
        from hermes_cli.tips import get_random_tip
        _tip_line = t("gateway.reset.tip", tip=get_random_tip())
    except Exception:
        _tip_line = ""

    if session_info:
        return EphemeralReply(f"{header}\n\n{session_info}{_tip_line}")
    return EphemeralReply(f"{header}{_tip_line}")
async def handle_usage_command(runner, event) -> str:
    """Handle /usage command -- show token usage for the current session.

    Checks both _running_agents (mid-turn) and _agent_cache (between turns)
    so that rate limits, cost estimates, and detailed token breakdowns are
    available whenever the user asks, not only while the agent is running.
    """
    import asyncio

    # Import at function level to avoid circular imports and follow gateway/run.py pattern
    from agent.account_usage import fetch_account_usage, render_account_usage_lines
    from agent.rate_limit_tracker import format_rate_limit_compact
    from agent.usage_pricing import CanonicalUsage, estimate_usage_cost
    from agent.i18n import t
    from agent.model_metadata import estimate_messages_tokens_rough
    source = event.source
    session_key = runner._session_key_for_source(source)

    # Try running agent first (mid-turn), then cached agent (between turns)
    agent = runner._running_agents.get(session_key)
    if not agent or agent is _AGENT_PENDING_SENTINEL:
        _cache_lock = getattr(runner, "_agent_cache_lock", None)
        _cache = getattr(runner, "_agent_cache", None)
        if _cache_lock and _cache is not None:
            with _cache_lock:
                cached = _cache.get(session_key)
                if cached:
                    agent = cached[0]

    # Resolve provider/base_url/api_key for the account-usage fetch.
    # Prefer the live agent; fall back to persisted billing data on the
    # SessionDB row so `/usage` still returns account info between turns
    # when no agent is resident.
    provider = getattr(agent, "provider", None) if agent and agent is not _AGENT_PENDING_SENTINEL else None
    base_url = getattr(agent, "base_url", None) if agent and agent is not _AGENT_PENDING_SENTINEL else None
    api_key = getattr(agent, "api_key", None) if agent and agent is not _AGENT_PENDING_SENTINEL else None
    if not provider and getattr(runner, "_session_db", None) is not None:
        try:
            _entry_for_billing = runner.session_store.get_or_create_session(source)
            persisted = runner._session_db.get_session(_entry_for_billing.session_id) or {}
        except Exception:
            persisted = {}
        provider = provider or persisted.get("billing_provider")
        base_url = base_url or persisted.get("billing_base_url")

    # Fetch account usage off the event loop so slow provider APIs don't
    # block the gateway. Failures are non-fatal -- account_lines stays [].
    account_lines: list[str] = []
    if provider:
        try:
            account_snapshot = await asyncio.to_thread(
                fetch_account_usage,
                provider,
                base_url=base_url,
                api_key=api_key,
            )
        except Exception:
            account_snapshot = None
        if account_snapshot:
            account_lines = render_account_usage_lines(account_snapshot, markdown=True)

    if agent and hasattr(agent, "session_total_tokens") and agent.session_api_calls > 0:
        lines = []

        # Rate limits (when available from provider headers)
        rl_state = agent.get_rate_limit_state()
        if rl_state and rl_state.has_data:
            from agent.rate_limit_tracker import format_rate_limit_compact
            lines.append(t("gateway.usage.rate_limits", state=format_rate_limit_compact(rl_state)))
            lines.append("")

        # Session token usage — detailed breakdown matching CLI
        input_tokens = getattr(agent, "session_input_tokens", 0) or 0
        output_tokens = getattr(agent, "session_output_tokens", 0) or 0
        cache_read = getattr(agent, "session_cache_read_tokens", 0) or 0
        cache_write = getattr(agent, "session_cache_write_tokens", 0) or 0

        lines.append(t("gateway.usage.header_session"))
        lines.append(t("gateway.usage.label_model", model=agent.model))
        lines.append(t("gateway.usage.label_input_tokens", count=f"{input_tokens:,}"))
        if cache_read:
            lines.append(t("gateway.usage.label_cache_read", count=f"{cache_read:,}"))
        if cache_write:
            lines.append(t("gateway.usage.label_cache_write", count=f"{cache_write:,}"))
        lines.append(t("gateway.usage.label_output_tokens", count=f"{output_tokens:,}"))
        lines.append(t("gateway.usage.label_total", count=f"{agent.session_total_tokens:,}"))
        lines.append(t("gateway.usage.label_api_calls", count=agent.session_api_calls))

        # Cost estimation
        try:
            from agent.usage_pricing import CanonicalUsage, estimate_usage_cost
            cost_result = estimate_usage_cost(
                agent.model,
                CanonicalUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read,
                    cache_write_tokens=cache_write,
                ),
                provider=getattr(agent, "provider", None),
                base_url=getattr(agent, "base_url", None),
            )
            if cost_result.amount_usd is not None:
                prefix = "~" if cost_result.status == "estimated" else ""
                lines.append(t("gateway.usage.label_cost", prefix=prefix, amount=f"{float(cost_result.amount_usd):.4f}"))
            elif cost_result.status == "included":
                lines.append(t("gateway.usage.label_cost_included"))
        except Exception:
            pass

        # Context window and compressions
        ctx = agent.context_compressor
        if ctx.last_prompt_tokens:
            pct = min(100, ctx.last_prompt_tokens / ctx.context_length * 100) if ctx.context_length else 0
            lines.append(t("gateway.usage.label_context", used=f"{ctx.last_prompt_tokens:,}", total=f"{ctx.context_length:,}", pct=f"{pct:.0f}"))
        if ctx.compression_count:
            lines.append(t("gateway.usage.label_compressions", count=ctx.compression_count))

        if account_lines:
            lines.append("")
            lines.extend(account_lines)

        return "\n".join(lines)

    # No agent at all -- check session history for a rough count
    session_entry = runner.session_store.get_or_create_session(source)
    history = runner.session_store.load_transcript(session_entry.session_id)
    if history:
        from agent.model_metadata import estimate_messages_tokens_rough
        msgs = [m for m in history if m.get("role") in {"user", "assistant"} and m.get("content")]
        approx = estimate_messages_tokens_rough(msgs)
        lines = [
            t("gateway.usage.header_session_info"),
            t("gateway.usage.label_messages", count=len(msgs)),
            t("gateway.usage.label_estimated_context", count=f"{approx:,}"),
            t("gateway.usage.detailed_after_first"),
        ]
        if account_lines:
            lines.append("")
            lines.extend(account_lines)
        return "\n".join(lines)
    if account_lines:
        return "\n".join(account_lines)
    return t("gateway.usage.no_data")


async def handle_reasoning_command(
    runner,
    event,
    hermes_home,
    atomic_yaml_write,
    platform_config_key,
    load_show_reasoning,
    load_reasoning_config,
    resolve_session_reasoning_config,
    session_key_for_source,
    set_session_reasoning_override,
    evict_cached_agent,
    parse_reasoning_command_args,
) -> str:
    """Handle /reasoning command — manage reasoning effort and display toggle.

    Usage:
        /reasoning                       Show current effort level and display state
        /reasoning <level>               Set reasoning effort for this session only
        /reasoning <level> --global      Persist reasoning effort to config.yaml
        /reasoning reset                 Clear this session's reasoning override
        /reasoning show|on               Show model reasoning in responses
        /reasoning hide|off              Hide model reasoning from responses

    Args:
        runner: GatewayRunner instance (for state)
        event: MessageEvent containing the command
        hermes_home: Path to hermes home directory
        atomic_yaml_write: Function to atomically write YAML
        platform_config_key: Function to get platform config key
        load_show_reasoning: Function to load show_reasoning setting
        load_reasoning_config: Function to load reasoning config
        resolve_session_reasoning_config: Function to resolve session reasoning config
        session_key_for_source: Function to get session key from source
        set_session_reasoning_override: Function to set session override
        evict_cached_agent: Function to evict cached agent
        parse_reasoning_command_args: Function to parse command args

    Returns:
        Response message to send to the user
    """
    import yaml

    raw_args = event.get_command_args().strip()
    args, persist_global = parse_reasoning_command_args(raw_args)
    config_path = hermes_home / "config.yaml"
    session_key = session_key_for_source(event.source)
    runner._show_reasoning = load_show_reasoning()
    runner._reasoning_config = resolve_session_reasoning_config(
        source=event.source,
        session_key=session_key,
    )

    def _save_config_key(key_path: str, value):
        """Save a dot-separated key to config.yaml."""
        try:
            user_config = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    user_config = yaml.safe_load(f) or {}
            keys = key_path.split(".")
            current = user_config
            for k in keys[:-1]:
                if k not in current or not isinstance(current[k], dict):
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
            atomic_yaml_write(config_path, user_config)
            return True
        except Exception as e:
            logger.error("Failed to save config key %s: %s", key_path, e)
            return False

    if not raw_args:
        # Show current state
        rc = runner._reasoning_config
        if rc is None:
            level = t("gateway.reasoning.level_default")
        elif rc.get("enabled") is False:
            level = t("gateway.reasoning.level_disabled")
        else:
            level = rc.get("effort", "medium")
        display_state = (
            t("gateway.reasoning.display_on")
            if runner._show_reasoning
            else t("gateway.reasoning.display_off")
        )
        has_session_override = session_key in (getattr(runner, "_session_reasoning_overrides", {}) or {})
        scope = (
            t("gateway.reasoning.scope_session")
            if has_session_override
            else t("gateway.reasoning.scope_global")
        )
        return t(
            "gateway.reasoning.status",
            level=level,
            scope=scope,
            display=display_state,
        )

    # Display toggle (per-platform)
    platform_key = platform_config_key(event.source.platform)
    if args in {"show", "on"}:
        runner._show_reasoning = True
        _save_config_key(f"display.platforms.{platform_key}.show_reasoning", True)
        return t("gateway.reasoning.display_set_on", platform=platform_key)

    if args in {"hide", "off"}:
        runner._show_reasoning = False
        _save_config_key(f"display.platforms.{platform_key}.show_reasoning", False)
        return t("gateway.reasoning.display_set_off", platform=platform_key)

    # Effort level change
    effort = args.strip()
    if effort == "reset":
        if persist_global:
            return t("gateway.reasoning.reset_global_unsupported")
        set_session_reasoning_override(session_key, None)
        runner._reasoning_config = load_reasoning_config()
        evict_cached_agent(session_key)
        return t("gateway.reasoning.reset_done")
    if effort == "none":
        parsed = {"enabled": False}
    elif effort in {"minimal", "low", "medium", "high", "xhigh"}:
        parsed = {"enabled": True, "effort": effort}
    else:
        return t(
            "gateway.reasoning.unknown_arg",
            arg=effort or raw_args.lower(),
        )

    runner._reasoning_config = parsed
    if persist_global:
        if _save_config_key("agent.reasoning_effort", effort):
            set_session_reasoning_override(session_key, None)
            evict_cached_agent(session_key)
            return t("gateway.reasoning.set_global", effort=effort)
        set_session_reasoning_override(session_key, parsed)
        evict_cached_agent(session_key)
        return t("gateway.reasoning.set_global_save_failed", effort=effort)

    set_session_reasoning_override(session_key, parsed)
    evict_cached_agent(session_key)
    return t("gateway.reasoning.set_session", effort=effort)



def _resolve_hermes_bin():
    """Resolve the hermes CLI binary path.

    Copied from gateway/run.py to avoid circular imports.
    """
    import shutil

    hermes_cmd = shutil.which("hermes")
    if hermes_cmd:
        # On Windows, use python to run the entry point
        if sys.platform == "win32":
            # Try to use the same Python interpreter that's running the gateway
            return [sys.executable, "-m", "hermes_cli"]
        return [hermes_cmd]

    # Fallback: use the current Python interpreter
    return [sys.executable, "-m", "hermes_cli"]


async def handle_update_command(runner, event: MessageEvent) -> str:
    """Handle /update command — update Hermes Agent to the latest version.

    Spawns ``hermes update`` in a detached session (via ``setsid``) so it
    survives the gateway restart that ``hermes update`` may trigger. Marker
    files are written so either the current gateway process or the next one
    can notify the user when the update finishes.

    Args:
        runner: GatewayRunner instance (for accessing methods and config)
        event: The message event that triggered the command

    Returns:
        Response message to send to the user
    """
    import json
    import shlex
    import shutil
    import subprocess

    # Block non-messaging platforms (API server, webhooks, ACP)
    platform = event.source.platform
    _allowed = runner._UPDATE_ALLOWED_PLATFORMS
    # Plugin platforms with allow_update_command=True are also allowed
    if platform not in _allowed:
        try:
            from gateway.platform_registry import platform_registry
            entry = platform_registry.get(platform.value)
            if not entry or not entry.allow_update_command:
                return t("gateway.update.platform_not_messaging")
        except Exception:
            return t("gateway.update.platform_not_messaging")

    if is_managed():
        return f"✗ {format_managed_message('update Hermes Agent')}"

    project_root = Path(__file__).parent.parent.resolve()
    git_dir = project_root / '.git'

    if not git_dir.exists():
        return t("gateway.update.not_git_repo")

    hermes_cmd = _resolve_hermes_bin()
    if not hermes_cmd:
        return t("gateway.update.hermes_cmd_not_found")

    pending_path = _hermes_home / ".update_pending.json"
    output_path = _hermes_home / ".update_output.txt"
    exit_code_path = _hermes_home / ".update_exit_code"
    session_key = runner._session_key_for_source(event.source)
    pending = {
        "platform": event.source.platform.value,
        "chat_id": event.source.chat_id,
        "chat_type": event.source.chat_type,
        "user_id": event.source.user_id,
        "session_key": session_key,
        "timestamp": datetime.now().isoformat(),
    }
    if event.source.thread_id:
        pending["thread_id"] = event.source.thread_id
    if event.message_id:
        pending["message_id"] = event.message_id
    _tmp_pending = pending_path.with_suffix(".tmp")
    _tmp_pending.write_text(json.dumps(pending))
    _tmp_pending.replace(pending_path)
    exit_code_path.unlink(missing_ok=True)

    # Spawn `hermes update --gateway` detached so it survives gateway restart.
    # --gateway enables file-based IPC for interactive prompts (stash
    # restore, config migration) so the gateway can forward them to the
    # user instead of silently skipping them.
    # Use setsid for portable session detach (works under system services
    # where systemd-run --user fails due to missing D-Bus session).
    # PYTHONUNBUFFERED ensures output is flushed line-by-line so the
    # gateway can stream it to the messenger in near-real-time.
    #
    # Windows: no bash/setsid chain.  Run `hermes update --gateway`
    # directly via sys.executable; redirect stdout/stderr to the same
    # output files via Popen file handles; write the exit code in a
    # follow-up write.  A tiny Python watcher would be cleaner but
    # we're already inside gateway/run.py's update path which is async,
    # so the simplest correct thing is: launch an inline Python helper
    # that runs the command and writes both outputs.
    try:
        if sys.platform == "win32":
            import textwrap
            from hermes_cli._subprocess_compat import windows_detach_popen_kwargs

            # hermes_cmd is a list of argv parts we can pass directly
            # (no shell-quoting needed).
            helper = textwrap.dedent(
                """
                import os, subprocess, sys
                output_path = sys.argv[1]
                exit_code_path = sys.argv[2]
                cmd = sys.argv[3:]
                env = dict(os.environ)
                env["PYTHONUNBUFFERED"] = "1"
                with open(output_path, "wb") as f:
                    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
                    rc = proc.wait()
                with open(exit_code_path, "w") as f:
                    f.write(str(rc))
                """
            ).strip()
            subprocess.Popen(
                [
                    sys.executable, "-c", helper,
                    str(output_path), str(exit_code_path),
                    *hermes_cmd, "update", "--gateway",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **windows_detach_popen_kwargs(),
            )
        else:
            hermes_cmd_str = " ".join(shlex.quote(part) for part in hermes_cmd)
            update_cmd = (
                f"PYTHONUNBUFFERED=1 {hermes_cmd_str} update --gateway"
                f" > {shlex.quote(str(output_path))} 2>&1; "
                # Avoid `status=$?`: `status` is a read-only special parameter
                # in zsh, and this command string is copied/reused in macOS/zsh
                # operator wrappers. Keep the template zsh-safe even though this
                # specific subprocess currently runs under bash.
                f"rc=$?; printf '%s' \"$rc\" > {shlex.quote(str(exit_code_path))}"
            )
            setsid_bin = shutil.which("setsid")
            if setsid_bin:
                # Preferred: setsid creates a new session, fully detached
                subprocess.Popen(
                    [setsid_bin, "bash", "-c", update_cmd],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            else:
                # Fallback: start_new_session=True calls os.setsid() in child
                subprocess.Popen(
                    ["bash", "-c", update_cmd],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
    except Exception as e:
        pending_path.unlink(missing_ok=True)
        exit_code_path.unlink(missing_ok=True)
        return t("gateway.update.start_failed", error=e)

    runner._schedule_update_notification_watch()
    return t("gateway.update.starting")


async def handle_codex_runtime_command(runner, event: "MessageEvent") -> str:
    """Handle /codex-runtime command in the gateway.

    Same surface as the CLI handler in cli.py:
        /codex-runtime                  — show current state
        /codex-runtime auto             — Hermes default runtime
        /codex-runtime codex_app_server — codex subprocess runtime
        /codex-runtime on / off         — synonyms

    On change, the cached agent for this session is evicted so the next
    message creates a fresh AIAgent with the new api_mode wired in
    (avoids prompt-cache invalidation mid-session)."""
    from hermes_cli import codex_runtime_switch as crs

    raw_args = event.get_command_args().strip() if event else ""
    new_value, errors = crs.parse_args(raw_args)
    if errors:
        return "❌ " + "\n❌ ".join(errors)

    # Load + persist via the same helpers used for /model and /yolo
    try:
        from hermes_cli.config import load_config, save_config
    except Exception as exc:
        return f"❌ Could not load config: {exc}"
    cfg = load_config()

    result = crs.apply(
        cfg,
        new_value,
        persist_callback=(save_config if new_value is not None else None),
    )

    # On a real change, evict the cached agent so the new runtime takes
    # effect on the next message rather than waiting for cache TTL.
    if result.success and new_value is not None and result.requires_new_session:
        try:
            session_key = runner._session_key_for_source(event.source)
            runner._evict_cached_agent(session_key)
        except Exception:
            logger.debug("could not evict cached agent after codex-runtime change",
                         exc_info=True)

    prefix = "✓" if result.success else "✗"
    return f"{prefix} {result.message}"


async def handle_retry_command(runner, event: "MessageEvent") -> str:
    """Handle /retry command - re-send the last user message."""
    from gateway.platforms.base import MessageType, MessageEvent
    from hermes_cli.i18n import t as _

    t = _
    source = event.source
    session_entry = runner.session_store.get_or_create_session(source)
    history = runner.session_store.load_transcript(session_entry.session_id)
    
    # Find the last user message
    last_user_msg = None
    last_user_idx = None
    for i in range(len(history) - 1, -1, -1):
        if history[i].get("role") == "user":
            last_user_msg = history[i].get("content", "")
            last_user_idx = i
            break
    
    if not last_user_msg:
        return t("gateway.retry.no_previous")
    
    # Truncate history to before the last user message and persist
    truncated = history[:last_user_idx]
    runner.session_store.rewrite_transcript(session_entry.session_id, truncated)
    # Reset stored token count — transcript was truncated
    session_entry.last_prompt_tokens = 0
    
    # Re-send by creating a fake text event with the old message
    retry_event = MessageEvent(
        text=last_user_msg,
        message_type=MessageType.TEXT,
        source=source,
        raw_message=event.raw_message,
        channel_prompt=event.channel_prompt,
    )
    
    # Let the normal message handler process it
    return await runner._handle_message(retry_event)


async def handle_reload_skills_command(runner, event: "MessageEvent") -> str:
    """Handle /reload-skills — rescan skills dir, queue a note for next turn.

    Skills don't need to be in the system prompt for the model to use
    them (they're invoked via ``/skill-name``, ``skills_list``, or
    ``skill_view`` at runtime), so this does NOT clear the prompt cache
    — prefix caching stays intact.

    If any skills were added or removed, a one-shot note is queued on
    ``runner._pending_skills_reload_notes[session_key]``. The gateway
    prepends it to the NEXT user message in this session, then clears it.
    """
    import inspect
    from agent.i18n import t as _

    t = _
    loop = asyncio.get_running_loop()
    try:
        from agent.skill_commands import reload_skills

        result = await loop.run_in_executor(None, reload_skills)
        added = result.get("added", [])      # [{"name", "description"}, ...]
        removed = result.get("removed", [])  # [{"name", "description"}, ...]
        total = result.get("total", 0)

        # Let each connected adapter refresh any platform-side state
        for adapter in list(runner.adapters.values()):
            refresh = getattr(adapter, "refresh_skill_group", None)
            if not callable(refresh):
                continue
            try:
                maybe = refresh()
                if inspect.isawaitable(maybe):
                    await maybe
            except Exception as exc:
                logger.warning(
                    "Adapter %s refresh_skill_group raised: %s",
                    getattr(adapter, "name", adapter), exc,
                )

        lines = [t("gateway.reload_skills.header")]
        if not added and not removed:
            lines.append(t("gateway.reload_skills.no_new"))
            lines.append(t("gateway.reload_skills.total", count=total))
            return "\n".join(lines)

        def _fmt_line(item: dict) -> str:
            nm = item.get("name", "")
            desc = item.get("description", "")
            if desc:
                return t("gateway.reload_skills.item_with_desc", name=nm, desc=desc)
            return t("gateway.reload_skills.item_no_desc", name=nm)

        if added:
            lines.append(t("gateway.reload_skills.added_header"))
            for item in added:
                lines.append(_fmt_line(item))
        if removed:
            lines.append(t("gateway.reload_skills.removed_header"))
            for item in removed:
                lines.append(_fmt_line(item))
        lines.append(t("gateway.reload_skills.total", count=total))

        # Queue the one-shot note for the next user turn in this session.
        sections = ["[USER INITIATED SKILLS RELOAD:"]
        if added:
            sections.append("")
            sections.append("Added Skills:")
            for item in added:
                sections.append(_fmt_line(item))
        if removed:
            sections.append("")
            sections.append("Removed Skills:")
            for item in removed:
                sections.append(_fmt_line(item))
        sections.append("")
        sections.append("Use skills_list to see the updated catalog.]")
        note = "\n".join(sections)

        session_key = runner._session_key_for_source(event.source)
        if not hasattr(runner, "_pending_skills_reload_notes"):
            runner._pending_skills_reload_notes = {}
        if session_key:
            runner._pending_skills_reload_notes[session_key] = note

        return "\n".join(lines)

    except Exception as e:
        logger.warning("Skills reload failed: %s", e)
        return t("gateway.reload_skills.failed", error=e)


async def handle_personality_command(runner, event: MessageEvent) -> str:
    """Handle /personality command - list or set a personality."""
    from hermes_constants import display_hermes_home

    args = event.get_command_args().strip().lower()
    config_path = _hermes_home / 'config.yaml'

    try:
        config = _load_gateway_config()
        personalities = cfg_get(config, "agent", "personalities", default={})
    except Exception:
        config = {}
        personalities = {}

    if not personalities:
        return t("gateway.personality.none_configured", path=display_hermes_home())

    if not args:
        lines = [t("gateway.personality.header")]
        lines.append(t("gateway.personality.none_option"))
        for name, prompt in personalities.items():
            if isinstance(prompt, dict):
                preview = prompt.get("description") or prompt.get("system_prompt", "")[:50]
            else:
                preview = prompt[:50] + "..." if len(prompt) > 50 else prompt
            lines.append(t("gateway.personality.item", name=name, preview=preview))
        lines.append(t("gateway.personality.usage"))
        return "\n".join(lines)

    def _resolve_prompt(value):
        if isinstance(value, dict):
            parts = [value.get("system_prompt", "")]
            if value.get("tone"):
                parts.append(f'Tone: {value["tone"]}')
            if value.get("style"):
                parts.append(f'Style: {value["style"]}')
            return "\n".join(p for p in parts if p)
        return str(value)

    if args in {"none", "default", "neutral"}:
        try:
            if "agent" not in config or not isinstance(config.get("agent"), dict):
                config["agent"] = {}
            config["agent"]["system_prompt"] = ""
            atomic_yaml_write(config_path, config)
        except Exception as e:
            return t("gateway.personality.save_failed", error=str(e))
        runner._ephemeral_system_prompt = ""
        return t("gateway.personality.cleared")
    elif args in personalities:
        new_prompt = _resolve_prompt(personalities[args])

        # Write to config.yaml, same pattern as CLI save_config_value.
        try:
            if "agent" not in config or not isinstance(config.get("agent"), dict):
                config["agent"] = {}
            config["agent"]["system_prompt"] = new_prompt
            atomic_yaml_write(config_path, config)
        except Exception as e:
            return t("gateway.personality.save_failed", error=str(e))

        # Update in-memory so it takes effect on the very next message.
        runner._ephemeral_system_prompt = new_prompt

        return t("gateway.personality.set_to", name=args)

    available = "`none`, " + ", ".join(f"`{n}`" for n in personalities)
    return t("gateway.personality.unknown", name=args, available=available)


async def handle_goal_command(runner, event: "MessageEvent") -> str:
    """Handle /goal for gateway platforms.

    Subcommands: ``/goal`` / ``/goal status`` / ``/goal pause`` /
    ``/goal resume`` / ``/goal clear``. Any other text becomes the
    new goal.

    Setting a new goal queues the goal text as the next turn so the
    agent starts working on it immediately — the post-turn
    continuation hook then takes over from there.
    """
    args = (event.get_command_args() or "").strip()
    lower = args.lower()

    mgr, session_entry = runner._get_goal_manager_for_event(event)
    if mgr is None:
        return t("gateway.goal.unavailable")

    if not args or lower == "status":
        return mgr.status_line()

    if lower == "pause":
        state = mgr.pause(reason="user-paused")
        if state is None:
            return t("gateway.goal.no_goal_set")
        try:
            adapter = runner.adapters.get(event.source.platform) if event.source else None
            _quick_key = runner._session_key_for_source(event.source) if event.source else None
            if adapter and _quick_key:
                runner._clear_goal_pending_continuations(_quick_key, adapter)
        except Exception as exc:
            logger.debug("goal pause: pending continuation cleanup failed: %s", exc)
        return t("gateway.goal.paused", goal=state.goal)

    if lower == "resume":
        state = mgr.resume()
        if state is None:
            return t("gateway.goal.no_resume")
        return t("gateway.goal.resumed", goal=state.goal)

    if lower in {"clear", "stop", "done"}:
        had = mgr.has_goal()
        mgr.clear()
        try:
            adapter = runner.adapters.get(event.source.platform) if event.source else None
            _quick_key = runner._session_key_for_source(event.source) if event.source else None
            if adapter and _quick_key:
                runner._clear_goal_pending_continuations(_quick_key, adapter)
        except Exception as exc:
            logger.debug("goal clear: pending continuation cleanup failed: %s", exc)
        return t("gateway.goal_cleared") if had else t("gateway.no_active_goal")

    # Otherwise — treat the remaining text as the new goal.
    try:
        state = mgr.set(args)
    except ValueError as exc:
        return t("gateway.goal.invalid", error=str(exc))

    # Queue the goal text as an immediate first turn so the agent
    # starts making progress. The post-turn hook takes over after.
    adapter = runner.adapters.get(event.source.platform) if event.source else None
    _quick_key = runner._session_key_for_source(event.source) if event.source else None
    if adapter and _quick_key:
        try:
            kickoff_event = MessageEvent(
                text=state.goal,
                message_type=MessageType.TEXT,
                source=event.source,
                message_id=event.message_id,
                channel_prompt=event.channel_prompt,
            )
            runner._enqueue_fifo(_quick_key, kickoff_event, adapter)
        except Exception as exc:
            logger.debug("goal kickoff enqueue failed: %s", exc)

    return t("gateway.goal.set", budget=state.max_turns, goal=state.goal)


async def handle_subgoal_command(runner, event: "MessageEvent") -> str:
    """Handle /subgoal for gateway platforms (mirror of CLI handler).

    Subgoals are extra criteria appended to the active goal mid-loop.
    They modify state read at the next turn boundary, so this is safe
    to invoke while the agent is running.
    """
    args = (event.get_command_args() or "").strip()
    mgr, _session_entry = runner._get_goal_manager_for_event(event)
    if mgr is None:
        return t("gateway.goal.unavailable")
    if not mgr.has_goal():
        return "No active goal. Set one with /goal <text>."

    # No args → list current subgoals.
    if not args:
        return f"{mgr.status_line()}\n{mgr.render_subgoals()}"

    tokens = args.split(None, 1)
    verb = tokens[0].lower()
    rest = tokens[1].strip() if len(tokens) > 1 else ""

    if verb == "remove":
        if not rest:
            return "Usage: /subgoal remove <n>"
        try:
            idx = int(rest.split()[0])
        except ValueError:
            return "/subgoal remove: <n> must be an integer (1-based index)."
        try:
            removed = mgr.remove_subgoal(idx)
        except (IndexError, RuntimeError) as exc:
            return f"/subgoal remove: {exc}"
        return f"✓ Removed subgoal {idx}: {removed}"

    if verb == "clear":
        try:
            prev = mgr.clear_subgoals()
        except RuntimeError as exc:
            return f"/subgoal clear: {exc}"
        if prev:
            return f"✓ Cleared {prev} subgoal{'s' if prev != 1 else ''}."
        return "No subgoals to clear."

    try:
        text = mgr.add_subgoal(args)
    except (ValueError, RuntimeError) as exc:
        return f"/subgoal: {exc}"
    idx = len(mgr.state.subgoals) if mgr.state else 0
    return f"✓ Added subgoal {idx}: {text}"


async def handle_voice_command(runner, event: MessageEvent) -> str:
    """Handle /voice [on|off|tts|channel|leave|status] command."""
    args = event.get_command_args().strip().lower()
    chat_id = event.source.chat_id
    platform = event.source.platform
    voice_key = runner._voice_key(platform, chat_id)

    adapter = runner.adapters.get(platform)

    if args in {"on", "enable"}:
        runner._voice_mode[voice_key] = "voice_only"
        runner._save_voice_modes()
        if adapter:
            runner._set_adapter_auto_tts_enabled(adapter, chat_id, enabled=True)
        return t("gateway.voice.enabled_voice_only")
    elif args in {"off", "disable"}:
        runner._voice_mode[voice_key] = "off"
        runner._save_voice_modes()
        if adapter:
            runner._set_adapter_auto_tts_disabled(adapter, chat_id, disabled=True)
        return t("gateway.voice.disabled_text")
    elif args == "tts":
        runner._voice_mode[voice_key] = "all"
        runner._save_voice_modes()
        if adapter:
            runner._set_adapter_auto_tts_enabled(adapter, chat_id, enabled=True)
        return t("gateway.voice.tts_enabled")
    elif args in {"channel", "join"}:
        return await runner._handle_voice_channel_join(event)
    elif args == "leave":
        return await runner._handle_voice_channel_leave(event)
    elif args == "status":
        mode = runner._voice_mode.get(voice_key, "off")
        labels = {
            "off": t("gateway.voice.label_off"),
            "voice_only": t("gateway.voice.label_voice_only"),
            "all": t("gateway.voice.label_all"),
        }
        # Append voice channel info if connected
        adapter = runner.adapters.get(event.source.platform)
        guild_id = runner._get_guild_id(event)
        if guild_id and hasattr(adapter, "get_voice_channel_info"):
            info = adapter.get_voice_channel_info(guild_id)
            if info:
                lines = [
                    t("gateway.voice.status_mode", label=labels.get(mode, mode)),
                    t("gateway.voice.status_channel", channel=info['channel_name']),
                    t("gateway.voice.status_participants", count=info['member_count']),
                ]
                for m in info["members"]:
                    status = t("gateway.voice.speaking") if m.get("is_speaking") else ""
                    lines.append(t("gateway.voice.status_member", name=m['display_name'], status=status))
                return "\n".join(lines)
        return t("gateway.voice.status_mode", label=labels.get(mode, mode))
    else:
        # Toggle: off → on, on/all → off
        current = runner._voice_mode.get(voice_key, "off")
        if current == "off":
            runner._voice_mode[voice_key] = "voice_only"
            runner._save_voice_modes()
            if adapter:
                runner._set_adapter_auto_tts_enabled(adapter, chat_id, enabled=True)
            return t("gateway.voice.enabled_short")
        else:
            runner._voice_mode[voice_key] = "off"
            runner._save_voice_modes()
            if adapter:
                runner._set_adapter_auto_tts_disabled(adapter, chat_id, disabled=True)
            return t("gateway.voice.disabled_short")


async def handle_footer_command(runner, event: MessageEvent) -> str:
    """Handle /footer command — toggle the runtime-metadata footer.

    Usage:
        /footer           → toggle on/off
        /footer on        → enable globally
        /footer off       → disable globally
        /footer status    → show current state + fields

    The footer is saved to ``display.runtime_footer.enabled`` (global).
    Per-platform overrides under ``display.platforms.<platform>.runtime_footer``
    are respected but not modified here — edit config.yaml directly for
    per-platform control.
    """
    from gateway.runtime_footer import resolve_footer_config

    config_path = _hermes_home / "config.yaml"
    platform_key = _platform_config_key(event.source.platform)

    # --- parse argument -------------------------------------------------
    arg = ""
    try:
        text = (getattr(event, "message", None) or "").strip()
        if text.startswith("/"):
            parts = text.split(None, 1)
            if len(parts) > 1:
                arg = parts[1].strip().lower()
    except Exception:
        arg = ""

    # --- load config ----------------------------------------------------
    try:
        user_config: dict = _load_gateway_config()
    except Exception as e:
        return t("gateway.config_read_failed", error=e)

    effective = resolve_footer_config(user_config, platform_key)

    if arg in {"status", "?"}:
        state = t("gateway.footer.state_on") if effective["enabled"] else t("gateway.footer.state_off")
        fields = ", ".join(effective.get("fields") or [])
        return t(
            "gateway.footer.status",
            state=state,
            fields=fields,
            platform=platform_key,
        )

    if arg in {"on", "enable", "true", "1"}:
        new_state = True
    elif arg in {"off", "disable", "false", "0"}:
        new_state = False
    elif arg == "":
        new_state = not effective["enabled"]
    else:
        return t("gateway.footer.usage")

    # --- write global flag ---------------------------------------------
    try:
        if not isinstance(user_config.get("display"), dict):
            user_config["display"] = {}
        display = user_config["display"]
        if not isinstance(display.get("runtime_footer"), dict):
            display["runtime_footer"] = {}
        display["runtime_footer"]["enabled"] = new_state
        atomic_yaml_write(config_path, user_config)
    except Exception as e:
        logger.warning("Failed to save runtime_footer.enabled: %s", e)
        return t("gateway.config_save_failed", error=e)

    state = t("gateway.footer.state_on") if new_state else t("gateway.footer.state_off")
    example = ""
    if new_state:
        # Show a preview using current agent state if available.
        from gateway.runtime_footer import format_runtime_footer
        preview = format_runtime_footer(
            model=_resolve_gateway_model(user_config) or None,
            context_tokens=0,
            context_length=None,
            fields=effective.get("fields") or ["model", "context_pct", "cwd"],
        )
        if preview:
            example = t("gateway.footer.example_line", preview=preview)
    return t("gateway.footer.saved", state=state, example=example)


async def handle_resume_command(runner, event: MessageEvent) -> str:
    """Handle /resume command — list or switch to a previous session."""
    if not runner._session_db:
        from hermes_state import format_session_db_unavailable
        return format_session_db_unavailable(prefix=t("gateway.shared.session_db_unavailable_prefix"))

    source = event.source
    session_key = runner._session_key_for_source(source)
    name = event.get_command_args().strip()

    # Strip common outer brackets/quotes users may type literally from the
    # usage hint (e.g. ``/resume <abc123>``). Mirrors the CLI behavior.
    if len(name) >= 2 and (
        (name[0] == "<" and name[-1] == ">")
        or (name[0] == "[" and name[-1] == "]")
        or (name[0] == '"' and name[-1] == '"')
        or (name[0] == "'" and name[-1] == "'")
    ):
        name = name[1:-1].strip()

    def _list_titled_sessions() -> list[dict]:
        user_source = source.platform.value if source.platform else None
        sessions = runner._session_db.list_sessions_rich(source=user_source, limit=10)
        return [s for s in sessions if s.get("title")][:10]

    if not name:
        # List recent titled sessions for this user/platform
        try:
            titled = _list_titled_sessions()
            if not titled:
                return t("gateway.resume.no_named_sessions")
            lines = [t("gateway.resume.list_header")]
            for idx, s in enumerate(titled[:10], start=1):
                title = s["title"]
                preview = s.get("preview", "")[:40]
                preview_part = t("gateway.resume.list_preview_suffix", preview=preview) if preview else ""
                lines.append(t("gateway.resume.list_item_numbered", index=idx, title=title, preview_part=preview_part))
            lines.append(t("gateway.resume.list_footer_numbered"))
            return "\n".join(lines)
        except Exception as e:
            logger.debug("Failed to list titled sessions: %s", e)
            return t("gateway.resume.list_failed", error=e)

    # Resolve a numbered choice or a title to a session ID.
    if name.isdigit():
        try:
            titled = _list_titled_sessions()
        except Exception as e:
            logger.debug("Failed to list titled sessions for numeric resume: %s", e)
            return t("gateway.resume.list_failed", error=e)
        index = int(name)
        if index < 1 or index > len(titled):
            return t("gateway.resume.out_of_range", index=index)
        target = titled[index - 1]
        target_id = target.get("id")
        name = target.get("title") or name
    else:
        # Try direct session ID lookup first (so `/resume <session_id>`
        # works in the gateway, not just `/resume <title>`).
        session = runner._session_db.get_session(name)
        if session:
            target_id = session["id"]
        else:
            target_id = runner._session_db.resolve_session_by_title(name)
    if not target_id:
        return t("gateway.resume.not_found", name=name)
    # Compression creates child continuations that hold the live transcript.
    # Follow that chain so gateway /resume matches CLI behavior (#15000).
    try:
        target_id = runner._session_db.resolve_resume_session_id(target_id)
    except Exception as e:
        logger.debug("Failed to resolve resume continuation for %s: %s", target_id, e)

    # Check if already on that session
    current_entry = runner.session_store.get_or_create_session(source)
    if current_entry.session_id == target_id:
        return t("gateway.resume.already_on", name=name)

    # Clear any running agent for this session key
    runner._release_running_agent_state(session_key)

    # Switch the session entry to point at the old session
    new_entry = runner.session_store.switch_session(session_key, target_id)
    if not new_entry:
        return t("gateway.resume.switch_failed")
    runner._clear_session_boundary_security_state(session_key)

    # Evict any cached agent for this session so the next message
    # rebuilds with the correct session_id end-to-end — mirrors
    # /branch and /reset. Without this, the cached AIAgent (and its
    # memory provider, which cached `_session_id` during initialize())
    # keeps writing into the wrong session's record. See #6672.
    runner._evict_cached_agent(session_key)

    # Get the title for confirmation
    title = runner._session_db.get_session_title(target_id) or name

    # Count messages for context
    history = runner.session_store.load_transcript(target_id)
    msg_count = len([m for m in history if m.get("role") == "user"]) if history else 0
    if not msg_count:
        return t("gateway.resume.resumed_no_count", title=title)
    if msg_count == 1:
        return t("gateway.resume.resumed_one", title=title, count=msg_count)
    return t("gateway.resume.resumed_many", title=title, count=msg_count)


async def handle_reload_mcp_command(runner, event: MessageEvent) -> Optional[str]:
    """Handle /reload-mcp — reconnect MCP servers and rebuild the cached agent.

    Reloading MCP tools invalidates the provider prompt cache for the
    active session (tool schemas are baked into the system prompt).  The
    next message re-sends full input tokens, which is expensive on
    long-context or high-reasoning models.

    To surface that cost, the command routes through the slash-confirm
    primitive: users get an Approve Once / Always Approve / Cancel
    prompt before the reload actually runs.  "Always Approve" persists
    ``approvals.mcp_reload_confirm: false`` so the prompt is silenced
    for subsequent reloads in any session.

    Users can also skip the confirm by flipping the config key directly.
    """
    source = event.source
    session_key = runner._session_key_for_source(source)

    # Read the gate fresh from disk so a prior "always" click takes
    # effect on the next invocation without restarting the gateway.
    user_config = runner._read_user_config()
    approvals = user_config.get("approvals") if isinstance(user_config, dict) else None
    confirm_required = True
    if isinstance(approvals, dict):
        confirm_required = bool(approvals.get("mcp_reload_confirm", True))

    if not confirm_required:
        return await runner._execute_mcp_reload(event)

    # Route through slash-confirm.  The primitive sends the prompt and
    # stores the resume handler; the button/text response triggers
    # ``_resolve_slash_confirm`` which invokes the handler with the
    # chosen outcome.
    async def _on_confirm(choice: str) -> Optional[str]:
        if choice == "cancel":
            return t("gateway.reload_mcp.cancelled")
        if choice == "always":
            # Persist the opt-out and run the reload.
            try:
                from cli import save_config_value
                save_config_value("approvals.mcp_reload_confirm", False)
                logger.info(
                    "User opted out of /reload-mcp confirmation (session=%s)",
                    session_key,
                )
            except Exception as exc:
                logger.warning("Failed to persist mcp_reload_confirm=false: %s", exc)
        # once / always → run the reload
        result = await runner._execute_mcp_reload(event)
        if choice == "always":
            return f"{result}\n\n" + t("gateway.reload_mcp.always_followup")
        return result

    prompt_message = t("gateway.reload_mcp.confirm_prompt")
    return await runner._request_slash_confirm(
        event=event,
        command="reload-mcp",
        title="/reload-mcp",
        message=prompt_message,
        handler=_on_confirm,
    )


async def execute_mcp_reload(runner, event: MessageEvent) -> str:
    """Actually disconnect, reconnect, and notify MCP tool changes.

    Split out from ``_handle_reload_mcp_command`` so the confirmation
    wrapper can invoke the same path whether the user confirmed via
    button, text reply, or has the confirm gate disabled.
    """
    loop = asyncio.get_running_loop()
    try:
        from tools.mcp_tool import shutdown_mcp_servers, discover_mcp_tools, _servers, _lock

        # Capture old server names before shutdown
        with _lock:
            old_servers = set(_servers.keys())

        # Read new config before shutting down, so we know what will be added/removed
        # Shutdown existing connections
        await loop.run_in_executor(None, shutdown_mcp_servers)

        # Reconnect by discovering tools (reads config.yaml fresh)
        new_tools = await loop.run_in_executor(None, discover_mcp_tools)

        # Compute what changed
        with _lock:
            connected_servers = set(_servers.keys())

        added = connected_servers - old_servers
        removed = old_servers - connected_servers
        reconnected = connected_servers & old_servers

        lines = [t("gateway.reload_mcp.header")]
        if reconnected:
            lines.append(t("gateway.reload_mcp.reconnected", names=", ".join(sorted(reconnected))))
        if added:
            lines.append(t("gateway.reload_mcp.added", names=", ".join(sorted(added))))
        if removed:
            lines.append(t("gateway.reload_mcp.removed", names=", ".join(sorted(removed))))
        if not connected_servers:
            lines.append(t("gateway.reload_mcp.none_connected"))
        else:
            lines.append(t("gateway.reload_mcp.tools_available", tools=len(new_tools), servers=len(connected_servers)))

        # Refresh cached agents so existing sessions see new MCP tools on
        # their next turn — without this, the user has to `/new` (which
        # discards conversation history) to pick up tools from a server
        # that was just added or reconnected. The user has already
        # consented to the prompt-cache invalidation via the slash-confirm
        # gate in _handle_reload_mcp_command before we reach this point.
        try:
            from model_tools import get_tool_definitions
            _cache = getattr(runner, "_agent_cache", None)
            _cache_lock = getattr(runner, "_agent_cache_lock", None)
            if _cache_lock is not None and _cache:
                with _cache_lock:
                    for _sess_key, _entry in list(_cache.items()):
                        try:
                            _agent = _entry[0] if isinstance(_entry, tuple) else _entry
                        except Exception:
                            continue
                        if _agent is None:
                            continue
                        new_defs = get_tool_definitions(
                            enabled_toolsets=getattr(_agent, "enabled_toolsets", None),
                            disabled_toolsets=getattr(_agent, "disabled_toolsets", None),
                            quiet_mode=True,
                        )
                        _agent.tools = new_defs
                        _agent.valid_tool_names = {
                            t["function"]["name"] for t in new_defs
                        } if new_defs else set()
        except Exception as _exc:
            logger.debug(
                "Failed to update cached agent tools after MCP reload: %s",
                _exc,
            )

        # Inject a message at the END of the session history so the
        # model knows tools changed on its next turn.  Appended after
        # all existing messages to preserve prompt-cache for the prefix.
        change_parts = []
        if added:
            change_parts.append(f"Added servers: {', '.join(sorted(added))}")
        if removed:
            change_parts.append(f"Removed servers: {', '.join(sorted(removed))}")
        if reconnected:
            change_parts.append(f"Reconnected servers: {', '.join(sorted(reconnected))}")
        tool_summary = f"{len(new_tools)} MCP tool(s) now available" if new_tools else "No MCP tools available"
        change_detail = ". ".join(change_parts) + ". " if change_parts else ""
        reload_msg = {
            "role": "user",
            "content": f"[IMPORTANT: MCP servers have been reloaded. {change_detail}{tool_summary}. The tool list for this conversation has been updated accordingly.]",
        }
        try:
            session_entry = runner.session_store.get_or_create_session(event.source)
            runner.session_store.append_to_transcript(
                session_entry.session_id, reload_msg
            )
        except Exception:
            pass  # Best-effort; don't fail the reload over a transcript write

        return "\n".join(lines)

    except Exception as e:
        logger.warning("MCP reload failed: %s", e)
        return t("gateway.reload_mcp.failed", error=e)


async def handle_undo_command(runner, event: MessageEvent) -> str:
    """Handle /undo [N] — back up N user turns (default 1), soft-deleting
    the truncated rows on disk and echoing the backed-up message text so
    the user can copy/edit and resend.

    Mirrors the CLI/TUI /undo: rewound rows stay in state.db (active=0)
    for audit and are hidden from re-prompts and search. The cached agent
    is evicted so the next message rebuilds context from the truncated
    (active-only) transcript — the gateway's equivalent of the CLI's
    in-place history surgery + memory-cache invalidation.
    """
    source = event.source

    # Parse optional turn count: "/undo" → 1, "/undo 3" → 3.
    n = 1
    raw_args = event.get_command_args().strip()
    if raw_args:
        try:
            n = int(raw_args.split()[0])
        except (ValueError, IndexError):
            return t("gateway.undo.invalid_count", arg=raw_args.split()[0])
        if n < 1:
            n = 1

    session_entry = runner.session_store.get_or_create_session(source)
    result = runner.session_store.rewind_session(session_entry.session_id, n)

    if result is None:
        return t("gateway.undo.nothing")

    # Reset stored token count — transcript was truncated.
    session_entry.last_prompt_tokens = 0
    # Evict the cached agent so the next turn rebuilds from the active-only
    # transcript and memory providers refresh their per-session caches.
    try:
        session_key = build_session_key(source)
        runner._evict_cached_agent(session_key)
    except Exception as e:
        logger.debug("undo: cached-agent eviction skipped: %s", e)

    target_text = result["target_text"]
    preview = target_text[:200] + "..." if len(target_text) > 200 else target_text
    return t(
        "gateway.undo.removed",
        turns=result["turns_undone"],
        count=result["rewound_count"],
        preview=preview,
    )


async def handle_set_home_command(runner, event: MessageEvent) -> str:
    """Handle /sethome command -- set the current chat as the platform's home channel."""
    source = event.source
    platform_name = source.platform.value if source.platform else "unknown"
    chat_id = source.chat_id
    chat_name = source.chat_name or chat_id

    env_key = _home_target_env_var(platform_name)
    thread_env_key = _home_thread_env_var(platform_name)
    thread_id = source.thread_id

    # Save to .env so it persists across restarts
    try:
        from hermes_cli.config import save_env_value
        save_env_value(env_key, str(chat_id))
        # Keep thread/topic routing explicit and clear stale values when
        # /sethome is run from the parent chat instead of a thread.
        save_env_value(thread_env_key, str(thread_id or ""))
    except Exception as e:
        return t("gateway.set_home.save_failed", error=e)

    # Keep the running gateway config in sync too. The pre-restart
    # notification path reads self.config before the process reloads env.
    if source.platform:
        platform_config = runner.config.platforms.setdefault(
            source.platform,
            PlatformConfig(enabled=True),
        )
        platform_config.home_channel = HomeChannel(
            platform=source.platform,
            chat_id=str(chat_id),
            name=chat_name,
            thread_id=str(thread_id) if thread_id else None,
        )

    return t("gateway.set_home.success", name=chat_name, chat_id=chat_id)


async def handle_help_command(runner, event: MessageEvent) -> str:
    """Handle /help command - list available commands."""
    from hermes_cli.commands import gateway_help_lines
    lines = [
        t("gateway.help.header"),
        *gateway_help_lines(),
    ]
    try:
        from agent.skill_commands import get_skill_commands
        skill_cmds = get_skill_commands()
        if skill_cmds:
            lines.append(t("gateway.help.skill_header", count=len(skill_cmds)))
            # Show first 10, then point to /commands for the rest
            sorted_cmds = sorted(skill_cmds)
            for cmd in sorted_cmds[:10]:
                lines.append(f"`{cmd}` — {skill_cmds[cmd]['description']}")
            if len(sorted_cmds) > 10:
                lines.append(t("gateway.help.more_use_commands", count=len(sorted_cmds) - 10))
    except Exception:
        pass
    return _telegramize_command_mentions(
        "\n".join(lines),
        getattr(getattr(event, "source", None), "platform", None),
    )


async def handle_commands_command(runner, event: MessageEvent) -> str:
    from hermes_cli.commands import gateway_help_lines

    raw_args = event.get_command_args().strip()
    if raw_args:
        try:
            requested_page = int(raw_args)
        except ValueError:
            return t("gateway.commands.usage")
    else:
        requested_page = 1

    # Build combined entry list: built-in commands + skill commands
    entries = list(gateway_help_lines())
    try:
        from agent.skill_commands import get_skill_commands
        skill_cmds = get_skill_commands()
        if skill_cmds:
            entries.append("")
            entries.append(t("gateway.commands.skill_header"))
            for cmd in sorted(skill_cmds):
                desc = skill_cmds[cmd].get("description", "").strip() or t("gateway.commands.default_desc")
                entries.append(f"`{cmd}` — {desc}")
    except Exception:
        pass

    if not entries:
        return t("gateway.commands.none")

    from gateway.config import Platform
    page_size = 15 if event.source.platform == Platform.TELEGRAM else 20
    total_pages = max(1, (len(entries) + page_size - 1) // page_size)
    page = max(1, min(requested_page, total_pages))
    start = (page - 1) * page_size
    page_entries = entries[start:start + page_size]

    lines = [
        t("gateway.commands.header", total=len(entries), page=page, total_pages=total_pages),
        "",
        *page_entries,
    ]
    if total_pages > 1:
        nav_parts = []
        if page > 1:
            nav_parts.append(t("gateway.commands.nav_prev", page=page - 1))
        if page < total_pages:
            nav_parts.append(t("gateway.commands.nav_next", page=page + 1))
        lines.extend(["", " | ".join(nav_parts)])
    if page != requested_page:
        lines.append(t("gateway.commands.out_of_range", requested=requested_page, page=page))
    return _telegramize_command_mentions(
        "\n".join(lines),
        getattr(getattr(event, "source", None), "platform", None),
    )


async def handle_profile_command(runner, event: MessageEvent) -> str:
    """Handle /profile — show active profile name and home directory."""
    from hermes_constants import display_hermes_home
    from hermes_cli.profiles import get_active_profile_name

    display = display_hermes_home()
    profile_name = get_active_profile_name()

    lines = [
        t("gateway.profile.header", profile=profile_name),
        t("gateway.profile.home", home=display),
    ]

    return "\n".join(lines)


async def handle_whoami_command(runner, event: MessageEvent) -> str:
    """Handle /whoami — show the user's slash command access on this scope.

    Always works (it's in the always-allowed floor of slash_access).
    Reports: platform, scope (DM vs group), the user's tier
    (admin / user / unrestricted), and the slash commands they can
    actually run on this scope.
    """
    from gateway.slash_access import policy_for_source as _policy_for_source

    source = event.source
    policy = _policy_for_source(runner.config, source)
    platform = source.platform.value if source and source.platform else "?"
    chat_type = (source.chat_type if source else "") or "dm"
    scope = "DM" if chat_type.lower() in {"dm", "direct", "private", ""} else "group/channel"
    user_id = (source.user_id if source else None) or "?"

    if not policy.enabled:
        return (
            f"**You** — {platform} ({scope})\n"
            f"User ID: `{user_id}`\n"
            f"Tier: unrestricted (no admin list configured for this scope)\n"
            f"Slash commands: all available"
        )

    if policy.is_admin(user_id):
        return (
            f"**You** — {platform} ({scope})\n"
            f"User ID: `{user_id}`\n"
            f"Tier: **admin**\n"
            f"Slash commands: all available"
        )

    # Non-admin user. Show what's actually reachable.
    floor = ["help", "whoami"]  # mirrors slash_access._ALWAYS_ALLOWED_FOR_USERS
    configured = sorted(policy.user_allowed_commands)
    # Combine + dedupe, preserve order: floor first, then operator additions.
    seen: set[str] = set()
    runnable: list[str] = []
    for c in floor + configured:
        if c not in seen:
            seen.add(c)
            runnable.append(c)
    runnable_str = ", ".join(f"/{c}" for c in runnable) if runnable else "(none)"
    return (
        f"**You** — {platform} ({scope})\n"
        f"User ID: `{user_id}`\n"
        f"Tier: user\n"
        f"Slash commands you can run: {runnable_str}"
    )


async def handle_kanban_command(runner, event: MessageEvent) -> str:
    """Handle /kanban — delegate to the shared kanban CLI.

    Run the potentially-blocking DB work in a thread pool so the
    gateway event loop stays responsive.  Read operations (list,
    show, context, tail) are permitted while an agent is running;
    mutations are allowed too because the board is profile-agnostic
    and does not touch the running agent's state.

    For ``/kanban create`` invocations we also auto-subscribe the
    originating gateway source (platform + chat + thread) to the new
    task's terminal events, so the user hears back when the worker
    completes / blocks / auto-blocks / crashes without having to poll.
    """
    import asyncio
    import re
    import shlex
    from hermes_cli.kanban import run_slash

    text = (event.text or "").strip()
    # Strip the leading "/kanban" (with or without slash), leaving args.
    if text.startswith("/"):
        text = text.lstrip("/")
    if text.startswith("kanban"):
        text = text[len("kanban"):].lstrip()

    tokens = shlex.split(text) if text else []
    requested_board = None
    action = None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--board":
            if i + 1 >= len(tokens):
                break
            requested_board = tokens[i + 1]
            i += 2
            continue
        if tok.startswith("--board="):
            requested_board = tok.split("=", 1)[1]
            i += 1
            continue
        action = tok
        break

    is_create = action == "create"

    try:
        output = await asyncio.to_thread(run_slash, text)
    except Exception as exc:  # pragma: no cover - defensive
        return t("gateway.kanban.error_prefix", error=exc)

    # Auto-subscribe on create. Parse the task id from the CLI's standard
    # success line ("Created t_abcd  (ready, assignee=...)"). If the user
    # passed --json we don't subscribe; they're clearly scripting and
    # can call /kanban notify-subscribe explicitly.
    if is_create and output:
        m = re.search(r"Created\s+(t_[0-9a-f]+)\b", output)
        if m:
            task_id = m.group(1)
            try:
                source = event.source
                platform = getattr(source, "platform", None)
                platform_str = (
                    platform.value if hasattr(platform, "value") else str(platform or "")
                ).lower()
                chat_id = str(getattr(source, "chat_id", "") or "")
                thread_id = str(getattr(source, "thread_id", "") or "")
                user_id = str(getattr(source, "user_id", "") or "") or None
                if platform_str and chat_id:
                    def _sub():
                        from hermes_cli import kanban_db as _kb
                        conn = _kb.connect(board=requested_board)
                        try:
                            _kb.add_notify_sub(
                                conn, task_id=task_id,
                                platform=platform_str, chat_id=chat_id,
                                thread_id=thread_id or None,
                                user_id=user_id,
                                notifier_profile=getattr(runner, "_kanban_notifier_profile", None) or runner._active_profile_name(),
                            )
                        finally:
                            conn.close()
                    await asyncio.to_thread(_sub)
                    output = (
                        output.rstrip()
                        + "\n"
                        + t("gateway.kanban.subscribed_suffix", task_id=task_id)
                    )
            except Exception as exc:
                logger.warning("kanban create auto-subscribe failed: %s", exc)

    # Gateway messages have practical length caps; truncate long
    # listings to keep the UX reasonable.
    if len(output) > 3800:
        output = output[:3800] + "\n" + t("gateway.kanban.truncated_suffix")
    return output or t("gateway.kanban.no_output")


async def handle_status_command(runner, event: MessageEvent) -> str:
    """Handle /status command."""
    source = event.source
    session_entry = runner.session_store.get_or_create_session(source)

    connected_platforms = [p.value for p in runner.adapters.keys()]

    # Check if there's an active agent
    session_key = session_entry.session_key
    is_running = session_key in runner._running_agents

    # Count pending /queue follow-ups (slot + overflow).
    adapter = runner.adapters.get(source.platform) if source else None
    queue_depth = runner._queue_depth(session_key, adapter=adapter)

    title = None
    # Pull token totals from the SQLite session DB rather than the
    # in-memory SessionStore.  The agent's per-turn token deltas are
    # persisted into sessions_db (run_agent.py), not into SessionEntry,
    # so session_entry.total_tokens is always 0.  SessionDB is the
    # single source of truth; reading it here keeps /status accurate
    # without duplicating token writes into two stores.
    db_total_tokens = 0
    if runner._session_db:
        try:
            title = runner._session_db.get_session_title(session_entry.session_id)
        except Exception:
            title = None
        try:
            row = runner._session_db.get_session(session_entry.session_id)
            if row:
                db_total_tokens = (
                    (row.get("input_tokens") or 0)
                    + (row.get("output_tokens") or 0)
                    + (row.get("cache_read_tokens") or 0)
                    + (row.get("cache_write_tokens") or 0)
                    + (row.get("reasoning_tokens") or 0)
                )
        except Exception:
            db_total_tokens = 0

    lines = [
        t("gateway.status.header"),
        "",
        t("gateway.status.session_id", session_id=session_entry.session_id),
    ]
    if title:
        lines.append(t("gateway.status.title", title=title))
    lines.extend([
        t("gateway.status.created", timestamp=session_entry.created_at.strftime('%Y-%m-%d %H:%M')),
        t("gateway.status.last_activity", timestamp=session_entry.updated_at.strftime('%Y-%m-%d %H:%M')),
        t("gateway.status.tokens", tokens=f"{db_total_tokens:,}"),
        t("gateway.status.agent_running", state=t("gateway.status.state_yes") if is_running else t("gateway.status.state_no")),
    ])
    if queue_depth:
        lines.append(t("gateway.status.queued", count=queue_depth))
    lines.extend([
        "",
        t("gateway.status.platforms", platforms=', '.join(connected_platforms)),
    ])

    return "\n".join(lines)


async def handle_agents_command(runner, event: MessageEvent) -> str:
    """Handle /agents command - list active agents and running tasks."""
    from tools.process_registry import format_uptime_short, process_registry

    now = time.time()
    current_session_key = runner._session_key_for_source(event.source)

    running_agents: dict = getattr(runner, "_running_agents", {}) or {}
    running_started: dict = getattr(runner, "_running_agents_ts", {}) or {}

    agent_rows: list[dict] = []
    for session_key, agent in running_agents.items():
        started = float(running_started.get(session_key, now))
        elapsed = max(0, int(now - started))
        is_pending = agent is _AGENT_PENDING_SENTINEL
        agent_rows.append(
            {
                "session_key": session_key,
                "elapsed": elapsed,
                "state": t("gateway.agents.state_starting") if is_pending else t("gateway.agents.state_running"),
                "session_id": "" if is_pending else str(getattr(agent, "session_id", "") or ""),
                "model": "" if is_pending else str(getattr(agent, "model", "") or ""),
            }
        )

    agent_rows.sort(key=lambda row: row["elapsed"], reverse=True)

    running_processes: list[dict] = []
    try:
        running_processes = [
            p for p in process_registry.list_sessions()
            if p.get("status") == "running"
        ]
    except Exception:
        running_processes = []

    background_tasks = [
        t for t in (getattr(runner, "_background_tasks", set()) or set())
        if hasattr(t, "done") and not t.done()
    ]

    lines = [
        t("gateway.agents.header"),
        "",
        t("gateway.agents.active_agents", count=len(agent_rows)),
    ]

    if agent_rows:
        for idx, row in enumerate(agent_rows[:12], 1):
            current = t("gateway.agents.this_chat") if row["session_key"] == current_session_key else ""
            sid = f" · `{row['session_id']}`" if row["session_id"] else ""
            model = f" · `{row['model']}`" if row["model"] else ""
            lines.append(
                f"{idx}. `{row['session_key']}` · {row['state']} · "
                f"{format_uptime_short(row['elapsed'])}{sid}{model}{current}"
            )
        if len(agent_rows) > 12:
            lines.append(t("gateway.agents.more", count=len(agent_rows) - 12))

    lines.extend(
        [
            "",
            t("gateway.agents.running_processes", count=len(running_processes)),
        ]
    )
    if running_processes:
        for proc in running_processes[:12]:
            cmd = " ".join(str(proc.get("command", "")).split())
            if len(cmd) > 90:
                cmd = cmd[:87] + "..."
            lines.append(
                f"- `{proc.get('session_id', '?')}` · "
                f"{format_uptime_short(int(proc.get('uptime_seconds', 0)))} · `{cmd}`"
            )
        if len(running_processes) > 12:
            lines.append(t("gateway.agents.more", count=len(running_processes) - 12))

    lines.extend(
        [
            "",
            t("gateway.agents.async_jobs", count=len(background_tasks)),
        ]
    )

    if not agent_rows and not running_processes and not background_tasks:
        lines.append("")
        lines.append(t("gateway.agents.none"))

    return "\n".join(lines)


async def handle_stop_command(runner, event: MessageEvent) -> Union[str, EphemeralReply]:
    """Handle /stop command - interrupt a running agent.

    When an agent is truly hung (blocked thread that never checks
    _interrupt_requested), the early intercept in _handle_message()
    handles /stop before this method is reached.  This handler fires
    only through normal command dispatch (no running agent) or as a
    fallback.  Force-clean the session lock in all cases for safety.

    The session is preserved so the user can continue the conversation.
    """
    from gateway.agent_execution import _AGENT_PENDING_SENTINEL, _INTERRUPT_REASON_STOP

    source = event.source
    session_entry = runner.session_store.get_or_create_session(source)
    session_key = session_entry.session_key

    agent = runner._running_agents.get(session_key)
    if agent is _AGENT_PENDING_SENTINEL:
        # Force-clean the sentinel so the session is unlocked.
        await runner._interrupt_and_clear_session(
            session_key,
            source,
            interrupt_reason="stop",
            invalidation_reason="stop_command_pending",
        )
        logger.info("STOP (pending) for session %s — sentinel cleared", session_key)
        return EphemeralReply(t("gateway.stop.stopped_pending"))
    if agent:
        # Force-clean the session lock so a truly hung agent doesn't
        # keep it locked forever.
        await runner._interrupt_and_clear_session(
            session_key,
            source,
            interrupt_reason=_INTERRUPT_REASON_STOP,
            invalidation_reason="stop_command_handler",
        )
        return EphemeralReply(t("gateway.stop.stopped"))

    # No run under the caller's own session key.  In a per-user thread
    # (thread_sessions_per_user=True) each participant is isolated even
    # inside one shared thread, so a run another user started lives under
    # a different key.  Authorized users should still be able to /stop it
    # (#bernard-thread-stop).  Fall back to interrupting any running
    # agent(s) that share this thread, gated on authorization.
    sibling_keys = runner._sibling_thread_run_keys(source, session_key)
    if sibling_keys and runner._is_user_authorized(source):
        for sibling_key in sibling_keys:
            await runner._interrupt_and_clear_session(
                sibling_key,
                source,
                interrupt_reason=_INTERRUPT_REASON_STOP,
                invalidation_reason="stop_command_thread_sibling",
            )
        logger.info(
            "STOP (thread sibling) by %s — interrupted %d run(s) in thread: %s",
            session_key,
            len(sibling_keys),
            ", ".join(sibling_keys),
        )
        return EphemeralReply(t("gateway.stop.stopped"))

    return t("gateway.stop.no_active")


async def handle_platform_command(runner, event: MessageEvent) -> str:
    """Handle ``/platform list|pause|resume [name]`` — surface and
    manually control failed/paused gateway adapters.

    Examples:
        ``/platform list``           — show connected + failed/paused platforms
        ``/platform pause whatsapp`` — stop the reconnect watcher hammering whatsapp
        ``/platform resume whatsapp`` — re-queue a paused platform for retry
    """
    from gateway.config import Platform

    text = (getattr(event, "content", "") or "").strip()
    # Strip the leading "/platform" (or "/PLATFORM") token if present
    parts = text.split(maxsplit=2)
    if parts and parts[0].lower().lstrip("/").startswith("platform"):
        parts = parts[1:]
    action = (parts[0] if parts else "list").lower()
    target = parts[1].lower() if len(parts) > 1 else ""

    # Resolve platform name (case-insensitive, value match)
    def _resolve_platform(name: str):
        if not name:
            return None
        for p in Platform.__members__.values():
            if p.value.lower() == name:
                return p
        return None

    if action == "list":
        lines = ["**Gateway platforms**"]
        connected = sorted(p.value for p in runner.adapters.keys())
        if connected:
            lines.append("Connected: " + ", ".join(connected))
        else:
            lines.append("Connected: (none)")
        failed = getattr(runner, "_failed_platforms", {}) or {}
        if failed:
            for p, info in failed.items():
                if info.get("paused"):
                    reason = info.get("pause_reason") or "paused"
                    lines.append(
                        f"  · {p.value} — PAUSED ({reason}). "
                        f"Resume with `/platform resume {p.value}`."
                    )
                else:
                    attempts = info.get("attempts", 0)
                    lines.append(
                        f"  · {p.value} — retrying (attempt {attempts})"
                    )
        else:
            lines.append("Failed/paused: (none)")
        return "\n".join(lines)

    if action in {"pause", "resume"}:
        if not target:
            return f"Usage: /platform {action} <name>"
        platform = _resolve_platform(target)
        if platform is None:
            return f"Unknown platform: {target}"
        failed = getattr(runner, "_failed_platforms", {}) or {}
        if action == "pause":
            if platform not in failed:
                return (
                    f"{platform.value} is not in the retry queue "
                    f"(it's either connected or not enabled)."
                )
            if failed[platform].get("paused"):
                return f"{platform.value} is already paused."
            runner._pause_failed_platform(platform, reason="paused via /platform pause")
            return (
                f"✓ {platform.value} paused. "
                f"Resume with `/platform resume {platform.value}` or "
                f"`hermes gateway restart` to reset."
            )
        # action == "resume"
        if platform not in failed:
            return (
                f"{platform.value} is not in the retry queue — "
                f"nothing to resume."
            )
        if not failed[platform].get("paused"):
            return (
                f"{platform.value} is already retrying — "
                f"no resume needed."
            )
        runner._resume_paused_platform(platform)
        return f"✓ {platform.value} resumed — retrying on next watcher tick."

    return (
        "Usage: /platform <list|pause|resume> [name]\n"
        "  /platform list — show platform status\n"
        "  /platform pause <name> — stop retrying a failing platform\n"
        "  /platform resume <name> — re-queue a paused platform"
    )


async def handle_restart_command(runner, event: MessageEvent) -> Union[str, EphemeralReply]:
    """Handle /restart command - drain active work, then restart the gateway."""
    import dataclasses
    import os

    # Defensive idempotency check: if the previous gateway process
    # recorded this same /restart (same platform + update_id) and the new
    # process is seeing it *again*, this is a re-delivery caused by PTB's
    # graceful-shutdown `get_updates` ACK failing on the way out ("Error
    # while calling `get_updates` one more time to mark all fetched
    # updates. Suppressing error to ensure graceful shutdown. When
    # polling for updates is restarted, updates may be received twice."
    # in gateway.log).  Ignoring the stale redelivery prevents a
    # self-perpetuating restart loop where every fresh gateway
    # re-processes the same /restart command and immediately restarts
    # again.
    if runner._is_stale_restart_redelivery(event):
        logger.info(
            "Ignoring redelivered /restart (platform=%s, update_id=%s) — "
            "already processed by a previous gateway instance.",
            event.source.platform.value if event.source and event.source.platform else "?",
            event.platform_update_id,
        )
        return ""

    if runner._restart_requested or runner._draining:
        count = runner._running_agent_count()
        if count:
            return t("gateway.draining", count=count)
        return EphemeralReply(t("gateway.restart.in_progress"))

    # Save the requester's routing info so the new gateway process can
    # notify them once it comes back online.
    try:
        notify_data = {
            "platform": event.source.platform.value if event.source.platform else None,
            "chat_id": event.source.chat_id,
            "chat_type": event.source.chat_type,
        }
        if event.source.thread_id:
            notify_data["thread_id"] = event.source.thread_id
        if event.message_id:
            notify_data["message_id"] = event.message_id
        if event.source is not None:
            try:
                runner._restart_command_source = dataclasses.replace(
                    event.source,
                    message_id=str(event.message_id)
                    if event.message_id is not None
                    else event.source.message_id,
                )
            except Exception:
                runner._restart_command_source = event.source
        atomic_json_write(
            _hermes_home / ".restart_notify.json",
            notify_data,
            indent=None,
        )
    except Exception as e:
        logger.debug("Failed to write restart notify file: %s", e)

    # Record the triggering platform + update_id in a dedicated dedup
    # marker.  Unlike .restart_notify.json (which gets unlinked once the
    # new gateway sends the "gateway restarted" notification), this
    # marker persists so the new gateway can still detect a delayed
    # /restart redelivery from Telegram.  Overwritten on every /restart.
    try:
        dedup_data = {
            "platform": event.source.platform.value if event.source.platform else None,
            "requested_at": time.time(),
        }
        if event.platform_update_id is not None:
            dedup_data["update_id"] = event.platform_update_id
        atomic_json_write(
            _hermes_home / ".restart_last_processed.json",
            dedup_data,
            indent=None,
        )
    except Exception as e:
        logger.debug("Failed to write restart dedup marker: %s", e)

    active_agents = runner._running_agent_count()
    # When running under a service manager (systemd/launchd) or inside a
    # Docker/Podman container, use the service restart path: exit with
    # code 75 so the service manager / container restart policy restarts
    # us.  The detached subprocess approach (setsid + bash) doesn't work
    # under systemd (KillMode=mixed kills the cgroup) or Docker (tini
    # exits when the gateway dies, taking the detached helper with it).
    _under_service = bool(os.environ.get("INVOCATION_ID"))  # systemd sets this
    _in_container = os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")
    if _under_service or _in_container:
        runner.request_restart(detached=False, via_service=True)
    else:
        runner.request_restart(detached=True, via_service=False)
    if active_agents:
        return t("gateway.draining", count=active_agents)
    return EphemeralReply(t("gateway.restart.restarting"))
