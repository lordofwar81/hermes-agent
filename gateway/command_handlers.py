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
