"""
Gateway runner - entry point for messaging platform integrations.

This module provides:
- start_gateway(): Start all configured platform adapters
- GatewayRunner: Main class managing the gateway lifecycle

Usage:
    # Start the gateway
    python -m gateway.run
    
    # Or from CLI
    python cli.py --gateway
"""

# IMPORTANT: hermes_bootstrap must be the very first import — UTF-8 stdio
# on Windows.  No-op on POSIX.  See hermes_bootstrap.py for full rationale.
try:
    import hermes_bootstrap  # noqa: F401
except ModuleNotFoundError:
    # Graceful fallback when hermes_bootstrap isn't registered in the venv
    # yet — happens during partial ``hermes update`` where git-reset landed
    # new code but ``uv pip install -e .`` didn't finish.  Missing bootstrap
    # means UTF-8 stdio setup is skipped on Windows; POSIX is unaffected.
    pass

import asyncio
import dataclasses
import inspect
import json
import logging
import os
import re
import shlex
import site
import sys
import signal
import tempfile
import threading
import time
import sqlite3
from collections import OrderedDict
from contextvars import copy_context
from pathlib import Path
from datetime import datetime
from gateway.gateway_timing import _format_duration, _coerce_gateway_timestamp
from gateway.gateway_telegram_topics import (
    _is_telegram_dm_topic_target,
    _telegram_topic_root_lobby_message,
    _telegram_topic_root_new_message,
    _sanitize_telegram_topic_title,
    _telegram_topic_help_text,
)
from gateway.gateway_cache_busting import (
    _CACHE_BUSTING_CONFIG_KEYS,
    _extract_cache_busting_config,
    _extract_honcho_cache_busting_config,
    _empty_honcho_cache_busting_config,
)
from gateway.gateway_agent_cache import (
    _agent_config_signature,
)
from gateway.gateway_voice import (
    _voice_key,
    _set_adapter_auto_tts_enabled,
    _set_adapter_auto_tts_disabled,
)
from gateway.gateway_gateway_env import (
    _adapter_disconnect_timeout_secs,
    _platform_connect_timeout_secs,
    _get_proxy_url,
    _active_profile_name,
    _has_setup_skill,
    _clear_session_env,
)
from gateway.gateway_events import (
    _is_goal_continuation_event,
    _parse_reasoning_command_args,
    _agent_has_active_subagents,
    _get_guild_id,
    _init_cached_agent_for_turn,
)
from gateway.gateway_lifecycle import (
    _cleanup_agent_resources,
    _launch_detached_restart_command,
    _launch_systemd_restart_shortcut,
    _is_stale_restart_redelivery,
)
from gateway.gateway_command_delegates import (
    _handle_suggestions_command,
    _handle_blueprint_command,
)
from gateway.gateway_session_info import (
    _format_session_info,
)
from gateway.gateway_async_utils import (
    _run_in_executor_with_context,
    _safe_adapter_disconnect,
    _connect_adapter_with_timeout,
)
from gateway.gateway_agent_mgmt import (
    _finalize_shutdown_agents,
    _release_evicted_agent_soft,
    _bind_adapter_run_generation,
    _update_platform_runtime_status,
    _goal_still_active_for_session,
)
from gateway.gateway_message_pipeline import (
    _decide_image_input_mode,
    _enrich_message_with_vision,
    _set_session_env,
    _enrich_async_delegation_routing,
    _read_user_config,
    _thread_metadata_for_target,
)
from gateway.gateway_message_pipeline import (
    _thread_metadata_for_source,
)
from gateway.gateway_config_loaders import (
    _load_prefill_messages,
    _load_ephemeral_system_prompt,
    _load_reasoning_config,
    _load_service_tier,
    _load_show_reasoning,
    _load_busy_input_mode,
    _load_busy_text_mode,
    _load_restart_drain_timeout,
    _load_background_notifications_mode,
    _load_provider_routing,
    _load_fallback_model,
    _load_voice_modes,
)
from gateway.gateway_skills import (
    _skill_slug_from_frontmatter,
    _check_unavailable_skill,
)
from gateway.gateway_message_builders import (
    _ASSISTANT_REPLAY_FIELDS,
    _TELEGRAM_OBSERVED_CONTEXT_PROMPT_MARKER,
    _OBSERVED_GROUP_CONTEXT_HEADER,
    _CURRENT_ADDRESSED_MESSAGE_HEADER,
    _build_replay_entry,
    _uses_telegram_observed_group_context,
    _build_gateway_agent_history,
    _wrap_current_message_with_observed_context,
    _last_transcript_timestamp,
    _build_media_placeholder,
    _build_document_context_note,
)
from gateway.gateway_transcript import (
    _AUTO_CONTINUE_NOTE_PREFIX,
    _AUTO_CONTINUE_FALLBACK_PREFIX,
    _is_interrupted_tool_result,
    _strip_interrupted_tool_tails,
    _is_auto_continue_noise,
    _strip_auto_continue_noise,
)
from gateway.gateway_response import (
    _GATEWAY_PROVIDER_POLICY_RE,
    _GATEWAY_AUTH_ERROR_RE,
    _GATEWAY_RATE_LIMIT_RE,
    _GATEWAY_SECRET_PATTERNS,
    _GATEWAY_PROVIDER_ERROR_SHAPE_RE,
    _gateway_platform_value,
    _redact_gateway_user_facing_secrets,
    _gateway_provider_error_reply,
    _looks_like_gateway_provider_error,
    _sanitize_gateway_final_response,
    _resolve_progress_thread_id,
    render_notice_line,
)
from typing import Dict, Optional, Any, List, Union

# account_usage imports the OpenAI SDK chain (~230 ms). Only needed by
# /usage; we still import it at module top in the gateway because test
# patches (tests/gateway/test_usage_command.py) target
# `gateway.run.fetch_account_usage` as a module-level attribute. The
# gateway is a long-running daemon, so its boot cost matters less than
# preserving the established test-patch surface.
from agent.account_usage import fetch_account_usage, render_account_usage_lines
from agent.async_utils import safe_schedule_threadsafe
from agent.i18n import t
from hermes_cli.config import cfg_get
from hermes_cli.fallback_config import get_fallback_chain

# --- Agent cache tuning ---------------------------------------------------
# Bounds the per-session AIAgent cache to prevent unbounded growth in
# long-lived gateways (each AIAgent holds LLM clients, tool schemas,
# memory providers, etc.).  LRU order + idle TTL eviction are enforced
# from _enforce_agent_cache_cap() and _session_expiry_watcher() below.
_AGENT_CACHE_MAX_SIZE = 128
_AGENT_CACHE_IDLE_TTL_SECS = 3600.0  # evict agents idle for >1h
_PLATFORM_CONNECT_TIMEOUT_SECS_DEFAULT = 30.0
_ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT = 5.0
_TELEGRAM_COMMAND_MENTION_RE = re.compile(r"(?<![\w:/])/([A-Za-z0-9][A-Za-z0-9_-]*)")

_TELEGRAM_NOISY_STATUS_RE = re.compile(
    r"("  # transient/auxiliary status that should stay in logs, not Telegram chat
    r"auxiliary\s+.+\s+failed"
    r"|compression\s+summary\s+failed"
    r"|fallback\s+context\s+marker"
    r"|configured\s+compression\s+model\s+.+\s+failed"
    r"|no\s+auxiliary\s+llm\s+provider\s+configured"
    r"|auto-lowered\s+compression\s+threshold"
    r"|compacting\s+context\s+[—-]\s+summarizing\s+earlier\s+conversation"
    r"|preflight\s+compression"
    r"|rate\s+limited\.\s+waiting\s+\d"
    r"|retrying\s+in\s+\d"
    r"|max\s+retries\s+\(\d+\).*(?:trying\s+fallback|exhausted|invalid\s+responses)"
    r"|stream\s+(?:drop|drop\s+mid\s+tool-call).+retry\s+\d"
    r"|stale\s+connections\s+from\s+a\s+previous\s+provider\s+issue"
    r")",
    re.IGNORECASE | re.DOTALL,
)

_GATEWAY_PROVIDER_ERROR_RE = re.compile(
    r"("  # infrastructure/provider error preambles, not ordinary assistant prose
    r"api\s+(?:call\s+)?failed"
    r"|provider\s+authentication\s+failed"
    r"|non-retryable\s+error"
    r"|rate\s+limited\s+after\s+\d+\s+retries"
    r"|error\s+code\s*:"
    r"|\bhttp\s*\d{3}\b"
    r"|incorrect\s+api\s+key"
    r"|invalid\s+api\s+key"
    r")",
    re.IGNORECASE,
)

def _ensure_windows_gateway_venv_imports() -> None:
    """Make detached Windows gateway runs see the Hermes venv packages.

    Some Windows restart paths run the gateway under uv's base ``pythonw.exe``
    to avoid the venv launcher respawning a visible console interpreter.  That
    mode can import the source tree via cwd/PYTHONPATH but still miss optional
    packages installed only in ``venv/Lib/site-packages`` (notably the MCP SDK).
    Patch the live process before MCP discovery so tool injection does not
    depend on every launcher preserving PYTHONPATH perfectly.
    """
    if sys.platform != "win32":
        return

    project_root = Path(__file__).resolve().parent.parent
    candidates: list[Path] = []
    if os.environ.get("VIRTUAL_ENV"):
        candidates.append(Path(os.environ["VIRTUAL_ENV"]))
    candidates.append(project_root / "venv")

    seen: set[str] = set()
    for venv_dir in candidates:
        try:
            resolved_venv = venv_dir.resolve()
        except OSError:
            resolved_venv = venv_dir
        venv_key = str(resolved_venv).lower()
        if venv_key in seen:
            continue
        seen.add(venv_key)

        site_packages = resolved_venv / "Lib" / "site-packages"
        if not site_packages.exists():
            continue

        project_entry = str(project_root)
        site_entry = str(site_packages)
        if project_entry not in sys.path:
            sys.path.insert(0, project_entry)
        # addsitepackages() semantics matter here: pywin32, used by the MCP
        # SDK on Windows, relies on .pth processing to expose pywintypes.
        site.addsitedir(site_entry)
        if site_entry in sys.path:
            sys.path.remove(site_entry)
        insert_at = 1 if sys.path and sys.path[0] == project_entry else 0
        sys.path.insert(insert_at, site_entry)

        os.environ["VIRTUAL_ENV"] = str(resolved_venv)
        pythonpath = [project_entry, site_entry]
        if os.environ.get("PYTHONPATH"):
            pythonpath.append(os.environ["PYTHONPATH"])
        os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath))
        return


def _is_transient_network_error(exc: BaseException) -> bool:
    """Return True for transient network errors safe to log + swallow.

    The crash class targeted by #31066 / #31110: an unhandled Telegram
    ``TimedOut`` (or peer ``NetworkError`` / ``httpx`` connection error)
    propagating to the event loop and killing the entire gateway
    process. These are by definition transient — the next poll cycle or
    user action recovers — so they must never crash the process.

    Walk the exception cause chain so wrapped errors (e.g. PTB's
    ``NetworkError`` wrapping ``httpx.ConnectError``) are still
    classified. The chain is bounded to avoid pathological cycles.
    """
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    depth = 0
    transient_class_names = {
        "TimedOut",
        "NetworkError",
        "ReadError",
        "WriteError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "RemoteProtocolError",
        "ServerDisconnectedError",
        "ClientConnectorError",
        "ClientOSError",
    }
    while cur is not None and depth < 12:
        ident = id(cur)
        if ident in seen:
            break
        seen.add(ident)
        depth += 1
        name = type(cur).__name__
        if name in transient_class_names:
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _redact_approval_command(cmd: "str | None") -> str:
    """Redact credentials from a command before it goes into an approval prompt.

    Tirith's *findings* are already redacted, but the gateway approval prompt
    is built from the raw command string, so a credential-shaped value Tirith
    flagged would otherwise be echoed verbatim to the chat platform (#48456).
    Uses ``redact_sensitive_text(force=True)`` - the same Tirith-grade redactor
    - so the prompt honors redaction even when ``security.redact_secrets`` is
    off. Module-level so the wiring is unit-testable.
    """
    from agent.redact import redact_sensitive_text

    return redact_sensitive_text(str(cmd or ""), force=True)


def _resolve_progress_thread_id(platform: Any, source_thread_id: Any, event_message_id: Any) -> Optional[str]:
    """Return thread/root ID that progress/status bubbles should target."""
    platform_value = getattr(platform, "value", platform)
    platform_key = str(platform_value or "").lower()
    if source_thread_id:
        return str(source_thread_id)
    if platform_key in {"slack", "mattermost"} and event_message_id:
        return str(event_message_id)
    return None


def _has_platform_display_override(user_config: dict, platform_key: str, setting: str) -> bool:
    """Return True when display.platforms.<platform> explicitly sets setting."""
    display = user_config.get("display") if isinstance(user_config, dict) else None
    if not isinstance(display, dict):
        return False
    platforms = display.get("platforms")
    if not isinstance(platforms, dict):
        return False
    platform_cfg = platforms.get(platform_key)
    return isinstance(platform_cfg, dict) and setting in platform_cfg


def _gateway_platform_value_local(platform: Any) -> str:
    """Local shim delegating to gateway.gateway_response (where our refactor
    moved _gateway_platform_value). Keeps upstream helper signatures intact."""
    from gateway.gateway_response import _gateway_platform_value
    return _gateway_platform_value(platform)


def _resolve_gateway_display_bool(
    user_config: dict,
    platform_key: str,
    setting: str,
    *,
    default: bool = False,
    platform: Any = None,
    require_platform_override_for: set = None,
) -> bool:
    """Resolve a boolean display setting with optional platform-only opt-in.

    Some display features expose assistant scratch text rather than deliberate
    user-facing output.  For high-noise threaded chat surfaces such as
    Mattermost, a global opt-in is too broad: they must be enabled with an
    explicit display.platforms.<platform>.<setting> override.
    """
    current_platform = _gateway_platform_value_local(platform or platform_key)
    platform_only = {
        _gateway_platform_value_local(candidate)
        for candidate in (require_platform_override_for or set())
    }
    if (
        current_platform in platform_only
        and not _has_platform_display_override(user_config, platform_key, setting)
    ):
        return False

    from gateway.display_config import resolve_display_setting

    value = resolve_display_setting(user_config, platform_key, setting, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "on"}
    if value is None:
        return bool(default)
    return bool(value)


def _gateway_loop_exception_handler(
    loop: "asyncio.AbstractEventLoop", context: Dict[str, Any]
) -> None:
    """Loop-level safety net for transient network errors.

    Installed once during :func:`start_gateway`. Catches the
    ``telegram.error.TimedOut`` crash class (issues #31066 / #31110)
    and any peer transient network error before it can kill the
    gateway process. Logs at WARNING with full traceback so the
    originating call site stays diagnosable; non-transient errors
    are forwarded to the default loop handler so real bugs still
    surface.
    """
    exc = context.get("exception")
    if exc is not None and _is_transient_network_error(exc):
        message = context.get("message") or "transient network error"
        task = context.get("future") or context.get("task")
        task_name = ""
        if task is not None:
            try:
                task_name = task.get_name() if hasattr(task, "get_name") else repr(task)
            except Exception:
                task_name = repr(task)
        logger.warning(
            "Gateway swallowed transient network error from %s: %s: %s",
            task_name or "<unknown task>",
            type(exc).__name__,
            exc,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return
    # Fall back to the default handler for anything we don't recognise.
    loop.default_exception_handler(context)


def _prepare_gateway_status_message(platform: Any, event_type: str, message: str) -> Optional[str]:
    """Filter/sanitize agent status callbacks before platform delivery."""
    text = str(message or "").strip()
    if not text:
        return None
    if _gateway_platform_value(platform) != "telegram":
        return text

    text = _redact_gateway_user_facing_secrets(text)
    if _TELEGRAM_NOISY_STATUS_RE.search(text):
        return None
    if _looks_like_gateway_provider_error(text):
        return _gateway_provider_error_reply(text)
    return text


async def _send_or_update_status_coro(adapter, chat_id, status_key, content, metadata):
    """Route a status message through adapter.send_or_update_status when supported.

    Issue #30045: adapters that implement send_or_update_status (currently
    Telegram) edit the previous bubble for the same status_key instead of
    appending a new one. Adapters without the method fall back to plain send.
    """
    sender = getattr(adapter, "send_or_update_status", None)
    if callable(sender):
        return await sender(chat_id, status_key, content, metadata=metadata)
    return await adapter.send(chat_id, content, metadata=metadata)


def _telegramize_command_mentions(text: str, platform: Any) -> str:
    """Rewrite slash-command mentions to Telegram-valid command names.

    Telegram Bot API command names allow only lowercase letters, digits, and
    underscores.  Keep other platform renderings unchanged, but normalize
    Telegram help text so command mentions remain clickable/valid there.
    """
    platform_value = getattr(platform, "value", platform)
    if platform_value != "telegram":
        return text

    from hermes_cli.commands import _sanitize_telegram_name

    def _replace(match: re.Match[str]) -> str:
        sanitized = _sanitize_telegram_name(match.group(1))
        return f"/{sanitized}" if sanitized else match.group(0)

    return _TELEGRAM_COMMAND_MENTION_RE.sub(_replace, text)


# Only auto-continue interrupted gateway turns while the interruption is fresh.
# Stale tool-tail/resume markers can otherwise revive an unrelated old task
# after a gateway restart when the user's next message starts new work.
#
# The freshness signal is the timestamp of the last transcript row, which
# ``hermes_state.get_messages`` carries on every persisted message.  This
# handles the two auto-continue cases uniformly:
#   * resume_pending (gateway restart/shutdown watchdog marked the session)
#   * tool-tail     (last persisted message is a tool result the agent
#                    never got to reply to)
# In both cases "when did we last do anything on this transcript" is the
# correct freshness question, so one signal replaces two divergent ones.
#
# Default window: 1 hour.  This comfortably covers ``agent.gateway_timeout``
# (30 min default) plus runtime slack — a legitimate long-running turn that
# gets interrupted near its timeout boundary and is resumed shortly after
# is still classified fresh.  Override via
# ``config.yaml`` ``agent.gateway_auto_continue_freshness``.
_AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT = 60 * 60


def _auto_continue_freshness_window() -> float:
    """Return the configured auto-continue freshness window in seconds.

    Reads ``HERMES_AUTO_CONTINUE_FRESHNESS`` (bridged from
    ``config.yaml`` ``agent.gateway_auto_continue_freshness`` at gateway
    startup, same pattern as ``HERMES_AGENT_TIMEOUT``).  Falls back to the
    module default when unset or malformed.  Non-positive values disable
    the freshness gate (restores the pre-fix "always fresh" behaviour for
    users who want to opt out).
    """
    raw = os.environ.get("HERMES_AUTO_CONTINUE_FRESHNESS")
    if raw is None or raw == "":
        return float(_AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(_AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT)


def _float_env(name: str, default: float) -> float:
    """Read an env var as float, falling back to ``default`` on typos/empty.

    A misconfigured env var (e.g. ``HERMES_AGENT_TIMEOUT=abc``) must not
    crash the gateway or an agent turn.  Unset/empty also falls back.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _is_fresh_gateway_interruption(
    value: Any,
    *,
    now: Optional[float] = None,
    window_secs: Optional[float] = None,
) -> bool:
    """Return True when an interruption marker is fresh enough to auto-continue.

    Unknown timestamps are treated as fresh for backward compatibility with
    legacy transcripts (pre-dating timestamp persistence) and with in-memory
    test scaffolding that constructs history entries without timestamps.

    A non-positive ``window_secs`` disables the gate (always fresh), which
    restores the pre-fix behaviour for users who opt out via config.
    """
    window = (
        float(window_secs)
        if window_secs is not None
        else float(_AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT)
    )
    if window <= 0:
        return True
    timestamp = _coerce_gateway_timestamp(value)
    if timestamp is None:
        return True
    current = time.time() if now is None else now
    return current - timestamp <= window


# Assistant-message fields that must survive transcript replay so multi-turn
# reasoning context, prefix-cache hits, and provider-specific echo
# requirements all behave the same on the gateway as they do in the CLI.
#
# ``reasoning`` and ``reasoning_details`` were the original three preserved
# by PR #2974 (schema v6).  ``reasoning_content``, ``codex_reasoning_items``,
# ``codex_message_items``, and ``finish_reason`` were added to the DB later
# but the gateway's replay whitelist was never expanded to match — so any
# pure-text assistant turn (no ``tool_calls``) silently dropped them on
# replay, regressing the CLI-vs-gateway behavioural parity.
#
# Why each field matters on replay:
#   * ``reasoning`` / ``reasoning_content``: provider-facing thinking text.
#     ``_copy_reasoning_content_for_api`` promotes ``reasoning`` →
#     ``reasoning_content`` at send time, but only when the strings happen to
#     match.  Carrying the original ``reasoning_content`` verbatim avoids
#     reconstruction loss for providers that return them as distinct fields
#     (DeepSeek/Kimi/Moonshot thinking modes).
#   * ``reasoning_details``: opaque structured array (signature,
#     encrypted_content) used by OpenRouter/Anthropic to maintain reasoning
#     continuity across turns.
#   * ``codex_reasoning_items``: encrypted reasoning blobs for the OpenAI
#     Codex Responses API.
#   * ``codex_message_items``: exact assistant message items with ``phase``.
#     OpenAI docs: "preserve and resend phase on all assistant messages —
#     dropping it can degrade performance."  Required for prefix cache hits.
#   * ``finish_reason``: informational; cheap to keep so transcripts replay
#     identically across CLI and gateway.
# Tool results can contain literal MEDIA: examples in docs, logs, or other
# ordinary outputs. Only tools that intentionally create deliverable media
# artifacts should be eligible for automatic append when the model omits them
# from the final gateway reply.
_AUTO_APPEND_MEDIA_TOOL_NAMES = {
    "text_to_speech",
    "text_to_speech_tool",
    "image_generate",
}

# ---- helpers: detect interrupted tool tails & auto-continue noise ----------

# Tools in this set return their deliverable artifact as a JSON payload with a
# local-file path field rather than a literal ``MEDIA:`` tag (e.g. image_generate
# returns ``{"success": true, "image": "/abs/path.png"}``). The auto-append path
# extracts the path from these fields so delivery is deterministic and does not
# depend on the model restating the path in its final reply.
_JSON_MEDIA_TOOL_PATH_FIELDS = ("host_image", "image", "agent_visible_image")


# Extension-anchored MEDIA: matcher for tool results. Mirrors the dispatch-site
# pattern so a bare ``MEDIA:`` token in prose (no deliverable extension) is never
# auto-appended. Kept local to the auto-append path; the producer-tool allowlist
# below is the primary guard, this is the secondary precision guard.
_TOOL_MEDIA_RE = re.compile(
    r'MEDIA:((?:[A-Za-z]:[/\\]|/|~\/)\S+\.(?:png|jpe?g|gif|webp|'
    r'mp4|mov|avi|mkv|webm|ogg|opus|mp3|wav|m4a|'
    r'flac|epub|pdf|zip|rar|7z|docx?|xlsx?|pptx?|'
    r'txt|csv|apk|ipa))',
    re.IGNORECASE,
)


def _collect_auto_append_media_tags(
    messages: List[Dict[str, Any]],
    history_offset: int = 0,
    history_media_paths: Optional[set] = None,
) -> tuple[List[str], bool]:
    """Collect real media tags from current-turn producer-tool results only.

    Two layered guards keep stale/example MEDIA: strings out of the reply:

    1. Producer-tool allowlist: only tools that intentionally emit deliverable
       artifacts (TTS) are eligible. Documentation, logs, and search results can
       contain example strings such as MEDIA:/absolute/path/to/file, which must
       never be delivered as attachments. (Fixes the original report behind #16721.)
    2. Current-turn isolation: only messages produced this turn are scanned, so a
       tool result from an earlier turn (still present in the full message list)
       cannot leak onto a later text-only reply (#34608).

    Mid-run context compression can rewrite/shrink the message list below the
    original history length. When that happens the slice boundary is no longer
    trustworthy, so fall back to scanning every message and rely on
    ``history_media_paths`` for dedup, preserving the compression-safe behaviour
    of #160. The producer-tool allowlist still applies on the fallback path.
    """
    history_media_paths = history_media_paths or set()
    # Only trust the slice boundary when the message list still contains the
    # full history prefix. Otherwise scan everything (compression-safe fallback).
    if history_offset and len(messages) >= history_offset:
        new_messages = messages[history_offset:]
    else:
        new_messages = messages

    tool_name_by_call_id: Dict[str, str] = {}
    for msg in new_messages:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            call_id = call.get("id") or call.get("call_id")
            fn = call.get("function") or {}
            name = str(fn.get("name") or call.get("name") or "")
            if call_id and name:
                tool_name_by_call_id[str(call_id)] = name

    media_tags: List[str] = []
    has_voice_directive = False
    for msg in new_messages:
        if msg.get("role") not in ("tool", "function"):
            continue
        call_id = str(msg.get("tool_call_id") or msg.get("call_id") or "")
        if tool_name_by_call_id.get(call_id) not in _AUTO_APPEND_MEDIA_TOOL_NAMES:
            continue
        content = str(msg.get("content") or "")
        tool_name = tool_name_by_call_id.get(call_id)
        # JSON-payload tools (image_generate) return a local-file path in a
        # known field rather than a MEDIA: tag. Extract it so delivery is
        # deterministic even when the model omits the path from its reply.
        if tool_name == "image_generate" and "MEDIA:" not in content:
            try:
                payload = json.loads(content)
            except Exception:
                payload = None
            if isinstance(payload, dict) and payload.get("success"):
                for field in _JSON_MEDIA_TOOL_PATH_FIELDS:
                    path = payload.get(field)
                    if (isinstance(path, str)
                            and _TOOL_MEDIA_RE.fullmatch(f"MEDIA:{path}")
                            and path not in history_media_paths):
                        media_tags.append(f"MEDIA:{path}")
                        break
            continue
        if "MEDIA:" not in content:
            continue
        for match in _TOOL_MEDIA_RE.finditer(content):
            path = match.group(1).strip().rstrip('",}')
            if path and path not in history_media_paths:
                media_tags.append(f"MEDIA:{path}")
        if "[[audio_as_voice]]" in content:
            has_voice_directive = True

    return media_tags, has_voice_directive


def _collect_history_media_paths(agent_history: List[Dict[str, Any]]) -> set:
    """Collect every media path already delivered in prior tool results.

    Used to dedup auto-appended MEDIA tags so the same file is not re-sent on
    later turns. Must cover BOTH delivery shapes:
      * ``MEDIA:<path>`` text tags in tool results, and
      * ``image_generate`` JSON-payload paths (``host_image`` / ``image`` /
        ``agent_visible_image``), which carry no MEDIA: tag.

    Missing the JSON-payload shape caused #46627: after a compression
    boundary the auto-append fallback rescans full history, re-discovers an
    earlier ``image_generate`` result whose path was never in the dedup set,
    and re-emits the MEDIA tag every turn.
    """
    paths: set = set()
    tool_name_by_call_id: Dict[str, str] = {}
    for msg in agent_history:
        if msg.get("role") == "assistant":
            for call in msg.get("tool_calls") or []:
                cid = call.get("id") or call.get("call_id")
                fn = call.get("function") or {}
                name = str(fn.get("name") or call.get("name") or "")
                if cid and name:
                    tool_name_by_call_id[str(cid)] = name
    for msg in agent_history:
        if msg.get("role") not in {"tool", "function"}:
            continue
        content = str(msg.get("content", "") or "")
        if "MEDIA:" in content:
            for match in _TOOL_MEDIA_RE.finditer(content):
                p = match.group(1).strip().rstrip('",}')
                if p:
                    paths.add(p)
            continue
        cid = str(msg.get("tool_call_id") or msg.get("call_id") or "")
        if tool_name_by_call_id.get(cid) == "image_generate":
            try:
                payload = json.loads(content)
            except Exception:
                payload = None
            if isinstance(payload, dict) and payload.get("success"):
                for field in _JSON_MEDIA_TOOL_PATH_FIELDS:
                    jp = payload.get(field)
                    if isinstance(jp, str) and jp:
                        paths.add(jp)
                        break
    return paths

# ---------------------------------------------------------------------------
# SSL certificate auto-detection for NixOS and other non-standard systems.
# Must run BEFORE any HTTP library (discord, aiohttp, etc.) is imported.
# ---------------------------------------------------------------------------
def _ensure_ssl_certs() -> None:
    """Set SSL_CERT_FILE if the system doesn't expose CA certs to Python.

    Windows startup paths (Desktop, Scheduled Tasks, installer children) can
    occasionally inherit a stale SSL_CERT_FILE. Returning just because the
    variable is present makes every later httpx/OpenAI client construction fail
    with FileNotFoundError from ssl.load_verify_locations(). Treat a missing
    path as unset and fall back to certifi instead.
    """
    configured_cert = os.environ.get("SSL_CERT_FILE")
    if configured_cert:
        if os.path.exists(configured_cert):
            return  # user already configured it to a real file
        logging.getLogger(__name__).warning(
            "Ignoring stale SSL_CERT_FILE=%r because the path does not exist",
            configured_cert,
        )
        os.environ.pop("SSL_CERT_FILE", None)

    import ssl

    # 1. Python's compiled-in defaults
    paths = ssl.get_default_verify_paths()
    for candidate in (paths.cafile, paths.openssl_cafile):
        if candidate and os.path.exists(candidate):
            os.environ["SSL_CERT_FILE"] = candidate
            return

    # 2. certifi (ships its own Mozilla bundle)
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
        return
    except ImportError:
        pass

    # 3. Common distro / macOS locations
    for candidate in (
        "/etc/ssl/certs/ca-certificates.crt",               # Debian/Ubuntu/Gentoo
        "/etc/pki/tls/certs/ca-bundle.crt",                 # RHEL/CentOS 7
        "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem", # RHEL/CentOS 8+
        "/etc/ssl/ca-bundle.pem",                            # SUSE/OpenSUSE
        "/etc/ssl/cert.pem",                                 # Alpine / macOS
        "/etc/pki/tls/cert.pem",                             # Fedora
        "/usr/local/etc/openssl@1.1/cert.pem",               # macOS Homebrew Intel
        "/opt/homebrew/etc/openssl@1.1/cert.pem",            # macOS Homebrew ARM
    ):
        if os.path.exists(candidate):
            os.environ["SSL_CERT_FILE"] = candidate
            return

def _home_target_env_var(platform_name: str) -> str:
    """Return the configured home-target env var for a platform.

    Consults built-in ``_HOME_TARGET_ENV_VARS`` first, then the plugin
    registry via ``cron.scheduler._resolve_home_env_var``, then falls back
    to ``<PLATFORM>_HOME_CHANNEL`` for unknown names.
    """
    from cron.scheduler import _resolve_home_env_var

    resolved = _resolve_home_env_var(platform_name)
    if resolved:
        return resolved
    return f"{platform_name.upper()}_HOME_CHANNEL"


def _home_thread_env_var(platform_name: str) -> str:
    """Return the optional thread/topic env var for a platform home target."""
    return f"{_home_target_env_var(platform_name)}_THREAD_ID"


def _restart_notification_pending() -> bool:
    """Return True when a /restart completion marker is waiting to be delivered."""
    return (_hermes_home / ".restart_notify.json").exists()


def _planned_restart_notification_path() -> Path:
    return _hermes_home / ".restart_pending.json"


def _planned_restart_notification_pending() -> bool:
    """Return True when a non-chat planned restart should notify home channels."""
    return _planned_restart_notification_path().exists()


def _clear_planned_restart_notification() -> None:
    _planned_restart_notification_path().unlink(missing_ok=True)


# Mark this process as a gateway so cli.py's module-level load_cli_config()
# knows not to clobber TERMINAL_CWD if lazily imported.
os.environ["_HERMES_GATEWAY"] = "1"

_ensure_ssl_certs()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Resolve Hermes home directory (respects HERMES_HOME override)
from hermes_constants import get_hermes_home
from utils import atomic_json_write, atomic_yaml_write, base_url_host_matches, is_truthy_value
_hermes_home = get_hermes_home()

# Load environment variables from ~/.hermes/.env first.
# User-managed env files should override stale shell exports on restart.
from dotenv import load_dotenv  # noqa: F401  # backward-compat for tests that monkeypatch this symbol
from hermes_cli.env_loader import load_hermes_dotenv
_env_path = _hermes_home / '.env'
load_hermes_dotenv(hermes_home=_hermes_home, project_env=Path(__file__).resolve().parents[1] / '.env')


def _reload_runtime_env_preserving_config_authority() -> None:
    """Reload .env for fresh credentials without letting stale .env override config.

    Gateway processes are long-lived, so per-turn code reloads ~/.hermes/.env to
    pick up rotated API keys. config.yaml remains authoritative for agent budget
    settings such as agent.max_turns; otherwise a stale HERMES_MAX_ITERATIONS in
    .env can replace the startup bridge on later turns.

    In multiplex mode this is a NO-OP for the credential reload: secrets come
    from the per-turn ``set_secret_scope`` (installed by ``_profile_runtime_scope``)
    which loads the routed profile's ``.env`` into an isolated mapping. Mutating
    the process-global ``os.environ`` here would defeat that isolation and leak
    the default profile's keys to every profile's turns and subprocesses.
    """
    from agent.secret_scope import is_multiplex_active
    if is_multiplex_active():
        # Credentials are resolved from the active profile's secret scope, not
        # os.environ. Still honor config.yaml's agent.max_turns bridge below
        # using the scoped home, but never reload .env into global env.
        _bridge_max_turns_from_config(_hermes_home)
        return

    load_hermes_dotenv(
        hermes_home=_hermes_home,
        project_env=Path(__file__).resolve().parents[1] / '.env',
    )
    _bridge_max_turns_from_config(_hermes_home)


def _bridge_max_turns_from_config(home: "Path") -> None:
    """Bridge config.yaml agent.max_turns into HERMES_MAX_ITERATIONS (a global)."""
    config_path = home / 'config.yaml'
    if not config_path.exists():
        return
    try:
        import yaml as _yaml
        with open(config_path, encoding="utf-8") as f:
            cfg = _yaml.safe_load(f) or {}
        from hermes_cli.config import _expand_env_vars
        cfg = _expand_env_vars(cfg)
        # Managed scope: keep administrator-pinned values authoritative on every
        # turn too. This per-turn reload re-bridges config→env, so without the
        # overlay a managed agent.max_turns / timezone / redact_secrets would be
        # replaced by the user's value after the first turn. Fail-open.
        try:
            from hermes_cli import managed_scope
            cfg = managed_scope.apply_managed_overlay(cfg)
        except Exception:
            pass
    except Exception:
        return

    agent_cfg = cfg.get("agent", {})
    if isinstance(agent_cfg, dict) and "max_turns" in agent_cfg:
        os.environ["HERMES_MAX_ITERATIONS"] = str(agent_cfg["max_turns"])


def _current_max_iterations() -> int:
    """Return the current per-turn iteration budget after runtime env refresh."""
    _reload_runtime_env_preserving_config_authority()
    try:
        return int(os.getenv("HERMES_MAX_ITERATIONS", "90"))
    except (TypeError, ValueError):
        return 90


from contextlib import contextmanager as _contextmanager


# Platforms that bind a host TCP port (HTTP/webhook listeners). In a profile
# multiplexer the default profile owns the single shared listener and serves
# every profile through the /p/<profile>/ URL prefix, so a SECONDARY profile
# enabling one of these is always a misconfiguration: it would try to bind a
# port already held by the default's listener. We hard-error on it rather than
# silently dropping the adapter (see _start_one_profile_adapters).
# Stored as platform .value strings since the Platform enum is imported below.
_PORT_BINDING_PLATFORM_VALUES = frozenset({
    "webhook",
    "api_server",
    "msgraph_webhook",
    "feishu",
    "wecom_callback",
    "bluebubbles",
    "sms",
})


class MultiplexConfigError(RuntimeError):
    """A profile multiplexer config is invalid (fail-fast at startup).

    Distinct from a transient adapter-connect failure: a transient error is
    logged and the gateway stays alive to retry, but a config error means the
    operator must fix config.yaml, so it aborts startup cleanly.
    """


@_contextmanager
def _profile_runtime_scope(profile_home: "Path"):
    """Scope config/skills/memory AND credentials to a profile for one turn.

    Combines the two seams the multiplexer needs:
      1. ``set_hermes_home_override`` — redirects ``get_hermes_home()`` (config,
         skills, memory, SOUL, sessions) to the profile's home. Contextvar, so
         it propagates into the agent worker thread via ``copy_context()``.
      2. ``set_secret_scope`` — installs the profile's ``.env`` secrets as the
         authoritative credential source, so ``get_secret`` reads this profile's
         keys and never the process-global ``os.environ`` (which in a
         multiplexer may hold another profile's values).

    Only used on the multiplexed inbound path. Single-profile gateways never
    enter this scope, so their behavior is unchanged. Loading the profile's
    ``.env`` here does NOT mutate ``os.environ`` — ``build_profile_secret_scope``
    returns an isolated dict — which is what keeps subprocesses (MCP, kanban)
    from inheriting cross-profile secrets.
    """
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from agent.secret_scope import (
        build_profile_secret_scope,
        set_secret_scope,
        reset_secret_scope,
    )

    home_token = set_hermes_home_override(str(profile_home))
    secret_token = set_secret_scope(build_profile_secret_scope(Path(profile_home)))
    try:
        yield
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


_DOCKER_VOLUME_SPEC_RE = re.compile(r"^(?P<host>.+):(?P<container>/[^:]+?)(?::(?P<options>[^:]+))?$")
_DOCKER_MEDIA_OUTPUT_CONTAINER_PATHS = {"/output", "/outputs"}

# Bridge config.yaml values into the environment so os.getenv() picks them up.
# config.yaml is authoritative for terminal settings — overrides .env.
_config_path = _hermes_home / 'config.yaml'
if _config_path.exists():
    try:
        import yaml as _yaml
        with open(_config_path, encoding="utf-8") as _f:
            _cfg = _yaml.safe_load(_f) or {}
        # Expand ${ENV_VAR} references before bridging to env vars.
        from hermes_cli.config import _expand_env_vars
        _cfg = _expand_env_vars(_cfg)
        # Managed scope: overlay administrator-pinned values BEFORE bridging to
        # env vars, so a managed timezone / redact_secrets / max_turns / terminal
        # setting wins over the user's value at the env layer too. This bridge
        # reads config.yaml directly (not via load_config), so without the
        # overlay every HERMES_*/TERMINAL_* env var below would carry the user's
        # value even when an administrator pinned it. Fail-open via the helper.
        try:
            from hermes_cli import managed_scope
            _cfg = managed_scope.apply_managed_overlay(_cfg)
        except Exception:
            pass
        # Top-level simple values (fallback only — don't override .env)
        for _key, _val in _cfg.items():
            if isinstance(_val, (str, int, float, bool)) and _key not in os.environ:
                os.environ[_key] = str(_val)
        # Terminal config is nested — bridge to TERMINAL_* env vars.
        # config.yaml overrides .env for these since it's the documented config path.
        _terminal_cfg = _cfg.get("terminal", {})
        if _terminal_cfg and isinstance(_terminal_cfg, dict):
            _terminal_env_map = {
                "backend": "TERMINAL_ENV",
                "cwd": "TERMINAL_CWD",
                "timeout": "TERMINAL_TIMEOUT",
                "home_mode": "TERMINAL_HOME_MODE",
                "lifetime_seconds": "TERMINAL_LIFETIME_SECONDS",
                "docker_image": "TERMINAL_DOCKER_IMAGE",
                "docker_forward_env": "TERMINAL_DOCKER_FORWARD_ENV",
                "singularity_image": "TERMINAL_SINGULARITY_IMAGE",
                "modal_image": "TERMINAL_MODAL_IMAGE",
                "daytona_image": "TERMINAL_DAYTONA_IMAGE",
                "ssh_host": "TERMINAL_SSH_HOST",
                "ssh_user": "TERMINAL_SSH_USER",
                "ssh_port": "TERMINAL_SSH_PORT",
                "ssh_key": "TERMINAL_SSH_KEY",
                "container_cpu": "TERMINAL_CONTAINER_CPU",
                "container_memory": "TERMINAL_CONTAINER_MEMORY",
                "container_disk": "TERMINAL_CONTAINER_DISK",
                "container_persistent": "TERMINAL_CONTAINER_PERSISTENT",
                "docker_volumes": "TERMINAL_DOCKER_VOLUMES",
                "docker_env": "TERMINAL_DOCKER_ENV",
                "docker_extra_args": "TERMINAL_DOCKER_EXTRA_ARGS",
                "docker_mount_cwd_to_workspace": "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE",
                "docker_run_as_host_user": "TERMINAL_DOCKER_RUN_AS_HOST_USER",
                "docker_persist_across_processes": "TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES",
                "docker_orphan_reaper": "TERMINAL_DOCKER_ORPHAN_REAPER",
                "sandbox_dir": "TERMINAL_SANDBOX_DIR",
                "persistent_shell": "TERMINAL_PERSISTENT_SHELL",
            }
            for _cfg_key, _env_var in _terminal_env_map.items():
                if _cfg_key in _terminal_cfg:
                    _val = _terminal_cfg[_cfg_key]
                    # Skip cwd placeholder values (".", "auto", "cwd") — the
                    # gateway resolves these to Path.home() later (line ~255).
                    # Writing the raw placeholder here would just be noise.
                    # Only bridge explicit absolute paths from config.yaml.
                    if _cfg_key == "cwd" and str(_val) in {".", "auto", "cwd"}:
                        continue
                    # Expand shell tilde in cwd so subprocess.Popen never
                    # receives a literal "~/" which the kernel rejects.
                    if _cfg_key == "cwd" and isinstance(_val, str):
                        _val = os.path.expanduser(_val)
                    if isinstance(_val, (list, dict)):
                        os.environ[_env_var] = json.dumps(_val)
                    else:
                        os.environ[_env_var] = str(_val)
        # Compression config is read directly from config.yaml by run_agent.py
        # and auxiliary_client.py — no env var bridging needed.
        # Auxiliary model/direct-endpoint overrides (vision, web_extract,
        # approval, plus any plugin-registered auxiliary tasks).
        # Each task has provider/model/base_url/api_key; bridge non-default
        # values to env vars named AUXILIARY_<KEY_UPPER>_*. The legacy
        # hard-coded list (vision/web_extract/approval) is replaced by a
        # dynamic loop so plugin-registered tasks benefit from the same
        # config→env bridging without core knowing about each one.
        _auxiliary_cfg = _cfg.get("auxiliary", {})
        if _auxiliary_cfg and isinstance(_auxiliary_cfg, dict):
            # Built-in tasks that previously had explicit env-var bridging.
            # Kept here as the canonical bridged set; plugin tasks are added
            # below via the plugin auxiliary registry.
            _aux_bridged_keys = {"vision", "web_extract", "approval"}
            try:
                from hermes_cli.plugins import get_plugin_auxiliary_tasks
                for _entry in get_plugin_auxiliary_tasks():
                    _aux_bridged_keys.add(_entry["key"])
            except Exception:
                # Plugin discovery failure must not break gateway startup;
                # built-in bridging stays intact.
                pass

            for _task_key in _aux_bridged_keys:
                _task_cfg = _auxiliary_cfg.get(_task_key, {})
                if not isinstance(_task_cfg, dict):
                    continue
                _prov = str(_task_cfg.get("provider", "")).strip()
                _model = str(_task_cfg.get("model", "")).strip()
                _base_url = str(_task_cfg.get("base_url", "")).strip()
                _api_key = str(_task_cfg.get("api_key", "")).strip()
                _upper = _task_key.upper()
                if _prov and _prov != "auto":
                    os.environ[f"AUXILIARY_{_upper}_PROVIDER"] = _prov
                if _model:
                    os.environ[f"AUXILIARY_{_upper}_MODEL"] = _model
                if _base_url:
                    os.environ[f"AUXILIARY_{_upper}_BASE_URL"] = _base_url
                if _api_key:
                    os.environ[f"AUXILIARY_{_upper}_API_KEY"] = _api_key
        # config.yaml is the documented, authoritative source for these
        # settings — it unconditionally wins over .env values. Previously
        # the guards below read `if X not in os.environ` and let stale
        # .env entries (e.g. HERMES_MAX_ITERATIONS=60 written by an old
        # `hermes setup` run) silently shadow the user's current config.
        # See PR #18413 / the 60-vs-500 max_turns incident.
        _agent_cfg = _cfg.get("agent", {})
        if _agent_cfg and isinstance(_agent_cfg, dict):
            if "max_turns" in _agent_cfg:
                os.environ["HERMES_MAX_ITERATIONS"] = str(_agent_cfg["max_turns"])
            if "gateway_timeout" in _agent_cfg:
                os.environ["HERMES_AGENT_TIMEOUT"] = str(_agent_cfg["gateway_timeout"])
            if "gateway_timeout_warning" in _agent_cfg:
                os.environ["HERMES_AGENT_TIMEOUT_WARNING"] = str(_agent_cfg["gateway_timeout_warning"])
            if "gateway_notify_interval" in _agent_cfg:
                os.environ["HERMES_AGENT_NOTIFY_INTERVAL"] = str(_agent_cfg["gateway_notify_interval"])
            if "restart_drain_timeout" in _agent_cfg:
                os.environ["HERMES_RESTART_DRAIN_TIMEOUT"] = str(_agent_cfg["restart_drain_timeout"])
            if "gateway_auto_continue_freshness" in _agent_cfg:
                os.environ["HERMES_AUTO_CONTINUE_FRESHNESS"] = str(
                    _agent_cfg["gateway_auto_continue_freshness"]
                )
        _display_cfg = _cfg.get("display", {})
        if _display_cfg and isinstance(_display_cfg, dict):
            if "busy_input_mode" in _display_cfg:
                os.environ["HERMES_GATEWAY_BUSY_INPUT_MODE"] = str(_display_cfg["busy_input_mode"])
            if "busy_text_mode" in _display_cfg:
                os.environ["HERMES_GATEWAY_BUSY_TEXT_MODE"] = str(_display_cfg["busy_text_mode"])
            if "busy_ack_enabled" in _display_cfg:
                os.environ["HERMES_GATEWAY_BUSY_ACK_ENABLED"] = str(_display_cfg["busy_ack_enabled"])
        # Timezone: bridge config.yaml → HERMES_TIMEZONE env var.
        _tz_cfg = _cfg.get("timezone", "")
        if _tz_cfg and isinstance(_tz_cfg, str):
            os.environ["HERMES_TIMEZONE"] = _tz_cfg.strip()
        # Security settings
        _security_cfg = _cfg.get("security", {})
        if isinstance(_security_cfg, dict):
            _redact = _security_cfg.get("redact_secrets")
            if _redact is not None:
                os.environ["HERMES_REDACT_SECRETS"] = str(_redact).lower()
        # Gateway settings (media delivery allowlist + recency trust + strict mode)
        _gateway_cfg = _cfg.get("gateway", {})
        if isinstance(_gateway_cfg, dict):
            _strict = _gateway_cfg.get("strict")
            if _strict is not None:
                os.environ["HERMES_MEDIA_DELIVERY_STRICT"] = (
                    "1" if _strict else "0"
                )
            _allow_dirs = _gateway_cfg.get("media_delivery_allow_dirs")
            if _allow_dirs:
                if isinstance(_allow_dirs, str):
                    _allow_dirs_str = _allow_dirs
                elif isinstance(_allow_dirs, (list, tuple)):
                    _allow_dirs_str = os.pathsep.join(str(p) for p in _allow_dirs if p)
                else:
                    _allow_dirs_str = ""
                if _allow_dirs_str:
                    os.environ["HERMES_MEDIA_ALLOW_DIRS"] = _allow_dirs_str
            _trust_recent = _gateway_cfg.get("trust_recent_files")
            if _trust_recent is not None:
                os.environ["HERMES_MEDIA_TRUST_RECENT_FILES"] = (
                    "1" if _trust_recent else "0"
                )
            _trust_recent_seconds = _gateway_cfg.get("trust_recent_files_seconds")
            if _trust_recent_seconds is not None:
                os.environ["HERMES_MEDIA_TRUST_RECENT_SECONDS"] = str(_trust_recent_seconds)
    except Exception as _bridge_err:
        # Previously this was silent (`except Exception: pass`), which
        # hid partial bridge failures and let .env defaults shadow
        # config.yaml values — users observed max_turns=500 in config
        # but a 60-iteration cap in practice. Surface the failure to
        # stderr so operators see it even though `logger` is not yet
        # initialized at module-import time (logger is defined further
        # down this module).
        print(
            f"  Warning: config.yaml → env bridge failed: "
            f"{type(_bridge_err).__name__}: {_bridge_err}",
            file=sys.stderr,
        )
        print(
            "  Gateway will fall back to .env values, which may not match "
            "your current config.yaml. Run `hermes doctor` to investigate.",
            file=sys.stderr,
        )

# Apply IPv4 preference if configured (before any HTTP clients are created).
try:
    from hermes_constants import apply_ipv4_preference
    _network_cfg = (_cfg if '_cfg' in dir() else {}).get("network", {})
    if isinstance(_network_cfg, dict) and _network_cfg.get("force_ipv4"):
        apply_ipv4_preference(force=True)
except Exception as _bootstrap_exc:
    print(f"  Warning: IPv4 preference application failed: {_bootstrap_exc}", file=sys.stderr)

# Validate config structure early — log warnings so gateway operators see problems
try:
    from hermes_cli.config import print_config_warnings
    print_config_warnings()
except Exception as _bootstrap_exc:
    print(f"  Warning: config validation failed: {_bootstrap_exc}", file=sys.stderr)

# Warn if user has deprecated MESSAGING_CWD / TERMINAL_CWD in .env
try:
    from hermes_cli.config import warn_deprecated_cwd_env_vars
    warn_deprecated_cwd_env_vars()
except Exception as _bootstrap_exc:
    print(f"  Warning: deprecation check failed: {_bootstrap_exc}", file=sys.stderr)

# Gateway runs in quiet mode - suppress debug output and use cwd directly (no temp dirs)
os.environ["HERMES_QUIET"] = "1"

# Enable interactive exec approval for dangerous commands on messaging platforms
os.environ["HERMES_EXEC_ASK"] = "1"

# Set terminal working directory for messaging platforms.
# config.yaml terminal.cwd is the canonical source (bridged to TERMINAL_CWD
# by the config bridge above).  When it's unset or a placeholder, default
# to home directory.  MESSAGING_CWD is accepted as a backward-compat
# fallback (deprecated — the warning above tells users to migrate).
_configured_cwd = os.environ.get("TERMINAL_CWD", "")
if not _configured_cwd or _configured_cwd in {".", "auto", "cwd"}:
    _fallback = os.getenv("MESSAGING_CWD") or str(Path.home())
    os.environ["TERMINAL_CWD"] = _fallback

from gateway.config import (
    Platform,
    _BUILTIN_PLATFORM_VALUES,
    GatewayConfig,
    HomeChannel,
    PlatformConfig,
    load_gateway_config,
)
from gateway.session import (
    SessionStore,
    SessionSource,
    SessionContext,
    build_session_context,
    build_session_context_prompt,
    build_session_key,
    is_shared_multi_user_session,
)
from gateway.delivery import DeliveryRouter
from gateway.authz_mixin import GatewayAuthorizationMixin
from gateway.session_cache_mixin import GatewaySessionCacheMixin
from gateway.process_mixin import GatewayProcessMixin
from gateway.telegram_topics_mixin import GatewayTelegramTopicsMixin
from gateway.voice_mixin import GatewayVoiceMixin
from gateway.startup_mixin import GatewayStartupMixin
from gateway.running_mixin import GatewayRunningMixin
from gateway.exit_request_mixin import GatewayExitRequestMixin
from gateway.active_session_mixin import GatewayActiveSessionMixin
from gateway.session_key_mixin import GatewaySessionKeyMixin
from gateway.agent_cache_mixin import GatewayAgentCacheMixin
from gateway.update_progress_mixin import GatewayUpdateProgressMixin
from gateway.drain_queue_mixin import GatewayDrainQueueMixin
from gateway.transcription_mixin import GatewayTranscriptionMixin
from gateway.platform_failover_mixin import GatewayPlatformFailoverMixin
from gateway.session_misc_mixin import GatewaySessionMiscMixin
from gateway.misc_tiny_mixin import MiscTinyMixin
from gateway.restart_notify_mixin import RestartNotifyMixin
from gateway.goal_continuation_mixin import GoalContinuationMixin
from gateway.background_task_mixin import BackgroundTaskMixin
from gateway.create_adapter_mixin import CreateAdapterMixin
from gateway.session_expiry_mixin import SessionExpiryMixin
from gateway.teams_docker_media_mixin import TeamsDockerMediaMixin
from gateway.turn_agent_config_mixin import TurnAgentConfigMixin
from gateway.platform_reconnect_mixin import PlatformReconnectMixin
from gateway.async_delegation_mixin import AsyncDelegationMixin
from gateway.slash_confirm_mixin import SlashConfirmMixin
from gateway.mcp_reload_mixin import McpReloadMixin
from gateway.media_delivery_mixin import MediaDeliveryMixin
from gateway.inbound_text_mixin import InboundTextMixin
from gateway.startup_preflight_mixin import StartupPreflightMixin
from gateway.stop_mixin import GatewayStopMixin
from gateway.busy_agent_dispatch_mixin import BusyAgentDispatchMixin
from gateway.handle_message_with_agent_mixin import HandleMessageWithAgentMixin
from gateway.run_agent_mixin import RunAgentMixin
from gateway.run_agent_via_proxy_mixin import RunAgentViaProxyMixin
from gateway.start_mixin_r54 import GatewayStartMixin
from gateway.gateway_entrypoint import (  # R57: re-export for backward compat
    _run_planned_stop_watcher,
    _start_cron_ticker,
    main,
    start_gateway,
)
from gateway.handle_message_mixin import HandleMessageMixin
from gateway.goals_mixin import GatewayGoalsMixin
from gateway.kanban_watchers import GatewayKanbanWatchersMixin
from gateway.slash_commands import GatewaySlashCommandsMixin
from gateway.platforms.base import (
    BasePlatformAdapter,
    EphemeralReply,
    MessageEvent,
    MessageType,
    _reply_anchor_for_event,
    merge_pending_message_event,
)
from gateway.restart import (
    DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
    GATEWAY_SERVICE_RESTART_EXIT_CODE,
    parse_restart_drain_timeout,
)


from gateway.whatsapp_identity import (
    canonical_whatsapp_identifier as _canonical_whatsapp_identifier,  # noqa: F401
    expand_whatsapp_aliases as _expand_whatsapp_auth_aliases,
    normalize_whatsapp_identifier as _normalize_whatsapp_identifier,
)


logger = logging.getLogger(__name__)


# Sentinel placed into _running_agents immediately when a session starts
# processing, *before* any await.  Prevents a second message for the same
# session from bypassing the "already running" guard during the async gap
# between the guard check and actual agent creation.
_AGENT_PENDING_SENTINEL = object()


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
    from hermes_cli.runtime_provider import (
        resolve_runtime_provider,
        format_runtime_provider_error,
        _get_model_config,
    )
    from hermes_cli.auth import AuthError, is_rate_limited_auth_error

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

    model_cfg = _get_model_config()
    max_tokens = None
    _env_mt = os.environ.get("HERMES_MAX_TOKENS")
    if _env_mt:
        try:
            max_tokens = int(_env_mt)
        except (ValueError, TypeError):
            max_tokens = None
    elif isinstance(model_cfg, dict):
        mt = model_cfg.get("max_tokens")
        if isinstance(mt, int):
            max_tokens = mt
    # Fall back to a per-provider output cap (custom_providers max_output_tokens)
    # only when the documented global model.max_tokens isn't set, so the global
    # key always wins.
    if max_tokens is None:
        _runtime_mot = runtime.get("max_output_tokens")
        if isinstance(_runtime_mot, int) and _runtime_mot > 0:
            max_tokens = _runtime_mot

    return {
        "api_key": runtime.get("api_key"),
        "base_url": runtime.get("base_url"),
        "provider": runtime.get("provider"),
        "api_mode": runtime.get("api_mode"),
        "command": runtime.get("command"),
        "args": list(runtime.get("args") or []),
        "credential_pool": runtime.get("credential_pool"),
        "max_tokens": max_tokens,
    }


def _try_resolve_fallback_provider() -> dict | None:
    """Attempt to resolve credentials from the fallback_model/fallback_providers config."""
    from hermes_cli.runtime_provider import resolve_runtime_provider
    try:
        import yaml as _y
        cfg_path = _hermes_home / "config.yaml"
        if not cfg_path.exists():
            return None
        with open(cfg_path, encoding="utf-8") as _f:
            cfg = _y.safe_load(_f) or {}
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
                continue
    except Exception:
        pass
    return None


async def _probe_audio_duration(path: str) -> Optional[str]:
    """Best-effort duration probe. Returns formatted MM:SS / HH:MM:SS, or None on failure."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".wav":
        try:
            def _wav_duration() -> float:
                import wave
                with wave.open(path, "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate() or 1
                    return frames / float(rate)
            secs = await asyncio.to_thread(_wav_duration)
            return _format_duration(secs)
        except Exception:
            pass

    if ext in (".ogg", ".opus", ".oga"):
        try:
            def _ogg_duration() -> float:
                from mutagen.oggopus import OggOpus
                return float(OggOpus(path).info.length)
            secs = await asyncio.to_thread(_ogg_duration)
            return _format_duration(secs)
        except Exception:
            pass

    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        if proc.returncode == 0:
            return _format_duration(float(stdout.decode().strip()))
    except Exception:
        pass

    return None


def _dequeue_pending_event(adapter, session_key: str) -> MessageEvent | None:
    """Consume and return the full pending event for a session.

    Queued follow-ups must preserve their media metadata so they can re-enter
    the normal image/STT/document preprocessing path instead of being reduced
    to a placeholder string.
    """
    return adapter.get_pending_message(session_key)


_INTERRUPT_REASON_STOP = "Stop requested"
_INTERRUPT_REASON_RESET = "Session reset requested"
_INTERRUPT_REASON_TIMEOUT = "Execution timed out (inactivity)"
_INTERRUPT_REASON_SSE_DISCONNECT = "SSE client disconnected"
_INTERRUPT_REASON_GATEWAY_SHUTDOWN = "Gateway shutting down"
_INTERRUPT_REASON_GATEWAY_RESTART = "Gateway restarting"

_CONTROL_INTERRUPT_MESSAGES = frozenset(
    {
        _INTERRUPT_REASON_STOP.lower(),
        _INTERRUPT_REASON_RESET.lower(),
        _INTERRUPT_REASON_TIMEOUT.lower(),
        _INTERRUPT_REASON_SSE_DISCONNECT.lower(),
        _INTERRUPT_REASON_GATEWAY_SHUTDOWN.lower(),
        _INTERRUPT_REASON_GATEWAY_RESTART.lower(),
    }
)


def _is_control_interrupt_message(message: Optional[str]) -> bool:
    """Return True when an interrupt message is internal control flow."""
    if not message:
        return False
    normalized = " ".join(str(message).strip().split()).lower()
    return normalized in _CONTROL_INTERRUPT_MESSAGES


def _platform_config_key(platform: "Platform") -> str:
    """Map a Platform enum to its config.yaml key (LOCAL→"cli", rest→enum value)."""
    return "cli" if platform == Platform.LOCAL else platform.value


def _teams_pipeline_plugin_enabled() -> bool:
    """Return True when the standalone Teams pipeline plugin is enabled."""
    config = _load_gateway_config()
    enabled = cfg_get(config, "plugins", "enabled", default=[])
    if not isinstance(enabled, list):
        return False
    return "teams_pipeline" in enabled or "teams-pipeline" in enabled


def _load_gateway_config() -> dict:
    """Load and parse ~/.hermes/config.yaml, returning {} on any error.

    Uses the module-level ``_hermes_home`` (so tests that monkeypatch it
    still see their fixture) and shares the mtime-keyed raw-yaml cache
    from ``hermes_cli.config.read_raw_config`` when the paths match.

    Managed scope is overlaid on the result (via the shared helper) so the
    gateway honors administrator-pinned values — neither read_raw_config nor a
    direct yaml.safe_load carries the managed merge on its own. Fail-open.
    """
    config_path = _hermes_home / 'config.yaml'
    raw: dict = {}
    used_canonical = False
    try:
        from hermes_cli.config import get_config_path, read_raw_config
        # Fast path: if _hermes_home agrees with the canonical config
        # location, reuse the shared cache. Otherwise fall through to a
        # direct read (keeps test fixtures with a monkeypatched
        # _hermes_home working).
        if config_path == get_config_path():
            raw = read_raw_config()
            used_canonical = True
    except Exception:
        pass

    if not used_canonical:
        try:
            if config_path.exists():
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    raw = yaml.safe_load(f) or {}
        except Exception:
            logger.debug("Could not load gateway config from %s", config_path)
            raw = {}

    # Overlay managed scope. read_raw_config() returns the user's raw YAML
    # WITHOUT the managed merge (that lives in load_config/_load_config_impl),
    # so the overlay is required on both paths for the gateway to honor pinned
    # values. Helper is fail-open and a no-op when no managed scope exists.
    try:
        from hermes_cli import managed_scope
        raw = managed_scope.apply_managed_overlay(raw if isinstance(raw, dict) else {})
    except Exception:
        pass
    return raw if isinstance(raw, dict) else {}


def _load_gateway_runtime_config() -> dict:
    """Load gateway config for runtime reads, expanding supported ``${VAR}`` refs.

    Runtime helpers should honor the same env-template expansion documented for
    ``config.yaml`` while still respecting tests that monkeypatch
    ``gateway.run._hermes_home``. Build on ``_load_gateway_config()`` rather
    than calling the canonical loader directly so both behaviors stay aligned.

    Expansion failures are intentionally NOT swallowed — silently returning
    the unexpanded dict would mask the very bug this helper exists to fix.
    """
    cfg = _load_gateway_config()
    if not isinstance(cfg, dict) or not cfg:
        return {}
    from hermes_cli.config import _expand_env_vars

    expanded = _expand_env_vars(cfg)
    return expanded if isinstance(expanded, dict) else {}


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


def _resolve_hermes_bin() -> Optional[list[str]]:
    """Resolve the Hermes update command as argv parts.

    Tries in order:
    1. ``shutil.which("hermes")`` — standard PATH lookup
    2. ``sys.executable -m hermes_cli.main`` — fallback when Hermes is running
       from a venv/module invocation and the ``hermes`` shim is not on PATH

    Returns argv parts ready for quoting/joining, or ``None`` if neither works.
    """
    import shutil

    hermes_bin = shutil.which("hermes")
    if hermes_bin:
        return [hermes_bin]

    try:
        import importlib.util

        if importlib.util.find_spec("hermes_cli") is not None:
            return [sys.executable, "-m", "hermes_cli.main"]
    except Exception:
        pass

    return None


def _parse_session_key(session_key: str) -> "dict | None":
    """Parse a session key into its component parts.

    Session keys follow the format
    ``agent:main:{platform}:{chat_type}:{chat_id}[:{extra}...]``.
    Returns a dict with ``platform``, ``chat_type``, ``chat_id``, and
    optionally ``thread_id`` keys, or None if the key doesn't match.

    The 6th element is only returned as ``thread_id`` for chat types where
    it is unambiguous (``dm`` and ``thread``).  For group/channel sessions
    the suffix may be a user_id (per-user isolation) rather than a
    thread_id, so we leave ``thread_id`` out to avoid mis-routing.
    """
    parts = session_key.split(":")
    if len(parts) >= 5 and parts[0] == "agent" and parts[1] == "main":
        result = {
            "platform": parts[2],
            "chat_type": parts[3],
            "chat_id": parts[4],
        }
        if len(parts) > 5 and parts[3] in {"dm", "thread"}:
            result["thread_id"] = parts[5]
        return result
    return None


def _format_gateway_process_notification(evt: dict) -> "str | None":
    """Format a watch pattern event from completion_queue into a [IMPORTANT:] message."""
    evt_type = evt.get("type", "completion")
    _sid = evt.get("session_id", "unknown")
    _cmd = evt.get("command", "unknown")

    if evt_type == "watch_disabled":
        return f"[IMPORTANT: {evt.get('message', '')}]"

    if evt_type == "watch_match":
        _pat = evt.get("pattern", "?")
        _out = evt.get("output", "")
        _sup = evt.get("suppressed", 0)
        text = (
            f"[IMPORTANT: Background process {_sid} matched "
            f"watch pattern \"{_pat}\".\n"
            f"Command: {_cmd}\n"
            f"Matched output:\n{_out}"
        )
        if _sup:
            text += f"\n({_sup} earlier matches were suppressed by rate limit)"
        text += "]"
        return text

    if evt_type == "async_delegation":
        # Reuse the shared rich formatter (self-contained task-source block).
        from tools.process_registry import format_process_notification
        return format_process_notification(evt)

    return None


def _drain_gateway_watch_events(completion_queue) -> "list[dict]":
    """Drain gateway-owned watch events without spinning on requeued events.

    Watch events are handled by the post-turn gateway drain. Process
    completions are owned by their per-process watcher task, and async
    delegation completions are owned by ``_async_delegation_watcher``.
    Requeueing async events inside ``while not queue.empty()`` would make the
    loop non-terminating, so detach the current batch first, then requeue any
    events this drain does not own after the queue is empty.
    """
    watch_events: list[dict] = []
    requeue: list[dict] = []
    while not completion_queue.empty():
        try:
            evt = completion_queue.get_nowait()
        except Exception:
            break
        evt_type = evt.get("type", "completion")
        if evt_type in {"watch_match", "watch_disabled"}:
            watch_events.append(evt)
        elif evt_type == "async_delegation":
            requeue.append(evt)
        # else: process completion events are handled by the watcher task
    for evt in requeue:
        completion_queue.put(evt)
    return watch_events


# Module-level weak reference to the active GatewayRunner instance.
# Used by tools (e.g. send_message) that need to route through a live
# adapter for plugin platforms.  Set in GatewayRunner.__init__().
import weakref as _weakref
_gateway_runner_ref: _weakref.ref = lambda: None


def _normalize_empty_agent_response(
    agent_result: dict,
    response: str,
    *,
    history_len: int = 0,
) -> str:
    """Normalize empty/None agent responses into user-facing messages.

    Consolidates the existing ``failed`` handler and adds a catch-all for
    the case where the agent did work (api_calls > 0) but returned no text.
    Fix for #18765.
    """
    if response:
        return response

    if agent_result.get("failed"):
        error_detail = agent_result.get("error", "unknown error")
        error_str = str(error_detail).lower()
        is_context_failure = any(
            p in error_str
            for p in ("context", "token", "too large", "too long", "exceed", "payload")
        ) or ("400" in error_str and history_len > 50)
        if is_context_failure:
            return (
                "⚠️ Session too large for the model's context window.\n"
                "Use /compact to compress the conversation, or "
                "/reset to start fresh."
            )
        return (
            f"The request failed: {str(error_detail)[:300]}\n"
            "Try again or use /reset to start a fresh session."
        )

    api_calls = int(agent_result.get("api_calls", 0) or 0)
    if api_calls > 0 and not agent_result.get("interrupted"):
        if agent_result.get("partial"):
            err = agent_result.get("error", "processing incomplete")
            return f"⚠️ Processing stopped: {str(err)[:200]}. Try again."
        return (
            "⚠️ Processing completed but no response was generated. "
            "This may be a transient error — try sending your message again."
        )

    return response


def _should_clear_resume_pending_after_turn(agent_result: dict) -> bool:
    """Return True only when a gateway turn really completed successfully.

    Restart recovery uses ``resume_pending`` as a durable marker for sessions
    interrupted during gateway drain.  A soft interrupt can still bubble out as
    a syntactically normal agent result with an empty final response; clearing
    the marker in that case loses the recovery signal and startup auto-resume
    has nothing to schedule.
    """
    if not isinstance(agent_result, dict):
        return False
    if agent_result.get("interrupted"):
        return False
    if agent_result.get("failed") or agent_result.get("partial") or agent_result.get("error"):
        return False
    if agent_result.get("completed") is False:
        return False
    return True


def _preserve_queued_followup_history_offset(
    current_result: dict,
    followup_result: dict,
) -> dict:
    """Carry the outer history offset through queued follow-up drains.

    ``_process_message_background()`` persists transcript rows only once, after the
    entire in-band queued-follow-up chain returns.  Each recursive ``_run_agent()``
    call advances ``history_offset`` to the history it received, so without
    correction the outermost persistence step sees only the *last* queued turn as
    "new" and silently drops earlier turns from the same drain chain.

    Preserve the earliest (outermost) history offset so the final transcript slice
    still includes every queued turn that ran during the chain.
    """
    if not isinstance(followup_result, dict):
        return followup_result
    if not isinstance(current_result, dict):
        return followup_result

    current_offset = current_result.get("history_offset")
    followup_offset = followup_result.get("history_offset")
    if not isinstance(current_offset, int):
        return followup_result
    if isinstance(followup_offset, int) and followup_offset <= current_offset:
        return followup_result

    merged = dict(followup_result)
    merged["history_offset"] = current_offset
    return merged


async def _dispose_unused_adapter(adapter: "BasePlatformAdapter | None") -> None:
    """Best-effort dispose for an adapter that never made it onto ``self.adapters``.

    The reconnect watcher in ``GatewayRunner._platform_reconnect_watcher``
    constructs a fresh adapter on every retry attempt. When the connect
    call fails — for any of the three reasons (non-retryable error,
    retryable error, exception during connect) — the adapter is dropped
    without ever being installed, so nothing else will call its
    ``disconnect()``. Any resources the adapter opened in ``__init__``
    (e.g. ``APIServerAdapter`` opens a SQLite ``ResponseStore`` that
    holds 2 fds — the db file and its WAL sidecar) stay open until
    garbage collection sweeps the unreachable object, which Python's
    cyclic GC does not do promptly for asyncio-bound objects with
    native handles. The cumulative leak is 2 fds × every retry at the
    300s backoff cap ≈ 12 fds/hour, and the default 2560-fd ulimit
    is exhausted in ~12h of continuous failure, after which every
    open() call on the gateway raises ``OSError: [Errno 24] Too many
    open files`` and the gateway becomes a zombie (#37011).

    This helper centralises the dispose-with-suppression so the three
    failure paths in the reconnect watcher can all call it without
    each one having to know that ``disconnect()`` may itself raise
    on a half-constructed adapter.

    ``adapter`` may be ``None``: the reconnect watcher initialises
    ``adapter = None`` before the ``try`` so the ``except Exception``
    arm can dispose a half-constructed object, and also early-returns
    here when ``_create_adapter()`` returned ``None``.
    """
    if adapter is None:
        return
    try:
        await adapter.disconnect()
    except Exception:
        # Half-constructed adapters (e.g. APIServerAdapter that
        # crashed during aiohttp app setup) can raise from
        # disconnect() on objects that never finished initializing.
        # We must not let that escape and abort the watcher loop.
        #
        # On Python 3.8+, ``asyncio.CancelledError`` inherits from
        # ``BaseException`` (not ``Exception``), so this ``except
        # Exception`` does not swallow task cancellation. We don't
        # re-raise explicitly because the watcher loop intentionally
        # treats dispose failures as best-effort: a failed ``disconnect``
        # call should not take down the reconnect watcher that
        # itself is what's keeping the gateway alive during a partial
        # outage.
        logger.debug(
            "Adapter dispose raised on unowned adapter %r",
            getattr(adapter, "name", type(adapter).__name__),
            exc_info=True,
        )


class GatewayRunner(GatewayAuthorizationMixin, GatewayKanbanWatchersMixin, GatewaySlashCommandsMixin, GatewayGoalsMixin, GatewayRunningMixin, GatewayStartupMixin, GatewayVoiceMixin, GatewayTelegramTopicsMixin, GatewayProcessMixin, GatewaySessionCacheMixin, GatewayExitRequestMixin, GatewayActiveSessionMixin, GatewaySessionKeyMixin, GatewayAgentCacheMixin, GatewayUpdateProgressMixin, GatewayDrainQueueMixin, GatewayTranscriptionMixin, GatewayPlatformFailoverMixin, GatewaySessionMiscMixin, MiscTinyMixin, RestartNotifyMixin, GoalContinuationMixin, BackgroundTaskMixin, CreateAdapterMixin, SessionExpiryMixin, TeamsDockerMediaMixin, TurnAgentConfigMixin, PlatformReconnectMixin, AsyncDelegationMixin, SlashConfirmMixin, McpReloadMixin, MediaDeliveryMixin, InboundTextMixin, StartupPreflightMixin, GatewayStopMixin, BusyAgentDispatchMixin, HandleMessageWithAgentMixin, RunAgentMixin, RunAgentViaProxyMixin, GatewayStartMixin, HandleMessageMixin):
    """
    Main gateway controller.

    Manages the lifecycle of all platform adapters and routes
    messages to/from the agent.
    """

    # Class-level defaults so partial construction in tests doesn't
    # blow up on attribute access.
    _running_agents_ts: Dict[str, float] = {}
    _busy_input_mode: str = "interrupt"
    _busy_text_mode: str = "interrupt"
    _restart_drain_timeout: float = DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    _exit_code: Optional[int] = None
    _draining: bool = False
    _restart_requested: bool = False
    _restart_task_started: bool = False
    _restart_detached: bool = False
    _restart_via_service: bool = False
    _restart_command_source: Optional[SessionSource] = None
    _stop_task: Optional[asyncio.Task] = None
    _session_model_overrides: Dict[str, Dict[str, str]] = {}
    _session_reasoning_overrides: Dict[str, Dict[str, Any]] = {}
    _startup_restore_in_progress: bool = False

    # Re-bind module-level helpers extracted in earlier decomposition rounds
    # (gateway_lifecycle / gateway_agent_mgmt / gateway_config_loaders) back
    # onto the class so legacy ``self.X`` call sites in slash_commands.py and
    # ``GatewayRunner.X`` / ``patch("gateway.run.GatewayRunner.X")`` test
    # references keep resolving. The functions are stateless (no ``self``),
    # so staticmethod preserves their original call signature verbatim.
    _cleanup_agent_resources = staticmethod(_cleanup_agent_resources)
    _finalize_shutdown_agents = staticmethod(_finalize_shutdown_agents)
    _is_stale_restart_redelivery = staticmethod(_is_stale_restart_redelivery)
    _launch_detached_restart_command = staticmethod(_launch_detached_restart_command)
    _launch_systemd_restart_shortcut = staticmethod(_launch_systemd_restart_shortcut)
    _load_busy_input_mode = staticmethod(_load_busy_input_mode)
    _load_busy_text_mode = staticmethod(_load_busy_text_mode)
    _load_restart_drain_timeout = staticmethod(_load_restart_drain_timeout)

    # R48: re-bind the remaining module-extracted helpers (rounds 8-28) that
    # tests still reach via ``GatewayRunner.X`` / ``patch(...GatewayRunner.X)``.
    # Same root cause as R47 — these were classmethods/methods hoisted to module
    # functions; the class bindings were never re-added. All are stateless (no
    # ``self``); ``_CACHE_BUSTING_CONFIG_KEYS`` is an immutable module constant,
    # so it is assigned directly rather than wrapped in staticmethod.
    _CACHE_BUSTING_CONFIG_KEYS = _CACHE_BUSTING_CONFIG_KEYS
    _agent_config_signature = staticmethod(_agent_config_signature)
    _agent_has_active_subagents = staticmethod(_agent_has_active_subagents)
    _bind_adapter_run_generation = staticmethod(_bind_adapter_run_generation)
    _enrich_message_with_vision = staticmethod(_enrich_message_with_vision)
    _extract_cache_busting_config = staticmethod(_extract_cache_busting_config)
    _extract_honcho_cache_busting_config = staticmethod(_extract_honcho_cache_busting_config)
    _format_session_info = staticmethod(_format_session_info)
    _init_cached_agent_for_turn = staticmethod(_init_cached_agent_for_turn)
    _load_background_notifications_mode = staticmethod(_load_background_notifications_mode)
    _load_fallback_model = staticmethod(_load_fallback_model)
    _load_prefill_messages = staticmethod(_load_prefill_messages)
    _load_reasoning_config = staticmethod(_load_reasoning_config)
    _load_show_reasoning = staticmethod(_load_show_reasoning)
    _load_voice_modes = staticmethod(_load_voice_modes)
    _parse_reasoning_command_args = staticmethod(_parse_reasoning_command_args)
    _thread_metadata_for_source = staticmethod(_thread_metadata_for_source)
    # Additional module helpers surfaced via parametrised / indirect test
    # references (not in the literal scan that drove this round).
    _load_ephemeral_system_prompt = staticmethod(_load_ephemeral_system_prompt)
    _load_service_tier = staticmethod(_load_service_tier)
    _voice_key = staticmethod(_voice_key)

    # R48 batch 2: additional module helpers surfaced by scanning BOTH prod
    # source (mixins/slash_commands self.X refs) AND test source (runner.X /
    # self.X / GatewayRunner.X refs) for module-fn names never re-bound on
    # the class. Same root cause as batch 1; these fail either at runtime in
    # production code paths or in test fixtures built via object.__new__.
    _active_profile_name = staticmethod(_active_profile_name)
    _clear_session_env = staticmethod(_clear_session_env)
    _connect_adapter_with_timeout = staticmethod(_connect_adapter_with_timeout)
    _decide_image_input_mode = staticmethod(_decide_image_input_mode)
    _get_guild_id = staticmethod(_get_guild_id)
    _get_proxy_url = staticmethod(_get_proxy_url)
    _has_setup_skill = staticmethod(_has_setup_skill)
    _read_user_config = staticmethod(_read_user_config)
    _release_evicted_agent_soft = staticmethod(_release_evicted_agent_soft)
    _run_in_executor_with_context = staticmethod(_run_in_executor_with_context)
    _safe_adapter_disconnect = staticmethod(_safe_adapter_disconnect)
    _set_adapter_auto_tts_disabled = staticmethod(_set_adapter_auto_tts_disabled)
    _set_adapter_auto_tts_enabled = staticmethod(_set_adapter_auto_tts_enabled)
    _set_session_env = staticmethod(_set_session_env)
    _telegram_topic_help_text = staticmethod(_telegram_topic_help_text)
    _update_platform_runtime_status = staticmethod(_update_platform_runtime_status)

    def __init__(self, config: Optional[GatewayConfig] = None):
        global _gateway_runner_ref
        self.config = config or load_gateway_config()
        # Mark the process as a profile multiplexer when configured. This flips
        # agent.secret_scope.get_secret() to fail-closed on any unscoped
        # credential read, so a missed migration crashes loudly instead of
        # leaking a cross-profile value (Workstream A). Inert when off.
        try:
            from agent.secret_scope import set_multiplex_active
            set_multiplex_active(bool(getattr(self.config, "multiplex_profiles", False)))
        except Exception:
            logger.debug("could not set multiplex-active flag", exc_info=True)
        self.adapters: Dict[Platform, BasePlatformAdapter] = {}
        # Multi-profile multiplexing: adapters for NON-default profiles live
        # here, keyed by profile name then Platform. self.adapters stays the
        # default/active profile's map so the ~93 existing self.adapters[...]
        # sites are untouched when multiplexing is off (this dict is empty).
        # Populated by _start_secondary_profile_adapters().
        self._profile_adapters: Dict[str, Dict[Platform, BasePlatformAdapter]] = {}
        self._warn_if_docker_media_delivery_is_risky()
        _gateway_runner_ref = _weakref.ref(self)

        # Load ephemeral config from config.yaml / env vars.
        # Both are injected at API-call time only and never persisted.
        self._prefill_messages = _load_prefill_messages()
        self._ephemeral_system_prompt = _load_ephemeral_system_prompt()
        self._reasoning_config = _load_reasoning_config()
        self._service_tier = _load_service_tier()
        self._show_reasoning = _load_show_reasoning()
        self._busy_input_mode = _load_busy_input_mode()
        self._busy_text_mode = _load_busy_text_mode()
        self._restart_drain_timeout = _load_restart_drain_timeout()
        self._provider_routing = _load_provider_routing()
        self._fallback_model = _load_fallback_model()

        # Wire process registry into session store for reset protection
        from tools.process_registry import process_registry
        self.session_store = SessionStore(
            self.config.sessions_dir, self.config,
            has_active_processes_fn=lambda key: process_registry.has_active_for_session(key),
        )
        self.delivery_router = DeliveryRouter(self.config)
        self._running = False
        self._gateway_loop: Optional[asyncio.AbstractEventLoop] = None
        self._shutdown_event = asyncio.Event()
        self._exit_cleanly = False
        self._exit_with_failure = False
        self._exit_reason: Optional[str] = None
        self._exit_code: Optional[int] = None
        self._draining = False
        self._restart_requested = False
        # Set by shutdown_signal_handler when a SIGTERM/SIGINT arrived
        # WITHOUT a planned-stop / takeover marker — i.e. an unexpected
        # external signal (container/s6 SIGTERM on `docker restart` or
        # image upgrade, OOM-killer, bare `kill`). Distinct from an
        # operator-requested stop, which writes a marker first. Used by
        # _stop_impl to decide whether to persist gateway_state=stopped
        # (see issue #42675): an unexpected signal must NOT persist
        # "stopped", or container_boot refuses to auto-start the gateway
        # on the next boot.
        self._signal_initiated_shutdown = False
        self._restart_task_started = False
        self._restart_detached = False
        self._restart_via_service = False
        self._restart_command_source: Optional[SessionSource] = None
        self._stop_task: Optional[asyncio.Task] = None
        
        # Track running agents per session for interrupt support
        # Key: session_key, Value: AIAgent instance
        self._running_agents: Dict[str, Any] = {}
        self._running_agents_ts: Dict[str, float] = {}  # start timestamp per session
        self._active_session_leases: Dict[str, Any] = {}
        self._pending_messages: Dict[str, str] = {}  # Queued messages during interrupt
        # Last successfully-resolved (non-empty) model, keyed by session. Used
        # as a fallback when a fresh config read transiently returns an empty
        # model (e.g. an mtime-keyed config-cache miss during a post-interrupt
        # recovery turn). Without this, the agent is built with model="" and
        # every API call fails HTTP 400 "No models provided" — the session goes
        # silent until the user manually re-sends. See #35314. ``"*"`` holds a
        # process-wide last-known-good for sessions seen for the first time.
        self._last_resolved_model: Dict[str, str] = {}
        # Overflow buffer for explicit /queue commands.  The adapter-level
        # _pending_messages dict is a single slot per session (designed for
        # "next-turn" follow-ups where repeated sends collapse into one
        # event).  /queue has different semantics: each invocation must
        # produce its own full agent turn, in FIFO order, with no merging.
        # When the slot is occupied, additional /queue items land here and
        # are promoted one-at-a-time after each run's drain.  Cleared on
        # /new and /reset.  /model and other mid-session operations
        # preserve the queue.
        self._queued_events: Dict[str, List[MessageEvent]] = {}
        self._pending_native_image_paths_by_session: Dict[str, List[str]] = {}
        self._busy_ack_ts: Dict[str, float] = {}  # last busy-ack timestamp per session (debounce)
        self._session_run_generation: Dict[str, int] = {}
        # Startup restore gate: while restart-interrupted sessions are being
        # auto-resumed, real inbound messages are queued instead of competing
        # with the synthetic resume turns for the same session.  The queued
        # events drain only after all startup resume tasks have finished.
        self._startup_restore_in_progress = False
        self._startup_restore_queue: List[MessageEvent] = []
        self._startup_restore_tasks: List[asyncio.Task] = []
        # LRU cache of live SessionSources keyed by session_key. Used by
        # fallback routing paths (shutdown notifications, synthetic
        # background-process events) when the persisted origin is missing
        # and _parse_session_key can't recover thread_id. Capped so it
        # cannot grow unbounded over a long-running gateway lifetime.
        self._session_sources: "OrderedDict[str, SessionSource]" = OrderedDict()
        self._session_sources_max = 512

        # Cache AIAgent instances per session to preserve prompt caching.
        # Without this, a new AIAgent is created per message, rebuilding the
        # system prompt (including memory) every turn — breaking prefix cache
        # and costing ~10x more on providers with prompt caching (Anthropic).
        # Key: session_key, Value: (AIAgent, config_signature_str)
        #
        # OrderedDict so _enforce_agent_cache_cap() can pop the least-recently-
        # used entry (move_to_end() on cache hits, popitem(last=False) for
        # eviction).  Hard cap via _AGENT_CACHE_MAX_SIZE, idle TTL enforced
        # from _session_expiry_watcher().
        import threading as _threading
        self._agent_cache: "OrderedDict[str, tuple]" = OrderedDict()
        self._agent_cache_lock = _threading.Lock()

        # Per-session model overrides from /model command.
        # Key: session_key, Value: dict with model/provider/api_key/base_url/api_mode
        self._session_model_overrides: Dict[str, Dict[str, str]] = {}
        # Per-session reasoning effort overrides from /reasoning.
        # Key: session_key, Value: parsed reasoning config dict.
        self._session_reasoning_overrides: Dict[str, Dict[str, Any]] = {}
        self._kanban_notifier_profile = _active_profile_name()
        # Teams meeting pipeline runtime (bound later when msgraph_webhook adapter exists).
        self._teams_pipeline_runtime = None
        self._teams_pipeline_runtime_error: Optional[str] = None
        # Track pending exec approvals per session
        # Key: session_key, Value: {"command": str, "pattern_key": str, ...}
        self._pending_approvals: Dict[str, Dict[str, Any]] = {}

        # Track platforms that failed to connect for background reconnection.
        # Key: Platform enum, Value: {"config": platform_config, "attempts": int, "next_retry": float}
        self._failed_platforms: Dict[Platform, Dict[str, Any]] = {}

        # Track pending /update prompt responses per session.
        # Key: session_key, Value: True when a prompt is waiting for user input.
        self._update_prompt_pending: Dict[str, bool] = {}

        # Slash-confirm state lives in tools.slash_confirm (module-level),
        # so platform adapters can resolve callbacks without a backref to
        # this runner.  Keep a local counter for confirm_id generation so
        # IDs stay compact (button callback_data has a 64-byte cap on
        # some platforms).
        import itertools as _itertools
        self._slash_confirm_counter = _itertools.count(1)

        # Persistent Honcho managers keyed by gateway session key.
        # This preserves write_frequency="session" semantics across short-lived
        # per-message AIAgent instances.


        # Ensure tirith security scanner is available (downloads if needed)
        try:
            from tools.tirith_security import ensure_installed
            ensure_installed(log_failures=False)
        except Exception:
            pass  # Non-fatal — fail-open at scan time if unavailable

        # Startup heads-up (#30882): a gateway in manual approval mode with no
        # automated risk assessor (tirith disabled AND no auxiliary.approval
        # model) can only gate dangerous commands / execute_code scripts via
        # live in-chat approval. With approval routing fixed, those actions now
        # fail closed (block) rather than silently auto-running — surface that
        # so operators knowingly enable tirith or configure auxiliary.approval
        # for unattended gateways.
        try:
            from hermes_cli.config import load_config as _load_full_config
            _appr_cfg = _load_full_config()
            _appr_mode = str(
                cfg_get(_appr_cfg, "approvals", "mode", default="manual") or "manual"
            ).strip().lower()
            _tirith_on = bool(cfg_get(_appr_cfg, "security", "tirith_enabled", default=True))
            _aux_approval = cfg_get(_appr_cfg, "auxiliary", "approval", default=None)
            if _appr_mode == "manual" and not _tirith_on and not _aux_approval:
                logger.warning(
                    "Gateway approvals.mode=manual with no automated risk "
                    "assessor (security.tirith_enabled is false and "
                    "auxiliary.approval is unset): dangerous commands and "
                    "execute_code scripts will BLOCK until a human approves "
                    "them in chat. Enable security.tirith_enabled or configure "
                    "auxiliary.approval for unattended operation."
                )
        except Exception:
            logger.debug("approvals.mode startup check skipped", exc_info=True)

        # Initialize session database for session_search tool support
        self._session_db = None
        try:
            from hermes_state import SessionDB
            self._session_db = SessionDB()
        except Exception as e:
            # WARNING (not DEBUG) so the failure appears in errors.log — matches
            # cli.py's handling of the same init path.  Users hitting NFS-mounted
            # HERMES_HOME silently lost /resume, /title, /history, /branch, and
            # session search without this.  The underlying cause (usually
            # "locking protocol" from NFS) is now also captured by
            # hermes_state.get_last_init_error() for slash-command error strings.
            logger.warning("SQLite session store not available: %s", e)

        # Opportunistic state.db maintenance: prune ended sessions older
        # than sessions.retention_days + optional VACUUM. Tracks last-run
        # in state_meta so it only actually executes once per
        # sessions.min_interval_hours.  Gateway is long-lived so blocking
        # a few seconds once per day is acceptable; failures are logged
        # but never raised.
        if self._session_db is not None:
            try:
                from hermes_cli.config import load_config as _load_full_config
                _sess_cfg = (_load_full_config().get("sessions") or {})
                if _sess_cfg.get("auto_prune", False):
                    self._session_db.maybe_auto_prune_and_vacuum(
                        retention_days=int(_sess_cfg.get("retention_days", 90)),
                        min_interval_hours=int(_sess_cfg.get("min_interval_hours", 24)),
                        vacuum=bool(_sess_cfg.get("vacuum_after_prune", True)),
                        sessions_dir=self.config.sessions_dir,
                    )
            except Exception as exc:
                logger.debug("state.db auto-maintenance skipped: %s", exc)

        # Opportunistic shadow-repo cleanup — deletes orphan/stale
        # checkpoint repos under ~/.hermes/checkpoints/.  Opt-in via
        # checkpoints.auto_prune, idempotent via .last_prune marker.
        try:
            from hermes_cli.config import load_config as _load_full_config
            _ckpt_cfg = (_load_full_config().get("checkpoints") or {})
            if _ckpt_cfg.get("auto_prune", False):
                from tools.checkpoint_manager import maybe_auto_prune_checkpoints
                maybe_auto_prune_checkpoints(
                    retention_days=int(_ckpt_cfg.get("retention_days", 7)),
                    min_interval_hours=int(_ckpt_cfg.get("min_interval_hours", 24)),
                    delete_orphans=bool(_ckpt_cfg.get("delete_orphans", True)),
                    max_total_size_mb=int(_ckpt_cfg.get("max_total_size_mb", 500)),
                )
        except Exception as exc:
            logger.debug("checkpoint auto-maintenance skipped: %s", exc)

        # DM pairing store for code-based user authorization
        from gateway.pairing import PairingStore
        self.pairing_store = PairingStore()
        
        # Event hook system
        from gateway.hooks import HookRegistry
        self.hooks = HookRegistry()

        # Per-chat voice reply mode: "off" | "voice_only" | "all"
        self._voice_mode: Dict[str, str] = _load_voice_modes(self._VOICE_MODE_PATH)
        # Recent voice transcripts per (guild,user) for duplicate suppression.
        # Protects against the same utterance being emitted twice by the voice
        # capture / STT pipeline, which otherwise produces a second delayed reply.
        self._recent_voice_transcripts: Dict[tuple[int, int], List[tuple[float, str]]] = {}

        # Track background tasks to prevent garbage collection mid-execution
        self._background_tasks: set = set()






    # -- Setup skill availability ----------------------------------------

    # -- Voice mode persistence ------------------------------------------

    _VOICE_MODE_PATH = _hermes_home / "gateway_voice_mode.json"






    # Telegram's General (pinned top) topic in forum-enabled private chats.
    # Bot API behavior varies: some clients omit message_thread_id for
    # General, others send "1". Treat both as "root" for lobby/lane purposes.
    _TELEGRAM_GENERAL_TOPIC_IDS = frozenset({"", "1"})

    _TELEGRAM_LOBBY_REMINDER_COOLDOWN_S = 30.0









    # -------- /queue FIFO helpers --------------------------------------
    # /queue must produce one full agent turn per invocation, in FIFO
    # order, with no merging.  The adapter's _pending_messages dict is a
    # single "next-up" slot (shared with photo-burst follow-ups), so we
    # use it for the head of the queue and an overflow list for the
    # tail.  Enqueue puts new items in the slot when free, otherwise in
    # the overflow.  Promotion (called after each run's drain) moves the
    # next overflow item into the slot so the following recursion picks
    # it up.  Clearing happens on /new and /reset via
    # _handle_reset_command.





    # ------------------------------------------------------------------
    # Per-platform circuit breaker (pause/resume) — used by the reconnect
    # watcher when a retryable failure recurs past a threshold, and by the
    # /platform pause|resume slash command for manual control.
    # ------------------------------------------------------------------






    # Hard cap on per-session pending follow-ups for busy_input_mode=queue
    # (and the draining/steer-fallback/subagent-demotion paths that share
    # this entry point).  Without a cap, a stuck agent + a rapid-fire user
    # could grow the overflow list unboundedly.  32 turns of queued
    # follow-ups is far beyond any realistic conversational backlog while
    # still small enough to never threaten memory.
    _BUSY_QUEUE_MAX_PENDING = 32






    _STUCK_LOOP_THRESHOLD = 3  # restarts while active before auto-suspend
    _STUCK_LOOP_FILE = ".restart_failure_counts"





    # Drain-timeout reasons set by _stop_impl() when a still-running turn is
    # force-interrupted; "restart_interrupted" is set by
    # SessionStore.suspend_recently_active() on crash recovery (no
    # .clean_shutdown marker).  All three mean "the agent was mid-turn and
    # we killed it" — eligible for startup auto-resume.
    _AUTO_RESUME_REASONS = frozenset(
        {"restart_timeout", "shutdown_timeout", "restart_interrupted"}
    )


    # R54: start lifted to GatewayStartMixin.
    # async def start(...) -> see gateway/start_mixin_r54.py



    # ── Kanban board watchers ───────────────────────────────────────────
    # The kanban notifier/dispatcher watcher loops + their helpers live in
    # GatewayKanbanWatchersMixin (gateway/kanban_watchers.py). They use only
    # self state, so inheriting the mixin keeps every self._kanban_* call site
    # working unchanged while lifting ~1,000 LOC out of this file.


    # R55: _handle_message lifted to HandleMessageMixin.
    # async def _handle_message(...) -> see gateway/handle_message_mixin.py

    # R51: _handle_message_with_agent lifted to HandleMessageWithAgentMixin.
    # Whole-method verbatim lift (1207ln). Called from _handle_message's
    # finally block under the _running_agents sentinel guard; resolves
    # unchanged via MRO. Lazy-imports run.py module globals to avoid the
    # circular import. See gateway/handle_message_with_agent_mixin.py.
    # ────────────────────────────────────────────────────────────────
    # /goal — persistent cross-turn goals (Ralph-style loop)
    # ────────────────────────────────────────────────────────────────







    _TELEGRAM_CAPABILITY_HINT_COOLDOWN_S = 300.0



    # ------------------------------------------------------------------
    # Slash-command confirmation primitive (generic)
    # ------------------------------------------------------------------
    # Used by slash commands that have a non-destructive but expensive
    # side effect worth an explicit user confirmation (currently only
    # /reload-mcp, which invalidates the prompt cache).  Two delivery
    # paths:
    #   1. Button UI — adapters that override ``send_slash_confirm``
    #      (Telegram, Discord, Slack, Matrix, Feishu) render three
    #      inline buttons.  The adapter routes the button click back via
    #      ``tools.slash_confirm.resolve(session_key, confirm_id, choice)``.
    #   2. Text fallback — adapters that don't override the hook get a
    #      plain text prompt.  Users reply with /approve, /always, or
    #      /cancel; the early intercept in ``_handle_message`` matches
    #      those replies against ``tools.slash_confirm.get_pending()``.





    # ------------------------------------------------------------------
    # /approve & /deny — explicit dangerous-command approval
    # ------------------------------------------------------------------

    _APPROVAL_TIMEOUT_SECONDS = 300  # 5 minutes


    # Built-in messaging platforms where the ``/update`` command is allowed.
    # ACP, API server, and webhooks are programmatic interfaces that should
    # not trigger system updates.  Plugin-migrated platforms (discord,
    # mattermost, teams, irc, line, …) are NOT listed here — they declare
    # ``allow_update_command=True`` on their ``PlatformEntry`` and are
    # honored via the registry fallback at ``_handle_update_command`` below.
    _UPDATE_ALLOWED_PLATFORMS = frozenset({
        Platform.TELEGRAM, Platform.SLACK, Platform.WHATSAPP,
        Platform.SIGNAL, Platform.MATRIX,
        Platform.EMAIL, Platform.SMS, Platform.DINGTALK,
        Platform.FEISHU, Platform.WECOM, Platform.WECOM_CALLBACK, Platform.WEIXIN, Platform.BLUEBUBBLES, Platform.QQBOT, Platform.LOCAL,
    })











    _MAX_INTERRUPT_DEPTH = 3  # Cap recursive interrupt handling (#816)

    # Config keys whose values MUST invalidate the gateway's cached agent
    # when they change.  The agent bakes these into its compressor / context
    # handling at construction time, so a mid-running-gateway config edit
    # would otherwise be silently ignored until the user triggers a
    # different cache eviction (model switch, /reset, etc.).
    #
    # Each entry is a tuple of (section, key) read from the raw config dict.
    # Add more here as new baked-at-construction config settings are added.









    # ------------------------------------------------------------------
    # Proxy mode: forward messages to a remote Hermes API server
    # ------------------------------------------------------------------

    # R53: _run_agent_via_proxy lifted to RunAgentViaProxyMixin.
    # async def _run_agent_via_proxy(...) → see gateway/run_agent_via_proxy_mixin.py


    # ------------------------------------------------------------------

    # R52: _run_agent lifted to RunAgentMixin.
    # async def _run_agent(...) → see gateway/run_agent_mixin.py


# R57: process entrypoint cluster lifted to gateway.gateway_entrypoint.
# start_gateway, main, _run_planned_stop_watcher, _start_cron_ticker moved
# verbatim. Re-exported below for backward compat (from gateway.run import
# start_gateway + monkeypatch patches still resolve).


if __name__ == "__main__":
    main()
