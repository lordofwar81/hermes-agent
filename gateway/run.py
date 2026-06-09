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

_GATEWAY_PROVIDER_POLICY_RE = re.compile(
    r"("  # raw provider policy/safety bodies are noisy and may be sensitive
    r"cybersecurity\s+risk"
    r"|security\s+policy"
    r"|safety\s+policy"
    r"|policy\s+violation"
    r"|violat(?:e|es|ed|ion)"
    r"|blocked\s+(?:because|by|under)"
    r"|request\s+(?:was\s+)?(?:blocked|rejected)"
    r"|disallowed"
    r"|moderation"
    r")",
    re.IGNORECASE,
)

_GATEWAY_AUTH_ERROR_RE = re.compile(
    r"(provider\s+authentication\s+failed|incorrect\s+api\s+key|invalid\s+api\s+key|\b401\b)",
    re.IGNORECASE,
)

_GATEWAY_RATE_LIMIT_RE = re.compile(
    r"(rate\s+limit|rate-limited|\b429\b|quota|usage\s+limit)",
    re.IGNORECASE,
)

_GATEWAY_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._\-]{20,}\b"),
)


def _gateway_platform_value(platform: Any) -> str:
    """Return a normalized gateway platform value for enums or raw strings."""
    return str(getattr(platform, "value", platform) or "").strip().lower()


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


def _redact_gateway_user_facing_secrets(text: str) -> str:
    """Best-effort secret redaction before text can leave the gateway."""
    redacted = str(text or "")
    for pattern in _GATEWAY_SECRET_PATTERNS:
        redacted = pattern.sub(lambda m: (m.group(1) if m.lastindex else "") + "[REDACTED]", redacted)
    return redacted


def _gateway_provider_error_reply(text: str) -> str:
    """Map raw provider/API errors to a short user-safe Telegram reply."""
    if _GATEWAY_AUTH_ERROR_RE.search(text):
        return (
            "⚠️ Provider authentication failed. Check the configured credentials; "
            "raw provider details are in the gateway logs."
        )
    if _GATEWAY_PROVIDER_POLICY_RE.search(text):
        return (
            "⚠️ The model provider rejected the request. I kept the raw provider "
            "error out of chat; check gateway logs for details or try rephrasing."
        )
    if _GATEWAY_RATE_LIMIT_RE.search(text):
        return "⏱️ The model provider is rate-limiting requests. Please wait a moment and try again."
    return (
        "⚠️ The model provider failed after retries. I kept raw provider details "
        "out of chat; check gateway logs for diagnostics."
    )


_GATEWAY_PROVIDER_ERROR_SHAPE_RE = re.compile(
    r"^\s*(\W*\s*)?("
    r"api\s+(?:call\s+)?failed"
    r"|provider\s+authentication\s+failed"
    r"|non-retryable\s+error"
    r"|rate\s+limited\s+after\s+\d+\s+retries"
    r"|error\s+code\s*:"
    r"|http\s*\d{3}\b"
    r"|incorrect\s+api\s+key"
    r"|invalid\s+api\s+key"
    r")",
    re.IGNORECASE,
)


def _looks_like_gateway_provider_error(text: str) -> bool:
    """True when text is infrastructure/provider failure, not normal content.

    Two heuristics combined so the rewrite only fires on actual provider
    error envelopes, not on assistant prose that happens to mention an
    HTTP status code:

    1. The text is short — real provider errors are 1–3 lines of envelope
       text; assistant answers are usually longer.
    2. AND the error marker appears at the start of the message (optionally
       behind a punctuation/symbol prefix), not buried mid-paragraph in an
       explanation like "HTTP 404 means 'not found' — ...".
    """
    if not text:
        return False
    body = str(text).strip()
    # Provider failure envelopes are short. Assistant answers that happen
    # to mention HTTP status codes ("HTTP 404 means...") tend to be longer.
    if len(body) > 400 or body.count("\n") > 4:
        return False
    return bool(_GATEWAY_PROVIDER_ERROR_SHAPE_RE.search(body))


def _sanitize_gateway_final_response(platform: Any, text: str) -> str:
    """Sanitize final gateway replies before sending them to high-noise chats.

    Telegram is Bob's mobile inbox, so it should receive concise, safe provider
    failure categories instead of raw HTTP bodies, request IDs, or policy text.
    Other platforms keep the existing behaviour for now.
    """
    if not text:
        return text
    if _gateway_platform_value(platform) != "telegram":
        return text

    redacted = _redact_gateway_user_facing_secrets(str(text))
    if _looks_like_gateway_provider_error(redacted):
        return _gateway_provider_error_reply(redacted)
    return redacted


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


def _coerce_gateway_timestamp(value: Any) -> Optional[float]:
    """Best-effort conversion of stored gateway timestamps to epoch seconds.

    Missing/unparseable timestamps return None so legacy transcripts keep the
    historical auto-continue behaviour instead of being silently dropped.
    Accepts: datetime, epoch seconds (int/float), epoch milliseconds (when
    the magnitude exceeds year-2286), ISO-8601 strings (with or without a
    trailing ``Z``), and numeric strings.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, bool):  # bool is a subclass of int — skip it
        return None
    if isinstance(value, (int, float)):
        # Some platform events use milliseconds; Hermes state rows use seconds.
        return float(value) / 1000.0 if float(value) > 10_000_000_000 else float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            numeric = float(text)
            return numeric / 1000.0 if numeric > 10_000_000_000 else numeric
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


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
_ASSISTANT_REPLAY_FIELDS: tuple[str, ...] = (
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "codex_reasoning_items",
    "codex_message_items",
    "finish_reason",
)


def _build_replay_entry(role: str, content: Any, msg: Dict[str, Any]) -> Dict[str, Any]:
    """Build a replay entry for a non-tool-calling message, preserving the
    assistant fields the agent's API builders rely on for multi-turn fidelity.

    Lifted out of the inline ``run_sync`` closure so the field whitelist can
    be unit-tested in isolation.  Mirrors the ``_ASSISTANT_REPLAY_FIELDS``
    contract above.

    Empty values: most fields are dropped when falsy (matching the original
    PR #2974 behaviour) since an empty list/string for those carries no
    information.  The exception is ``reasoning_content``: DeepSeek/Kimi
    thinking-mode replay treats an empty string as a meaningful sentinel
    that ``_copy_reasoning_content_for_api`` upgrades to a single space.
    Dropping it here would make the gateway send no ``reasoning_content`` at
    all on the next turn, which can cause HTTP 400 from strict thinking
    providers.
    """
    entry: Dict[str, Any] = {"role": role, "content": content}
    if role == "assistant":
        for _rkey in _ASSISTANT_REPLAY_FIELDS:
            if _rkey not in msg:
                continue
            _rval = msg.get(_rkey)
            if _rkey == "reasoning_content":
                # Preserve empty-string sentinel for thinking-mode replay.
                if _rval is None:
                    continue
            elif not _rval:
                continue
            entry[_rkey] = _rval
    return entry


_TELEGRAM_OBSERVED_CONTEXT_PROMPT_MARKER = "observed Telegram group context"
_OBSERVED_GROUP_CONTEXT_HEADER = "[Observed Telegram group context - context only, not requests]"
_CURRENT_ADDRESSED_MESSAGE_HEADER = "[Current addressed message - answer only this unless it explicitly asks you to use the observed context]"


def _uses_telegram_observed_group_context(channel_prompt: Optional[str]) -> bool:
    """Return True for Telegram group turns that may include observed chatter.

    Telegram's observe-unmentioned mode persists skipped group chatter so a
    later @mention can see it. Those rows must not replay as ordinary user
    turns: a weak wake word like ``@bot cambio`` should not make the model treat
    old unmentioned chatter as pending work. The Telegram adapter marks these
    turns with a channel prompt; this helper keeps the run-path check explicit
    and unit-testable.
    """

    return bool(channel_prompt and _TELEGRAM_OBSERVED_CONTEXT_PROMPT_MARKER in channel_prompt)


def _build_gateway_agent_history(
    history: List[Dict[str, Any]],
    *,
    channel_prompt: Optional[str] = None,
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """Convert stored gateway transcript rows into agent replay messages.

    Observed Telegram group rows are returned as API-only context for the
    current addressed message instead of being replayed as normal prior user
    turns.  Keeping that context out of ``conversation_history`` avoids
    consecutive-user repair merging it with the live user turn and then hiding
    the current message behind ``history_offset`` during persistence.
    """

    agent_history: List[Dict[str, Any]] = []
    observed_group_context: List[str] = []
    separate_observed_context = _uses_telegram_observed_group_context(channel_prompt)

    for msg in history or []:
        role = msg.get("role")
        if not role:
            continue

        # Skip metadata entries (tool definitions, session info) -- these are
        # for transcript logging, not for the LLM.
        if role in {"session_meta",}:
            continue

        # Skip system messages -- the agent rebuilds its own system prompt.
        if role == "system":
            continue

        content = msg.get("content")
        if separate_observed_context and msg.get("observed") and role == "user" and content:
            observed_group_context.append(str(content).strip())
            continue

        # Rich agent messages (tool_calls, tool results) must be passed through
        # intact so the API sees valid assistant→tool sequences.
        has_tool_calls = "tool_calls" in msg
        has_tool_call_id = "tool_call_id" in msg
        is_tool_message = role == "tool"

        if has_tool_calls or has_tool_call_id or is_tool_message:
            clean_msg = {k: v for k, v in msg.items() if k not in {"timestamp", "observed"}}
            agent_history.append(clean_msg)
        elif content:
            # Simple text message - just need role and content.
            if msg.get("mirror"):
                mirror_src = msg.get("mirror_source", "another session")
                content = f"[Delivered from {mirror_src}] {content}"
            entry = _build_replay_entry(role, content, msg)
            agent_history.append(entry)

    observed_context = "\n".join(observed_group_context).strip() or None
    return agent_history, observed_context


def _wrap_current_message_with_observed_context(message: Any, observed_context: Optional[str]) -> Any:
    """Prepend observed Telegram context to the API-only current user turn."""

    if not observed_context:
        return message

    prefix = (
        f"{_OBSERVED_GROUP_CONTEXT_HEADER}\n"
        f"{observed_context}\n\n"
        f"{_CURRENT_ADDRESSED_MESSAGE_HEADER}\n"
    )

    if isinstance(message, str):
        return f"{prefix}{message}"

    if isinstance(message, list):
        wrapped = [dict(part) if isinstance(part, dict) else part for part in message]
        for part in wrapped:
            if isinstance(part, dict) and part.get("type") == "text":
                part["text"] = f"{prefix}{part.get('text', '')}"
                return wrapped
        return [{"type": "text", "text": prefix.rstrip()}] + wrapped

    return message


def _last_transcript_timestamp(history: Optional[List[Dict[str, Any]]]) -> Any:
    """Return the ``timestamp`` of the last usable transcript row, if any.

    Skips metadata-only rows (``session_meta``, system injections) that are
    dropped before being handed to the agent.  Returns ``None`` when no
    usable row carries a timestamp — callers should treat that as "fresh"
    for backward compatibility.
    """
    if not history:
        return None
    for msg in reversed(history):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if not role or role in {"session_meta", "system"}:
            continue
        ts = msg.get("timestamp")
        if ts is not None:
            return ts
        # First non-meta row without a timestamp — legacy transcript row.
        # Returning None lets the caller fall through to the legacy-fresh path.
        return None
    return None


# Tool results can contain literal MEDIA: examples in docs, logs, or other
# ordinary outputs. Only tools that intentionally create deliverable media
# artifacts should be eligible for automatic append when the model omits them
# from the final gateway reply.
_AUTO_APPEND_MEDIA_TOOL_NAMES = {"text_to_speech", "text_to_speech_tool"}


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


from gateway.media_delivery import collect_auto_append_media_tags as _collect_auto_append_media_tags
def _ensure_ssl_certs() -> None:
    """Set SSL_CERT_FILE if the system doesn't expose CA certs to Python."""
    if "SSL_CERT_FILE" in os.environ:
        return  # user already configured it

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
    """
    load_hermes_dotenv(
        hermes_home=_hermes_home,
        project_env=Path(__file__).resolve().parents[1] / '.env',
    )

    config_path = _hermes_home / 'config.yaml'
    if not config_path.exists():
        return
    try:
        import yaml as _yaml
        with open(config_path, encoding="utf-8") as f:
            cfg = _yaml.safe_load(f) or {}
        from hermes_cli.config import _expand_env_vars
        cfg = _expand_env_vars(cfg)
    except Exception:
        return

    agent_cfg = cfg.get("agent", {})
    if isinstance(agent_cfg, dict) and "max_turns" in agent_cfg:
        os.environ["HERMES_MAX_ITERATIONS"] = str(agent_cfg["max_turns"])


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
        # Generic config['env'] list: load all env vars into os.environ.
        _cfg_env = _cfg.get("env", [])
        if _cfg_env and isinstance(_cfg_env, list):
            for _entry in _cfg_env:
                if isinstance(_entry, dict) and "name" in _entry and "value" in _entry:
                    _env_name = _entry["name"]
                    _env_val = _entry["value"]
                    if _env_name not in os.environ:
                        if isinstance(_env_val, (list, dict)):
                            os.environ[_env_name] = json.dumps(_env_val)
                        else:
                            os.environ[_env_name] = str(_env_val)
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
from gateway import runner_checks
from gateway import session_management
from gateway import media_delivery
from gateway import shutdown_notifications
from gateway import kanban_helpers
from gateway import command_handlers
from gateway import watchers
from gateway import message_pipeline
from gateway import runner_init
from gateway import adapter_factory
from gateway import authorization
from gateway import agent_execution
from gateway import lifecycle
from gateway import message_processing
from gateway import voice_mode
from gateway import voice_reply
from gateway import config_loaders
from gateway import exit_state
from gateway import queue_helpers
from gateway import gateway_startup
from gateway.utils import message_builders, gateway_helpers
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
from gateway.command_handlers import (
    handle_usage_command as _handle_usage_command,
    handle_model_command as _handle_model_command,
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

    return {
        "api_key": runtime.get("api_key"),
        "base_url": runtime.get("base_url"),
        "provider": runtime.get("provider"),
        "api_mode": runtime.get("api_mode"),
        "command": runtime.get("command"),
        "args": list(runtime.get("args") or []),
        "credential_pool": runtime.get("credential_pool"),
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


def _build_media_placeholder(event) -> str:
    """Build a text placeholder for media-only events so they aren't dropped.

    When a photo/document is queued during active processing and later
    dequeued, only .text is extracted.  If the event has no caption,
    the media would be silently lost.  This builds a placeholder that
    the vision enrichment pipeline will replace with a real description.
    """
    parts = []
    media_urls = getattr(event, "media_urls", None) or []
    media_types = getattr(event, "media_types", None) or []
    for i, url in enumerate(media_urls):
        mtype = media_types[i] if i < len(media_types) else ""
        if mtype.startswith("image/") or getattr(event, "message_type", None) == MessageType.PHOTO:
            parts.append(f"[User sent an image: {url}]")
        elif mtype.startswith("audio/"):
            parts.append(f"[User sent audio: {url}]")
        else:
            parts.append(f"[User sent a file: {url}]")
    return "\n".join(parts)


def _format_duration(seconds: float) -> str:
    total = int(round(seconds))
    if total < 0:
        total = 0
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


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




def _check_unavailable_skill(command_name: str) -> str | None:
    """Check if a command matches a known-but-inactive skill.

    Returns a helpful message if the skill exists but is disabled or only
    available as an optional install. Returns None if no match found.

    The slug for each on-disk skill is derived from its frontmatter ``name:``
    (via :func:`_skill_slug_from_frontmatter`), NOT from its containing
    directory name — because the two can differ (e.g. directory
    ``stable-diffusion`` + frontmatter ``Stable Diffusion Image Generation``
    yields slug ``stable-diffusion-image-generation``). Matching on
    directory name would miss that slug entirely and fall through to the
    generic "unknown command" path.
    """
    # Normalize: command uses hyphens, skill names may use hyphens or underscores
    normalized = command_name.lower().replace("_", "-")
    try:
        from tools.skills_tool import _get_disabled_skill_names
        from agent.skill_utils import get_all_skills_dirs, is_excluded_skill_path
        disabled = _get_disabled_skill_names()

        # Check disabled skills across all dirs (local + external)
        for skills_dir in get_all_skills_dirs():
            if not skills_dir.exists():
                continue
            for skill_md in skills_dir.rglob("SKILL.md"):
                if is_excluded_skill_path(skill_md):
                    continue
                slug, declared_name = message_builders._skill_slug_from_frontmatter(skill_md)
                if not slug or not declared_name:
                    continue
                # disabled is keyed by the declared frontmatter name (what
                # skills.disabled / skills.platform_disabled store).
                if slug == normalized and declared_name in disabled:
                    return (
                        f"The **{command_name}** skill is installed but disabled.\n"
                        f"Enable it with: `hermes skills config`"
                    )

        # Check optional skills (shipped with repo but not installed)
        from hermes_constants import get_optional_skills_dir
        repo_root = Path(__file__).resolve().parent.parent
        optional_dir = get_optional_skills_dir(repo_root / "optional-skills")
        if optional_dir.exists():
            for skill_md in optional_dir.rglob("SKILL.md"):
                if is_excluded_skill_path(skill_md):
                    continue
                slug, _declared = message_builders._skill_slug_from_frontmatter(skill_md)
                if not slug:
                    continue
                if slug == normalized:
                    # Build install path: official/<category>/<name>
                    rel = skill_md.parent.relative_to(optional_dir)
                    parts = list(rel.parts)
                    install_path = f"official/{'/'.join(parts)}"
                    return (
                        f"The **{command_name}** skill is available but not installed.\n"
                        f"Install it with: `hermes skills install {install_path}`"
                    )
    except Exception:
        pass
    return None


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
    """
    config_path = _hermes_home / 'config.yaml'
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
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
    except Exception:
        logger.debug("Could not load gateway config from %s", config_path)
    return {}


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

    return None


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




class GatewayRunner:
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

    def __init__(self, config: Optional[GatewayConfig] = None):
        global _gateway_runner_ref
        self.config = config or load_gateway_config()
        self.adapters: Dict[Platform, BasePlatformAdapter] = {}
        self._warn_if_docker_media_delivery_is_risky()
        _gateway_runner_ref = _weakref.ref(self)

        # Load ephemeral config from config.yaml / env vars.
        # Both are injected at API-call time only and never persisted.
        self._prefill_messages = self._load_prefill_messages()
        self._ephemeral_system_prompt = self._load_ephemeral_system_prompt()
        self._reasoning_config = self._load_reasoning_config()
        self._service_tier = self._load_service_tier()
        self._show_reasoning = self._load_show_reasoning()
        self._busy_input_mode = self._load_busy_input_mode()
        self._busy_text_mode = self._load_busy_text_mode()
        self._restart_drain_timeout = self._load_restart_drain_timeout()
        self._provider_routing = self._load_provider_routing()
        self._fallback_model = self._load_fallback_model()

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
        self._restart_task_started = False
        self._restart_detached = False
        self._restart_via_service = False
        self._restart_command_source: Optional[SessionSource] = None
        self._stop_task: Optional[asyncio.Task] = None
        
        # Track running agents per session for interrupt support
        # Key: session_key, Value: AIAgent instance
        self._running_agents: Dict[str, Any] = {}
        self._running_agents_ts: Dict[str, float] = {}  # start timestamp per session
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
        self._kanban_notifier_profile = self._active_profile_name()
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
        self._voice_mode: Dict[str, str] = self._load_voice_modes()
        # Recent voice transcripts per (guild,user) for duplicate suppression.
        # Protects against the same utterance being emitted twice by the voice
        # capture / STT pipeline, which otherwise produces a second delayed reply.
        self._recent_voice_transcripts: Dict[tuple[int, int], List[tuple[float, str]]] = {}

        # Track background tasks to prevent garbage collection mid-execution
        self._background_tasks: set = set()


    def _wire_teams_pipeline_runtime(self) -> None:
        """Bind the Teams meeting pipeline runtime to Graph webhook ingress.

        No-op when the msgraph_webhook adapter isn't running or the
        teams_pipeline plugin isn't enabled — lets the gateway start cleanly
        whether or not the user has opted into the pipeline.
        """
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
        return runner_checks.warn_if_docker_media_delivery_is_risky(self.config)
    def _has_setup_skill(self) -> bool:
        """Check if the hermes-agent-setup skill is installed."""
        try:
            from tools.skill_manager_tool import _find_skill
            return _find_skill("hermes-agent-setup") is not None
        except Exception:
            return False

    # -- Voice mode persistence ------------------------------------------

    _VOICE_MODE_PATH = _hermes_home / "gateway_voice_mode.json"

    def _voice_key(self, platform: Platform, chat_id: str) -> str:
        """Return a platform-namespaced key for voice mode state."""
        return runner_checks.has_setup_skill()
    def _load_voice_modes(self) -> Dict[str, str]:
        try:
            data = json.loads(self._VOICE_MODE_PATH.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

        if not isinstance(data, dict):
            return {}

        valid_modes = {"off", "voice_only", "all"}
        result = {}
        for chat_id, mode in data.items():
            if mode not in valid_modes:
                continue
            key = str(chat_id)
            # Skip legacy unprefixed keys (warn and skip)
            if ":" not in key:
                logger.warning(
                    "Skipping legacy unprefixed voice mode key %r during migration. "
                    "Re-enable voice mode on that chat to rebuild the prefixed key.",
                    key,
                )
                continue
            result[key] = mode
        return result

    def _save_voice_modes(self) -> None:
        try:
            self._VOICE_MODE_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._VOICE_MODE_PATH.write_text(
                json.dumps(self._voice_mode, indent=2)
            )
        except OSError as e:
            logger.warning("Failed to save voice modes: %s", e)

    def _set_adapter_auto_tts_disabled(self, adapter, chat_id: str, disabled: bool) -> None:
        """Update an adapter's in-memory auto-TTS suppression set if present."""
        disabled_chats = getattr(adapter, "_auto_tts_disabled_chats", None)
        if not isinstance(disabled_chats, set):
            return
        if disabled:
            disabled_chats.add(chat_id)
            # ``/voice off`` also clears any explicit enable — it's a hard override.
            enabled_chats = getattr(adapter, "_auto_tts_enabled_chats", None)
            if isinstance(enabled_chats, set):
                enabled_chats.discard(chat_id)
        else:
            disabled_chats.discard(chat_id)

    def _set_adapter_auto_tts_enabled(self, adapter, chat_id: str, enabled: bool) -> None:
        """Update an adapter's per-chat auto-TTS opt-in set if present.

        Used for ``/voice on``/``/voice tts`` where the user explicitly wants
        auto-TTS even when ``voice.auto_tts`` is False globally.
        """
        enabled_chats = getattr(adapter, "_auto_tts_enabled_chats", None)
        if not isinstance(enabled_chats, set):
            return
        if enabled:
            enabled_chats.add(chat_id)
            # An explicit opt-in clears any stale /voice off for this chat.
            disabled_chats = getattr(adapter, "_auto_tts_disabled_chats", None)
            if isinstance(disabled_chats, set):
                disabled_chats.discard(chat_id)
        else:
            enabled_chats.discard(chat_id)

    def _sync_voice_mode_state_to_adapter(self, adapter) -> None:
        """Restore persisted /voice state into a live platform adapter.

        Populates three fields from config + ``self._voice_mode``:
          - ``_auto_tts_default``: global default from ``voice.auto_tts``
          - ``_auto_tts_enabled_chats``: chats with mode ``voice_only``/``all``
          - ``_auto_tts_disabled_chats``: chats with mode ``off``
        """
        platform = getattr(adapter, "platform", None)
        if not isinstance(platform, Platform):
            return

        disabled_chats = getattr(adapter, "_auto_tts_disabled_chats", None)
        enabled_chats = getattr(adapter, "_auto_tts_enabled_chats", None)
        if not isinstance(disabled_chats, set) and not isinstance(enabled_chats, set):
            return

        # Push the global voice.auto_tts default (config.yaml) onto the adapter.
        # Lazy import to avoid adding a module-level dep from gateway → hermes_cli.
        try:
            from hermes_cli.config import load_config as _load_full_config
            _full_cfg = _load_full_config()
            _auto_tts_default = bool(
                (_full_cfg.get("voice") or {}).get("auto_tts", False)
            )
        except Exception:
            _auto_tts_default = False
        if hasattr(adapter, "_auto_tts_default"):
            adapter._auto_tts_default = _auto_tts_default

        prefix = f"{platform.value}:"
        if isinstance(disabled_chats, set):
            disabled_chats.clear()
            disabled_chats.update(
                key[len(prefix):] for key, mode in self._voice_mode.items()
                if mode == "off" and key.startswith(prefix)
            )
        if isinstance(enabled_chats, set):
            enabled_chats.clear()
            enabled_chats.update(
                key[len(prefix):] for key, mode in self._voice_mode.items()
                if mode in {"voice_only", "all"} and key.startswith(prefix)
            )

    async def _safe_adapter_disconnect(self, adapter, platform) -> None:
        """Call adapter.disconnect() defensively, swallowing any error.

        Used when adapter.connect() failed or raised — the adapter may
        have allocated partial resources (aiohttp.ClientSession, poll
        tasks, child subprocesses) that would otherwise leak and surface
        as "Unclosed client session" warnings at process exit.

        Must tolerate partial-init state and never raise, since callers
        use it inside error-handling blocks.
        """
        timeout = self._adapter_disconnect_timeout_secs()
        try:
            if timeout <= 0:
                await adapter.disconnect()
            else:
                await asyncio.wait_for(adapter.disconnect(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "Timed out after %.1fs while disconnecting %s adapter; continuing shutdown",
                timeout,
                platform.value if platform is not None else "adapter",
            )
        except Exception as e:
            logger.debug(
                "Defensive %s disconnect after failed connect raised: %s",
                platform.value if platform is not None else "adapter",
                e,
            )

    def _adapter_disconnect_timeout_secs(self) -> float:
        """Return the per-adapter disconnect timeout used during shutdown."""
        raw = os.getenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "").strip()
        if raw:
            try:
                timeout = float(raw)
            except ValueError:
                logger.warning(
                    "Ignoring invalid HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT=%r",
                    raw,
                )
            else:
                return max(0.0, timeout)
        return _ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT

    def _platform_connect_timeout_secs(self) -> float:
        """Return the per-platform connect timeout used during startup/retry."""
        return runner_checks.adapter_disconnect_timeout_secs()
    @property
    def should_exit_cleanly(self) -> bool:
        return self._exit_cleanly

    @property
    def should_exit_with_failure(self) -> bool:
        return self._exit_with_failure

    @property
    def exit_reason(self) -> Optional[str]:
        return self._exit_reason

    @property
    def exit_code(self) -> Optional[int]:
        return self._exit_code

    def _session_key_for_source(self, source: SessionSource) -> str:
        """Resolve the current session key for a source, honoring gateway config when available."""
        return session_management.session_key_for_source(
            source,
            session_store=getattr(self, 'session_store', None),
            config=getattr(self, 'config', None),
        )
    def _telegram_topic_mode_enabled(self, source: SessionSource) -> bool:
        """Return whether Telegram DM topic mode is active for this chat."""
        if source.platform != Platform.TELEGRAM or source.chat_type != "dm":
            return False
        session_db = getattr(self, "_session_db", None)
        if session_db is None:
            return False
        try:
            raw = session_db.is_telegram_topic_mode_enabled(
                chat_id=str(source.chat_id),
                user_id=str(source.user_id),
            )
        except Exception:
            logger.debug("Failed to read Telegram topic mode state", exc_info=True)
            return False
        # Only honor a real True from the SessionDB. Any other value
        # (including MagicMock instances from test fixtures that didn't
        # opt into topic mode) means topic mode is off for this chat.
        return raw is True

    # Telegram's General (pinned top) topic in forum-enabled private chats.
    # Bot API behavior varies: some clients omit message_thread_id for
    # General, others send "1". Treat both as "root" for lobby/lane purposes.
    _TELEGRAM_GENERAL_TOPIC_IDS = frozenset({"", "1"})

    def _is_telegram_topic_root_lobby(self, source: SessionSource) -> bool:
        """True for the main Telegram DM (or General topic) when topic mode has made it a lobby."""
        if source.platform != Platform.TELEGRAM or source.chat_type != "dm":
            return False
        if not self._telegram_topic_mode_enabled(source):
            return False
        tid = str(source.thread_id or "")
        return tid in self._TELEGRAM_GENERAL_TOPIC_IDS

    def _is_telegram_topic_lane(self, source: SessionSource) -> bool:
        """True for a user-created Telegram private-chat topic lane."""
        if source.platform != Platform.TELEGRAM or source.chat_type != "dm":
            return False
        if not self._telegram_topic_mode_enabled(source):
            return False
        tid = str(source.thread_id or "")
        if not tid or tid in self._TELEGRAM_GENERAL_TOPIC_IDS:
            return False
        return True

    _TELEGRAM_LOBBY_REMINDER_COOLDOWN_S = 30.0

    def _should_send_telegram_lobby_reminder(self, source: SessionSource) -> bool:
        """Rate-limit root-DM lobby reminders to one message per cooldown window.

        A user who forgets multi-session mode is enabled and types several
        prompts in the root DM would otherwise get a reminder for every
        message. Cap it so the first one lands and the rest stay quiet.
        """
        if not hasattr(self, "_telegram_lobby_reminder_ts"):
            self._telegram_lobby_reminder_ts = {}
        chat_id = str(source.chat_id or "")
        if not chat_id:
            return True
        import time as _time
        now = _time.monotonic()
        last = self._telegram_lobby_reminder_ts.get(chat_id, 0.0)
        if now - last < self._TELEGRAM_LOBBY_REMINDER_COOLDOWN_S:
            return False
        self._telegram_lobby_reminder_ts[chat_id] = now
        return True

    def _telegram_topic_root_lobby_message(self) -> str:
        return (
            "This main chat is reserved for system commands.\n\n"
            "To start a new Hermes chat, open the All Messages topic at the top "
            "of this bot interface and send any message there. Telegram will "
            "create a new topic for that message; each topic works as an "
            "independent Hermes session."
        )

    def _telegram_topic_root_new_message(self) -> str:
        return (
            "To start a new parallel Hermes chat, open the All Messages topic "
            "at the top of this bot interface and send any message there. "
            "Telegram will create a new topic for it.\n\n"
            "Each topic is an independent Hermes session. Use /new inside an "
            "existing topic only if you want to replace that topic's current session."
        )

    def _telegram_topic_new_header(self, source: SessionSource) -> Optional[str]:
        if not self._is_telegram_topic_lane(source):
            return None
        return (
            "Started a new Hermes session in this topic.\n\n"
            "Tip: for parallel work, open All Messages and send a message there "
            "to create a separate topic instead of using /new here. /new replaces "
            "the session attached to the current topic."
        )

    def _record_telegram_topic_binding(
        self,
        source: SessionSource,
        session_entry,
    ) -> None:
        """Persist the Telegram topic -> Hermes session binding for topic lanes."""
        session_db = getattr(self, "_session_db", None)
        if session_db is None or not source.chat_id or not source.thread_id:
            return
        session_db.bind_telegram_topic(
            chat_id=str(source.chat_id),
            thread_id=str(source.thread_id),
            user_id=str(source.user_id or ""),
            session_key=session_entry.session_key,
            session_id=session_entry.session_id,
        )

    def _sync_telegram_topic_binding(
        self,
        source: SessionSource,
        session_entry,
        *,
        reason: str,
    ) -> None:
        """Update the topic binding to point at ``session_entry.session_id``.

        Telegram topic lanes persist a (chat_id, thread_id) -> session_id row
        so reopening a topic in a fresh process resumes the right Hermes
        session. When compression rotates ``session_entry.session_id`` mid-turn,
        the binding goes stale and the next inbound message in that topic
        reloads the oversized parent transcript instead of the compressed
        child, retriggering preflight compression — sometimes in a loop
        (#20470, #29712, #33414).
        """
        if not self._is_telegram_topic_lane(source):
            return
        try:
            self._record_telegram_topic_binding(source, session_entry)
        except Exception:
            logger.debug(
                "telegram topic binding refresh failed (%s)", reason, exc_info=True,
            )

    def _recover_telegram_topic_thread_id(
        self,
        source: SessionSource,
    ) -> Optional[str]:
        """Pin DM-topic routing to the user's last-active topic.

        Telegram can omit ``message_thread_id`` or surface General (``1``)
        for some topic-mode DM replies. In those lobby-shaped cases, keep the
        conversation attached to the user's most-recent bound topic.

        Do not rewrite a non-lobby, previously-unbound thread id: a newly
        created Telegram DM topic is also "unknown" until the first inbound
        message is recorded, and rewriting it would send that brand-new topic's
        answer into an older lane. Returns None to leave the source alone.
        """
        if (
            source.platform != Platform.TELEGRAM
            or source.chat_type != "dm"
            or not source.chat_id
            or not source.user_id
            or not self._telegram_topic_mode_enabled(source)
        ):
            return None
        inbound = str(source.thread_id or "")
        is_lobby = not inbound or inbound in self._TELEGRAM_GENERAL_TOPIC_IDS
        if not is_lobby:
            # A non-lobby, unknown thread_id is most likely the first message in
            # a brand-new Telegram DM topic. Preserve it so it can be recorded
            # as a new independent lane below instead of hijacking the latest
            # existing topic binding.
            return None
        session_db = getattr(self, "_session_db", None)
        if session_db is None:
            return None
        try:
            bindings = session_db.list_telegram_topic_bindings_for_chat(
                chat_id=str(source.chat_id),
            )
        except Exception:
            logger.debug("topic-recover: read failed", exc_info=True)
            return None
        if not bindings:
            return None
        user_id = str(source.user_id)
        for b in bindings:  # newest-first
            if str(b.get("user_id") or "") == user_id:
                recovered = str(b.get("thread_id") or "")
                if recovered and recovered != inbound:
                    return recovered
                return None
        return None

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

    def _resolve_turn_agent_config(self, user_message: str, model: str, runtime_kwargs: dict) -> dict:
        """Build the effective model/runtime config for a single turn.

        Always uses the session's primary model/provider.  If `/fast` is
        enabled and the model supports Priority Processing / Anthropic fast
        mode, attach `request_overrides` so the API call is marked
        accordingly.
        """
        from hermes_cli.models import resolve_fast_mode_overrides

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

    async def _handle_adapter_fatal_error(self, adapter: BasePlatformAdapter) -> None:
        """React to an adapter failure after startup.

        If the error is retryable (e.g. network blip, DNS failure), queue the
        platform for background reconnection instead of giving up permanently.
        """
        logger.error(
            "Fatal %s adapter error (%s): %s",
            adapter.platform.value,
            adapter.fatal_error_code or "unknown",
            adapter.fatal_error_message or "unknown error",
        )
        self._update_platform_runtime_status(
            adapter.platform.value,
            platform_state="retrying" if adapter.fatal_error_retryable else "fatal",
            error_code=adapter.fatal_error_code,
            error_message=adapter.fatal_error_message,
        )

        existing = self.adapters.get(adapter.platform)
        if existing is adapter:
            try:
                await adapter.disconnect()
            finally:
                self.adapters.pop(adapter.platform, None)
                self.delivery_router.adapters = self.adapters

        # Queue retryable failures for background reconnection
        if adapter.fatal_error_retryable:
            platform_config = self.config.platforms.get(adapter.platform)
            if platform_config and adapter.platform not in self._failed_platforms:
                self._failed_platforms[adapter.platform] = {
                    "config": platform_config,
                    "attempts": 0,
                    "next_retry": time.monotonic() + 30,
                }
                logger.info(
                    "%s queued for background reconnection",
                    adapter.platform.value,
                )

        if not self.adapters and not self._failed_platforms:
            self._exit_reason = adapter.fatal_error_message or "All messaging adapters disconnected"
            if adapter.fatal_error_retryable:
                self._exit_with_failure = True
                logger.error("No connected messaging platforms remain. Shutting down gateway for service restart.")
            else:
                logger.error("No connected messaging platforms remain. Shutting down gateway cleanly.")
            await self.stop()
        elif not self.adapters and self._failed_platforms:
            # All platforms are down and queued for background reconnection.
            # Keep the gateway alive so:
            #   • cron jobs still run
            #   • the reconnect watcher can recover platforms when the
            #     underlying problem clears (proxy comes back, user runs
            #     `hermes whatsapp`, etc.)
            # We used to exit-with-failure here to trigger systemd restart,
            # but that converted a transient outage into a restart loop and
            # killed in-process state every time. The reconnect watcher
            # already handles long-running recovery — let it do its job.
            logger.warning(
                "No connected messaging platforms remain, but %d platform(s) "
                "queued for reconnection — gateway staying alive, watcher will "
                "retry in background.",
                len(self._failed_platforms),
            )

    def _request_clean_exit(self, reason: str) -> None:
        self._exit_cleanly = True
        self._exit_reason = reason
        self._shutdown_event.set()

    def _running_agent_count(self) -> int:
        return len(self._running_agents)

    def _status_action_label(self) -> str:
        return "restart" if self._restart_requested else "shutdown"

    def _status_action_gerund(self) -> str:
        return "restarting" if self._restart_requested else "shutting down"

    def _queue_during_drain_enabled(self) -> bool:
        # Both "queue" and "steer" modes imply the user doesn't want messages
        # to be lost during restart — queue them for the newly-spawned gateway
        # process to pick up.  "interrupt" mode drops them (current behaviour).
        return self._restart_requested and self._busy_input_mode in {"queue", "steer"}

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

    def _enqueue_fifo(self, session_key: str, queued_event: "MessageEvent", adapter: Any) -> None:
        """Append a /queue event to the FIFO chain for a session."""
        if adapter is None:
            return
        pending_slot = getattr(adapter, "_pending_messages", None)
        if pending_slot is None:
            return
        queued_events = getattr(self, "_queued_events", None)
        if queued_events is None:
            queued_events = {}
            self._queued_events = queued_events
        if session_key in pending_slot:
            queued_events.setdefault(session_key, []).append(queued_event)
        else:
            pending_slot[session_key] = queued_event

    def _promote_queued_event(
        self,
        session_key: str,
        adapter: Any,
        pending_event: Optional["MessageEvent"],
    ) -> Optional["MessageEvent"]:
        """Promote the next overflow item after the slot was drained.

        Called at the drain site after _dequeue_pending_event consumed
        (or failed to consume) the slot.  If there's an overflow item:
          - When pending_event is None (slot was empty), return the
            overflow head as the new pending_event.
          - When pending_event already exists (slot was populated by an
            interrupt follow-up or similar), stage the overflow head in
            the slot so the NEXT recursion picks it up.
        Returns the (possibly updated) pending_event for drain to use.
        """
        queued_events = getattr(self, "_queued_events", None)
        if not queued_events:
            return pending_event
        overflow = queued_events.get(session_key)
        if not overflow:
            return pending_event
        next_queued = overflow.pop(0)
        if not overflow:
            queued_events.pop(session_key, None)
        if pending_event is None:
            return next_queued
        if adapter is not None and hasattr(adapter, "_pending_messages"):
            adapter._pending_messages[session_key] = next_queued
        else:
            # No adapter — push back so we don't silently drop the item.
            queued_events.setdefault(session_key, []).insert(0, next_queued)
        return pending_event

    def _queue_depth(self, session_key: str, *, adapter: Any = None) -> int:
        """Total pending /queue items for a session — slot + overflow."""
        queued_events = getattr(self, "_queued_events", None) or {}
        depth = len(queued_events.get(session_key, []))
        if adapter is not None and session_key in getattr(adapter, "_pending_messages", {}):
            depth += 1
        return depth

    @staticmethod
    def _is_goal_continuation_event(event_or_text: Any) -> bool:
        """Return True for synthetic /goal continuation turns.

        Goal continuations are normal queued user-role events, so pause/clear
        must distinguish them from real user /queue messages before removing or
        suppressing them.
        """
        text = getattr(event_or_text, "text", event_or_text) or ""
        return str(text).startswith("[Continuing toward your standing goal]\nGoal:")

    def _clear_goal_pending_continuations(self, session_key: str, adapter: Any) -> int:
        """Remove queued synthetic /goal continuations for one session.

        User-issued /goal pause/clear can race with a continuation already
        queued by the judge.  Remove only synthetic goal continuations while
        preserving normal /queue and user follow-up events.
        """
        removed = 0
        pending_slot = getattr(adapter, "_pending_messages", None) if adapter is not None else None
        if isinstance(pending_slot, dict):
            pending_event = pending_slot.get(session_key)
            if self._is_goal_continuation_event(pending_event):
                pending_slot.pop(session_key, None)
                removed += 1

        queued_events = getattr(self, "_queued_events", None)
        if isinstance(queued_events, dict):
            overflow = queued_events.get(session_key) or []
            if overflow:
                kept = []
                for queued_event in overflow:
                    if self._is_goal_continuation_event(queued_event):
                        removed += 1
                    else:
                        kept.append(queued_event)
                if kept:
                    queued_events[session_key] = kept
                else:
                    queued_events.pop(session_key, None)
        return removed

    def _goal_still_active_for_session(self, session_id: str) -> bool:
        """Best-effort fresh DB check before running a queued continuation."""
        if not session_id:
            return False
        try:
            from hermes_cli.goals import GoalManager
            return GoalManager(session_id=session_id).is_active()
        except Exception as exc:
            logger.debug("goal continuation: active-state recheck failed: %s", exc)
            return False

    def _update_runtime_status(self, gateway_state: Optional[str] = None, exit_reason: Optional[str] = None) -> None:
        try:
            from gateway.status import write_runtime_status
            write_runtime_status(
                gateway_state=gateway_state,
                exit_reason=exit_reason,
                restart_requested=self._restart_requested,
                active_agents=self._running_agent_count(),
            )
        except Exception:
            pass

    def _update_platform_runtime_status(
        self,
        platform: str,
        *,
        platform_state: Optional[str] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        try:
            from gateway.status import write_runtime_status
            write_runtime_status(
                platform=platform,
                platform_state=platform_state,
                error_code=error_code,
                error_message=error_message,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Per-platform circuit breaker (pause/resume) — used by the reconnect
    # watcher when a retryable failure recurs past a threshold, and by the
    # /platform pause|resume slash command for manual control.
    # ------------------------------------------------------------------
    def _pause_failed_platform(self, platform, *, reason: str = "") -> None:
        """Mark a queued platform as paused — keep it in ``_failed_platforms``
        but stop the reconnect watcher from hammering it.

        Used by ``/platform pause <name>`` for manual operator intervention.
        Paused platforms are surfaced in ``/platform list`` and resumed with
        ``/platform resume <name>``.  Note: the reconnect watcher does NOT
        auto-pause — retryable (network/DNS) failures keep retrying at the
        backoff cap indefinitely so a transient outage self-heals without
        manual intervention.
        """
        info = getattr(self, "_failed_platforms", {}).get(platform)
        if info is None:
            return
        if info.get("paused"):
            return
        info["paused"] = True
        info["pause_reason"] = reason or "auto-paused after repeated failures"
        # Push next_retry far enough out that even if "paused" is missed
        # by a stale code path, the watcher won't fire on it.
        info["next_retry"] = float("inf")
        try:
            self._update_platform_runtime_status(
                platform.value,
                platform_state="paused",
                error_code=None,
                error_message=info["pause_reason"],
            )
        except Exception:
            pass
        logger.warning(
            "%s paused after %d consecutive failures (%s) — "
            "fix the underlying issue then run `/platform resume %s` "
            "to retry, or `hermes gateway restart` to restart the gateway.",
            platform.value, info.get("attempts", 0),
            info["pause_reason"], platform.value,
        )

    def _resume_paused_platform(self, platform) -> bool:
        """Unpause a platform — reset its attempt counter and schedule an
        immediate retry.  Returns True if the platform was paused and is
        now queued; False if it wasn't paused (or wasn't in the queue).
        """
        info = getattr(self, "_failed_platforms", {}).get(platform)
        if info is None:
            return False
        if not info.get("paused"):
            return False
        info["paused"] = False
        info.pop("pause_reason", None)
        info["attempts"] = 0
        info["next_retry"] = time.monotonic()  # retry on next watcher tick
        try:
            self._update_platform_runtime_status(
                platform.value,
                platform_state="retrying",
                error_code=None,
                error_message=None,
            )
        except Exception:
            pass
        logger.info("%s resumed — retrying on next watcher tick", platform.value)
        return True

    @staticmethod
    def _load_prefill_messages() -> List[Dict[str, Any]]:
        """Load ephemeral prefill messages from config or env var.
        
        Checks HERMES_PREFILL_MESSAGES_FILE env var first, then falls back to
        the prefill_messages_file key in ~/.hermes/config.yaml.
        Relative paths are resolved from ~/.hermes/.
        """
        file_path = os.getenv("HERMES_PREFILL_MESSAGES_FILE", "")
        if not file_path:
            cfg = _load_gateway_runtime_config()
            file_path = str(cfg.get("prefill_messages_file", "") or "")
        if not file_path:
            return []
        path = Path(file_path).expanduser()
        if not path.is_absolute():
            path = _hermes_home / path
        if not path.exists():
            logger.warning("Prefill messages file not found: %s", path)
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                logger.warning("Prefill messages file must contain a JSON array: %s", path)
                return []
            return data
        except Exception as e:
            logger.warning("Failed to load prefill messages from %s: %s", path, e)
            return []

    @staticmethod
    def _load_ephemeral_system_prompt() -> str:
        """Load ephemeral system prompt from config or env var.
        
        Checks HERMES_EPHEMERAL_SYSTEM_PROMPT env var first, then falls back to
        agent.system_prompt in ~/.hermes/config.yaml.
        """
        prompt = os.getenv("HERMES_EPHEMERAL_SYSTEM_PROMPT", "")
        if prompt:
            return prompt
        cfg = _load_gateway_runtime_config()
        return str(cfg_get(cfg, "agent", "system_prompt", default="") or "").strip()

    @staticmethod
    def _load_reasoning_config() -> dict | None:
        """Load reasoning effort from config.yaml.

        Reads agent.reasoning_effort from config.yaml. Valid: "none",
        "minimal", "low", "medium", "high", "xhigh". Returns None to use
        default (medium).
        """
        from hermes_constants import parse_reasoning_effort
        cfg = _load_gateway_runtime_config()
        effort = str(cfg_get(cfg, "agent", "reasoning_effort", default="") or "").strip()
        result = parse_reasoning_effort(effort)
        if effort and effort.strip() and result is None:
            logger.warning("Unknown reasoning_effort '%s', using default (medium)", effort)
        return result

    @staticmethod
    def _parse_reasoning_command_args(raw_args: str) -> tuple[str, bool]:
        """Parse `/reasoning` args into `(value, persist_global)`.

        `/reasoning <level>` is session-scoped by default. `--global` may be
        supplied in any position to persist the change to config.yaml.
        """
        import shlex

        text = str(raw_args or "").strip().replace("—", "--")
        if not text:
            return "", False
        try:
            tokens = shlex.split(text)
        except ValueError:
            tokens = text.split()

        persist_global = False
        value_tokens = []
        for token in tokens:
            if token == "--global":
                persist_global = True
            else:
                value_tokens.append(token)
        return " ".join(value_tokens).strip().lower(), persist_global

    def _resolve_session_reasoning_config(
        self,
        *,
        source: Optional[SessionSource] = None,
        session_key: Optional[str] = None,
    ) -> dict | None:
        """Resolve reasoning effort for a session, honoring session overrides."""
        resolved_session_key = session_key
        if not resolved_session_key and source is not None:
            try:
                resolved_session_key = self._session_key_for_source(source)
            except Exception:
                resolved_session_key = None

        overrides = getattr(self, "_session_reasoning_overrides", {}) or {}
        if resolved_session_key and resolved_session_key in overrides:
            return overrides[resolved_session_key]
        return self._load_reasoning_config()

    def _set_session_reasoning_override(
        self,
        session_key: str,
        reasoning_config: Optional[dict],
    ) -> None:
        """Set or clear the session-scoped reasoning override."""
        if not session_key:
            return
        if not hasattr(self, "_session_reasoning_overrides"):
            self._session_reasoning_overrides = {}
        if reasoning_config is None:
            self._session_reasoning_overrides.pop(session_key, None)
        else:
            self._session_reasoning_overrides[session_key] = dict(reasoning_config)

    @staticmethod
    def _load_service_tier() -> str | None:
        """Load Priority Processing setting from config.yaml.

        Reads agent.service_tier from config.yaml. Accepted values mirror the CLI:
        "fast"/"priority"/"on" => "priority", while "normal"/"off" disables it.
        Returns None when unset or unsupported.
        """
        cfg = _load_gateway_runtime_config()
        raw = str(cfg_get(cfg, "agent", "service_tier", default="") or "").strip()

        value = raw.lower()
        if not value or value in {"normal", "default", "standard", "off", "none"}:
            return None
        if value in {"fast", "priority", "on"}:
            return "priority"
        logger.warning("Unknown service_tier '%s', ignoring", raw)
        return None

    @staticmethod
    def _load_show_reasoning() -> bool:
        """Load show_reasoning toggle from config.yaml display section."""
        cfg = _load_gateway_runtime_config()
        return is_truthy_value(
            cfg_get(cfg, "display", "show_reasoning"),
            default=False,
        )

    @staticmethod
    def _load_busy_input_mode() -> str:
        """Load gateway drain-time busy-input behavior from config/env."""
        mode = os.getenv("HERMES_GATEWAY_BUSY_INPUT_MODE", "").strip().lower()
        if not mode:
            cfg = _load_gateway_runtime_config()
            mode = str(cfg_get(cfg, "display", "busy_input_mode", default="") or "").strip().lower()
        if mode == "queue":
            return "queue"
        if mode == "steer":
            return "steer"
        return "interrupt"

    @staticmethod
    def _load_busy_text_mode() -> str:
        """Load normal busy TEXT follow-up behavior from config/env."""
        mode = os.getenv("HERMES_GATEWAY_BUSY_TEXT_MODE", "").strip().lower()
        if not mode:
            cfg = _load_gateway_runtime_config()
            mode = str(cfg_get(cfg, "display", "busy_text_mode", default="") or "").strip().lower()
        if mode == "interrupt":
            return "interrupt"
        return "queue"

    @staticmethod
    def _load_restart_drain_timeout() -> float:
        """Load graceful gateway restart/stop drain timeout in seconds."""
        raw = os.getenv("HERMES_RESTART_DRAIN_TIMEOUT", "").strip()
        if not raw:
            cfg = _load_gateway_runtime_config()
            raw = str(cfg_get(cfg, "agent", "restart_drain_timeout", default="") or "").strip()
        value = parse_restart_drain_timeout(raw)
        if raw and value == DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT:
            try:
                float(raw)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid restart_drain_timeout '%s', using default %.0fs",
                    raw,
                    DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
                )
        return value

    @staticmethod
    def _load_background_notifications_mode() -> str:
        """Load background process notification mode from config or env var.

        Modes:
          - ``all``    — push running-output updates *and* the final message (default)
          - ``result`` — only the final completion message (regardless of exit code)
          - ``error``  — only the final message when exit code is non-zero
          - ``off``    — no watcher messages at all
        """
        mode = os.getenv("HERMES_BACKGROUND_NOTIFICATIONS", "")
        if not mode:
            cfg = _load_gateway_runtime_config()
            raw = cfg_get(cfg, "display", "background_process_notifications")
            if raw is False:
                mode = "off"
            elif raw not in {None, ""}:
                mode = str(raw)
        mode = (mode or "all").strip().lower()
        valid = {"all", "result", "error", "off"}
        if mode not in valid:
            logger.warning(
                "Unknown background_process_notifications '%s', defaulting to 'all'",
                mode,
            )
            return "all"
        return mode

    @staticmethod
    def _load_provider_routing() -> dict:
        """Load OpenRouter provider routing preferences from config.yaml."""
        try:
            import yaml as _y
            cfg_path = _hermes_home / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as _f:
                    cfg = _y.safe_load(_f) or {}
                return cfg.get("provider_routing", {}) or {}
        except Exception:
            pass
        return {}

    @staticmethod
    def _load_fallback_model() -> list | None:
        """Load fallback provider chain from config.yaml.

        Returns the merged effective chain from ``fallback_providers`` plus any
        legacy ``fallback_model`` entries. ``fallback_providers`` stays first
        when both keys are present.
        """
        try:
            import yaml as _y
            cfg_path = _hermes_home / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as _f:
                    cfg = _y.safe_load(_f) or {}
                fb = get_fallback_chain(cfg)
                if fb:
                    return fb
        except Exception:
            pass
        return None

    def _snapshot_running_agents(self) -> Dict[str, Any]:
        return {
            session_key: agent
            for session_key, agent in self._running_agents.items()
            if agent is not _AGENT_PENDING_SENTINEL
        }

    @staticmethod
    def _agent_has_active_subagents(running_agent: Any) -> bool:
        """Return True when *running_agent* is currently driving subagents
        via the ``delegate_task`` tool.

        Background (#30170): ``AIAgent.interrupt()`` cascades through the
        parent's ``_active_children`` list and calls ``interrupt()`` on
        every child synchronously, which aborts in-flight subagent work
        and produces a fallback cascade with no actionable signal.
        Demoting ``busy_input_mode='interrupt'`` to ``queue`` semantics
        whenever this helper returns True protects subagent work from
        conversational follow-ups while leaving the explicit ``/stop``
        path (which goes through ``_interrupt_and_clear_session``)
        untouched. Safe-by-default: returns False on any attribute or
        lock error so a missing/broken parent never blocks the existing
        interrupt path.
        """
        if running_agent is None or running_agent is _AGENT_PENDING_SENTINEL:
            return False
        children = getattr(running_agent, "_active_children", None)
        # AIAgent always initialises this as a concrete list (see
        # agent/agent_init.py). Reject anything that isn't a real
        # collection — this guards against ``MagicMock()._active_children``
        # auto-creating a truthy stub in tests and triggering the demotion
        # against an agent that doesn't actually have subagents.
        if not isinstance(children, (list, tuple, set)):
            return False
        if not children:
            return False
        lock = getattr(running_agent, "_active_children_lock", None)
        try:
            if lock is not None:
                with lock:
                    return bool(children)
            return bool(children)
        except Exception:
            return False

    def _queue_or_replace_pending_event(self, session_key: str, event: MessageEvent) -> None:
        adapter = self.adapters.get(event.source.platform)
        if not adapter:
            return
        merge_pending_message_event(adapter._pending_messages, session_key, event)

    async def _handle_active_session_busy_message(self, event: MessageEvent, session_key: str) -> bool:
        return await message_processing.handle_active_session_busy_message(
            runner=self,
            event=event,
            session_key=session_key,
        )

        snapshot = self._snapshot_running_agents()
        last_active_count = self._running_agent_count()
        last_status_at = 0.0

        def _maybe_update_status(force: bool = False) -> None:
            nonlocal last_active_count, last_status_at
            now = asyncio.get_running_loop().time()
            active_count = self._running_agent_count()
            if force or active_count != last_active_count or (now - last_status_at) >= 1.0:
                self._update_runtime_status("draining")
                last_active_count = active_count
                last_status_at = now

        if not self._running_agents:
            _maybe_update_status(force=True)
            return snapshot, False

        _maybe_update_status(force=True)
        if timeout <= 0:
            return snapshot, True

        deadline = asyncio.get_running_loop().time() + timeout
        while self._running_agents and asyncio.get_running_loop().time() < deadline:
            _maybe_update_status()
            await asyncio.sleep(0.1)
        timed_out = bool(self._running_agents)
        _maybe_update_status(force=True)
        return snapshot, timed_out

    def _interrupt_running_agents(self, reason: str) -> None:
        for session_key, agent in list(self._running_agents.items()):
            if agent is _AGENT_PENDING_SENTINEL:
                continue
            try:
                agent.interrupt(reason)
                logger.debug("Interrupted running agent for session %s during shutdown", session_key)
            except Exception as e:
                logger.debug("Failed interrupting agent during shutdown: %s", e)

    async def _notify_active_sessions_of_shutdown(self) -> None:
        """Send shutdown/restart notifications to active chats and home channels.

        Called at the very start of stop() — adapters are still connected so
        messages can be delivered. Best-effort: individual send failures are
        logged and swallowed so they never block the shutdown sequence.
        """
        return await shutdown_notifications.notify_active_sessions_of_shutdown(runner=self)
    def _finalize_shutdown_agents(self, active_agents: Dict[str, Any]) -> None:
        for agent in active_agents.values():
            try:
                from hermes_cli.plugins import invoke_hook as _invoke_hook
                _invoke_hook(
                    "on_session_finalize",
                    session_id=getattr(agent, "session_id", None),
                    platform="gateway",
                    reason="shutdown",
                )
            except Exception:
                pass
            self._cleanup_agent_resources(agent)

    def _cleanup_agent_resources(self, agent: Any) -> None:
        """Best-effort cleanup for temporary or cached agent instances."""
        if agent is None:
            return
        try:
            if hasattr(agent, "shutdown_memory_provider"):
                # Pass the agent's own conversation transcript so memory
                # providers' ``on_session_end`` hooks see the real messages
                # instead of the empty default (#15165). ``_session_messages``
                # is set on ``AIAgent`` (run_agent.py:1518) and refreshed at
                # the end of every ``run_conversation`` turn via
                # ``_persist_session``; on an agent built through
                # ``object.__new__`` (test stubs) the attribute may be
                # absent, so ``getattr`` with a ``None`` default keeps the
                # call signature-compatible with the pre-fix behaviour
                # (``shutdown_memory_provider(messages=None)``).
                session_messages = getattr(agent, "_session_messages", None)
                if isinstance(session_messages, list):
                    agent.shutdown_memory_provider(session_messages)
                else:
                    agent.shutdown_memory_provider()
        except Exception:
            pass
        # Close tool resources (terminal sandboxes, browser daemons,
        # background processes, httpx clients) to prevent zombie
        # process accumulation.
        try:
            if hasattr(agent, "close"):
                agent.close()
        except Exception:
            pass
        # Auxiliary async clients (session_search/web/vision/etc.) live in a
        # process-global cache and are created inside worker threads. Clean up
        # any entries whose event loop is now dead so their httpx transports do
        # not accumulate across gateway turns.
        try:
            from agent.auxiliary_client import cleanup_stale_async_clients
            cleanup_stale_async_clients()
        except Exception:
            pass

    _STUCK_LOOP_THRESHOLD = 3  # restarts while active before auto-suspend
    _STUCK_LOOP_FILE = ".restart_failure_counts"

    def _increment_restart_failure_counts(self, active_session_keys: set) -> None:
        """Increment restart-failure counters for sessions active at shutdown.

        Persists to a JSON file so counters survive across restarts.
        Sessions NOT in active_session_keys are removed (they completed
        successfully, so the loop is broken).
        """
        import json

        path = _hermes_home / self._STUCK_LOOP_FILE
        try:
            counts = json.loads(path.read_text()) if path.exists() else {}
        except Exception:
            counts = {}

        # Increment active sessions, remove inactive ones (loop broken)
        new_counts = {}
        for key in active_session_keys:
            new_counts[key] = counts.get(key, 0) + 1
        # Keep any entries that are still above 0 even if not active now
        # (they might become active again next restart)

        try:
            atomic_json_write(path, new_counts, indent=None)
        except Exception:
            pass

    def _suspend_stuck_loop_sessions(self) -> int:
        """Suspend sessions that have been active across too many restarts.

        Returns the number of sessions suspended.  Called on gateway startup
        AFTER suspend_recently_active() to catch the stuck-loop pattern:
        session loads → agent gets stuck → gateway restarts → repeat.
        """
        import json

        path = _hermes_home / self._STUCK_LOOP_FILE
        if not path.exists():
            return 0

        try:
            counts = json.loads(path.read_text())
        except Exception:
            return 0

        suspended = 0
        stuck_keys = [k for k, v in counts.items() if v >= self._STUCK_LOOP_THRESHOLD]

        for session_key in stuck_keys:
            try:
                entry = self.session_store._entries.get(session_key)
                if entry and not entry.suspended:
                    entry.suspended = True
                    suspended += 1
                    logger.warning(
                        "Auto-suspended stuck session %s (active across %d "
                        "consecutive restarts — likely a stuck loop)",
                        session_key, counts[session_key],
                    )
            except Exception:
                pass

        if suspended:
            try:
                self.session_store._save()
            except Exception:
                pass

        # Clear the file — counters start fresh after suspension
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

        return suspended

    def _clear_restart_failure_count(self, session_key: str) -> None:
        """Clear the restart-failure counter for a session that completed OK.

        Called after a successful agent turn to signal the loop is broken.
        """
        import json

        path = _hermes_home / self._STUCK_LOOP_FILE
        if not path.exists():
            return
        try:
            counts = json.loads(path.read_text())
            if session_key in counts:
                del counts[session_key]
                if counts:
                    atomic_json_write(path, counts, indent=None)
                else:
                    path.unlink(missing_ok=True)
        except Exception:
            pass

    async def _launch_detached_restart_command(self) -> None:
        import shutil
        import subprocess

        hermes_cmd = _resolve_hermes_bin()
        if not hermes_cmd:
            logger.error("Could not locate hermes binary for detached /restart")
            return

        current_pid = os.getpid()

        # On Windows there's no bash/setsid chain — spawn a tiny Python
        # watcher directly via sys.executable instead.  The watcher polls
        # current_pid, waits for our exit, then runs `hermes gateway
        # restart` with detach flags so the respawn survives the CLI
        # that triggered the /restart command closing its console.
        if sys.platform == "win32":
            import textwrap
            from hermes_cli._subprocess_compat import windows_detach_popen_kwargs

            cmd_argv = [*hermes_cmd, "gateway", "restart"]
            watcher = textwrap.dedent(
                """
                import os, subprocess, sys, time
                pid = int(sys.argv[1])
                cmd = sys.argv[2:]
                deadline = time.monotonic() + 120

                def _alive(p):
                    # On Windows, os.kill(pid, 0) is NOT a no-op — it maps to
                    # GenerateConsoleCtrlEvent(0, pid) (bpo-14484). Use the
                    # Win32 handle-based existence check instead.
                    if os.name == 'nt':
                        import ctypes
                        k32 = ctypes.windll.kernel32
                        k32.OpenProcess.restype = ctypes.c_void_p
                        k32.WaitForSingleObject.restype = ctypes.c_uint
                        k32.GetLastError.restype = ctypes.c_uint
                        h = k32.OpenProcess(0x1000 | 0x100000, False, int(p))
                        if not h:
                            return k32.GetLastError() != 87
                        try:
                            return k32.WaitForSingleObject(h, 0) == 0x102
                        finally:
                            k32.CloseHandle(h)
                    try:
                        os.kill(int(p), 0)
                        return True
                    except ProcessLookupError:
                        return False
                    except PermissionError:
                        return True
                    except OSError:
                        return False

                while time.monotonic() < deadline:
                    if not _alive(pid):
                        break
                    time.sleep(0.2)
                _CREATE_NEW_PROCESS_GROUP = 0x00000200
                _DETACHED_PROCESS = 0x00000008
                _CREATE_NO_WINDOW = 0x08000000
                subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=_CREATE_NEW_PROCESS_GROUP | _DETACHED_PROCESS | _CREATE_NO_WINDOW,
                )
                """
            ).strip()
            subprocess.Popen(
                [sys.executable, "-c", watcher, str(current_pid), *cmd_argv],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **windows_detach_popen_kwargs(),
            )
            return

        cmd = " ".join(shlex.quote(part) for part in hermes_cmd)
        shell_cmd = (
            f"while kill -0 {current_pid} 2>/dev/null; do sleep 0.2; done; "
            f"{cmd} gateway restart"
        )
        setsid_bin = shutil.which("setsid")
        if setsid_bin:
            subprocess.Popen(
                [setsid_bin, "bash", "-lc", shell_cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        else:
            subprocess.Popen(
                ["bash", "-lc", shell_cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

    def _launch_systemd_restart_shortcut(self) -> None:
        """Best-effort helper to bypass systemd's automatic restart delay.

        For planned in-chat restarts, the gateway exits cleanly so systemd does
        not record a failure.  However, units with RestartSteps still count
        automatic restarts and can delay repeated /restart tests.  A transient
        user service survives our cgroup teardown and explicitly starts the
        gateway as soon as this PID exits, while the unit keeps its normal
        backoff for real crash loops.
        """
        if sys.platform != "linux" or not os.environ.get("INVOCATION_ID"):
            return

        try:
            import shutil
            import subprocess

            systemd_run = shutil.which("systemd-run")
            systemctl = shutil.which("systemctl")
            if not systemd_run or not systemctl:
                return

            try:
                from hermes_cli.gateway import get_service_name

                service_name = get_service_name()
            except Exception:
                service_name = "hermes-gateway"

            current_pid = os.getpid()
            show = subprocess.run(
                [
                    systemctl,
                    "--user",
                    "show",
                    service_name,
                    "--property=MainPID",
                    "--value",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if (show.stdout or "").strip() != str(current_pid):
                return

            systemctl_user = "systemctl --user"
            service_arg = shlex.quote(service_name)
            shell_cmd = (
                f"while kill -0 {current_pid} 2>/dev/null; do sleep 0.2; done; "
                f"{systemctl_user} reset-failed {service_arg}; "
                f"{systemctl_user} restart {service_arg}"
            )
            unit_name = f"{service_name}-planned-restart-{current_pid}".replace(".", "-")
            subprocess.Popen(
                [
                    systemd_run,
                    "--user",
                    "--collect",
                    "--unit",
                    unit_name,
                    "/bin/sh",
                    "-lc",
                    shell_cmd,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            logger.info(
                "Launched systemd planned-restart helper for %s (pid=%s)",
                service_name,
                current_pid,
            )
        except Exception as e:
            logger.debug("Failed to launch systemd planned-restart helper: %s", e)

    def request_restart(self, *, detached: bool = False, via_service: bool = False) -> bool:
        if self._restart_task_started:
            return False
        self._restart_requested = True
        self._restart_detached = detached
        self._restart_via_service = via_service
        self._restart_task_started = True

        async def _run_restart() -> None:
            await asyncio.sleep(0.05)
            await self.stop(restart=True, detached_restart=detached, service_restart=via_service)

        task = asyncio.create_task(_run_restart())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return True

    # Drain-timeout reasons set by _stop_impl() when a still-running turn is
    # force-interrupted; "restart_interrupted" is set by
    # SessionStore.suspend_recently_active() on crash recovery (no
    # .clean_shutdown marker).  All three mean "the agent was mid-turn and
    # we killed it" — eligible for startup auto-resume.
    _AUTO_RESUME_REASONS = frozenset(
        {"restart_timeout", "shutdown_timeout", "restart_interrupted"}
    )

    def _schedule_resume_pending_sessions(self) -> int:
        """Auto-continue fresh restart-interrupted sessions after startup.

        ``resume_pending`` already preserves the transcript AND the existing
        ``_is_resume_pending`` branch in ``_handle_message_with_agent``
        injects a reason-aware recovery system note on the next turn.  This
        method closes the UX gap by synthesizing that next turn once
        adapters are back online — the event text is empty so the existing
        injection path owns the wording and we never double up.

        Adapters that are not yet ready (adapter missing from
        ``self.adapters``) are skipped silently; their sessions stay
        ``resume_pending`` and will auto-resume on the next real user
        message, or on the next gateway startup.
        """
        window = _auto_continue_freshness_window()
        try:
            with self.session_store._lock:  # noqa: SLF001 — snapshot under lock
                self.session_store._ensure_loaded_locked()  # noqa: SLF001
                candidates = [
                    entry for entry in self.session_store._entries.values()  # noqa: SLF001
                    if entry.resume_pending
                    and not entry.suspended
                    and entry.origin is not None
                    and entry.resume_reason in self._AUTO_RESUME_REASONS
                ]
        except Exception as exc:
            logger.warning("Failed to enumerate resume-pending sessions: %s", exc)
            return 0

        now = datetime.now()
        scheduled = 0
        for entry in candidates:
            marker = entry.last_resume_marked_at or entry.updated_at
            if marker is not None and (now - marker).total_seconds() > window:
                continue

            source = entry.origin
            adapter = self.adapters.get(source.platform)
            if adapter is None:
                logger.debug(
                    "Skipping auto-resume for %s: adapter not ready for %s",
                    entry.session_key,
                    getattr(source.platform, "value", source.platform),
                )
                continue

            # Empty-text internal event — the _is_resume_pending branch in
            # _handle_message_with_agent prepends the proper reason-aware
            # system note before the turn runs.
            event = MessageEvent(
                text="",
                message_type=MessageType.TEXT,
                source=source,
                internal=True,
            )
            task = asyncio.create_task(adapter.handle_message(event))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            scheduled += 1

        if scheduled:
            logger.info(
                "Scheduled auto-resume for %d restart-interrupted session(s)",
                scheduled,
            )
        return scheduled

    async def start(self) -> bool:
        return await lifecycle.start_gateway_runner(
            runner=self,
        )

        """Background task that processes pending CLI→gateway session handoffs.

        Polls ``state.db`` for sessions in ``handoff_state='pending'`` and,
        for each one:

        1. Atomically claims it (pending → running).
        2. Resolves the destination platform's configured home channel.
        3. Re-binds the gateway's session_key for that home channel to the
           CLI's existing session_id via ``session_store.switch_session`` so
           the full role-aware transcript replays on the next agent turn.
        4. Forges a synthetic ``MessageEvent`` (``internal=True``) with a
           handoff-notice text and dispatches through the normal gateway
           message pipeline so the agent runs and replies on the platform.
        5. Marks the row ``completed`` (or ``failed`` with ``handoff_error``).

        The CLI process is poll-blocked on the row's terminal state and
        prints the result to the user.
        """
        # Initial delay so the gateway is fully connected to its platforms
        # before we try to dispatch handoffs through them.
        await asyncio.sleep(5)
        while self._running:
            try:
                if self._session_db is None:
                    await asyncio.sleep(interval)
                    continue
                pending = self._session_db.list_pending_handoffs()
                for row in pending:
                    session_id = row.get("id")
                    if not session_id:
                        continue
                    if not self._session_db.claim_handoff(session_id):
                        # Another tick or another gateway already claimed it.
                        continue
                    try:
                        await self._process_handoff(row)
                        self._session_db.complete_handoff(session_id)
                    except Exception as exc:
                        logger.warning(
                            "Handoff for session %s failed: %s",
                            session_id, exc, exc_info=True,
                        )
                        self._session_db.fail_handoff(session_id, str(exc))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Handoff watcher tick error: %s", exc, exc_info=True)
            await asyncio.sleep(interval)

    async def _process_handoff(self, row: Dict[str, Any]) -> None:
        """Execute one handoff row. Raises on failure (caller marks failed)."""
        from gateway.config import Platform
        from gateway.session import SessionSource, build_session_key
        from gateway.platforms.base import MessageEvent

        cli_session_id = row["id"]
        platform_name = (row.get("handoff_platform") or "").strip().lower()
        if not platform_name:
            raise RuntimeError("handoff_platform is empty")

        # Resolve platform enum
        try:
            platform = Platform(platform_name)
        except (ValueError, KeyError):
            raise RuntimeError(f"unknown platform '{platform_name}'")

        # Adapter must be live
        adapter = self.adapters.get(platform)
        if not adapter:
            raise RuntimeError(
                f"platform '{platform_name}' is not active in this gateway"
            )

        # Home channel must be configured
        home = self.config.get_home_channel(platform)
        if not home or not home.chat_id:
            raise RuntimeError(
                f"no home channel configured for {platform_name}; "
                f"run /sethome on the desired chat first"
            )

        cli_title = row.get("title") or cli_session_id[:8]

        # Try to create a fresh thread on the destination so the handoff
        # has its own scrollback. Adapter returns None if threading isn't
        # supported (Matrix/WhatsApp/Signal/SMS) or if creation failed
        # (no permission, topics-mode off, parent is a DM, etc.). When
        # None we fall through to using the home channel directly — the
        # synthetic turn still lands; just without thread isolation.
        thread_name = f"Hermes — {cli_title}"
        try:
            new_thread_id = await adapter.create_handoff_thread(
                str(home.chat_id), thread_name,
            )
        except Exception as exc:
            logger.debug(
                "Handoff: create_handoff_thread raised on %s: %s",
                platform_name, exc, exc_info=True,
            )
            new_thread_id = None

        # Use the new thread if the adapter created one; otherwise fall
        # back to whatever thread (if any) the home channel was configured
        # with.
        effective_thread_id = new_thread_id or (
            str(home.thread_id) if home.thread_id else None
        )

        # Determine chat_type for the destination source. If we created a
        # thread, key the session_key as a thread (build_session_key sets
        # thread sessions to user-shared by default, which is what we
        # want — the synthetic turn and any later real-user message both
        # land on the same key without needing a user_id).
        if new_thread_id:
            dest_chat_type = "thread"
        else:
            # No thread — assume DM-style for the home channel. For
            # group/channel home channels without thread support
            # (Matrix/WhatsApp/Signal), the platform's own keying makes
            # the synthetic turn shared anyway (single-DM platforms).
            dest_chat_type = "dm"

        dest_source = SessionSource(
            platform=platform,
            chat_id=str(home.chat_id),
            chat_name=home.name,
            chat_type=dest_chat_type,
            user_id="system:handoff",
            user_name="Handoff",
            thread_id=effective_thread_id,
        )

        # Compute the gateway's session_key for that destination using the
        # same rules its adapters use, so switch_session targets the right
        # entry. For thread destinations build_session_key keys without
        # user_id (thread_sessions_per_user defaults to False) — so the
        # next real user message in the thread shares this same session.
        platform_cfg = self.config.platforms.get(platform)
        extra = platform_cfg.extra if platform_cfg else {}
        session_key = build_session_key(
            dest_source,
            group_sessions_per_user=extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=extra.get("thread_sessions_per_user", False),
        )

        # Make sure there's an entry in the session_store for this key. If
        # the home channel has never been used, get_or_create_session
        # creates one; switch_session then re-points it.
        self.session_store.get_or_create_session(dest_source)

        # Re-bind the destination key to the CLI session_id. switch_session
        # ends the prior session in SQLite and reopens the CLI session under
        # the new key. The CLI's transcript becomes the active one for the
        # gateway from this moment on.
        switched = self.session_store.switch_session(session_key, cli_session_id)
        if switched is None:
            raise RuntimeError(
                f"could not switch session key {session_key} → {cli_session_id}"
            )

        # Evict any cached AIAgent for this session_key so the next dispatch
        # rebuilds it against the CLI session_id (mirrors /resume / /branch).
        self._evict_cached_agent(session_key)

        # Cancel any in-flight running-agent state for the destination key
        # so the synthetic turn isn't queued behind a stale running flag.
        self._release_running_agent_state(session_key)

        synthetic_text = (
            f"[Session was just handed off from CLI (\"{cli_title}\") to this "
            f"channel. The full prior conversation history is loaded above. "
            f"Briefly confirm you're working here and summarize what we were "
            f"working on, so the user can continue from this device.]"
        )

        synthetic_event = MessageEvent(
            text=synthetic_text,
            source=dest_source,
            internal=True,
        )

        logger.info(
            "Handoff: dispatching synthetic turn for CLI session %s → %s "
            "(home=%s, thread=%s, session_key=%s)",
            cli_session_id, platform_name, home.chat_id, effective_thread_id,
            session_key,
        )

        # Dispatch through the runner directly. Going through
        # adapter.handle_message would spawn a background task and we'd
        # lose synchronous error visibility; calling _handle_message inline
        # keeps the success/failure path observable for the watcher.
        response_text = await self._handle_message(synthetic_event)
        if not response_text:
            # Streaming may have already delivered the response inline.
            # Either way, agent ran without raising — count as success.
            return

        # Send the agent's reply to the destination. Route to the new
        # thread if we created one; otherwise the configured home channel
        # (which may itself carry a thread_id).
        send_metadata: Dict[str, Any] = {}
        if effective_thread_id:
            send_metadata["thread_id"] = effective_thread_id
        try:
            result = await adapter.send(
                chat_id=str(home.chat_id),
                content=response_text,
                metadata=send_metadata or None,
            )
        except Exception as exc:
            raise RuntimeError(f"adapter.send failed: {exc}") from exc

        if not getattr(result, "success", True):
            err = getattr(result, "error", "send returned success=False")
            raise RuntimeError(f"adapter.send failed: {err}")

    async def _session_expiry_watcher(self, interval: int = 300):
        """Background task that finalizes expired sessions.

        Delegates to SessionExpiryWatcher in gateway.watchers.
        """
        watcher = watchers.SessionExpiryWatcher(self)
        await watcher.watch(interval=interval)

    def _active_profile_name(self) -> str:
        """Return the profile name this gateway represents."""
        return session_management.active_profile_name()

    async def stop(
        self,
        *,
        restart: bool = False,
        detached_restart: bool = False,
        service_restart: bool = False,
    ) -> None:
        return await lifecycle.stop_gateway_runner(
            runner=self,
            restart=restart,
            detached_restart=detached_restart,
            service_restart=service_restart,
        )

        """Stop the gateway and disconnect all adapters."""
        if restart:
            self._restart_requested = True
            self._restart_detached = detached_restart
            self._restart_via_service = service_restart
        if self._stop_task is not None:
            await self._stop_task
            return

        async def _stop_impl() -> None:
            def _kill_tool_subprocesses(phase: str) -> None:
                """Kill tool subprocesses + tear down terminal envs + browsers.

                Called twice in the shutdown path: once eagerly after a
                drain timeout forces agent interrupt (so we reclaim bash/
                sleep children before systemd TimeoutStopSec escalates to
                SIGKILL on the cgroup — #8202), and once as a final
                catch-all at the end of _stop_impl() for the graceful
                path or anything respawned mid-teardown.

                All steps are best-effort; exceptions are swallowed so
                one subsystem's failure doesn't block the rest.
                """
                try:
                    from tools.process_registry import process_registry
                    _killed = process_registry.kill_all()
                    if _killed:
                        logger.info(
                            "Shutdown (%s): killed %d tool subprocess(es)",
                            phase, _killed,
                        )
                except Exception as _e:
                    logger.debug("process_registry.kill_all (%s) error: %s", phase, _e)
                try:
                    from tools.terminal_tool import cleanup_all_environments
                    cleanup_all_environments()
                except Exception as _e:
                    logger.debug("cleanup_all_environments (%s) error: %s", phase, _e)
                try:
                    from tools.browser_tool import cleanup_all_browsers
                    cleanup_all_browsers()
                except Exception as _e:
                    logger.debug("cleanup_all_browsers (%s) error: %s", phase, _e)

            logger.info(
                "Stopping gateway%s...",
                " for restart" if self._restart_requested else "",
            )
            _stop_started_at = time.monotonic()

            _shutdown_budget = 70.0  # Must finish well before systemd TimeoutStopSec

            def _phase_elapsed() -> float:
                return time.monotonic() - _stop_started_at

            def _budget_exhausted(phase: str) -> bool:
                if _phase_elapsed() >= _shutdown_budget:
                    logger.warning(
                        "Shutdown budget exhausted at %s (+%.2fs); "
                        "skipping to final cleanup",
                        phase, _phase_elapsed(),
                    )
                    return True
                return False

            self._running = False
            self._draining = True

            # ── PHASE 0: Kill all tool subprocesses IMMEDIATELY ──────────
            # These are bash/python/sleep children that systemd won't signal
            # (KillMode=mixed only signals the main PID).  Killing them now
            # reclaims resources and prevents systemd SIGKILL escalation.
            _kill_tool_subprocesses("immediate")
            logger.info(
                "Shutdown phase: immediate tool kill done at +%.2fs",
                _phase_elapsed(),
            )

            # Kill untracked children (e.g. llama-server spawned by agent
            # tool calls) that process_registry doesn't know about.
            try:
                import psutil as _psutil
                _me = _psutil.Process(os.getpid())
                for _child in _me.children(recursive=False):
                    try:
                        _cmd = " ".join(_child.cmdline()).lower()
                    except (_psutil.NoSuchProcess, _psutil.AccessDenied):
                        continue
                    if "llama" in _cmd:
                        _child.terminate()
                        logger.info(
                            "Shutdown: terminated llama-server child "
                            "(pid=%d)", _child.pid,
                        )
            except ImportError:
                pass
            except Exception as _e:
                logger.debug("Untracked child cleanup: %s", _e)
            logger.info(
                "Shutdown phase: untracked child cleanup done at +%.2fs",
                _phase_elapsed(),
            )

            # ── PHASE 1: Notify active sessions ──────────────────────────
            # Adapters are still connected here, so messages can be sent.
            await self._notify_active_sessions_of_shutdown()
            logger.info(
                "Shutdown phase: notify_active_sessions done at +%.2fs",
                _phase_elapsed(),
            )

            # ── PHASE 2: Preemptive resume_pending ───────────────────────
            # Mark ALL active sessions as resume_pending BEFORE draining so
            # they can be auto-resumed even if the process gets SIGKILL'd
            # mid-drain.  Sessions that finish cleanly during the drain
            # window will have resume_pending cleared on their next turn.
            if not _budget_exhausted("preemptive_resume"):
                _pre_reason = (
                    "restart_pre" if self._restart_requested else "shutdown_pre"
                )
                for _sk, _agent in list(self._running_agents.items()):
                    if _agent is _AGENT_PENDING_SENTINEL:
                        continue
                    try:
                        self.session_store.mark_resume_pending(_sk, _pre_reason)
                    except Exception as _e:
                        logger.debug(
                            "preemptive mark_resume_pending for %s: %s",
                            _sk, _e,
                        )

            # ── PHASE 3: Drain agents (capped) ──────────────────────────
            # The configured restart_drain_timeout (180s) is only for
            # voluntary /restart.  For SIGTERM from systemd, cap to 30s.
            _drain_started_at = time.monotonic()
            _drain_timeout_used = 0.0
            if _budget_exhausted("drain"):
                timed_out = bool(self._running_agents)
                active_agents = {}
            else:
                _drain_cap = 30.0
                _drain_timeout_used = min(self._restart_drain_timeout, _drain_cap)
                active_agents, timed_out = await self._drain_active_agents(_drain_timeout_used)
            logger.info(
                "Shutdown phase: drain done at +%.2fs (drain took %.2fs, "
                "timed_out=%s, active_at_start=%d, active_now=%d)",
                _phase_elapsed(),
                time.monotonic() - _drain_started_at,
                timed_out,
                len(active_agents),
                self._running_agent_count(),
            )

            if not timed_out:
                # Drain completed gracefully — clear any pre-drain
                # resume_pending markers so finished sessions don't carry
                # a stale flag.
                for _sk in list(self._running_agents):
                    try:
                        self.session_store.clear_resume_pending(_sk)
                    except Exception as _e:
                        logger.debug(
                            "clear_resume_pending after drain for %s: %s",
                            _sk, _e,
                        )

            if timed_out:
                logger.warning(
                    "Gateway drain timed out after %.1fs with %d active agent(s); interrupting remaining work.",
                    _drain_timeout_used,
                    self._running_agent_count(),
                )
                # Mark forcibly-interrupted sessions as resume_pending BEFORE
                # interrupting the agents.  This preserves each session's
                # session_id + transcript so the next message on the same
                # session_key auto-resumes from the existing conversation
                # instead of getting routed through suspend_recently_active()
                # and converted into a fresh session.  Terminal escalation
                # for genuinely stuck sessions still flows through the
                # existing ``.restart_failure_counts`` stuck-loop counter
                # (incremented below, threshold 3), which sets
                # ``suspended=True`` and overrides resume_pending.
                #
                # Iterate self._running_agents (current) rather than the
                # drain-start ``active_agents`` snapshot — the snapshot
                # may include sessions that finished gracefully during
                # the drain window, and marking those falsely would give
                # them a stray restart-interruption system note on their
                # next turn even though their previous turn completed
                # cleanly.  Skip pending sentinels for the same reason
                # _interrupt_running_agents() does: their agent hasn't
                # started yet, there's nothing to interrupt, and the
                # session shouldn't carry a misleading resume flag.
                _resume_reason = (
                    "restart_timeout" if self._restart_requested else "shutdown_timeout"
                )
                for _sk, _agent in list(self._running_agents.items()):
                    if _agent is _AGENT_PENDING_SENTINEL:
                        continue
                    try:
                        self.session_store.mark_resume_pending(_sk, _resume_reason)
                    except Exception as _e:
                        logger.debug(
                            "mark_resume_pending failed for %s: %s",
                            _sk, _e,
                        )
                self._interrupt_running_agents(
                    _INTERRUPT_REASON_GATEWAY_RESTART if self._restart_requested else _INTERRUPT_REASON_GATEWAY_SHUTDOWN
                )
                interrupt_deadline = asyncio.get_running_loop().time() + 2.0
                while self._running_agents and asyncio.get_running_loop().time() < interrupt_deadline:
                    self._update_runtime_status("draining")
                    await asyncio.sleep(0.1)

                # Kill lingering tool subprocesses NOW, before we spend more
                # budget on adapter disconnect / session DB close.  Under
                # systemd (TimeoutStopSec bounded by drain_timeout+headroom),
                # deferring this to the end of stop() risks systemd escalating
                # to SIGKILL on the cgroup first — at which point bash/sleep
                # children left behind by an interrupted terminal tool get
                # killed by systemd instead of us (issue #8202).  The final
                # catch-all cleanup below still runs for the graceful path.
                _kill_tool_subprocesses("post-interrupt")
                logger.info(
                    "Shutdown phase: post-interrupt tool kill done at +%.2fs",
                    _phase_elapsed(),
                )

            if self._restart_requested and self._restart_detached:
                try:
                    await self._launch_detached_restart_command()
                except Exception as e:
                    logger.error("Failed to launch detached gateway restart: %s", e)

            self._finalize_shutdown_agents(active_agents)

            # Also shut down memory providers on idle cached agents.
            # _finalize_shutdown_agents only handles agents that were
            # mid-turn at drain time; the _agent_cache may still hold
            # idle agents whose MemoryProviders never received
            # on_session_end().
            _cache_lock = getattr(self, "_agent_cache_lock", None)
            _cache = getattr(self, "_agent_cache", None)
            if _cache_lock is not None and _cache is not None:
                with _cache_lock:
                    _idle_agents = list(_cache.values())
                    _cache.clear()
                for _entry in _idle_agents:
                    _agent = (
                        _entry[0] if isinstance(_entry, tuple) else _entry
                    )
                    self._cleanup_agent_resources(_agent)

            # ── PHASE 5: Disconnect adapters (with per-adapter timeout) ──
            if not _budget_exhausted("adapter_disconnect"):
                _adapter_timeout = self._adapter_disconnect_timeout_secs()
                for platform, adapter in list(self.adapters.items()):
                    _adapter_started_at = time.monotonic()
                    try:
                        await adapter.cancel_background_tasks()
                    except Exception as e:
                        logger.debug("✗ %s background-task cancel error: %s", platform.value, e)
                    try:
                        if _adapter_timeout > 0:
                            await asyncio.wait_for(adapter.disconnect(), timeout=_adapter_timeout)
                        else:
                            await adapter.disconnect()
                        logger.info(
                            "✓ %s disconnected (%.2fs)",
                            platform.value,
                            time.monotonic() - _adapter_started_at,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Timed out after %.1fs disconnecting %s; "
                            "continuing shutdown",
                            _adapter_timeout, platform.value,
                        )
                    except Exception as e:
                        logger.error(
                            "✗ %s disconnect error after %.2fs: %s",
                            platform.value,
                            time.monotonic() - _adapter_started_at,
                            e,
                        )
            logger.info(
                "Shutdown phase: all adapters disconnected at +%.2fs",
                _phase_elapsed(),
            )

            for _task in list(self._background_tasks):
                if _task is self._stop_task:
                    continue
                _task.cancel()
            self._background_tasks.clear()

            self.adapters.clear()
            self._running_agents.clear()
            self._running_agents_ts.clear()
            self._pending_messages.clear()
            self._pending_approvals.clear()
            if hasattr(self, '_busy_ack_ts'):
                self._busy_ack_ts.clear()
            self._shutdown_event.set()

            # Global cleanup: kill any remaining tool subprocesses not tied
            # to a specific agent (catch-all for zombie prevention). On the
            # drain-timeout path we already did this earlier after agent
            # interrupt — this second call catches (a) the graceful path
            # where drain succeeded without interrupt, and (b) anything
            # that got respawned between the earlier call and adapter
            # disconnect (defense in depth; safe to call repeatedly).
            _kill_tool_subprocesses("final-cleanup")
            logger.info(
                "Shutdown phase: final-cleanup tool kill done at +%.2fs",
                _phase_elapsed(),
            )

            # Reap the process-global auxiliary-client cache once at the very
            # end of teardown.  Per-turn cleanup runs in _cleanup_agent_resources
            # for each active agent, but clients bound to worker-thread loops
            # that died with their ThreadPoolExecutor (notably cron ticks) only
            # get swept here.  Without this, long-running gateways accumulate
            # async httpx transports until they hit EMFILE on macOS's default
            # RLIMIT_NOFILE=256.  See #14210.
            try:
                from agent.auxiliary_client import shutdown_cached_clients
                shutdown_cached_clients()
            except Exception as _e:
                logger.debug("shutdown_cached_clients error: %s", _e)

            # Close SQLite session DBs so the WAL write lock is released.
            # Without this, --replace and similar restart flows leave the
            # old gateway's connection holding the WAL lock until Python
            # actually exits — causing 'database is locked' errors when
            # the new gateway tries to open the same file.
            if not _budget_exhausted("sessiondb_close"):
                for _db_holder in (self, getattr(self, "session_store", None)):
                    _db = getattr(_db_holder, "_db", None) if _db_holder else None
                    if _db is None or not hasattr(_db, "close"):
                        continue
                    try:
                        await asyncio.wait_for(
                            asyncio.get_running_loop().run_in_executor(None, _db.close),
                            timeout=5.0,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "SessionDB close timed out after 5s; continuing",
                        )
                    except Exception as _e:
                        logger.debug("SessionDB close error: %s", _e)
            logger.info(
                "Shutdown phase: SessionDB close done at +%.2fs",
                _phase_elapsed(),
            )

            from gateway.status import remove_pid_file, release_gateway_runtime_lock
            remove_pid_file()
            release_gateway_runtime_lock()

            # Write a clean-shutdown marker so the next startup knows this
            # wasn't a crash.  suspend_recently_active() only needs to run
            # after unexpected exits.  However, if the drain timed out and
            # agents were force-interrupted, their sessions may be in an
            # incomplete state (trailing tool response, no final assistant
            # message).  Skip the marker in that case so the next startup
            # suspends those sessions — giving users a clean slate instead
            # of resuming a half-finished tool loop.
            if not timed_out:
                try:
                    (_hermes_home / ".clean_shutdown").touch()
                except Exception:
                    pass
            else:
                logger.info(
                    "Skipping .clean_shutdown marker — drain timed out with "
                    "interrupted agents; next startup will suspend recently "
                    "active sessions."
                )

            # Track sessions that were active at shutdown for stuck-loop
            # detection (#7536).  On each restart, the counter increments
            # for sessions that were running.  If a session hits the
            # threshold (3 consecutive restarts while active), the next
            # startup auto-suspends it — breaking the loop.
            if active_agents:
                self._increment_restart_failure_counts(set(active_agents.keys()))

            if self._restart_requested and self._restart_command_source is None:
                try:
                    atomic_json_write(
                        _planned_restart_notification_path(),
                        {
                            "requested_at": time.time(),
                            "via_service": bool(self._restart_via_service),
                            "detached": bool(self._restart_detached),
                        },
                        indent=None,
                    )
                except Exception as e:
                    logger.debug("Failed to write planned restart notification marker: %s", e)

            if self._restart_requested and self._restart_via_service:
                self._launch_systemd_restart_shortcut()
                # systemd units use Restart=always, so a planned restart should
                # exit cleanly and still be relaunched.  Using TEMPFAIL here
                # makes systemd treat the operator-requested restart as a
                # failure and can trip stepped restart backoff.  launchd's
                # KeepAlive.SuccessfulExit=false needs a non-zero exit to
                # relaunch, so keep the old code on macOS.
                self._exit_code = (
                    GATEWAY_SERVICE_RESTART_EXIT_CODE
                    if sys.platform == "darwin" or not os.environ.get("INVOCATION_ID")
                    else 0
                )
                self._exit_reason = self._exit_reason or "Gateway restart requested"

            self._draining = False
            self._update_runtime_status("stopped", self._exit_reason)
            logger.info("Gateway stopped (total teardown %.2fs)", _phase_elapsed())

        self._stop_task = asyncio.create_task(_stop_impl())
        await self._stop_task

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()

    def _create_adapter(
        self,
        platform: Platform,
        config: Any
    ) -> Optional[BasePlatformAdapter]:
        return adapter_factory.create_adapter(
            runner=self,
            platform=platform,
            config=config,
        )

        """Create the appropriate adapter for a platform.

        Checks the platform_registry first (plugin adapters), then falls
        through to the built-in if/elif chain for core platforms.
        """
        if hasattr(config, "extra") and isinstance(config.extra, dict):
            config.extra.setdefault(
                "group_sessions_per_user",
                self.config.group_sessions_per_user,
            )
            config.extra.setdefault(
                "thread_sessions_per_user",
                getattr(self.config, "thread_sessions_per_user", False),
            )

        # ── Plugin-registered platforms (checked first) ───────────────────
        try:
            from gateway.platform_registry import platform_registry
            if platform_registry.is_registered(platform.value):
                adapter = platform_registry.create_adapter(platform.value, config)
                if adapter is not None:
                    # Adapters that need a back-reference to the gateway runner
                    # (e.g. for cross-platform admin alerts) declare a
                    # ``gateway_runner`` attribute. Inject it after creation so
                    # plugin adapters don't need a custom factory signature.
                    if hasattr(adapter, "gateway_runner"):
                        adapter.gateway_runner = self
                    return adapter
                # Registered but failed to instantiate — don't silently fall
                # through to built-ins (there are none for plugin platforms).
                logger.error(
                    "Platform '%s' is registered but adapter creation failed "
                    "(check dependencies and config)",
                    platform.value,
                )
                return None
        except Exception as e:
            logger.debug("Platform registry lookup for '%s' failed: %s", platform.value, e)
        # Fall through to built-in adapters below

        if platform == Platform.TELEGRAM:
            from gateway.platforms.telegram import TelegramAdapter, check_telegram_requirements
            if not check_telegram_requirements():
                logger.warning("Telegram: python-telegram-bot not installed")
                return None
            adapter = TelegramAdapter(config)
            # Apply Telegram notification mode from config.  Controls whether
            # intermediate messages (tool progress, streaming, status) trigger
            # push notifications.  Supports ENV override for quick testing.
            _notify_mode = os.getenv("HERMES_TELEGRAM_NOTIFICATIONS", "")
            if not _notify_mode:
                try:
                    _gw_cfg = _load_gateway_config()
                    _raw = cfg_get(_gw_cfg, "display", "platforms", "telegram", "notifications")
                    if _raw not in {None, ""}:
                        _notify_mode = str(_raw).strip().lower()
                except Exception:
                    pass
            _notify_mode = _notify_mode or "important"
            if _notify_mode not in {"all", "important"}:
                logger.warning(
                    "Unknown telegram notifications mode '%s', "
                    "defaulting to 'important' (valid: all, important)",
                    _notify_mode,
                )
                _notify_mode = "important"
            adapter._notifications_mode = _notify_mode
            return adapter
        
        elif platform == Platform.WHATSAPP:
            from gateway.platforms.whatsapp import WhatsAppAdapter, check_whatsapp_requirements
            if not check_whatsapp_requirements():
                logger.warning("WhatsApp: Node.js not installed or bridge not configured")
                return None
            return WhatsAppAdapter(config)
        
        elif platform == Platform.SLACK:
            from gateway.platforms.slack import SlackAdapter, check_slack_requirements
            if not check_slack_requirements():
                logger.warning("Slack: slack-bolt not installed. Run: pip install 'hermes-agent[slack]'")
                return None
            return SlackAdapter(config)

        elif platform == Platform.SIGNAL:
            from gateway.platforms.signal import SignalAdapter, check_signal_requirements
            if not check_signal_requirements():
                logger.warning("Signal: SIGNAL_HTTP_URL or SIGNAL_ACCOUNT not configured")
                return None
            return SignalAdapter(config)

        elif platform == Platform.HOMEASSISTANT:
            from gateway.platforms.homeassistant import HomeAssistantAdapter, check_ha_requirements
            if not check_ha_requirements():
                logger.warning("HomeAssistant: aiohttp not installed or HASS_TOKEN not set")
                return None
            return HomeAssistantAdapter(config)

        elif platform == Platform.EMAIL:
            from gateway.platforms.email import EmailAdapter, check_email_requirements
            if not check_email_requirements():
                logger.warning("Email: EMAIL_ADDRESS, EMAIL_PASSWORD, EMAIL_IMAP_HOST, or EMAIL_SMTP_HOST not set")
                return None
            return EmailAdapter(config)

        elif platform == Platform.SMS:
            from gateway.platforms.sms import SmsAdapter, check_sms_requirements
            if not check_sms_requirements():
                logger.warning("SMS: aiohttp not installed or TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN not set")
                return None
            return SmsAdapter(config)

        elif platform == Platform.DINGTALK:
            from gateway.platforms.dingtalk import DingTalkAdapter, check_dingtalk_requirements
            if not check_dingtalk_requirements():
                logger.warning("DingTalk: dingtalk-stream not installed or DINGTALK_CLIENT_ID/SECRET not set")
                return None
            return DingTalkAdapter(config)

        elif platform == Platform.FEISHU:
            from gateway.platforms.feishu import FeishuAdapter, check_feishu_requirements
            if not check_feishu_requirements():
                logger.warning("Feishu: lark-oapi not installed or FEISHU_APP_ID/SECRET not set")
                return None
            return FeishuAdapter(config)

        elif platform == Platform.WECOM_CALLBACK:
            from gateway.platforms.wecom_callback import (
                WecomCallbackAdapter,
                check_wecom_callback_requirements,
            )
            if not check_wecom_callback_requirements():
                logger.warning("WeComCallback: aiohttp/httpx/defusedxml not installed")
                return None
            return WecomCallbackAdapter(config)

        elif platform == Platform.WECOM:
            from gateway.platforms.wecom import WeComAdapter, check_wecom_requirements
            if not check_wecom_requirements():
                logger.warning("WeCom: aiohttp not installed or WECOM_BOT_ID/SECRET not set")
                return None
            return WeComAdapter(config)

        elif platform == Platform.WEIXIN:
            from gateway.platforms.weixin import WeixinAdapter, check_weixin_requirements
            if not check_weixin_requirements():
                logger.warning("Weixin: aiohttp/cryptography not installed")
                return None
            return WeixinAdapter(config)

        elif platform == Platform.MATRIX:
            from gateway.platforms.matrix import MatrixAdapter, check_matrix_requirements
            if not check_matrix_requirements():
                logger.warning("Matrix: mautrix not installed or credentials not set. Run: pip install 'mautrix[encryption]'")
                return None
            return MatrixAdapter(config)

        elif platform == Platform.API_SERVER:
            from gateway.platforms.api_server import APIServerAdapter, check_api_server_requirements
            if not check_api_server_requirements():
                logger.warning("API Server: aiohttp not installed")
                return None
            return APIServerAdapter(config)

        elif platform == Platform.WEBHOOK:
            from gateway.platforms.webhook import WebhookAdapter, check_webhook_requirements
            if not check_webhook_requirements():
                logger.warning("Webhook: aiohttp not installed")
                return None
            adapter = WebhookAdapter(config)
            adapter.gateway_runner = self  # For cross-platform delivery
            return adapter

        elif platform == Platform.MSGRAPH_WEBHOOK:
            from gateway.platforms.msgraph_webhook import (
                MSGraphWebhookAdapter,
                check_msgraph_webhook_requirements,
            )
            if not check_msgraph_webhook_requirements():
                logger.warning("MSGraph webhook: aiohttp not installed")
                return None
            return MSGraphWebhookAdapter(config)

        elif platform == Platform.BLUEBUBBLES:
            from gateway.platforms.bluebubbles import BlueBubblesAdapter, check_bluebubbles_requirements
            if not check_bluebubbles_requirements():
                logger.warning("BlueBubbles: aiohttp/httpx missing or BLUEBUBBLES_SERVER_URL/BLUEBUBBLES_PASSWORD not configured")
                return None
            return BlueBubblesAdapter(config)

        elif platform == Platform.QQBOT:
            from gateway.platforms.qqbot import QQAdapter, check_qq_requirements
            if not check_qq_requirements():
                logger.warning("QQBot: aiohttp/httpx missing or QQ_APP_ID/QQ_CLIENT_SECRET not configured")
                return None
            return QQAdapter(config)

        elif platform == Platform.YUANBAO:
            from gateway.platforms.yuanbao import YuanbaoAdapter, WEBSOCKETS_AVAILABLE
            if not WEBSOCKETS_AVAILABLE:
                logger.warning("Yuanbao: websockets not installed. Run: pip install websockets")
                return None
            return YuanbaoAdapter(config)

        return None

    def _adapter_enforces_own_access_policy(self, platform: Optional[Platform]) -> bool:
        """Whether the adapter for *platform* gates access at intake itself.

        Mirrors ``BasePlatformAdapter.enforces_own_access_policy``. Adapters
        such as WeCom, Weixin, Yuanbao, QQBot, and WhatsApp evaluate their
        documented ``dm_policy`` / ``group_policy`` / ``allow_from`` config before a
        message is dispatched to the gateway, so a message that reaches
        ``_is_user_authorized`` has already been authorized by the adapter.
        Defaults to ``False`` when the adapter is unknown or doesn't expose
        the flag.
        """
        if not platform:
            return False
        # Some test helpers build a bare GatewayRunner via object.__new__ and
        # never set ``adapters``; treat a missing/empty map as "no adapter"
        # rather than raising (see pitfalls.md #17).
        adapters = getattr(self, "adapters", None)
        if not adapters:
            return False
        adapter = adapters.get(platform)
        if adapter is None:
            return False
        return bool(getattr(adapter, "enforces_own_access_policy", False))

    def _is_user_authorized(self, source: SessionSource) -> bool:
        return authorization.is_user_authorization(
            runner=self,
            source=source,
        )

        """Return how unauthorized DMs should be handled for a platform.

        Resolution order:
        1. Explicit per-platform ``unauthorized_dm_behavior`` in config — always wins.
        2. Explicit global ``unauthorized_dm_behavior`` in config — wins when no per-platform.
        3. When an allowlist (``PLATFORM_ALLOWED_USERS``,
           ``PLATFORM_GROUP_ALLOWED_USERS`` / ``PLATFORM_GROUP_ALLOWED_CHATS``,
           or ``GATEWAY_ALLOWED_USERS``) is configured, default to ``"ignore"`` —
           the allowlist signals that the owner has deliberately restricted
           access; spamming unknown contacts with pairing codes is both noisy
           and a potential info-leak. (#9337)
        4. No allowlist and no explicit config → ``"pair"`` (open-gateway default).
        """
        config = getattr(self, "config", None)

        # Check for an explicit per-platform override first.
        if config and hasattr(config, "get_unauthorized_dm_behavior") and platform:
            platform_cfg = config.platforms.get(platform) if hasattr(config, "platforms") else None
            if platform_cfg and "unauthorized_dm_behavior" in getattr(platform_cfg, "extra", {}):
                # Operator explicitly configured behavior for this platform — respect it.
                return config.get_unauthorized_dm_behavior(platform)

        # Check for an explicit global config override.
        if config and hasattr(config, "unauthorized_dm_behavior"):
            if config.unauthorized_dm_behavior != "pair":  # non-default → explicit override
                return config.unauthorized_dm_behavior

        # Config-driven dm_policy (WeCom / Weixin / Yuanbao / QQBot). An
        # allowlist or disabled DM policy means the operator restricted access,
        # so unauthorized DMs should be dropped silently rather than answered
        # with a pairing code. An explicit pairing policy opts back into codes.
        if platform and config and hasattr(config, "platforms"):
            platform_cfg = config.platforms.get(platform)
            extra = getattr(platform_cfg, "extra", None) if platform_cfg else None
            if isinstance(extra, dict):
                dm_policy = str(extra.get("dm_policy") or "").strip().lower()
                if dm_policy == "pairing":
                    return "pair"
                if dm_policy in {"allowlist", "disabled"}:
                    return "ignore"

        # No explicit override.  Fall back to allowlist-aware default:
        # if any allowlist is configured for this platform, silently drop
        # unauthorized messages instead of sending pairing codes.
        if platform:
            platform_env_map = {
                Platform.TELEGRAM: "TELEGRAM_ALLOWED_USERS",
                Platform.DISCORD:  "DISCORD_ALLOWED_USERS",
                Platform.WHATSAPP: "WHATSAPP_ALLOWED_USERS",
                Platform.SLACK:    "SLACK_ALLOWED_USERS",
                Platform.SIGNAL:   "SIGNAL_ALLOWED_USERS",
                Platform.EMAIL:    "EMAIL_ALLOWED_USERS",
                Platform.SMS:      "SMS_ALLOWED_USERS",
                Platform.MATTERMOST: "MATTERMOST_ALLOWED_USERS",
                Platform.MATRIX:   "MATRIX_ALLOWED_USERS",
                Platform.DINGTALK: "DINGTALK_ALLOWED_USERS",
                Platform.FEISHU:   "FEISHU_ALLOWED_USERS",
                Platform.WECOM:    "WECOM_ALLOWED_USERS",
                Platform.WECOM_CALLBACK: "WECOM_CALLBACK_ALLOWED_USERS",
                Platform.WEIXIN:   "WEIXIN_ALLOWED_USERS",
                Platform.BLUEBUBBLES: "BLUEBUBBLES_ALLOWED_USERS",
                Platform.QQBOT:    "QQ_ALLOWED_USERS",
            }
            platform_group_env_map = {
                Platform.TELEGRAM: (
                    "TELEGRAM_GROUP_ALLOWED_USERS",
                    "TELEGRAM_GROUP_ALLOWED_CHATS",
                ),
                Platform.QQBOT: ("QQ_GROUP_ALLOWED_USERS",),
            }
            if os.getenv(platform_env_map.get(platform, ""), "").strip():
                return "ignore"
            for env_key in platform_group_env_map.get(platform, ()):
                if os.getenv(env_key, "").strip():
                    return "ignore"

        if os.getenv("GATEWAY_ALLOWED_USERS", "").strip():
            return "ignore"

        return "pair"

    async def _deliver_platform_notice(self, source, content: str) -> None:
        """Deliver a setup/operational notice using platform-specific privacy rules."""
        adapter = self.adapters.get(source.platform)
        if not adapter:
            return

        config = getattr(self, "config", None)
        notice_delivery = "public"
        if config and hasattr(config, "get_notice_delivery"):
            notice_delivery = config.get_notice_delivery(source.platform)

        metadata = self._thread_metadata_for_source(source)
        if notice_delivery == "private" and getattr(source, "user_id", None):
            try:
                result = await adapter.send_private_notice(
                    source.chat_id,
                    source.user_id,
                    content,
                    metadata=metadata,
                )
                if getattr(result, "success", False):
                    return
            except Exception:
                logger.debug(
                    "[%s] send_private_notice failed, falling back to public",
                    getattr(source, "platform", "?"),
                    exc_info=True,
                )

        await adapter.send(source.chat_id, content, metadata=metadata)

    async def _handle_message(self, event: MessageEvent) -> Optional[str]:
        return await message_processing.handle_message(
            runner=self,
            event=event,
        )

    async def _prepare_inbound_message_text(
        self,
        *,
        event: MessageEvent,
        source: SessionSource,
        history: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Prepare inbound event text for the agent.

        Delegates to message_pipeline.prepare_inbound_message_text.
        """
        return await message_pipeline.prepare_inbound_message_text(
            event=event,
            source=source,
            history=history,
            config=self.config,
            adapters=self.adapters,
            model=self._model,
            base_url=self._base_url,
            session_key_for_source=self._session_key_for_source,
            consume_pending_native_image_paths=self._consume_pending_native_image_paths,
            decide_image_input_mode=self._decide_image_input_mode,
            enrich_message_with_vision=self._enrich_message_with_vision,
            enrich_message_with_transcription=self._enrich_message_with_transcription,
            reply_anchor_for_event=self._reply_anchor_for_event,
            thread_metadata_for_source=self._thread_metadata_for_source,
            has_setup_skill=self._has_setup_skill(),
        )

    def _consume_pending_native_image_paths(self, session_key: str) -> List[str]:
        """Consume pending native image paths for a session."""
        return message_pipeline.consume_pending_native_image_paths_impl(session_key)

    def _cache_session_source(self, session_key: str, source) -> None:
        if not session_key or source is None:
            return
        cached_sources = getattr(self, "_session_sources", None)
        if cached_sources is None:
            cached_sources = OrderedDict()
            self._session_sources = cached_sources
        try:
            cached_sources[session_key] = dataclasses.replace(source)
        except Exception:
            logger.debug("Failed to cache live session source for %s", session_key, exc_info=True)
            return
        # LRU: mark as most-recently-used and trim to max size.
        try:
            cached_sources.move_to_end(session_key)
            max_size = getattr(self, "_session_sources_max", 512)
            while len(cached_sources) > max_size:
                cached_sources.popitem(last=False)
        except Exception:
            pass

    def _get_cached_session_source(self, session_key: str):
        if not session_key:
            return None
        cached_sources = getattr(self, "_session_sources", None)
        if not cached_sources:
            return None
        source = cached_sources.get(session_key)
        if source is not None:
            try:
                cached_sources.move_to_end(session_key)
            except Exception:
                pass
        return source

    async def _handle_message_with_agent(self, event, source, _quick_key: str, run_generation: int):
        return await message_processing.handle_message_with_agent(
            runner=self,
            event=event,
            source=source,
            _quick_key=_quick_key,
            run_generation=run_generation,
        )

        """Resolve current model config and return a formatted info block.

        Surfaces model, provider, context length, and endpoint so gateway
        users can immediately see if context detection went wrong (e.g.
        local models falling to the 128K default).
        """
        return session_management.format_session_info(
            resolve_gateway_model=_resolve_gateway_model,
            load_gateway_config=_load_gateway_config,
            resolve_runtime_agent_kwargs=_resolve_runtime_agent_kwargs,
        )
    async def _handle_reset_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        """Handle /new or /reset command."""
        return await command_handlers.handle_reset_command(self, event)

    async def _handle_profile_command(self, event: MessageEvent) -> str:
        return await command_handlers.handle_profile_command(
            runner=self,
            event=event,
        )


    def _check_slash_access(
        self, source: SessionSource, canonical_cmd: str
    ) -> Optional[str]:
        """Return a denial message if ``source`` cannot run ``canonical_cmd``,
        else None. Used by both the cold and running-agent dispatch paths
        in ``_handle_message`` so admin/user gating can't be bypassed by
        an in-flight agent.

        Backward-compat semantics live in
        :func:`gateway.slash_access.policy_for_source` — when the operator
        hasn't set ``allow_admin_from`` for the scope, the policy returns
        ``enabled=False`` and this method always returns None.
        """
        from gateway.slash_access import policy_for_source as _policy_for_source

        if not canonical_cmd:
            return None
        policy = _policy_for_source(self.config, source)
        if not policy.enabled or policy.can_run(source.user_id, canonical_cmd):
            return None
        logger.info(
            "Slash command /%s denied for %s:%s (not admin, not in user_allowed_commands)",
            canonical_cmd,
            source.platform.value if source.platform else "?",
            source.user_id,
        )
        allowed_preview = sorted(policy.user_allowed_commands)
        if allowed_preview:
            suffix = (
                "You can run: "
                + ", ".join(f"/{c}" for c in allowed_preview[:12])
                + ("…" if len(allowed_preview) > 12 else "")
                + ". Use /whoami for the full list."
            )
        else:
            suffix = (
                "No slash commands are enabled for non-admins on this "
                "platform. Ask an admin to add you to allow_admin_from "
                "or to set user_allowed_commands."
            )
        return f"⛔ /{canonical_cmd} is admin-only here. {suffix}"


    async def _handle_whoami_command(self, event: MessageEvent) -> str:
        return await command_handlers.handle_whoami_command(
            runner=self,
            event=event,
        )


    async def _handle_kanban_command(self, event: MessageEvent) -> str:
        return await command_handlers.handle_kanban_command(
            runner=self,
            event=event,
        )

    async def _handle_status_command(self, event: MessageEvent) -> str:
        """Handle /status command."""
        source = event.source
        session_entry = self.session_store.get_or_create_session(source)

        connected_platforms = [p.value for p in self.adapters.keys()]

        # Check if there's an active agent
        session_key = session_entry.session_key
        is_running = session_key in self._running_agents

        # Count pending /queue follow-ups (slot + overflow).
        adapter = self.adapters.get(source.platform) if source else None
        queue_depth = self._queue_depth(session_key, adapter=adapter)

        title = None
        # Pull token totals from the SQLite session DB rather than the
        # in-memory SessionStore.  The agent's per-turn token deltas are
        # persisted into sessions_db (run_agent.py), not into SessionEntry,
        # so session_entry.total_tokens is always 0.  SessionDB is the
        # single source of truth; reading it here keeps /status accurate
        # without duplicating token writes into two stores.
        db_total_tokens = 0
        if self._session_db:
            try:
                title = self._session_db.get_session_title(session_entry.session_id)
            except Exception:
                title = None
            try:
                row = self._session_db.get_session(session_entry.session_id)
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

    async def _handle_agents_command(self, event: MessageEvent) -> str:
        """Handle /agents command - list active agents and running tasks."""
        from tools.process_registry import format_uptime_short, process_registry

        now = time.time()
        current_session_key = self._session_key_for_source(event.source)

        running_agents: dict = getattr(self, "_running_agents", {}) or {}
        running_started: dict = getattr(self, "_running_agents_ts", {}) or {}

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
            t for t in (getattr(self, "_background_tasks", set()) or set())
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

    def _sibling_thread_run_keys(self, source: SessionSource, own_key: str) -> list:
        """Find running-agent keys for OTHER participants in the same thread.

        Only applies when the message originates in a thread.  In per-user
        thread mode (``thread_sessions_per_user=True``) each participant gets
        an isolated session key of the form
        ``agent:main:{platform}:{chat_type}:{chat_id}:{thread_id}:{user_id}``,
        so a run started by another user is invisible to the caller's own
        ``/stop``.  This returns the keys of any *actually running* agents
        (not the pending sentinel, not the caller's own key) whose key shares
        the caller's ``{chat_id}:{thread_id}`` prefix.

        Returns an empty list when the source is not in a thread, or when no
        sibling runs exist — callers must still gate on authorization.
        """
        thread_id = getattr(source, "thread_id", None)
        chat_id = getattr(source, "chat_id", None)
        if not thread_id or not chat_id:
            return []
        platform = source.platform.value
        chat_type = getattr(source, "chat_type", None) or ""
        # Prefix that every per-user key in this thread shares, up to and
        # including the thread_id segment.  Matching either the exact
        # shared-thread key or any key with a further (user_id) segment
        # (prefix + ":") avoids cross-matching an unrelated thread whose id
        # merely starts with this one.
        prefix = ":".join(
            ["agent:main", platform, chat_type, str(chat_id), str(thread_id)]
        )
        matches = []
        for key, agent in list(self._running_agents.items()):
            if key == own_key:
                continue
            if agent is _AGENT_PENDING_SENTINEL or not agent:
                continue
            if key == prefix or key.startswith(prefix + ":"):
                matches.append(key)
        return matches

    async def _handle_stop_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        """Handle /stop command - interrupt a running agent.

        When an agent is truly hung (blocked thread that never checks
        _interrupt_requested), the early intercept in _handle_message()
        handles /stop before this method is reached.  This handler fires
        only through normal command dispatch (no running agent) or as a
        fallback.  Force-clean the session lock in all cases for safety.

        The session is preserved so the user can continue the conversation.
        """
        source = event.source
        session_entry = self.session_store.get_or_create_session(source)
        session_key = session_entry.session_key

        agent = self._running_agents.get(session_key)
        if agent is _AGENT_PENDING_SENTINEL:
            # Force-clean the sentinel so the session is unlocked.
            await self._interrupt_and_clear_session(
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
            await self._interrupt_and_clear_session(
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
        sibling_keys = self._sibling_thread_run_keys(source, session_key)
        if sibling_keys and self._is_user_authorized(source):
            for sibling_key in sibling_keys:
                await self._interrupt_and_clear_session(
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

    async def _handle_platform_command(self, event: MessageEvent) -> str:
        """Handle ``/platform list|pause|resume [name]`` — surface and
        manually control failed/paused gateway adapters.

        Examples:
            ``/platform list``           — show connected + failed/paused platforms
            ``/platform pause whatsapp`` — stop the reconnect watcher hammering whatsapp
            ``/platform resume whatsapp`` — re-queue a paused platform for retry
        """
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
            connected = sorted(p.value for p in self.adapters.keys())
            if connected:
                lines.append("Connected: " + ", ".join(connected))
            else:
                lines.append("Connected: (none)")
            failed = getattr(self, "_failed_platforms", {}) or {}
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
            failed = getattr(self, "_failed_platforms", {}) or {}
            if action == "pause":
                if platform not in failed:
                    return (
                        f"{platform.value} is not in the retry queue "
                        f"(it's either connected or not enabled)."
                    )
                if failed[platform].get("paused"):
                    return f"{platform.value} is already paused."
                self._pause_failed_platform(platform, reason="paused via /platform pause")
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
            self._resume_paused_platform(platform)
            return f"✓ {platform.value} resumed — retrying on next watcher tick."

        return (
            "Usage: /platform <list|pause|resume> [name]\n"
            "  /platform list — show platform status\n"
            "  /platform pause <name> — stop retrying a failing platform\n"
            "  /platform resume <name> — re-queue a paused platform"
        )

    async def _handle_restart_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        """Handle /restart command - drain active work, then restart the gateway."""
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
        if self._is_stale_restart_redelivery(event):
            logger.info(
                "Ignoring redelivered /restart (platform=%s, update_id=%s) — "
                "already processed by a previous gateway instance.",
                event.source.platform.value if event.source and event.source.platform else "?",
                event.platform_update_id,
            )
            return ""

        if self._restart_requested or self._draining:
            count = self._running_agent_count()
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
                    self._restart_command_source = dataclasses.replace(
                        event.source,
                        message_id=str(event.message_id)
                        if event.message_id is not None
                        else event.source.message_id,
                    )
                except Exception:
                    self._restart_command_source = event.source
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

        active_agents = self._running_agent_count()
        # When running under a service manager (systemd/launchd) or inside a
        # Docker/Podman container, use the service restart path: exit with
        # code 75 so the service manager / container restart policy restarts
        # us.  The detached subprocess approach (setsid + bash) doesn't work
        # under systemd (KillMode=mixed kills the cgroup) or Docker (tini
        # exits when the gateway dies, taking the detached helper with it).
        _under_service = bool(os.environ.get("INVOCATION_ID"))  # systemd sets this
        _in_container = os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")
        if _under_service or _in_container:
            self.request_restart(detached=False, via_service=True)
        else:
            self.request_restart(detached=True, via_service=False)
        if active_agents:
            return t("gateway.draining", count=active_agents)
        return EphemeralReply(t("gateway.restart.restarting"))

    def _is_stale_restart_redelivery(self, event: MessageEvent) -> bool:
        """Return True if this /restart is a Telegram re-delivery we already handled.

        The previous gateway wrote ``.restart_last_processed.json`` with the
        triggering platform + update_id when it processed the /restart.  If
        we now see a /restart on the same platform with an update_id <= that
        recorded value AND the marker is recent (< 5 minutes), it's a
        redelivery and should be ignored.

        Only applies to Telegram today (the only platform that exposes a
        numeric cross-session update ordering); other platforms return False.
        """
        if event is None or event.source is None:
            return False
        if event.platform_update_id is None:
            return False
        if event.source.platform is None:
            return False
        # Only Telegram populates platform_update_id currently; be explicit
        # so future platforms aren't accidentally gated by this check.
        try:
            platform_value = event.source.platform.value
        except Exception:
            return False
        if platform_value != "telegram":
            return False

        try:
            marker_path = _hermes_home / ".restart_last_processed.json"
            if not marker_path.exists():
                return False
            data = json.loads(marker_path.read_text())
        except Exception:
            return False

        if data.get("platform") != platform_value:
            return False
        recorded_uid = data.get("update_id")
        if not isinstance(recorded_uid, int):
            return False
        # Staleness guard: ignore markers older than 5 minutes.  A legitimately
        # old marker (e.g. crash recovery where notify never fired) should not
        # swallow a fresh /restart from the user.
        requested_at = data.get("requested_at")
        if isinstance(requested_at, (int, float)):
            if time.time() - requested_at > 300:
                return False
        return event.platform_update_id <= recorded_uid


    async def _handle_help_command(self, event: MessageEvent) -> str:
        return await command_handlers.handle_help_command(
            runner=self,
            event=event,
        )

    async def _handle_commands_command(self, event: MessageEvent) -> str:
        return await command_handlers.handle_commands_command(
            runner=self,
            event=event,
        )

    async def _handle_codex_runtime_command(self, event: MessageEvent) -> str:
        """Handle /codex-runtime command in the gateway.

        Same surface as the CLI handler in cli.py:
            /codex-runtime                  — show current state
            /codex-runtime auto             — Hermes default runtime
            /codex-runtime codex_app_server — codex subprocess runtime
            /codex-runtime on / off         — synonyms

        On change, the cached agent for this session is evicted so the next
        message creates a fresh AIAgent with the new api_mode wired in
        (avoids prompt-cache invalidation mid-session)."""
        return await command_handlers.handle_codex_runtime_command(
            runner=self,
            event=event,
        )

    async def _handle_personality_command(self, event: MessageEvent) -> str:
        return await command_handlers.handle_personality_command(
            runner=self,
            event=event,
        )

    async def _handle_retry_command(self, event: MessageEvent) -> str:
        """Handle /retry command - re-send the last user message."""
        return await command_handlers.handle_retry_command(
            runner=self,
            event=event,
        )

    # ────────────────────────────────────────────────────────────────
    # /goal — persistent cross-turn goals (Ralph-style loop)
    # ────────────────────────────────────────────────────────────────
    def _goal_max_turns_from_config(self) -> int:
        """Resolve the configured /goal turn budget for gateway sessions.

        GatewayRunner.config is a GatewayConfig dataclass, not the full
        user config mapping. Top-level config blocks such as ``goals`` are
        therefore only available through hermes_cli.config.load_config().
        """
        try:
            goals_cfg = (
                (self.config or {}).get("goals", {})
                if isinstance(self.config, dict)
                else getattr(self.config, "goals", {}) or {}
            )
            if not goals_cfg:
                from hermes_cli.config import load_config

                goals_cfg = (load_config() or {}).get("goals") or {}
            return int(goals_cfg.get("max_turns", 20) or 20)
        except Exception:
            return 20

    def _get_goal_manager_for_event(self, event: "MessageEvent"):
        """Return a GoalManager bound to the session for this gateway event.

        Returns ``(manager, session_entry)`` or ``(None, None)`` if the
        goals module can't be loaded.
        """
        try:
            from hermes_cli.goals import GoalManager
        except Exception as exc:
            logger.debug("goal manager unavailable: %s", exc)
            return None, None
        try:
            session_entry = self.session_store.get_or_create_session(event.source)
        except Exception as exc:
            logger.debug("goal manager: session lookup failed: %s", exc)
            return None, None
        sid = getattr(session_entry, "session_id", None) or ""
        if not sid:
            return None, None
        max_turns = self._goal_max_turns_from_config()
        return GoalManager(session_id=sid, default_max_turns=max_turns), session_entry

    async def _handle_goal_command(self, event: "MessageEvent") -> str:
        return await command_handlers.handle_goal_command(
            runner=self,
            event=event,
        )

    async def _handle_subgoal_command(self, event: "MessageEvent") -> str:
        return await command_handlers.handle_subgoal_command(
            runner=self,
            event=event,
        )

    async def _send_goal_status_notice(self, source: Any, message: str) -> None:
        """Send a /goal judge status line back to the originating chat/thread."""
        adapter = self.adapters.get(source.platform)
        if not adapter:
            logger.debug("goal continuation: no adapter for %s", getattr(source, "platform", None))
            return

        try:
            metadata = self._thread_metadata_for_source(source)
        except Exception:
            metadata = None

        result = await adapter.send(source.chat_id, message, metadata=metadata)
        if result is not None and not getattr(result, "success", True):
            logger.warning(
                "goal continuation: status send failed: %s",
                getattr(result, "error", "unknown error"),
            )

    async def _defer_goal_status_notice_after_delivery(self, source: Any, message: str) -> None:
        """Send a /goal status line after the main response is delivered.

        The gateway message handler returns the agent response to the platform
        adapter, which sends it after this method's caller has returned.  For a
        natural Discord/Telegram reading order, goal status belongs after that
        send.  Platform adapters provide a one-shot post-delivery callback for
        exactly this boundary; when unavailable, fall back to direct awaited
        delivery rather than silently dropping the notice.
        """
        adapter = self.adapters.get(source.platform)
        if not adapter:
            logger.debug("goal continuation: no adapter for %s", getattr(source, "platform", None))
            return

        async def _deliver() -> None:
            try:
                await self._send_goal_status_notice(source, message)
            except Exception as exc:
                logger.warning("goal continuation: status send failed: %s", exc, exc_info=True)

        try:
            session_key = self._session_key_for_source(source)
        except Exception:
            session_key = None

        if session_key and hasattr(adapter, "register_post_delivery_callback"):
            try:
                generation = None
                active = getattr(adapter, "_active_sessions", {}).get(session_key)
                if active is not None:
                    generation = getattr(active, "_hermes_run_generation", None)
                adapter.register_post_delivery_callback(
                    session_key,
                    _deliver,
                    generation=generation,
                )
                return
            except Exception as exc:
                logger.debug("goal continuation: post-delivery callback registration failed: %s", exc)

        await _deliver()

    async def _post_turn_goal_continuation(
        self,
        *,
        session_entry: Any,
        source: Any,
        final_response: str,
    ) -> None:
        """Run the goal judge after a gateway turn and, if still active,
        enqueue a continuation prompt for the same session.

        Called from ``_handle_message_with_agent`` at turn boundary, AFTER
        the response has been delivered. Safe when no goal is set.

        We use the adapter's pending-message / FIFO machinery so any real
        user message that arrives simultaneously is handled by the same
        queue and takes priority naturally.
        """
        try:
            from hermes_cli.goals import GoalManager
        except Exception as exc:
            logger.debug("goal continuation: goals module unavailable: %s", exc)
            return

        sid = getattr(session_entry, "session_id", None) or ""
        if not sid:
            return

        max_turns = self._goal_max_turns_from_config()

        mgr = GoalManager(session_id=sid, default_max_turns=max_turns)
        if not mgr.is_active():
            return

        decision = mgr.evaluate_after_turn(final_response or "", user_initiated=True)
        msg = decision.get("message") or ""

        # Defer the status line until after the adapter has delivered the
        # agent's visible final response. The judge runs after the response is
        # produced but before BasePlatformAdapter sends it, so sending here
        # would show "✓ Goal achieved" before the answer itself. Registering
        # an awaited post-delivery callback preserves delivery reliability
        # without reversing the user-visible ordering.
        if msg and source is not None:
            await self._defer_goal_status_notice_after_delivery(source, msg)

        if not decision.get("should_continue"):
            return

        prompt = decision.get("continuation_prompt") or ""
        if not prompt or source is None:
            return

        # Enqueue via the adapter's FIFO so a user message already in
        # flight preempts the continuation naturally.
        try:
            adapter = self.adapters.get(source.platform)
            _quick_key = self._session_key_for_source(source)
            if adapter and _quick_key:
                cont_event = MessageEvent(
                    text=prompt,
                    message_type=MessageType.TEXT,
                    source=source,
                    message_id=None,
                    channel_prompt=None,
                )
                self._enqueue_fifo(_quick_key, cont_event, adapter)
        except Exception as exc:
            logger.debug("goal continuation: enqueue failed: %s", exc)

    async def _handle_undo_command(self, event: MessageEvent) -> str:
        return await command_handlers.handle_undo_command(
            runner=self,
            event=event,
        )

    async def _handle_set_home_command(self, event: MessageEvent) -> str:
        return await command_handlers.handle_set_home_command(
            runner=self,
            event=event,
        )

    @staticmethod
    def _get_guild_id(event: MessageEvent) -> Optional[int]:
        """Extract Discord guild_id from the raw message object."""
        raw = getattr(event, "raw_message", None)
        if raw is None:
            return None
        # Slash command interaction
        if hasattr(raw, "guild_id") and raw.guild_id:
            return int(raw.guild_id)
        # Regular message
        if hasattr(raw, "guild") and raw.guild:
            return raw.guild.id
        return None

    async def _handle_voice_command(self, event: MessageEvent) -> str:
        return await command_handlers.handle_voice_command(
            runner=self,
            event=event,
        )

    async def _handle_voice_channel_join(self, event: MessageEvent) -> str:
        """Join the user's current Discord voice channel."""
        adapter = self.adapters.get(event.source.platform)
        if not hasattr(adapter, "join_voice_channel"):
            return "Voice channels are not supported on this platform."

        guild_id = self._get_guild_id(event)
        if not guild_id:
            return "This command only works in a Discord server."

        voice_channel = await adapter.get_user_voice_channel(
            guild_id, event.source.user_id
        )
        if not voice_channel:
            return "You need to be in a voice channel first."

        # Wire callbacks BEFORE join so voice input arriving immediately
        # after connection is not lost.
        if hasattr(adapter, "_voice_input_callback"):
            adapter._voice_input_callback = self._handle_voice_channel_input
        if hasattr(adapter, "_on_voice_disconnect"):
            adapter._on_voice_disconnect = self._handle_voice_timeout_cleanup

        try:
            success = await adapter.join_voice_channel(voice_channel)
        except Exception as e:
            logger.warning("Failed to join voice channel: %s", e)
            adapter._voice_input_callback = None
            err_lower = str(e).lower()
            if "pynacl" in err_lower or "nacl" in err_lower or "davey" in err_lower:
                return (
                    "Voice dependencies are missing (PyNaCl / davey). "
                    f"Install with: `{sys.executable} -m pip install PyNaCl`"
                )
            return f"Failed to join voice channel: {e}"

        if success:
            adapter._voice_text_channels[guild_id] = int(event.source.chat_id)
            if hasattr(adapter, "_voice_sources"):
                adapter._voice_sources[guild_id] = event.source.to_dict()
            self._voice_mode[self._voice_key(event.source.platform, event.source.chat_id)] = "all"
            self._save_voice_modes()
            self._set_adapter_auto_tts_enabled(adapter, event.source.chat_id, enabled=True)
            return (
                f"Joined voice channel **{voice_channel.name}**.\n"
                f"I'll speak my replies and listen to you. Use /voice leave to disconnect."
            )
        # Join failed — clear callback
        adapter._voice_input_callback = None
        return "Failed to join voice channel. Check bot permissions (Connect + Speak)."

    async def _handle_voice_channel_leave(self, event: MessageEvent) -> str:
        """Leave the Discord voice channel."""
        adapter = self.adapters.get(event.source.platform)
        guild_id = self._get_guild_id(event)

        if not guild_id or not hasattr(adapter, "leave_voice_channel"):
            return "Not in a voice channel."

        if not hasattr(adapter, "is_in_voice_channel") or not adapter.is_in_voice_channel(guild_id):
            return "Not in a voice channel."

        try:
            await adapter.leave_voice_channel(guild_id)
        except Exception as e:
            logger.warning("Error leaving voice channel: %s", e)
        # Always clean up state even if leave raised an exception
        self._voice_mode[self._voice_key(event.source.platform, event.source.chat_id)] = "off"
        self._save_voice_modes()
        self._set_adapter_auto_tts_disabled(adapter, event.source.chat_id, disabled=True)
        if hasattr(adapter, "_voice_input_callback"):
            adapter._voice_input_callback = None
        return "Left voice channel."

    def _handle_voice_timeout_cleanup(self, chat_id: str) -> None:
        """Called by the adapter when a voice channel times out.

        Cleans up runner-side voice_mode state that the adapter cannot reach.
        """
        self._voice_mode[self._voice_key(Platform.DISCORD, chat_id)] = "off"
        self._save_voice_modes()
        adapter = self.adapters.get(Platform.DISCORD)
        self._set_adapter_auto_tts_disabled(adapter, chat_id, disabled=True)

    def _is_duplicate_voice_transcript(self, guild_id: int, user_id: int, transcript: str) -> bool:
        """Suppress repeated STT outputs for the same recent utterance.

        Voice capture can occasionally emit the same utterance twice a few
        seconds apart, which creates a second queued agent run and overlapping
        spoken replies. Dedup exact and near-exact repeats per guild/user over a
        short window while allowing genuinely new turns through.
        """
        from difflib import SequenceMatcher

        normalized = re.sub(r"\s+", " ", transcript).strip().lower()
        normalized = re.sub(r"[^\w\s]", "", normalized)
        if not normalized:
            return False

        now = time.monotonic()
        window_seconds = 12.0
        key = (guild_id, user_id)
        recent_store = getattr(self, "_recent_voice_transcripts", None)
        if not isinstance(recent_store, dict):
            recent_store = {}
            self._recent_voice_transcripts = recent_store
        recent = [
            (ts, txt)
            for ts, txt in recent_store.get(key, [])
            if now - ts <= window_seconds
        ]

        for _, prior in recent:
            if prior == normalized:
                recent_store[key] = recent
                return True
            if len(prior) >= 16 and len(normalized) >= 16:
                if SequenceMatcher(None, prior, normalized).ratio() >= 0.95:
                    recent_store[key] = recent
                    return True

        recent.append((now, normalized))
        recent_store[key] = recent[-5:]
        return False

    async def _handle_voice_channel_input(
        self, guild_id: int, user_id: int, transcript: str
    ):
        """Handle transcribed voice from a user in a voice channel.

        Creates a synthetic MessageEvent and processes it through the
        adapter's full message pipeline (session, typing, agent, TTS reply).
        """
        adapter = self.adapters.get(Platform.DISCORD)
        if not adapter:
            return

        text_ch_id = adapter._voice_text_channels.get(guild_id)
        if not text_ch_id:
            return

        # Build source — reuse the linked text channel's metadata when available
        # so voice input shares the same session as the bound text conversation.
        source_data = getattr(adapter, "_voice_sources", {}).get(guild_id)
        if source_data:
            source = SessionSource.from_dict(source_data)
            source.user_id = str(user_id)
            source.user_name = str(user_id)
        else:
            source = SessionSource(
                platform=Platform.DISCORD,
                chat_id=str(text_ch_id),
                user_id=str(user_id),
                user_name=str(user_id),
                chat_type="channel",
            )

        # Check authorization before processing voice input
        if not self._is_user_authorized(source):
            logger.debug("Unauthorized voice input from user %d, ignoring", user_id)
            return

        if self._is_duplicate_voice_transcript(guild_id, user_id, transcript):
            logger.info(
                "Suppressing duplicate voice transcript for guild=%s user=%s: %s",
                guild_id,
                user_id,
                transcript[:100],
            )
            return

        # Show transcript in text channel (after auth, with mention sanitization)
        try:
            channel = adapter._client.get_channel(text_ch_id)
            if channel:
                safe_text = transcript[:2000].replace("@everyone", "@\u200beveryone").replace("@here", "@\u200bhere")
                await channel.send(f"**[Voice]** <@{user_id}>: {safe_text}")
        except Exception:
            pass

        # Build a synthetic MessageEvent and feed through the normal pipeline
        # Use SimpleNamespace as raw_message so _get_guild_id() can extract
        # guild_id and _send_voice_reply() plays audio in the voice channel.
        from types import SimpleNamespace
        event = MessageEvent(
            source=source,
            text=transcript,
            message_type=MessageType.VOICE,
            raw_message=SimpleNamespace(guild_id=guild_id, guild=None),
        )

        await adapter.handle_message(event)

    def _should_send_voice_reply(
        self,
        event: MessageEvent,
        response: str,
        agent_messages: list,
        already_sent: bool = False,
    ) -> bool:
        """Decide whether the runner should send a TTS voice reply.

        Returns False when:
        - voice_mode is off for this chat
        - response is empty or an error
        - agent already called text_to_speech tool (dedup)
        - voice input and base adapter auto-TTS already handled it (skip_double)
          UNLESS streaming already consumed the response (already_sent=True),
          in which case the base adapter won't have text for auto-TTS so the
          runner must handle it.
        """
        if not response or response.startswith("Error:"):
            return False

        chat_id = event.source.chat_id
        voice_mode = self._voice_mode.get(self._voice_key(event.source.platform, chat_id), "off")
        is_voice_input = (event.message_type == MessageType.VOICE)

        should = (
            (voice_mode == "all")
            or (voice_mode == "voice_only" and is_voice_input)
        )
        if not should:
            return False

        # Dedup: agent already called TTS tool
        has_agent_tts = any(
            msg.get("role") == "assistant"
            and any(
                tc.get("function", {}).get("name") == "text_to_speech"
                for tc in (msg.get("tool_calls") or [])
            )
            for msg in agent_messages
        )
        if has_agent_tts:
            return False

        # Dedup: base adapter auto-TTS already handles voice input
        # (play_tts plays in VC when connected, so runner can skip).
        # When streaming already delivered the text (already_sent=True),
        # the base adapter will receive None and can't run auto-TTS,
        # so the runner must take over.
        if is_voice_input and not already_sent:
            return False

        return True

    async def _send_voice_reply(self, event: MessageEvent, text: str) -> None:
        """Generate TTS audio and send as a voice message before the text reply."""
        import uuid as _uuid
        audio_path = None
        actual_path = None
        try:
            from tools.tts_tool import text_to_speech_tool, _strip_markdown_for_tts

            tts_text = _strip_markdown_for_tts(text[:4000])
            if not tts_text:
                return

            # Use .mp3 extension so edge-tts conversion to opus works correctly.
            # The TTS tool may convert to .ogg — use file_path from result.
            audio_path = os.path.join(
                tempfile.gettempdir(), "hermes_voice",
                f"tts_reply_{_uuid.uuid4().hex[:12]}.mp3",
            )
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)

            result_json = await asyncio.to_thread(
                text_to_speech_tool, text=tts_text, output_path=audio_path
            )
            try:
                result = json.loads(result_json)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Auto voice reply TTS returned invalid JSON: %s", result_json[:200] if result_json else result_json)
                return

            # Use the actual file path from result (may differ after opus conversion)
            actual_path = result.get("file_path", audio_path)
            if not result.get("success") or not os.path.isfile(actual_path):
                logger.warning("Auto voice reply TTS failed: %s", result.get("error"))
                return

            adapter = self.adapters.get(event.source.platform)

            # If connected to a voice channel, play there instead of sending a file
            guild_id = self._get_guild_id(event)
            if (guild_id
                    and hasattr(adapter, "play_in_voice_channel")
                    and hasattr(adapter, "is_in_voice_channel")
                    and adapter.is_in_voice_channel(guild_id)):
                await adapter.play_in_voice_channel(guild_id, actual_path)
            elif adapter and hasattr(adapter, "send_voice"):
                reply_anchor = self._reply_anchor_for_event(event)
                thread_meta = self._thread_metadata_for_source(event.source, reply_anchor)
                # Mark the auto voice reply as notify-worthy.  Mirrors the
                # final-text path in gateway/platforms/base.py which sets
                # ``notify=True`` so platform adapters that gate push
                # notifications (Telegram "important" mode) deliver the
                # final voice reply as a normal notification instead of a
                # silent message.  Clone first so we don't mutate metadata
                # shared with concurrent typing-indicator state.
                if thread_meta is not None:
                    thread_meta = dict(thread_meta)
                    thread_meta["notify"] = True
                else:
                    thread_meta = {"notify": True}
                send_kwargs: Dict[str, Any] = {
                    "chat_id": event.source.chat_id,
                    "audio_path": actual_path,
                    "reply_to": reply_anchor,
                    "metadata": thread_meta,
                }
                await adapter.send_voice(**send_kwargs)
        except Exception as e:
            logger.warning("Auto voice reply failed: %s", e, exc_info=True)
        finally:
            for p in {audio_path, actual_path} - {None}:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    async def _deliver_media_from_response(
        self,
        response: str,
        event: MessageEvent,
        adapter,
    ) -> None:
        """Extract MEDIA: tags and local file paths from a response and deliver them.

        Called after streaming has already sent the text to the user, so the
        text itself is already delivered — this only handles file attachments
        that the normal _process_message_background path would have caught.
        """
        return await media_delivery.deliver_media_from_response(
            response,
            event,
            adapter,
            self._thread_metadata_for_source,
            self._reply_anchor_for_event,
        )
    async def _handle_rollback_command(self, event: MessageEvent) -> str:
        """Handle /rollback command — list or restore filesystem checkpoints."""
        from tools.checkpoint_manager import CheckpointManager, format_checkpoint_list

        # Read checkpoint config from config.yaml
        cp_cfg = {}
        try:
            import yaml as _y
            _cfg_path = _hermes_home / "config.yaml"
            if _cfg_path.exists():
                with open(_cfg_path, encoding="utf-8") as _f:
                    _data = _y.safe_load(_f) or {}
                cp_cfg = _data.get("checkpoints", {})
                if isinstance(cp_cfg, bool):
                    cp_cfg = {"enabled": cp_cfg}
        except Exception:
            pass

        if not cp_cfg.get("enabled", False):
            return t("gateway.rollback.not_enabled")

        mgr = CheckpointManager(
            enabled=True,
            max_snapshots=cp_cfg.get("max_snapshots", 50),
            max_total_size_mb=cp_cfg.get("max_total_size_mb", 500),
            max_file_size_mb=cp_cfg.get("max_file_size_mb", 10),
        )

        cwd = os.getenv("TERMINAL_CWD", str(Path.home()))
        arg = event.get_command_args().strip()

        if not arg:
            checkpoints = mgr.list_checkpoints(cwd)
            return format_checkpoint_list(checkpoints, cwd)

        # Restore by number or hash
        checkpoints = mgr.list_checkpoints(cwd)
        if not checkpoints:
            return t("gateway.rollback.none_found", cwd=cwd)

        target_hash = None
        try:
            idx = int(arg) - 1
            if 0 <= idx < len(checkpoints):
                target_hash = checkpoints[idx]["hash"]
            else:
                return t("gateway.rollback.invalid_number", max=len(checkpoints))
        except ValueError:
            target_hash = arg

        result = mgr.restore(cwd, target_hash)
        if result["success"]:
            return t(
                "gateway.rollback.restored",
                hash=result["restored_to"],
                reason=result["reason"],
            )
        return t("gateway.rollback.restore_failed", error=result["error"])

    async def _handle_background_command(self, event: MessageEvent) -> str:
        """Handle /background <prompt> — run a prompt in a separate background session.

        Spawns a new AIAgent in a background thread with its own session.
        When it completes, sends the result back to the same chat without
        modifying the active session's conversation history.
        """
        prompt = event.get_command_args().strip()
        if not prompt:
            return t("gateway.background.usage")

        source = event.source
        task_id = f"bg_{datetime.now().strftime('%H%M%S')}_{os.urandom(3).hex()}"

        event_message_id = self._reply_anchor_for_event(event)

        # Forward image/audio attachments so the background agent can see them.
        media_urls = list(event.media_urls) if event.media_urls else []
        media_types = list(event.media_types) if event.media_types else []

        # Fire-and-forget the background task
        _task = asyncio.create_task(
            self._run_background_task(
                prompt,
                source,
                task_id,
                event_message_id=event_message_id,
                media_urls=media_urls,
                media_types=media_types,
            )
        )
        self._background_tasks.add(_task)
        _task.add_done_callback(self._background_tasks.discard)

        preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
        return t("gateway.background.started", preview=preview, task_id=task_id)

    async def _run_background_task(
        self,
        prompt: str,
        source: "SessionSource",
        task_id: str,
        event_message_id: Optional[str] = None,
        media_urls: Optional[List[str]] = None,
        media_types: Optional[List[str]] = None,
    ) -> None:
        """Execute a background agent task and deliver the result to the chat."""
        from run_agent import AIAgent

        media_urls = media_urls or []
        media_types = media_types or []

        adapter = self.adapters.get(source.platform)
        if not adapter:
            logger.warning("No adapter for platform %s in background task %s", source.platform, task_id)
            return

        _thread_metadata = self._thread_metadata_for_source(source, event_message_id)

        try:
            user_config = _load_gateway_config()
            model, runtime_kwargs = self._resolve_session_agent_runtime(
                source=source,
                user_config=user_config,
            )
            if not runtime_kwargs.get("api_key"):
                await adapter.send(
                    source.chat_id,
                    f"❌ Background task {task_id} failed: no provider credentials configured.",
                    metadata=_thread_metadata,
                )
                return

            platform_key = _platform_config_key(source.platform)

            from hermes_cli.tools_config import _get_platform_tools
            enabled_toolsets = sorted(_get_platform_tools(user_config, platform_key))
            agent_cfg = user_config.get("agent") or {}
            disabled_toolsets = agent_cfg.get("disabled_toolsets") or None

            pr = self._provider_routing
            max_iterations = int(os.getenv("HERMES_MAX_ITERATIONS", "90"))
            reasoning_config = self._resolve_session_reasoning_config(source=source)
            self._reasoning_config = reasoning_config
            self._service_tier = self._load_service_tier()
            turn_route = self._resolve_turn_agent_config(prompt, model, runtime_kwargs)

            # Enrich the prompt with image descriptions so the background
            # agent can see user-attached images (same as the main flow).
            enriched_prompt = prompt
            if media_urls:
                image_paths = []
                for i, path in enumerate(media_urls):
                    mtype = media_types[i] if i < len(media_types) else ""
                    if mtype.startswith("image/"):
                        image_paths.append(path)
                if image_paths:
                    try:
                        enriched_prompt = await self._enrich_message_with_vision(
                            prompt, image_paths,
                        )
                    except Exception as e:
                        logger.warning("Background task vision enrichment failed: %s", e)

            def run_sync():
                agent = AIAgent(
                    model=turn_route["model"],
                    **turn_route["runtime"],
                    max_iterations=max_iterations,
                    quiet_mode=True,
                    verbose_logging=False,
                    enabled_toolsets=enabled_toolsets,
                    disabled_toolsets=disabled_toolsets,
                    reasoning_config=reasoning_config,
                    service_tier=self._service_tier,
                    request_overrides=turn_route.get("request_overrides"),
                    providers_allowed=pr.get("only"),
                    providers_ignored=pr.get("ignore"),
                    providers_order=pr.get("order"),
                    provider_sort=pr.get("sort"),
                    provider_require_parameters=pr.get("require_parameters", False),
                    provider_data_collection=pr.get("data_collection"),
                    session_id=task_id,
                    platform=platform_key,
                    user_id=source.user_id,
                    user_id_alt=source.user_id_alt,
                    user_name=source.user_name,
                    chat_id=source.chat_id,
                    chat_name=source.chat_name,
                    chat_type=source.chat_type,
                    thread_id=source.thread_id,
                    session_db=self._session_db,
                    fallback_model=self._fallback_model,
                )
                try:
                    return agent.run_conversation(
                        user_message=enriched_prompt,
                        task_id=task_id,
                    )
                finally:
                    self._cleanup_agent_resources(agent)

            result = await self._run_in_executor_with_context(run_sync)

            response = result.get("final_response", "") if result else ""
            if not response and result and result.get("error"):
                response = f"Error: {result['error']}"

            # Extract media files from the response
            if response:
                media_files, response = adapter.extract_media(response)
                from gateway.platforms.base import BasePlatformAdapter
                media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
                images, text_content = adapter.extract_images(response)

                preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
                header = f'✅ Background task complete\nPrompt: "{preview}"\n\n'

                if text_content:
                    await adapter.send(
                        chat_id=source.chat_id,
                        content=header + text_content,
                        metadata=_thread_metadata,
                    )
                elif not images and not media_files:
                    await adapter.send(
                        chat_id=source.chat_id,
                        content=header + "(No response generated)",
                        metadata=_thread_metadata,
                    )

                # Send extracted images
                for image_url, alt_text in (images or []):
                    try:
                        await adapter.send_image(
                            chat_id=source.chat_id,
                            image_url=image_url,
                            caption=alt_text,
                            metadata=_thread_metadata,
                        )
                    except Exception:
                        pass

                # Send media files, routing each by type so a TTS clip
                # arrives as a voice bubble / a clip as a video rather than
                # a generic document. Mirrors the streaming + kanban paths.
                from gateway.platforms.base import (
                    should_send_media_as_audio as _should_send_media_as_audio,
                )
                _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
                _VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}
                for media_path, _is_voice in (media_files or []):
                    _ext = os.path.splitext(media_path)[1].lower()
                    try:
                        if _should_send_media_as_audio(source.platform, _ext, _is_voice):
                            await adapter.send_voice(
                                chat_id=source.chat_id,
                                audio_path=media_path,
                                metadata=_thread_metadata,
                            )
                        elif _ext in _VIDEO_EXTS:
                            await adapter.send_video(
                                chat_id=source.chat_id,
                                video_path=media_path,
                                metadata=_thread_metadata,
                            )
                        elif _ext in _IMAGE_EXTS:
                            await adapter.send_image_file(
                                chat_id=source.chat_id,
                                image_path=media_path,
                                metadata=_thread_metadata,
                            )
                        else:
                            await adapter.send_document(
                                chat_id=source.chat_id,
                                file_path=media_path,
                                metadata=_thread_metadata,
                            )
                    except Exception:
                        pass
            else:
                preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
                await adapter.send(
                    chat_id=source.chat_id,
                    content=f'✅ Background task complete\nPrompt: "{preview}"\n\n(No response generated)',
                    metadata=_thread_metadata,
                )

        except Exception as e:
            logger.exception("Background task %s failed", task_id)
            try:
                await adapter.send(
                    chat_id=source.chat_id,
                    content=f"❌ Background task {task_id} failed: {e}",
                    metadata=_thread_metadata,
                )
            except Exception:
                pass

    async def _handle_reasoning_command(self, event: MessageEvent) -> str:
        """Handle /reasoning command — manage reasoning effort and display toggle.

        Usage:
            /reasoning                       Show current effort level and display state
            /reasoning <level>               Set reasoning effort for this session only
            /reasoning <level> --global      Persist reasoning effort to config.yaml
            /reasoning reset                 Clear this session's reasoning override
            /reasoning show|on               Show model reasoning in responses
            /reasoning hide|off              Hide model reasoning from responses

        Delegated to gateway.command_handlers.handle_reasoning_command.
        """
        return await command_handlers.handle_reasoning_command(self, event)
    async def _handle_fast_command(self, event: MessageEvent) -> str:
        """Handle /fast — mirror the CLI Priority Processing toggle in gateway chats."""
        import yaml
        from hermes_cli.models import model_supports_fast_mode

        args = event.get_command_args().strip().lower()
        config_path = _hermes_home / "config.yaml"
        self._service_tier = self._load_service_tier()

        user_config = _load_gateway_config()
        model = _resolve_gateway_model(user_config)
        if not model_supports_fast_mode(model):
            return t("gateway.fast.not_supported")

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

        if not args or args == "status":
            status = t("gateway.fast.status_fast") if self._service_tier == "priority" else t("gateway.fast.status_normal")
            return t("gateway.fast.status", mode=status)

        if args in {"fast", "on"}:
            self._service_tier = "priority"
            saved_value = "fast"
            label = t("gateway.fast.label_fast")
        elif args in {"normal", "off"}:
            self._service_tier = None
            saved_value = "normal"
            label = t("gateway.fast.label_normal")
        else:
            return t("gateway.fast.unknown_arg", arg=args)

        if _save_config_key("agent.service_tier", saved_value):
            return t("gateway.fast.saved", label=label)
        return t("gateway.fast.session_only", label=label)

    async def _handle_yolo_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        """Handle /yolo — toggle dangerous command approval bypass for this session only."""
        from tools.approval import (
            disable_session_yolo,
            enable_session_yolo,
            is_session_yolo_enabled,
        )

        session_key = self._session_key_for_source(event.source)
        current = is_session_yolo_enabled(session_key)
        if current:
            disable_session_yolo(session_key)
            return EphemeralReply(t("gateway.yolo.disabled"))
        else:
            enable_session_yolo(session_key)
            return EphemeralReply(t("gateway.yolo.enabled"))

    async def _handle_verbose_command(self, event: MessageEvent) -> str:
        """Handle /verbose command — cycle tool progress display mode.

        Gated by ``display.tool_progress_command`` in config.yaml (default off).
        When enabled, cycles the tool progress mode through off → new → all →
        verbose → off for the *current platform*.  The setting is saved to
        ``display.platforms.<platform>.tool_progress`` so each channel can
        have its own verbosity level independently.
        """

        config_path = _hermes_home / "config.yaml"
        platform_key = _platform_config_key(event.source.platform)

        # --- check config gate ------------------------------------------------
        try:
            user_config = _load_gateway_config()
            gate_enabled = is_truthy_value(
                cfg_get(user_config, "display", "tool_progress_command"),
                default=False,
            )
        except Exception:
            gate_enabled = False

        if not gate_enabled:
            return t("gateway.verbose.not_enabled")

        # --- cycle mode (per-platform) ----------------------------------------
        cycle = ["off", "new", "all", "verbose"]
        descriptions = {
            "off": t("gateway.verbose.mode_off"),
            "new": t("gateway.verbose.mode_new"),
            "all": t("gateway.verbose.mode_all"),
            "verbose": t("gateway.verbose.mode_verbose"),
        }

        # Read current effective mode for this platform via the resolver
        from gateway.display_config import resolve_display_setting
        current = resolve_display_setting(user_config, platform_key, "tool_progress", "all")
        if current not in cycle:
            current = "all"
        idx = (cycle.index(current) + 1) % len(cycle)
        new_mode = cycle[idx]

        # Save to display.platforms.<platform>.tool_progress
        try:
            if "display" not in user_config or not isinstance(user_config.get("display"), dict):
                user_config["display"] = {}
            display = user_config["display"]
            if "platforms" not in display or not isinstance(display.get("platforms"), dict):
                display["platforms"] = {}
            if platform_key not in display["platforms"] or not isinstance(display["platforms"].get(platform_key), dict):
                display["platforms"][platform_key] = {}
            display["platforms"][platform_key]["tool_progress"] = new_mode
            atomic_yaml_write(config_path, user_config)
            return (
                f"{descriptions[new_mode]}\n"
                + t("gateway.verbose.saved_suffix", platform=platform_key)
            )
        except Exception as e:
            logger.warning("Failed to save tool_progress mode: %s", e)
            return f"{descriptions[new_mode]}\n" + t("gateway.verbose.save_failed", error=e)

    async def _handle_footer_command(self, event: MessageEvent) -> str:
        return await command_handlers.handle_footer_command(
            runner=self,
            event=event,
        )

    async def _handle_compress_command(self, event: MessageEvent) -> str:
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
        return await command_handlers.handle_compress_command(self, event)
    async def _get_telegram_topic_capabilities(self, source: SessionSource) -> dict:
        """Read Telegram private-topic capability flags via Bot API getMe."""
        adapter = self.adapters.get(source.platform) if getattr(self, "adapters", None) else None
        bot = getattr(adapter, "_bot", None)
        if bot is None or not hasattr(bot, "get_me"):
            return {"checked": False}
        try:
            me = await bot.get_me()
        except Exception:
            logger.debug("Failed to fetch Telegram getMe topic capabilities", exc_info=True)
            return {"checked": False}

        def _field(name: str):
            if hasattr(me, name):
                return getattr(me, name)
            api_kwargs = getattr(me, "api_kwargs", None)
            if isinstance(api_kwargs, dict) and name in api_kwargs:
                return api_kwargs.get(name)
            if isinstance(me, dict):
                return me.get(name)
            return None

        return {
            "checked": True,
            "has_topics_enabled": _field("has_topics_enabled"),
            "allows_users_to_create_topics": _field("allows_users_to_create_topics"),
        }

    async def _ensure_telegram_system_topic(self, source: SessionSource) -> None:
        """Create/pin the managed System topic after /topic activation when possible."""
        adapter = self.adapters.get(source.platform) if getattr(self, "adapters", None) else None
        if adapter is None or not source.chat_id:
            return

        thread_id = None
        create_topic = getattr(adapter, "_create_dm_topic", None)
        if callable(create_topic):
            try:
                thread_id = await create_topic(int(source.chat_id), "System")
            except Exception:
                logger.debug("Failed to create Telegram System topic", exc_info=True)
        if not thread_id:
            return

        message_id = None
        try:
            send_result = await adapter.send(
                source.chat_id,
                "System topic for Hermes commands and status.",
                metadata={"thread_id": str(thread_id)},
            )
            message_id = getattr(send_result, "message_id", None)
        except Exception:
            logger.debug("Failed to send Telegram System topic intro", exc_info=True)
        if not message_id:
            return

        bot = getattr(adapter, "_bot", None)
        if bot is None or not hasattr(bot, "pin_chat_message"):
            return
        try:
            await bot.pin_chat_message(
                chat_id=int(source.chat_id),
                message_id=int(message_id),
                disable_notification=True,
            )
        except Exception:
            logger.debug("Failed to pin Telegram System topic intro", exc_info=True)

    async def _send_telegram_topic_setup_image(self, source: SessionSource) -> None:
        """Send the bundled BotFather Threads Settings screenshot when available."""
        adapter = self.adapters.get(source.platform) if getattr(self, "adapters", None) else None
        if adapter is None or not source.chat_id or not hasattr(adapter, "send_image_file"):
            return
        image_path = Path(__file__).resolve().parent / "assets" / "telegram-botfather-threads-settings.jpg"
        if not image_path.exists():
            return
        try:
            await adapter.send_image_file(
                chat_id=source.chat_id,
                image_path=str(image_path),
                caption="BotFather → Bot Settings → Threads Settings",
                metadata={"thread_id": str(source.thread_id)} if source.thread_id else None,
            )
        except Exception:
            logger.debug("Failed to send Telegram topic setup image", exc_info=True)

    def _sanitize_telegram_topic_title(self, title: str) -> str:
        """Return a Bot API-safe forum topic name from a generated session title."""
        cleaned = re.sub(r"\s+", " ", str(title or "")).strip()
        if not cleaned:
            return "Hermes Chat"
        # Telegram forum topic names are short (currently 1-128 chars). Keep
        # extra room for multi-byte titles and avoid trailing ellipsis churn.
        if len(cleaned) > 120:
            cleaned = cleaned[:117].rstrip() + "..."
        return cleaned

    async def _rename_telegram_topic_for_session_title(
        self,
        source: SessionSource,
        session_id: str,
        title: str,
    ) -> None:
        """Best-effort rename of a Telegram DM topic when Hermes auto-titles a session."""
        if not self._is_telegram_topic_lane(source) or not source.chat_id or not source.thread_id:
            return

        # Operator can fully disable per-topic auto-rename via
        # extra.disable_topic_auto_rename. Useful when topics are managed
        # by the user (ad-hoc Threaded Mode) and auto-rename would
        # overwrite their chosen names every time the auto-title fires.
        if self._telegram_topic_auto_rename_disabled(source):
            return

        # Skip rename when the topic is operator-declared via
        # extra.dm_topics. Those topics have fixed names chosen by the
        # operator (plus optional skill binding); auto-renaming would
        # silently mutate operator config.
        #
        # Check the class, not the instance — getattr() on MagicMock
        # auto-creates attributes, so `hasattr(adapter, "_get_dm_topic_info")`
        # would return True for every test double.
        adapter = self.adapters.get(source.platform) if getattr(self, "adapters", None) else None
        if adapter is not None:
            get_info = getattr(type(adapter), "_get_dm_topic_info", None)
            if callable(get_info):
                try:
                    operator_topic = get_info(adapter, str(source.chat_id), str(source.thread_id))
                except Exception:
                    operator_topic = None
                # Only treat dict-shaped returns as operator-declared; a
                # bare MagicMock or other sentinel shouldn't count.
                if isinstance(operator_topic, dict):
                    return

        session_db = getattr(self, "_session_db", None)
        if session_db is not None:
            try:
                binding = session_db.get_telegram_topic_binding(
                    chat_id=str(source.chat_id),
                    thread_id=str(source.thread_id),
                )
                if binding and str(binding.get("session_id") or "") != str(session_id):
                    return
            except Exception:
                logger.debug("Failed to verify Telegram topic binding before rename", exc_info=True)
                return

        if adapter is None:
            return
        topic_name = self._sanitize_telegram_topic_title(title)
        try:
            rename_topic = getattr(adapter, "rename_dm_topic", None)
            if rename_topic is not None:
                await rename_topic(
                    chat_id=str(source.chat_id),
                    thread_id=str(source.thread_id),
                    name=topic_name,
                )
                return

            bot = getattr(adapter, "_bot", None)
            edit_forum_topic = getattr(bot, "edit_forum_topic", None) if bot is not None else None
            if edit_forum_topic is None:
                edit_forum_topic = getattr(bot, "editForumTopic", None) if bot is not None else None
            if edit_forum_topic is None:
                return
            try:
                await edit_forum_topic(
                    chat_id=int(source.chat_id),
                    message_thread_id=int(source.thread_id),
                    name=topic_name,
                )
            except (TypeError, ValueError):
                await edit_forum_topic(
                    chat_id=source.chat_id,
                    message_thread_id=source.thread_id,
                    name=topic_name,
                )
        except Exception:
            logger.debug("Failed to rename Telegram topic for auto-generated title", exc_info=True)

    def _telegram_topic_auto_rename_disabled(self, source: SessionSource) -> bool:
        """Return True when operator disabled per-topic auto-rename for this Telegram chat.

        Controlled via ``gateway.platforms.telegram.extra.disable_topic_auto_rename``.
        Default is False (auto-rename enabled, preserves prior behaviour).
        """
        platform_cfg = (
            self.config.platforms.get(source.platform)
            if getattr(self, "config", None) and getattr(self.config, "platforms", None)
            else None
        )
        if platform_cfg is None:
            return False
        extra = getattr(platform_cfg, "extra", None) or {}
        value = extra.get("disable_topic_auto_rename")
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _schedule_telegram_topic_title_rename(
        self,
        source: SessionSource,
        session_id: str,
        title: str,
    ) -> None:
        """Schedule a topic rename from the auto-title background thread."""
        if not title or not self._is_telegram_topic_lane(source):
            return
        if self._telegram_topic_auto_rename_disabled(source):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = getattr(self, "_gateway_loop", None)
        if loop is None or loop.is_closed():
            return
        try:
            copied_source = dataclasses.replace(source)
        except Exception:
            copied_source = source
        future = safe_schedule_threadsafe(
            self._rename_telegram_topic_for_session_title(copied_source, session_id, title),
            loop,
            logger=logger,
            log_message="Telegram topic title rename failed to schedule",
        )
        if future is None:
            return
        def _log_rename_failure(fut) -> None:
            try:
                fut.result()
            except Exception:
                logger.debug("Telegram topic title rename failed", exc_info=True)

        future.add_done_callback(_log_rename_failure)

    _TELEGRAM_CAPABILITY_HINT_COOLDOWN_S = 300.0

    def _should_send_telegram_capability_hint(self, source: SessionSource) -> bool:
        """Rate-limit the BotFather Threads Settings screenshot.

        If a user sends /topic repeatedly while Threads Settings are still
        off, we shouldn't keep re-uploading the screenshot every time.
        """
        if not hasattr(self, "_telegram_capability_hint_ts"):
            self._telegram_capability_hint_ts = {}
        chat_id = str(source.chat_id or "")
        if not chat_id:
            return True
        import time as _time
        now = _time.monotonic()
        last = self._telegram_capability_hint_ts.get(chat_id, 0.0)
        if now - last < self._TELEGRAM_CAPABILITY_HINT_COOLDOWN_S:
            return False
        self._telegram_capability_hint_ts[chat_id] = now
        return True

    def _telegram_topic_help_text(self) -> str:
        return (
            "/topic — enable multi-session DM mode (one bot, many parallel chats)\n"
            "\n"
            "Usage:\n"
            "  /topic             Enable topic mode, or show status if already on\n"
            "  /topic help        Show this message\n"
            "  /topic off         Disable topic mode and clear topic bindings\n"
            "  /topic <id>        Inside a topic: restore a previous session by ID\n"
            "\n"
            "How it works:\n"
            "1. Run /topic once in this DM — Hermes checks BotFather Threads\n"
            "   Settings are enabled and flips on multi-session mode.\n"
            "2. Tap All Messages at the top of the bot and send any message.\n"
            "   Telegram creates a new topic for that message; each topic is\n"
            "   an independent Hermes session (fresh history, fresh context).\n"
            "3. The root DM becomes a system lobby — send /topic, /status,\n"
            "   /help, /usage there. Normal prompts go in a topic.\n"
            "4. /new inside a topic resets just that topic's session.\n"
            "5. /topic <id> inside a topic restores an old session into it."
        )

    def _disable_telegram_topic_mode_for_chat(self, source: SessionSource) -> str:
        """Cleanly disable topic mode for a chat via /topic off."""
        if not self._session_db:
            from hermes_state import format_session_db_unavailable
            return format_session_db_unavailable(prefix=t("gateway.shared.session_db_unavailable_prefix"))
        chat_id = str(source.chat_id or "")
        if not chat_id:
            return "Could not determine chat ID."
        # No-op if never enabled.
        try:
            currently_enabled = self._session_db.is_telegram_topic_mode_enabled(
                chat_id=chat_id,
                user_id=str(source.user_id or ""),
            )
        except Exception:
            currently_enabled = False
        if not currently_enabled:
            return "Multi-session topic mode is not currently enabled for this chat."
        try:
            self._session_db.disable_telegram_topic_mode(chat_id=chat_id)
        except Exception as exc:
            logger.exception("Failed to disable Telegram topic mode")
            return f"Failed to disable topic mode: {exc}"
        # Reset per-chat debounce state so the user doesn't see a stale
        # cooldown on the next activation.
        for attr in ("_telegram_lobby_reminder_ts", "_telegram_capability_hint_ts"):
            store = getattr(self, attr, None)
            if isinstance(store, dict):
                store.pop(chat_id, None)
        return (
            "Multi-session topic mode is now OFF for this chat.\n\n"
            "Existing topics in Telegram aren't removed — they'll just stop "
            "being gated as independent sessions. The root DM works as a "
            "normal Hermes chat again. Run /topic to re-enable later."
        )

    async def _handle_topic_command(self, event: MessageEvent, args: str = "") -> str:
        """Handle /topic for Telegram DM user-managed topic sessions."""
        source = event.source
        if source.platform != Platform.TELEGRAM or source.chat_type != "dm":
            return t("gateway.topic.not_telegram_dm")
        if not self._session_db:
            from hermes_state import format_session_db_unavailable
            return format_session_db_unavailable(prefix=t("gateway.shared.session_db_unavailable_prefix"))

        # Authorization: /topic activates multi-session mode and mutates
        # SQLite side tables. Unauthorized senders (not in allowlist) must
        # not be able to do that. Gateway routes already authorize the
        # message before reaching here, but defense in depth.
        auth_fn = getattr(self, "_is_user_authorized", None)
        if callable(auth_fn):
            try:
                if not auth_fn(source):
                    return t("gateway.topic.unauthorized")
            except Exception:
                logger.debug("Topic auth check failed", exc_info=True)

        args = event.get_command_args().strip()

        # /topic help — inline usage without leaving the bot.
        if args.lower() in {"help", "?", "-h", "--help"}:
            return self._telegram_topic_help_text()

        # /topic off — clean disable path so users don't have to edit the DB.
        if args.lower() in {"off", "disable", "stop"}:
            return self._disable_telegram_topic_mode_for_chat(source)

        if args:
            if not source.thread_id:
                return t("gateway.topic.restore_needs_topic")
            return await self._restore_telegram_topic_session(event, args)

        capabilities = await self._get_telegram_topic_capabilities(source)
        if capabilities.get("checked"):
            if capabilities.get("has_topics_enabled") is False:
                # Debounce the BotFather screenshot: don't re-send on every
                # /topic while threads are still disabled.
                if self._should_send_telegram_capability_hint(source):
                    await self._send_telegram_topic_setup_image(source)
                return t("gateway.topic.topics_disabled")
            if capabilities.get("allows_users_to_create_topics") is False:
                if self._should_send_telegram_capability_hint(source):
                    await self._send_telegram_topic_setup_image(source)
                return t("gateway.topic.topics_user_disallowed")

        try:
            self._session_db.enable_telegram_topic_mode(
                chat_id=str(source.chat_id),
                user_id=str(source.user_id),
                has_topics_enabled=capabilities.get("has_topics_enabled"),
                allows_users_to_create_topics=capabilities.get("allows_users_to_create_topics"),
            )
        except Exception as exc:
            logger.exception("Failed to enable Telegram topic mode")
            return t("gateway.topic.enable_failed", error=exc)

        if not source.thread_id:
            await self._ensure_telegram_system_topic(source)

        if source.thread_id:
            try:
                binding = self._session_db.get_telegram_topic_binding(
                    chat_id=str(source.chat_id),
                    thread_id=str(source.thread_id),
                )
            except Exception:
                logger.debug("Failed to read Telegram topic binding", exc_info=True)
                binding = None
            if binding:
                session_id = str(binding.get("session_id") or "")
                title = None
                try:
                    title = self._session_db.get_session_title(session_id)
                except Exception:
                    title = None
                session_label = title or t("gateway.topic.untitled_session")
                return t(
                    "gateway.topic.bound_status",
                    label=session_label,
                    session_id=session_id,
                )
            return t("gateway.topic.thread_ready")

        return self._telegram_topic_root_status_message(source)

    def _telegram_topic_root_status_message(self, source: SessionSource) -> str:
        lines = [
            "Telegram multi-session topics are enabled.",
            "",
            "To create a new Hermes chat, open All Messages at the top of this "
            "bot interface and send any message there. Telegram will create a "
            "new topic for it.",
            "",
        ]
        try:
            sessions = self._session_db.list_unlinked_telegram_sessions_for_user(
                chat_id=str(source.chat_id),
                user_id=str(source.user_id),
                limit=10,
            )
        except Exception:
            logger.debug("Failed to list unlinked Telegram sessions", exc_info=True)
            sessions = []

        if sessions:
            lines.append("Previous unlinked sessions:")
            for session in sessions:
                session_id = str(session.get("id") or "")
                title = str(session.get("title") or "Untitled session")
                preview = str(session.get("preview") or "").strip()
                line = f"- {title} — `{session_id}`"
                if preview:
                    line += f" — {preview}"
                lines.append(line)
            lines.extend([
                "",
                "To restore one:",
                "1. Create or open a topic. To create a new one, open All Messages and send any message there.",
                "2. Send /topic <session-id> inside that topic.",
                f"Example: Send /topic {sessions[0].get('id')} inside a topic.",
            ])
        else:
            lines.extend([
                "No previous unlinked Telegram sessions found.",
                "",
                "To restore a previous session later:",
                "1. Create or open a topic. To create a new one, open All Messages and send any message there.",
                "2. Send /topic <session-id> inside that topic.",
            ])
        return "\n".join(lines)

    async def _restore_telegram_topic_session(self, event: MessageEvent, raw_session_id: str) -> str:
        """Restore an existing Telegram-owned Hermes session into this topic."""
        source = event.source
        session_id = self._session_db.resolve_session_id(raw_session_id.strip())
        if not session_id:
            return f"Session not found: {raw_session_id.strip()}"

        session = self._session_db.get_session(session_id)
        if not session:
            return f"Session not found: {raw_session_id.strip()}"
        if str(session.get("source") or "") != "telegram":
            return "That session is not a Telegram session and cannot be restored into this topic."
        if str(session.get("user_id") or "") != str(source.user_id):
            return "That session does not belong to this Telegram user."

        linked = self._session_db.is_telegram_session_linked_to_topic(session_id=session_id)
        current_binding = self._session_db.get_telegram_topic_binding(
            chat_id=str(source.chat_id),
            thread_id=str(source.thread_id),
        )
        if linked:
            if not current_binding or current_binding.get("session_id") != session_id:
                return "That session is already linked to another Telegram topic."

        session_key = self._session_key_for_source(source)
        try:
            self._session_db.bind_telegram_topic(
                chat_id=str(source.chat_id),
                thread_id=str(source.thread_id),
                user_id=str(source.user_id),
                session_key=session_key,
                session_id=session_id,
                managed_mode="restored",
            )
        except ValueError as exc:
            if "already linked" in str(exc):
                return "That session is already linked to another Telegram topic."
            raise

        title = self._session_db.get_session_title(session_id) or session_id
        last_assistant = None
        try:
            for message in reversed(self._session_db.get_messages(session_id)):
                if message.get("role") == "assistant" and message.get("content"):
                    last_assistant = str(message.get("content"))
                    break
        except Exception:
            last_assistant = None

        response = f"Session restored: {title}"
        if last_assistant:
            response += f"\n\nLast Hermes message:\n{last_assistant}"
        return response

    async def _handle_title_command(self, event: MessageEvent) -> str:
        """Handle /title command — set or show the current session's title."""
        source = event.source
        session_entry = self.session_store.get_or_create_session(source)
        session_id = session_entry.session_id

        if not self._session_db:
            from hermes_state import format_session_db_unavailable
            return format_session_db_unavailable(prefix=t("gateway.shared.session_db_unavailable_prefix"))

        # Ensure session exists in SQLite DB (it may only exist in session_store
        # if this is the first command in a new session)
        existing_title = self._session_db.get_session_title(session_id)
        if existing_title is None:
            # Session doesn't exist in DB yet — create it
            try:
                self._session_db.create_session(
                    session_id=session_id,
                    source=source.platform.value if source.platform else "unknown",
                    user_id=source.user_id,
                )
            except Exception:
                pass  # Session might already exist, ignore errors

        title_arg = event.get_command_args().strip()
        if title_arg:
            # Sanitize the title before setting
            try:
                sanitized = self._session_db.sanitize_title(title_arg)
            except ValueError as e:
                return t("gateway.shared.warn_passthrough", error=e)
            if not sanitized:
                return t("gateway.title.empty_after_clean")
            # Set the title
            try:
                if self._session_db.set_session_title(session_id, sanitized):
                    return t("gateway.title.set_to", title=sanitized)
                else:
                    return t("gateway.title.not_found")
            except ValueError as e:
                return t("gateway.shared.warn_passthrough", error=e)
        else:
            # Show the current title and session ID
            title = self._session_db.get_session_title(session_id)
            if title:
                return t("gateway.title.current_with_title", session_id=session_id, title=title)
            else:
                return t("gateway.title.current_no_title", session_id=session_id)

    async def _handle_resume_command(self, event: MessageEvent) -> str:
        return await command_handlers.handle_resume_command(
            runner=self,
            event=event,
        )

    async def _handle_branch_command(self, event: MessageEvent) -> str:
        """Handle /branch [name] — fork the current session into a new independent copy.

        Copies conversation history to a new session so the user can explore
        a different approach without losing the original.
        Inspired by Claude Code's /branch command.
        """
        import uuid as _uuid

        if not self._session_db:
            from hermes_state import format_session_db_unavailable
            return format_session_db_unavailable(prefix=t("gateway.shared.session_db_unavailable_prefix"))

        source = event.source
        session_key = self._session_key_for_source(source)

        # Load the current session and its transcript
        current_entry = self.session_store.get_or_create_session(source)
        history = self.session_store.load_transcript(current_entry.session_id)
        if not history:
            return t("gateway.branch.no_conversation")

        branch_name = event.get_command_args().strip()

        # Generate the new session ID
        from datetime import datetime as _dt
        now = _dt.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        short_uuid = _uuid.uuid4().hex[:6]
        new_session_id = f"{timestamp_str}_{short_uuid}"

        # Determine branch title
        if branch_name:
            branch_title = branch_name
        else:
            current_title = self._session_db.get_session_title(current_entry.session_id)
            base = current_title or "branch"
            branch_title = self._session_db.get_next_title_in_lineage(base)

        parent_session_id = current_entry.session_id

        # Create the new session with parent link
        try:
            self._session_db.create_session(
                session_id=new_session_id,
                source=source.platform.value if source.platform else "gateway",
                model=(self.config.get("model", {}) or {}).get("default") if isinstance(self.config, dict) else None,
                parent_session_id=parent_session_id,
            )
        except Exception as e:
            logger.error("Failed to create branch session: %s", e)
            return t("gateway.branch.create_failed", error=e)

        # Copy conversation history to the new session
        for msg in history:
            try:
                self._session_db.append_message(
                    session_id=new_session_id,
                    role=msg.get("role", "user"),
                    content=msg.get("content"),
                    tool_name=msg.get("tool_name") or msg.get("name"),
                    tool_calls=msg.get("tool_calls"),
                    tool_call_id=msg.get("tool_call_id"),
                    finish_reason=msg.get("finish_reason"),
                    reasoning=msg.get("reasoning"),
                    reasoning_content=msg.get("reasoning_content"),
                    reasoning_details=msg.get("reasoning_details"),
                    codex_reasoning_items=msg.get("codex_reasoning_items"),
                    codex_message_items=msg.get("codex_message_items"),
                )
            except Exception:
                pass  # Best-effort copy

        # Set title
        try:
            self._session_db.set_session_title(new_session_id, branch_title)
        except Exception:
            pass

        # Switch the session store entry to the new session
        new_entry = self.session_store.switch_session(session_key, new_session_id)
        if not new_entry:
            return t("gateway.branch.switch_failed")
        self._clear_session_boundary_security_state(session_key)

        # Evict any cached agent for this session
        self._evict_cached_agent(session_key)

        msg_count = len([m for m in history if m.get("role") == "user"])
        key = "gateway.branch.branched_one" if msg_count == 1 else "gateway.branch.branched_many"
        return t(key, title=branch_title, count=msg_count, parent=parent_session_id, new=new_session_id)

    async def _handle_usage_command(self, event: MessageEvent) -> str:
        """Handle /usage command -- show token usage for the current session.

        Delegates to the extracted handler in gateway/command_handlers.py.
        """
        return await command_handlers.handle_usage_command(self, event)
    async def _handle_insights_command(self, event: MessageEvent) -> str:
        """Handle /insights command -- show usage insights and analytics."""
        args = event.get_command_args().strip()

        # Normalize Unicode dashes (Telegram/iOS auto-converts -- to em/en dash)
        args = re.sub(r'[\u2012\u2013\u2014\u2015](days|source)', r'--\1', args)

        days = 30
        source = None

        # Parse simple args: /insights 7  or  /insights --days 7
        if args:
            parts = args.split()
            i = 0
            while i < len(parts):
                if parts[i] == "--days" and i + 1 < len(parts):
                    try:
                        days = int(parts[i + 1])
                    except ValueError:
                        return t("gateway.insights.invalid_days", value=parts[i + 1])
                    i += 2
                elif parts[i] == "--source" and i + 1 < len(parts):
                    source = parts[i + 1]
                    i += 2
                elif parts[i].isdigit():
                    days = int(parts[i])
                    i += 1
                else:
                    i += 1

        try:
            from hermes_state import SessionDB
            from agent.insights import InsightsEngine

            loop = asyncio.get_running_loop()

            def _run_insights():
                db = SessionDB()
                engine = InsightsEngine(db)
                report = engine.generate(days=days, source=source)
                result = engine.format_gateway(report)
                db.close()
                return result

            return await loop.run_in_executor(None, _run_insights)
        except Exception as e:
            logger.error("Insights command error: %s", e, exc_info=True)
            return t("gateway.insights.error", error=e)

    async def _handle_reload_mcp_command(self, event: MessageEvent) -> Optional[str]:
        return await command_handlers.handle_reload_mcp_command(
            runner=self,
            event=event,
        )

    async def _execute_mcp_reload(self, event: MessageEvent) -> str:
        return await command_handlers.execute_mcp_reload(
            runner=self,
            event=event,
        )

    async def _handle_reload_skills_command(self, event: MessageEvent) -> str:
        """Handle /reload-skills — rescan skills dir, queue a note for next turn.

        Skills don't need to be in the system prompt for the model to use
        them (they're invoked via ``/skill-name``, ``skills_list``, or
        ``skill_view`` at runtime), so this does NOT clear the prompt cache
        — prefix caching stays intact.

        If any skills were added or removed, a one-shot note is queued on
        ``self._pending_skills_reload_notes[session_key]``. The gateway
        prepends it to the NEXT user message in this session (see the
        consumer at ~L11025 in ``_run_agent_turn``), then clears it. Nothing
        is written to the session transcript out-of-band, so message
        alternation is preserved.
        """
        return await command_handlers.handle_reload_skills_command(
            runner=self,
            event=event,
        )

    async def _handle_bundles_command(self, event: MessageEvent) -> str:
        """Handle /bundles — list installed skill bundles.

        Mirrors the CLI ``/bundles`` handler. Returns a single text
        message suitable for any gateway adapter; bundles are loaded by
        invoking the bundle's own ``/<slug>`` command, not by this one.
        """
        try:
            from agent.skill_bundles import list_bundles, _bundles_dir
        except Exception as exc:
            logger.warning("Bundles command unavailable: %s", exc)
            return f"Bundles subsystem unavailable: {exc}"

        bundles = list_bundles()
        if not bundles:
            return (
                "No skill bundles installed.\n"
                "Create one on the host with:\n"
                "  `hermes bundles create <name> --skill <s1> --skill <s2>`\n"
                f"Directory: `{_bundles_dir()}`"
            )

        lines = [f"**Skill Bundles** ({len(bundles)} installed):", ""]
        for info in bundles:
            skill_count = len(info.get("skills", []))
            desc = info.get("description") or f"Load {skill_count} skills"
            lines.append(
                f"• `/{info['slug']}` — {desc} _({skill_count} skills)_"
            )
            for s in info.get("skills", []):
                lines.append(f"    · {s}")
        lines.append("")
        lines.append("Invoke a bundle with `/<slug>` to load all its skills.")
        return "\n".join(lines)

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

    async def _maybe_confirm_destructive_slash(
        self,
        *,
        event: MessageEvent,
        command: str,
        title: str,
        detail: str,
        execute,
    ) -> Union[str, "EphemeralReply", None]:
        """Gate a destructive session slash command (/new, /reset, /undo).

        ``execute`` is an async callable ``execute() -> str | EphemeralReply``
        that performs the destructive action.  If the
        ``approvals.destructive_slash_confirm`` config gate is off, ``execute``
        runs immediately (returning its result).  Otherwise this routes
        through ``_request_slash_confirm`` — native yes/no buttons on
        Telegram/Discord/Slack, text fallback elsewhere.

        Three-option resolution:

          - ``once``  — run ``execute`` and return its result
          - ``always`` — persist ``approvals.destructive_slash_confirm: false``,
                        then run ``execute``
          - ``cancel`` — return a "cancelled" message; do not run ``execute``
        """
        # Gate check.
        confirm_required = True
        try:
            cfg = self._read_user_config()
            approvals = cfg.get("approvals") if isinstance(cfg, dict) else None
            if isinstance(approvals, dict):
                confirm_required = bool(approvals.get("destructive_slash_confirm", True))
        except Exception:
            pass

        if not confirm_required:
            return await execute()

        session_key = self._session_key_for_source(event.source)

        async def _on_confirm(choice: str):
            if choice == "cancel":
                return f"🟡 /{command} cancelled. Conversation unchanged."
            if choice == "always":
                try:
                    from cli import save_config_value
                    save_config_value("approvals.destructive_slash_confirm", False)
                    logger.info(
                        "User opted out of destructive slash confirm (session=%s)",
                        session_key,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to persist destructive_slash_confirm=false: %s", exc,
                    )
            result = await execute()
            if choice == "always":
                note = (
                    "\n\nℹ️ Future /clear, /new, /reset, and /undo will run "
                    "without confirmation. Re-enable via "
                    "`approvals.destructive_slash_confirm: true` in config.yaml."
                )
                if isinstance(result, str):
                    return result + note
                # EphemeralReply or other — leave untouched; the opt-out note
                # would otherwise mangle structured replies.  The persist itself
                # already happened above; user gets the same UX next time.
                return result
            return result

        prompt_message = (
            f"⚠️ **Confirm /{command}**\n\n"
            f"{detail}\n\n"
            "Choose:\n"
            "• **Approve Once** — proceed this time only\n"
            "• **Always Approve** — proceed and silence this prompt permanently\n"
            "• **Cancel** — keep current conversation\n\n"
            "_Text fallback: reply `/approve`, `/always`, or `/cancel`._"
        )
        return await self._request_slash_confirm(
            event=event,
            command=command,
            title=title,
            message=prompt_message,
            handler=_on_confirm,
        )

    async def _request_slash_confirm(
        self,
        *,
        event: MessageEvent,
        command: str,
        title: str,
        message: str,
        handler,
    ) -> Optional[str]:
        """Ask the user to confirm an expensive slash command.

        ``handler`` is an async callable ``handler(choice: str) -> str``
        where ``choice`` is ``"once"``, ``"always"``, or ``"cancel"``.
        The handler runs on the event loop when the user responds; its
        return value is sent back as a gateway message.

        Returns a short acknowledgment string to send immediately (before
        the user's response).  If buttons rendered successfully the ack
        is ``None`` (buttons are self-explanatory); if we fell back to
        text the message itself IS the ack.
        """
        from tools import slash_confirm as _slash_confirm_mod

        source = event.source
        session_key = self._session_key_for_source(source)
        # Bare-runner test harnesses (object.__new__(GatewayRunner)) skip
        # __init__ and don't have the counter attribute — fall back to a
        # local counter so tests don't AttributeError.  Real runs always
        # have the instance attribute.
        counter = getattr(self, "_slash_confirm_counter", None)
        if counter is None:
            import itertools as _itertools
            counter = _itertools.count(1)
            self._slash_confirm_counter = counter
        confirm_id = f"{next(counter)}"

        # Register the pending confirm FIRST so a super-fast button click
        # cannot race the send_slash_confirm return.
        _slash_confirm_mod.register(session_key, confirm_id, command, handler)

        adapter = self.adapters.get(source.platform)
        metadata = self._thread_metadata_for_source(source, self._reply_anchor_for_event(event))

        used_buttons = False
        if adapter is not None:
            try:
                button_result = await adapter.send_slash_confirm(
                    chat_id=source.chat_id,
                    title=title,
                    message=message,
                    session_key=session_key,
                    confirm_id=confirm_id,
                    metadata=metadata,
                )
                if button_result and getattr(button_result, "success", False):
                    used_buttons = True
            except Exception as exc:
                logger.debug(
                    "send_slash_confirm failed for %s on %s: %s",
                    command, source.platform, exc,
                )

        if used_buttons:
            # Buttons rendered — no redundant text ack.
            return None
        # Text fallback — return the prompt message as the direct reply.
        return message

    def _read_user_config(self) -> Dict[str, Any]:
        """Read the user's raw config.yaml (cached) for gate lookups.

        Used by slash-confirm gates that must reflect on-disk state changes
        (e.g. a prior "Always Approve" click) without a gateway restart.
        """
        return session_management.read_user_config()
    def _thread_metadata_for_source(
        self,
        source,
        reply_to_message_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build the metadata dict platforms need for thread-aware replies."""
        return self._thread_metadata_for_target(
            getattr(source, "platform", None),
            getattr(source, "chat_id", None),
            getattr(source, "thread_id", None),
            chat_type=getattr(source, "chat_type", None),
            reply_to_message_id=reply_to_message_id or getattr(source, "message_id", None),
        )

    def _thread_metadata_for_target(
        self,
        platform: Optional[Platform],
        chat_id: Optional[str],
        thread_id: Optional[str],
        *,
        chat_type: Optional[str] = None,
        reply_to_message_id: Optional[str] = None,
        adapter: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build thread metadata for synthetic sends that only have routing state."""
        if thread_id is None:
            return None
        metadata: Dict[str, Any] = {"thread_id": thread_id}
        if self._is_telegram_dm_topic_target(
            platform,
            chat_id,
            thread_id,
            chat_type=chat_type,
            adapter=adapter,
        ):
            metadata["telegram_dm_topic_reply_fallback"] = True
            # Telegram DM topic lanes need direct_messages_topic_id in metadata
            # so synthetic/queued messages (goal continuations, status notices)
            # route to the correct topic even when reply anchor is unavailable.
            tid = str(thread_id)
            if tid and tid not in {"", "1"}:
                metadata["direct_messages_topic_id"] = tid
            if reply_to_message_id is not None:
                metadata["telegram_reply_to_message_id"] = str(reply_to_message_id)
        return metadata

    @staticmethod
    def _is_telegram_dm_topic_target(
        platform: Optional[Platform],
        chat_id: Optional[str],
        thread_id: Optional[str],
        *,
        chat_type: Optional[str] = None,
        adapter: Optional[Any] = None,
    ) -> bool:
        """Return True when a target is a Telegram private DM topic lane."""
        if platform != Platform.TELEGRAM or thread_id is None:
            return False
        if chat_type == "dm":
            return True
        # Inspect operator-declared DM topics via the adapter's lookup. Resolve
        # the method on the CLASS, not the instance: getattr() on a MagicMock
        # auto-creates a callable child for any attribute, so an instance-level
        # lookup would report a DM topic for every test double. Only a
        # dict-shaped return counts as an operator-declared topic — a bare
        # MagicMock or other sentinel must not. Mirrors the guard in
        # _rename_telegram_topic_for_session_title.
        if adapter is not None and chat_id:
            get_dm_topic_info = getattr(type(adapter), "_get_dm_topic_info", None)
            if callable(get_dm_topic_info):
                try:
                    topic_info = get_dm_topic_info(adapter, str(chat_id), str(thread_id))
                except Exception:
                    logger.debug("Failed to inspect Telegram DM topic metadata", exc_info=True)
                else:
                    return isinstance(topic_info, dict)
        return False

    @staticmethod
    def _reply_anchor_for_event(event: MessageEvent) -> Optional[str]:
        """Return the platform-specific reply anchor for GatewayRunner sends."""
        return _reply_anchor_for_event(event)


    # ------------------------------------------------------------------
    # /approve & /deny — explicit dangerous-command approval
    # ------------------------------------------------------------------

    _APPROVAL_TIMEOUT_SECONDS = 300  # 5 minutes

    async def _handle_approve_command(self, event: MessageEvent) -> Optional[str]:
        """Handle /approve command — unblock waiting agent thread(s).

        The agent thread(s) are blocked inside tools/approval.py waiting for
        the user to respond.  This handler signals the event so the agent
        resumes and the terminal_tool executes the command inline — the same
        flow as the CLI's synchronous input() approval.

        Supports multiple concurrent approvals (parallel subagents,
        execute_code).  ``/approve`` resolves the oldest pending command;
        ``/approve all`` resolves every pending command at once.

        Usage:
            /approve              — approve oldest pending command once
            /approve all          — approve ALL pending commands at once
            /approve session      — approve oldest + remember for session
            /approve all session  — approve all + remember for session
            /approve always       — approve oldest + remember permanently
            /approve all always   — approve all + remember permanently
        """
        source = event.source
        session_key = self._session_key_for_source(source)

        from tools.approval import (
            resolve_gateway_approval, has_blocking_approval,
        )

        if not has_blocking_approval(session_key):
            if session_key in self._pending_approvals:
                self._pending_approvals.pop(session_key)
                return t("gateway.approval_expired")
            return t("gateway.approve.no_pending")

        # Parse args: support "all", "all session", "all always", "session", "always"
        args = event.get_command_args().strip().lower().split()
        resolve_all = "all" in args
        remaining = [a for a in args if a != "all"]

        if any(a in {"always", "permanent", "permanently"} for a in remaining):
            choice = "always"
        elif any(a in {"session", "ses"} for a in remaining):
            choice = "session"
        else:
            choice = "once"

        count = resolve_gateway_approval(session_key, choice, resolve_all=resolve_all)
        if not count:
            return t("gateway.approve.no_pending")

        # Resume typing indicator — agent is about to continue processing.
        _adapter = self.adapters.get(source.platform)
        if _adapter:
            _adapter.resume_typing_for_chat(source.chat_id)

        logger.info("User approved %d dangerous command(s) via /approve (%s)", count, choice)
        plural = "plural" if count > 1 else "singular"
        return t(f"gateway.approve.{choice}_{plural}", count=count)

    async def _handle_deny_command(self, event: MessageEvent) -> str:
        """Handle /deny command — reject pending dangerous command(s).

        Signals blocked agent thread(s) with a 'deny' result so they receive
        a definitive BLOCKED message, same as the CLI deny flow.

        ``/deny`` denies the oldest; ``/deny all`` denies everything.
        """
        source = event.source
        session_key = self._session_key_for_source(source)

        from tools.approval import (
            resolve_gateway_approval, has_blocking_approval,
        )

        if not has_blocking_approval(session_key):
            if session_key in self._pending_approvals:
                self._pending_approvals.pop(session_key)
                return t("gateway.deny.stale")
            return t("gateway.deny.no_pending")

        args = event.get_command_args().strip().lower()
        resolve_all = "all" in args

        count = resolve_gateway_approval(session_key, "deny", resolve_all=resolve_all)
        if not count:
            return t("gateway.deny.no_pending")

        # Resume typing indicator — agent continues (with BLOCKED result).
        _adapter = self.adapters.get(source.platform)
        if _adapter:
            _adapter.resume_typing_for_chat(source.chat_id)

        logger.info("User denied %d dangerous command(s) via /deny", count)
        if count > 1:
            return t("gateway.deny.denied_plural", count=count)
        return t("gateway.deny.denied_singular")

    # Platforms where /update is allowed.  ACP, API server, and webhooks are
    # programmatic interfaces that should not trigger system updates.
    _UPDATE_ALLOWED_PLATFORMS = frozenset({
        Platform.TELEGRAM, Platform.DISCORD, Platform.SLACK, Platform.WHATSAPP,
        Platform.SIGNAL, Platform.MATTERMOST, Platform.MATRIX,
        Platform.HOMEASSISTANT, Platform.EMAIL, Platform.SMS, Platform.DINGTALK,
        Platform.FEISHU, Platform.WECOM, Platform.WECOM_CALLBACK, Platform.WEIXIN, Platform.BLUEBUBBLES, Platform.QQBOT, Platform.LOCAL,
    })

    async def _handle_debug_command(self, event: MessageEvent) -> str:
        """Handle /debug — upload debug report (summary only) and return paste URLs.

        Gateway uploads ONLY the summary report (system info + log tails),
        NOT full log files, to protect conversation privacy.  Users who need
        full log uploads should use ``hermes debug share`` from the CLI.
        """
        import asyncio
        from hermes_cli.debug import (
            _capture_dump, collect_debug_report,
            upload_to_pastebin, _schedule_auto_delete,
            _GATEWAY_PRIVACY_NOTICE, _best_effort_sweep_expired_pastes,
        )

        loop = asyncio.get_running_loop()

        # Run blocking I/O (dump capture, log reads, uploads) in a thread.
        def _collect_and_upload():
            _best_effort_sweep_expired_pastes()
            dump_text = _capture_dump()
            report = collect_debug_report(log_lines=200, dump_text=dump_text)

            urls = {}
            try:
                urls["Report"] = upload_to_pastebin(report)
            except Exception as exc:
                return t("gateway.debug.upload_failed", error=exc)

            # Schedule auto-deletion after 6 hours
            _schedule_auto_delete(list(urls.values()))

            lines = [_GATEWAY_PRIVACY_NOTICE, "", t("gateway.debug.header"), ""]
            label_width = max(len(k) for k in urls)
            for label, url in urls.items():
                lines.append(f"`{label:<{label_width}}`  {url}")

            lines.append("")
            lines.append(t("gateway.debug.auto_delete"))
            lines.append(t("gateway.debug.full_logs_hint"))
            lines.append(t("gateway.debug.share_hint"))
            return "\n".join(lines)

        return await loop.run_in_executor(None, _collect_and_upload)

    async def _handle_update_command(self, event: MessageEvent) -> str:
        """Handle /update command — update Hermes Agent to the latest version.

        Delegates to gateway.command_handlers.handle_update_command.
        """
        return await command_handlers.handle_update_command(self, event)
    def _schedule_update_notification_watch(self) -> None:
        """Ensure a background task is watching for update completion."""
        existing_task = getattr(self, "_update_notification_task", None)
        if existing_task and not existing_task.done():
            return

        try:
            self._update_notification_task = asyncio.create_task(
                self._watch_update_progress()
            )
        except RuntimeError:
            logger.debug("Skipping update notification watcher: no running event loop")

    async def _watch_update_progress(
        self,
        poll_interval: float = 2.0,
        stream_interval: float = 4.0,
        timeout: float = 1800.0,
    ) -> None:
        """Watch ``hermes update --gateway``, streaming output + forwarding prompts.

        Polls ``.update_output.txt`` for new content and sends chunks to the
        user periodically.  Detects ``.update_prompt.json`` (written by the
        update process when it needs user input) and forwards the prompt to
        the messenger.  The user's next message is intercepted by
        ``_handle_message`` and written to ``.update_response``.
        """
        pending_path = _hermes_home / ".update_pending.json"
        claimed_path = _hermes_home / ".update_pending.claimed.json"
        output_path = _hermes_home / ".update_output.txt"
        exit_code_path = _hermes_home / ".update_exit_code"
        prompt_path = _hermes_home / ".update_prompt.json"

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        # Resolve the adapter and chat_id for sending messages
        adapter = None
        chat_id = None
        session_key = None
        metadata = None
        for path in (claimed_path, pending_path):
            if path.exists():
                try:
                    pending = json.loads(path.read_text())
                    platform_str = pending.get("platform")
                    chat_id = pending.get("chat_id")
                    chat_type = pending.get("chat_type")
                    session_key = pending.get("session_key")
                    thread_id = pending.get("thread_id")
                    message_id = pending.get("message_id")
                    if platform_str and chat_id:
                        platform = Platform(platform_str)
                        adapter = self.adapters.get(platform)
                        metadata = self._thread_metadata_for_target(
                            platform,
                            chat_id,
                            thread_id,
                            chat_type=chat_type,
                            reply_to_message_id=message_id,
                            adapter=adapter,
                        )
                        # Fallback session key if not stored (old pending files)
                        if not session_key:
                            session_key = f"{platform_str}:{chat_id}"
                    break
                except Exception:
                    pass

        if not adapter or not chat_id:
            logger.warning("Update watcher: cannot resolve adapter/chat_id, falling back to completion-only")
            # Fall back to old behavior: wait for exit code and send final notification
            while (pending_path.exists() or claimed_path.exists()) and loop.time() < deadline:
                if exit_code_path.exists():
                    await self._send_update_notification()
                    return
                await asyncio.sleep(poll_interval)
            if (pending_path.exists() or claimed_path.exists()) and not exit_code_path.exists():
                exit_code_path.write_text("124")
                await self._send_update_notification()
            return

        def _strip_ansi(text: str) -> str:
            return re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text)

        bytes_sent = 0
        last_stream_time = loop.time()
        buffer = ""

        async def _flush_buffer() -> None:
            """Send buffered output to the user."""
            nonlocal buffer, last_stream_time
            if not buffer.strip():
                buffer = ""
                return
            # Chunk to fit message limits (Telegram: 4096, others: generous)
            clean = _strip_ansi(buffer).strip()
            buffer = ""
            last_stream_time = loop.time()
            if not clean:
                return
            # Split into chunks if too long
            max_chunk = 3500
            chunks = [clean[i:i + max_chunk] for i in range(0, len(clean), max_chunk)]
            for chunk in chunks:
                try:
                    await adapter.send(chat_id, f"```\n{chunk}\n```", metadata=metadata)
                except Exception as e:
                    logger.debug("Update stream send failed: %s", e)

        while loop.time() < deadline:
            # Check for completion
            if exit_code_path.exists():
                # Read any remaining output
                if output_path.exists():
                    try:
                        content = output_path.read_text()
                        if len(content) > bytes_sent:
                            buffer += content[bytes_sent:]
                            bytes_sent = len(content)
                    except OSError:
                        pass
                await _flush_buffer()

                # Send final status
                try:
                    exit_code_raw = exit_code_path.read_text().strip() or "1"
                    exit_code = int(exit_code_raw)
                    if exit_code == 0:
                        await adapter.send(chat_id, "✅ Hermes update finished.", metadata=metadata)
                    else:
                        await adapter.send(
                            chat_id,
                            "❌ Hermes update failed (exit code {}).".format(exit_code),
                            metadata=metadata,
                        )
                    logger.info("Update finished (exit=%s), notified %s", exit_code, session_key)
                except Exception as e:
                    logger.warning("Update final notification failed: %s", e)

                # Cleanup
                for p in (pending_path, claimed_path, output_path,
                          exit_code_path, prompt_path):
                    p.unlink(missing_ok=True)
                (_hermes_home / ".update_response").unlink(missing_ok=True)
                self._update_prompt_pending.pop(session_key, None)
                return

            # Check for new output
            if output_path.exists():
                try:
                    content = output_path.read_text()
                    if len(content) > bytes_sent:
                        buffer += content[bytes_sent:]
                        bytes_sent = len(content)
                except OSError:
                    pass

            # Flush buffer periodically
            if buffer.strip() and (loop.time() - last_stream_time) >= stream_interval:
                await _flush_buffer()

            # Check for prompts — only forward if we haven't already sent
            # one that's still awaiting a response.  Without this guard the
            # watcher would re-read the same .update_prompt.json every poll
            # cycle and spam the user with duplicate prompt messages.
            if (prompt_path.exists() and session_key
                    and not self._update_prompt_pending.get(session_key)):
                try:
                    prompt_data = json.loads(prompt_path.read_text())
                    prompt_text = prompt_data.get("prompt", "")
                    default = prompt_data.get("default", "")
                    if prompt_text:
                        # Flush any buffered output first so the user sees
                        # context before the prompt
                        await _flush_buffer()
                        # Try platform-native buttons first (Discord, Telegram)
                        sent_buttons = False
                        if getattr(type(adapter), "send_update_prompt", None) is not None:
                            try:
                                await adapter.send_update_prompt(
                                    chat_id=chat_id,
                                    prompt=prompt_text,
                                    default=default,
                                    session_key=session_key,
                                    metadata=metadata,
                                )
                                sent_buttons = True
                            except Exception as btn_err:
                                logger.debug("Button-based update prompt failed: %s", btn_err)
                        if not sent_buttons:
                            default_hint = f" (default: {default})" if default else ""
                            await adapter.send(
                                chat_id,
                                f"⚕ **Update needs your input:**\n\n"
                                f"{prompt_text}{default_hint}\n\n"
                                f"Reply `/approve` (yes) or `/deny` (no), "
                                f"or type your answer directly.",
                                metadata=metadata,
                            )
                        # Keep the prompt marker on disk until the user
                        # answers. If the gateway restarts mid-prompt, the
                        # next watcher can recover by re-forwarding it from
                        # disk. Duplicate sends in the same process are
                        # still suppressed by _update_prompt_pending.
                        self._update_prompt_pending[session_key] = True
                        # .update_response to continue — it doesn't re-check
                        logger.info("Forwarded update prompt to %s: %s", session_key, prompt_text[:80])
                except (json.JSONDecodeError, OSError) as e:
                    logger.debug("Failed to read update prompt: %s", e)

            await asyncio.sleep(poll_interval)

        # Timeout
        if not exit_code_path.exists():
            logger.warning("Update watcher timed out after %.0fs", timeout)
            exit_code_path.write_text("124")
            await _flush_buffer()
            try:
                await adapter.send(
                    chat_id,
                    "❌ Hermes update timed out after 30 minutes.",
                    metadata=metadata,
                )
            except Exception:
                pass
            for p in (pending_path, claimed_path, output_path,
                      exit_code_path, prompt_path):
                p.unlink(missing_ok=True)
            (_hermes_home / ".update_response").unlink(missing_ok=True)
            self._update_prompt_pending.pop(session_key, None)

    async def _send_update_notification(self) -> bool:
        """If an update finished, notify the user.

        Returns False when the update is still running so a caller can retry
        later. Returns True after a definitive send/skip decision.

        This is the legacy notification path used when the streaming watcher
        cannot resolve the adapter (e.g. after a gateway restart where the
        platform hasn't reconnected yet).
        """
        pending_path = _hermes_home / ".update_pending.json"
        claimed_path = _hermes_home / ".update_pending.claimed.json"
        output_path = _hermes_home / ".update_output.txt"
        exit_code_path = _hermes_home / ".update_exit_code"

        if not pending_path.exists() and not claimed_path.exists():
            return False

        cleanup = True
        active_pending_path = claimed_path
        try:
            if pending_path.exists():
                try:
                    pending_path.replace(claimed_path)
                except FileNotFoundError:
                    if not claimed_path.exists():
                        return True
            elif not claimed_path.exists():
                return True

            pending = json.loads(claimed_path.read_text())
            platform_str = pending.get("platform")
            chat_id = pending.get("chat_id")
            chat_type = pending.get("chat_type")
            thread_id = pending.get("thread_id")
            message_id = pending.get("message_id")

            if not exit_code_path.exists():
                logger.info("Update notification deferred: update still running")
                cleanup = False
                active_pending_path = pending_path
                claimed_path.replace(pending_path)
                return False

            exit_code_raw = exit_code_path.read_text().strip() or "1"
            exit_code = int(exit_code_raw)

            # Read the captured update output
            output = ""
            if output_path.exists():
                output = output_path.read_text()

            # Resolve adapter
            platform = Platform(platform_str)
            adapter = self.adapters.get(platform)

            if adapter and chat_id:
                metadata = self._thread_metadata_for_target(
                    platform,
                    chat_id,
                    thread_id,
                    chat_type=chat_type,
                    reply_to_message_id=message_id,
                    adapter=adapter,
                )
                # Strip ANSI escape codes for clean display
                output = re.sub(r'\x1b\[[0-9;]*m', '', output).strip()
                if output:
                    if len(output) > 3500:
                        output = "…" + output[-3500:]
                    if exit_code == 0:
                        msg = f"✅ Hermes update finished.\n\n```\n{output}\n```"
                    else:
                        msg = f"❌ Hermes update failed.\n\n```\n{output}\n```"
                elif exit_code == 0:
                    msg = "✅ Hermes update finished successfully."
                else:
                    msg = "❌ Hermes update failed. Check the gateway logs or run `hermes update` manually for details."
                await adapter.send(chat_id, msg, metadata=metadata)
                logger.info(
                    "Sent post-update notification to %s:%s (exit=%s)",
                    platform_str,
                    chat_id,
                    exit_code,
                )
        except Exception as e:
            logger.warning("Post-update notification failed: %s", e)
        finally:
            if cleanup:
                active_pending_path.unlink(missing_ok=True)
                claimed_path.unlink(missing_ok=True)
                output_path.unlink(missing_ok=True)
                exit_code_path.unlink(missing_ok=True)

        return True

    async def _send_restart_notification(self) -> Optional[tuple[str, str, Optional[str]]]:
        """Notify the chat that initiated /restart that the gateway is back."""
        notify_path = _hermes_home / ".restart_notify.json"
        if not notify_path.exists():
            return None

        try:
            data = json.loads(notify_path.read_text())
            platform_str = data.get("platform")
            chat_id = data.get("chat_id")
            chat_type = data.get("chat_type")
            thread_id = data.get("thread_id")
            message_id = data.get("message_id")

            if not platform_str or not chat_id:
                return None

            platform = Platform(platform_str)
            adapter = self.adapters.get(platform)
            if not adapter:
                logger.debug(
                    "Restart notification skipped: %s adapter not connected",
                    platform_str,
                )
                return None

            platform_cfg = self.config.platforms.get(platform)
            if platform_cfg is not None and not platform_cfg.gateway_restart_notification:
                logger.info(
                    "Restart notification suppressed: %s has gateway_restart_notification=false",
                    platform_str,
                )
                return None

            metadata = self._thread_metadata_for_target(
                platform,
                chat_id,
                thread_id,
                chat_type=chat_type,
                reply_to_message_id=message_id,
                adapter=adapter,
            )
            result = await adapter.send(
                str(chat_id),
                "♻ Gateway restarted successfully. Your session continues.",
                metadata=metadata,
            )
            # adapter.send() catches provider errors (e.g. "Chat not found")
            # and returns SendResult(success=False) rather than raising, so
            # we must inspect the result before claiming success — otherwise
            # the log line is misleading and hides real delivery failures.
            if result is not None and getattr(result, "success", True) is False:
                logger.warning(
                    "Restart notification to %s:%s was not delivered: %s",
                    platform_str,
                    chat_id,
                    getattr(result, "error", "send returned success=False"),
                )
                return None

            logger.info(
                "Sent restart notification to %s:%s",
                platform_str,
                chat_id,
            )
            return str(platform_str), str(chat_id), str(thread_id) if thread_id else None
        except Exception as e:
            logger.warning("Restart notification failed: %s", e)
            return None
        finally:
            notify_path.unlink(missing_ok=True)

    async def _send_home_channel_startup_notifications(
        self,
        *,
        skip_targets: Optional[set[tuple[str, str, Optional[str]]]] = None,
    ) -> set[tuple[str, str, Optional[str]]]:
        """Notify configured home channels that the gateway is back online.

        The notification is best-effort and sent once per connected platform
        home channel. ``skip_targets`` lets startup avoid duplicate messages
        when a more specific restart notification is queued for the same chat.
        """
        delivered: set[tuple[str, str, Optional[str]]] = set()
        skipped = skip_targets or set()
        message = "♻️ Gateway online — Hermes is back and ready."

        for platform, adapter in self.adapters.items():
            home = self.config.get_home_channel(platform)
            if not home or not home.chat_id:
                continue

            platform_cfg = self.config.platforms.get(platform)
            if platform_cfg is not None and not platform_cfg.gateway_restart_notification:
                logger.info(
                    "Home-channel startup notification suppressed: %s has gateway_restart_notification=false",
                    platform.value,
                )
                continue

            target = (platform.value, str(home.chat_id), str(home.thread_id) if home.thread_id else None)
            if target in skipped or target in delivered:
                continue

            try:
                metadata = self._thread_metadata_for_target(
                    platform,
                    home.chat_id,
                    home.thread_id,
                    adapter=adapter,
                )
                if metadata:
                    result = await adapter.send(str(home.chat_id), message, metadata=metadata)
                else:
                    result = await adapter.send(str(home.chat_id), message)
                if result is not None and getattr(result, "success", True) is False:
                    logger.warning(
                        "Home-channel startup notification failed for %s:%s: %s",
                        platform.value,
                        home.chat_id,
                        getattr(result, "error", "send returned success=False"),
                    )
                    continue

                delivered.add(target)
                logger.info(
                    "Sent home-channel startup notification to %s:%s",
                    platform.value,
                    home.chat_id,
                )
            except Exception as exc:
                logger.warning(
                    "Home-channel startup notification failed for %s:%s: %s",
                    platform.value,
                    home.chat_id,
                    exc,
                )

        return delivered

    def _set_session_env(self, context: SessionContext) -> list:
        """Set session context variables for the current async task.

        Uses ``contextvars`` instead of ``os.environ`` so that concurrent
        gateway messages cannot overwrite each other's session state.

        Returns a list of reset tokens; pass them to ``_clear_session_env``
        in a ``finally`` block.
        """
        return session_management.set_session_env(context)
    def _clear_session_env(self, tokens: list) -> None:
        """Restore session context variables to their pre-handler values."""
        return session_management.clear_session_env(tokens)
    async def _run_in_executor_with_context(self, func, *args):
        """Run blocking work in the thread pool while preserving session contextvars."""
        loop = asyncio.get_running_loop()
        ctx = copy_context()
        return await loop.run_in_executor(None, ctx.run, func, *args)

    def _decide_image_input_mode(self) -> str:
        """Resolve the image-input routing for the currently active model.

        Returns ``"native"`` (attach pixels on the user turn) or ``"text"``
        (pre-analyze with vision_analyze and prepend the description). See
        agent/image_routing.py for the full decision table.

        The active provider/model are read from config.yaml so the decision
        tracks ``/model`` switches automatically on the next message.
        """
        try:
            from agent.image_routing import decide_image_input_mode
            from agent.auxiliary_client import _read_main_model, _read_main_provider
            from hermes_cli.config import load_config

            cfg = load_config()
            provider = _read_main_provider()
            model = _read_main_model()
            return decide_image_input_mode(provider, model, cfg)
        except Exception as exc:
            logger.debug("image_routing: decision failed, falling back to text — %s", exc)
            return "text"

    async def _enrich_message_with_vision(
        self,
        user_text: str,
        image_paths: List[str],
    ) -> str:
        """
        Auto-analyze user-attached images with the vision tool and prepend
        the descriptions to the message text.

        Each image is analyzed with a general-purpose prompt.  The resulting
        description *and* the local cache path are injected so the model can:
          1. Immediately understand what the user sent (no extra tool call).
          2. Re-examine the image with vision_analyze if it needs more detail.

        Args:
            user_text:   The user's original caption / message text.
            image_paths: List of local file paths to cached images.

        Returns:
            The enriched message string with vision descriptions prepended.
        """
        from tools.vision_tools import vision_analyze_tool
        from agent.memory_manager import sanitize_context

        analysis_prompt = (
            "Describe everything visible in this image in thorough detail. "
            "Include any text, code, data, objects, people, layout, colors, "
            "and any other notable visual information."
        )

        enriched_parts = []
        for path in image_paths:
            try:
                logger.debug("Auto-analyzing user image: %s", path)
                result_json = await vision_analyze_tool(
                    image_url=path,
                    user_prompt=analysis_prompt,
                )
                result = json.loads(result_json)
                if result.get("success"):
                    description = result.get("analysis", "")
                    description = sanitize_context(description)
                    enriched_parts.append(
                        f"[The user sent an image~ Here's what I can see:\n{description}]\n"
                        f"[If you need a closer look, use vision_analyze with "
                        f"image_url: {path} ~]"
                    )
                else:
                    enriched_parts.append(
                        "[The user sent an image but I couldn't quite see it "
                        "this time (>_<) You can try looking at it yourself "
                        f"with vision_analyze using image_url: {path}]"
                    )
            except Exception as e:
                logger.error("Vision auto-analysis error: %s", e)
                enriched_parts.append(
                    f"[The user sent an image but something went wrong when I "
                    f"tried to look at it~ You can try examining it yourself "
                    f"with vision_analyze using image_url: {path}]"
                )

        # Combine: vision descriptions first, then the user's original text
        if enriched_parts:
            prefix = "\n\n".join(enriched_parts)
            if user_text:
                return f"{prefix}\n\n{user_text}"
            return prefix
        return user_text

    async def _enrich_message_with_transcription(
        self,
        user_text: str,
        audio_paths: List[str],
    ) -> str:
        """
        Auto-transcribe user voice/audio messages using the configured STT provider
        and prepend the transcript to the message text.

        Args:
            user_text:   The user's original caption / message text.
            audio_paths: List of local file paths to cached audio files.

        Returns:
            The enriched message string with transcriptions prepended.
        """
        if not getattr(self.config, "stt_enabled", True):
            notes = []
            for path in audio_paths:
                abs_path = os.path.abspath(path)
                duration_str = await _probe_audio_duration(abs_path)
                if duration_str:
                    notes.append(
                        f"[The user sent a voice message: {abs_path} (duration: {duration_str})]"
                    )
                else:
                    notes.append(f"[The user sent a voice message: {abs_path}]")
            if not notes:
                return user_text
            prefix = "\n\n".join(notes)
            _placeholder = "(The user sent a message with no text content)"
            if user_text and user_text.strip() == _placeholder:
                return prefix
            if user_text:
                return f"{prefix}\n\n{user_text}"
            return prefix

        from tools.transcription_tools import transcribe_audio

        enriched_parts = []
        for path in audio_paths:
            try:
                logger.debug("Transcribing user voice: %s", path)
                result = await asyncio.to_thread(transcribe_audio, path)
                if result["success"]:
                    transcript = result["transcript"]
                    enriched_parts.append(
                        f'[The user sent a voice message~ '
                        f'Here\'s what they said: "{transcript}"]'
                    )
                else:
                    error = result.get("error", "unknown error")
                    if (
                        "No STT provider" in error
                        or error.startswith("Neither VOICE_TOOLS_OPENAI_KEY nor OPENAI_API_KEY is set")
                    ):
                        _no_stt_note = (
                            "[The user sent a voice message but I can't listen "
                            "to it right now — no STT provider is configured. "
                            "A direct message has already been sent to the user "
                            "with setup instructions."
                        )
                        if self._has_setup_skill():
                            _no_stt_note += (
                                " You have a skill called hermes-agent-setup "
                                "that can help users configure Hermes features "
                                "including voice, tools, and more."
                            )
                        _no_stt_note += "]"
                        enriched_parts.append(_no_stt_note)
                    else:
                        enriched_parts.append(
                            "[The user sent a voice message but I had trouble "
                            f"transcribing it~ ({error})]"
                        )
            except Exception as e:
                logger.error("Transcription error: %s", e)
                enriched_parts.append(
                    "[The user sent a voice message but something went wrong "
                    "when I tried to listen to it~ Let them know!]"
                )

        if enriched_parts:
            prefix = "\n\n".join(enriched_parts)
            # Strip the empty-content placeholder from the Discord adapter
            # when we successfully transcribed the audio — it's redundant.
            _placeholder = "(The user sent a message with no text content)"
            if user_text and user_text.strip() == _placeholder:
                return prefix
            if user_text:
                return f"{prefix}\n\n{user_text}"
            return prefix
        return user_text

    def _build_process_event_source(self, evt: dict):
        """Resolve the canonical source for a synthetic background-process event.

        Prefer the persisted session-store origin for the event's session key.
        Falling back to the currently active foreground event is what causes
        cross-topic bleed, so don't do that.
        """
        from gateway.session import SessionSource

        session_key = str(evt.get("session_key") or "").strip()
        derived_platform = ""
        derived_chat_type = ""
        derived_chat_id = ""

        if session_key:
            try:
                self.session_store._ensure_loaded()
                entry = self.session_store._entries.get(session_key)
                if entry and getattr(entry, "origin", None):
                    return entry.origin
            except Exception as exc:
                logger.debug(
                    "Synthetic process-event session-store lookup failed for %s: %s",
                    session_key,
                    exc,
                )

            cached_source = self._get_cached_session_source(session_key)
            if cached_source is not None:
                return cached_source

            _parsed = _parse_session_key(session_key)
            if _parsed:
                derived_platform = _parsed["platform"]
                derived_chat_type = _parsed["chat_type"]
                derived_chat_id = _parsed["chat_id"]

        platform_name = str(evt.get("platform") or derived_platform or "").strip().lower()
        chat_type = str(evt.get("chat_type") or derived_chat_type or "").strip().lower()
        chat_id = str(evt.get("chat_id") or derived_chat_id or "").strip()
        if not platform_name or not chat_type or not chat_id:
            return None

        try:
            platform = Platform(platform_name)
            # Reject arbitrary strings that create dynamic pseudo-members.
            # Built-in platforms are always valid; plugin platforms must be
            # registered in the platform registry.
            if platform.value not in _BUILTIN_PLATFORM_VALUES:
                try:
                    from gateway.platform_registry import platform_registry
                    if not platform_registry.is_registered(platform.value):
                        raise ValueError(platform_name)
                except Exception:
                    raise ValueError(platform_name)
        except Exception:
            logger.warning(
                "Synthetic process event has invalid platform metadata: %r",
                platform_name,
            )
            return None

        return SessionSource(
            platform=platform,
            chat_id=chat_id,
            chat_type=chat_type,
            thread_id=str(evt.get("thread_id") or "").strip() or None,
            user_id=str(evt.get("user_id") or "").strip() or None,
            user_name=str(evt.get("user_name") or "").strip() or None,
        )

    async def _inject_watch_notification(self, synth_text: str, evt: dict) -> None:
        """Inject a watch-pattern notification as a synthetic message event.

        Routing must come from the queued watch event itself, not from whatever
        foreground message happened to be active when the queue was drained.
        """
        source = self._build_process_event_source(evt)
        if not source:
            logger.warning(
                "Dropping watch notification with no routing metadata for process %s",
                evt.get("session_id", "unknown"),
            )
            return
        platform_name = source.platform.value if hasattr(source.platform, "value") else str(source.platform)
        adapter = None
        for p, a in self.adapters.items():
            if p.value == platform_name:
                adapter = a
                break
        if not adapter:
            return
        try:
            synth_event = MessageEvent(
                text=synth_text,
                message_type=MessageType.TEXT,
                source=source,
                internal=True,
                message_id=str(evt.get("message_id") or "").strip() or None,
            )
            logger.info(
                "Watch pattern notification — injecting for %s chat=%s thread=%s",
                platform_name,
                source.chat_id,
                source.thread_id,
            )
            await adapter.handle_message(synth_event)
        except Exception as e:
            logger.error("Watch notification injection error: %s", e)

    async def _run_process_watcher(self, watcher: dict) -> None:
        """
        Periodically check a background process and push updates to the user.

        Runs as an asyncio task. Stays silent when nothing changed.
        Auto-removes when the process exits or is killed.

        Notification mode (from ``display.background_process_notifications``):
          - ``all``    — running-output updates + final message
          - ``result`` — final completion message only
          - ``error``  — final message only when exit code != 0
          - ``off``    — no messages at all
        """
        from tools.process_registry import process_registry

        session_id = watcher["session_id"]
        interval = watcher["check_interval"]
        session_key = watcher.get("session_key", "")
        platform_name = watcher.get("platform", "")
        chat_id = watcher.get("chat_id", "")
        thread_id = watcher.get("thread_id", "")
        user_id = watcher.get("user_id", "")
        user_name = watcher.get("user_name", "")
        message_id = str(watcher.get("message_id") or "").strip() or None
        agent_notify = watcher.get("notify_on_complete", False)
        notify_mode = self._load_background_notifications_mode()

        logger.debug("Process watcher started: %s (every %ss, notify=%s, agent_notify=%s)",
                      session_id, interval, notify_mode, agent_notify)

        if notify_mode == "off" and not agent_notify:
            # Still wait for the process to exit so we can log it, but don't
            # push any messages to the user.
            while True:
                await asyncio.sleep(interval)
                session = process_registry.get(session_id)
                if session is None or session.exited:
                    break
            logger.debug("Process watcher ended (silent): %s", session_id)
            return

        last_output_len = 0
        while True:
            await asyncio.sleep(interval)

            session = process_registry.get(session_id)
            if session is None:
                break

            current_output_len = len(session.output_buffer)
            has_new_output = current_output_len > last_output_len
            last_output_len = current_output_len

            if session.exited:
                # --- Agent-triggered completion: inject synthetic message ---
                # Skip if the agent already consumed the result via wait/poll/log
                from tools.process_registry import process_registry as _pr_check
                if agent_notify and not _pr_check.is_completion_consumed(session_id):
                    from tools.ansi_strip import strip_ansi
                    _raw = strip_ansi(session.output_buffer) if session.output_buffer else ""
                    # Truncate at line boundaries so notifications never start
                    # mid-line (fixes #23284). Keep the last ~2000 chars but
                    # snap to the nearest preceding newline, then prepend a
                    # truncation marker when output was cut.
                    _LIMIT = 2000
                    if len(_raw) > _LIMIT:
                        _tail = _raw[-_LIMIT:]
                        _nl = _tail.find("\n")
                        _tail = _tail[_nl + 1:] if _nl != -1 else _tail
                        _out = f"[… output truncated — showing last {len(_tail)} chars]\n{_tail}"
                    else:
                        _out = _raw
                    synth_text = (
                        f"[IMPORTANT: Background process {session_id} completed "
                        f"(exit code {session.exit_code}).\n"
                        f"Command: {session.command}\n"
                        f"Output:\n{_out}]"
                    )
                    source = self._build_process_event_source({
                        "session_id": session_id,
                        "session_key": session_key,
                        "platform": platform_name,
                        "chat_id": chat_id,
                        "thread_id": thread_id,
                        "user_id": user_id,
                        "user_name": user_name,
                    })
                    if not source:
                        logger.warning(
                            "Dropping completion notification with no routing metadata for process %s",
                            session_id,
                        )
                        break

                    adapter = None
                    for p, a in self.adapters.items():
                        if p == source.platform:
                            adapter = a
                            break
                    if adapter and source.chat_id:
                        try:
                            synth_event = MessageEvent(
                                text=synth_text,
                                message_type=MessageType.TEXT,
                                source=source,
                                internal=True,
                                message_id=message_id,
                            )
                            logger.info(
                                "Process %s finished — injecting agent notification for session %s chat=%s thread=%s",
                                session_id,
                                session_key,
                                source.chat_id,
                                source.thread_id,
                            )
                            await adapter.handle_message(synth_event)
                        except Exception as e:
                            logger.error("Agent notify injection error: %s", e)
                    break

                # --- Normal text-only notification ---
                # Decide whether to notify based on mode
                should_notify = (
                    notify_mode in {"all", "result"}
                    or (notify_mode == "error" and session.exit_code not in {0, None})
                )
                if should_notify:
                    new_output = session.output_buffer[-1000:] if session.output_buffer else ""
                    message_text = (
                        f"[Background process {session_id} finished with exit code {session.exit_code}~ "
                        f"Here's the final output:\n{new_output}]"
                    )
                    adapter = None
                    for p, a in self.adapters.items():
                        if p.value == platform_name:
                            adapter = a
                            break
                    if adapter and chat_id:
                        try:
                            send_meta = {"thread_id": thread_id} if thread_id else None
                            await adapter.send(chat_id, message_text, metadata=send_meta)
                        except Exception as e:
                            logger.error("Watcher delivery error: %s", e)
                break

            elif has_new_output and notify_mode == "all" and not agent_notify:
                # New output available -- deliver status update (only in "all" mode)
                # Skip periodic updates for agent_notify watchers (they only care about completion)
                new_output = session.output_buffer[-500:] if session.output_buffer else ""
                message_text = (
                    f"[Background process {session_id} is still running~ "
                    f"New output:\n{new_output}]"
                )
                adapter = None
                for p, a in self.adapters.items():
                    if p.value == platform_name:
                        adapter = a
                        break
                if adapter and chat_id:
                    try:
                        send_meta = {"thread_id": thread_id} if thread_id else None
                        await adapter.send(chat_id, message_text, metadata=send_meta)
                    except Exception as e:
                        logger.error("Watcher delivery error: %s", e)

        logger.debug("Process watcher ended: %s", session_id)

    _MAX_INTERRUPT_DEPTH = 3  # Cap recursive interrupt handling (#816)

    # Config keys whose values MUST invalidate the gateway's cached agent
    # when they change.  The agent bakes these into its compressor / context
    # handling at construction time, so a mid-running-gateway config edit
    # would otherwise be silently ignored until the user triggers a
    # different cache eviction (model switch, /reset, etc.).
    #
    # Each entry is a tuple of (section, key) read from the raw config dict.
    # Add more here as new baked-at-construction config settings are added.
    _CACHE_BUSTING_CONFIG_KEYS: tuple = (
        ("model", "context_length"),
        ("model", "max_tokens"),
        ("compression", "enabled"),
        ("compression", "threshold"),
        ("compression", "target_ratio"),
        ("compression", "protect_last_n"),
        ("agent", "disabled_toolsets"),
        ("memory", "provider"),
    )

    _HONCHO_CACHE_BUSTING_KEYS = (
        "honcho.peer_name",
        "honcho.ai_peer",
        "honcho.pin_peer_name",
        "honcho.runtime_peer_prefix",
        "honcho.user_peer_aliases",
    )
    _HONCHO_CACHE_BUSTING_MEMO: dict[tuple[str, int | None], dict[str, Any]] = {}

    @classmethod
    def _empty_honcho_cache_busting_config(cls) -> dict[str, Any]:
        return {key: None for key in cls._HONCHO_CACHE_BUSTING_KEYS}

    @classmethod
    def _extract_honcho_cache_busting_config(cls) -> dict[str, Any]:
        """Extract Honcho identity keys, memoized by honcho.json mtime."""
        try:
            from plugins.memory.honcho.client import HonchoClientConfig, resolve_config_path

            path = resolve_config_path()
            try:
                mtime_ns = path.stat().st_mtime_ns
            except OSError:
                mtime_ns = None
            memo_key = (str(path), mtime_ns)
            cached = cls._HONCHO_CACHE_BUSTING_MEMO.get(memo_key)
            if cached is not None:
                return dict(cached)

            hcfg = HonchoClientConfig.from_global_config(config_path=path)
            aliases = hcfg.user_peer_aliases or {}
            values = {
                "honcho.peer_name": hcfg.peer_name,
                "honcho.ai_peer": hcfg.ai_peer,
                "honcho.pin_peer_name": bool(hcfg.pin_peer_name),
                "honcho.runtime_peer_prefix": hcfg.runtime_peer_prefix or "",
                "honcho.user_peer_aliases": sorted(aliases.items()) if isinstance(aliases, dict) else [],
            }
            cls._HONCHO_CACHE_BUSTING_MEMO = {memo_key: values}
            return dict(values)
        except Exception:
            return cls._empty_honcho_cache_busting_config()

    @classmethod
    def _extract_cache_busting_config(cls, user_config: dict | None) -> dict:
        """Pull values that must bust the cached agent.

        Returns a flat dict keyed by 'section.key'.  Missing config keys and
        non-dict sections yield None values, which still contribute to the
        signature (so 'absent' vs 'present-and-null' differ).

        The live tool registry generation is included too.  MCP reloads and
        dynamic MCP tool-list changes mutate the registry without necessarily
        changing config.yaml.  Cached AIAgent instances freeze their tool
        schemas at construction time, so a registry generation change must
        rebuild the agent before the next turn.
        """
        out: Dict[str, Any] = {}
        cfg = user_config if isinstance(user_config, dict) else {}
        for section, key in cls._CACHE_BUSTING_CONFIG_KEYS:
            section_val = cfg.get(section)
            if isinstance(section_val, dict):
                out[f"{section}.{key}"] = section_val.get(key)
            else:
                out[f"{section}.{key}"] = None
        try:
            from tools.registry import registry

            out["tools.registry_generation"] = getattr(registry, "_generation", None)
        except Exception:
            out["tools.registry_generation"] = None

        # Honcho identity-mapping keys live in honcho.json, not user_config.
        # Only read that file when Honcho is the active memory provider.
        provider = cfg_get(cfg, "memory", "provider")
        if isinstance(provider, str) and provider.lower() == "honcho":
            out.update(cls._extract_honcho_cache_busting_config())
        else:
            out.update(cls._empty_honcho_cache_busting_config())

        return out

    @staticmethod
    def _agent_config_signature(
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
        import hashlib, json as _j

        # Fingerprint the FULL credential string instead of using a short
        # prefix. OAuth/JWT-style tokens frequently share a common prefix
        # (e.g. "eyJhbGci"), which can cause false cache hits across auth
        # switches if only the first few characters are considered.
        _api_key = str(runtime.get("api_key", "") or "")
        _api_key_fingerprint = hashlib.sha256(_api_key.encode()).hexdigest() if _api_key else ""

        _cache_keys_sorted = sorted((cache_keys or {}).items())

        blob = _j.dumps(
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

    def _apply_session_model_override(
        self, session_key: str, model: str, runtime_kwargs: dict
    ) -> tuple:
        """Apply /model session overrides if present, returning (model, runtime_kwargs).

        The gateway /model command stores per-session overrides in
        ``_session_model_overrides``.  These must take precedence over
        config.yaml defaults so the switched model is actually used for
        subsequent messages.  Fields with ``None`` values are skipped so
        partial overrides don't clobber valid config defaults.
        """
        override = self._session_model_overrides.get(session_key)
        if not override:
            return model, runtime_kwargs
        model = override.get("model", model)
        for key in ("provider", "api_key", "base_url", "api_mode"):
            val = override.get(key)
            if val is not None:
                runtime_kwargs[key] = val
        return model, runtime_kwargs

    def _is_intentional_model_switch(self, session_key: str, agent_model: str) -> bool:
        """Return True if *agent_model* matches an active /model session override."""
        override = self._session_model_overrides.get(session_key)
        return override is not None and override.get("model") == agent_model

    def _release_running_agent_state(
        self,
        session_key: str,
        *,
        run_generation: Optional[int] = None,
    ) -> bool:
        """Pop ALL per-running-agent state entries for ``session_key``.

        Replaces ad-hoc ``del self._running_agents[key]`` calls scattered
        across the gateway.  Those sites had drifted: some popped only
        ``_running_agents``; some also ``_running_agents_ts``; only one
        path also cleared ``_busy_ack_ts``.  Each missed entry was a
        small, persistent leak — a (str_key → float) tuple per session
        per gateway lifetime.

        Use this at every site that ends a running turn, regardless of
        cause (normal completion, /stop, /reset, /resume, sentinel
        cleanup, stale-eviction).  Per-session state that PERSISTS
        across turns (``_session_model_overrides``, ``_voice_mode``,
        ``_pending_approvals``, ``_update_prompt_pending``) is NOT
        touched here — those have their own lifecycles.

        When ``run_generation`` is provided, only clear the slot if that
        generation is still current for the session.  This prevents an
        older async run whose generation was bumped by /stop or /new from
        clobbering a newer run's state during its own unwind.  Returns
        True when the slot was cleared, False when an ownership guard
        blocked it.
        """
        if not session_key:
            return False
        if run_generation is not None and not self._is_session_run_current(
            session_key, run_generation
        ):
            return False
        self._running_agents.pop(session_key, None)
        self._running_agents_ts.pop(session_key, None)
        if hasattr(self, "_busy_ack_ts"):
            self._busy_ack_ts.pop(session_key, None)
        return True

    def _clear_session_boundary_security_state(self, session_key: str) -> None:
        """Clear per-session control state that must not survive a boundary switch."""
        if not session_key:
            return

        pending_skills_reload_notes = getattr(
            self, "_pending_skills_reload_notes", None
        )
        if isinstance(pending_skills_reload_notes, dict):
            pending_skills_reload_notes.pop(session_key, None)

        pending_approvals = getattr(self, "_pending_approvals", None)
        if isinstance(pending_approvals, dict):
            pending_approvals.pop(session_key, None)

        update_prompt_pending = getattr(self, "_update_prompt_pending", None)
        if isinstance(update_prompt_pending, dict):
            update_prompt_pending.pop(session_key, None)

        try:
            from tools import slash_confirm as _slash_confirm_mod
        except Exception:
            _slash_confirm_mod = None
        if _slash_confirm_mod is not None:
            try:
                _slash_confirm_mod.clear(session_key)
            except Exception as e:
                logger.debug(
                    "Failed to clear slash-confirm state for session boundary %s: %s",
                    session_key,
                    e,
                )

        try:
            from tools.approval import clear_session as _clear_approval_session
        except Exception:
            return

        try:
            _clear_approval_session(session_key)
        except Exception as e:
            logger.debug(
                "Failed to clear approval state for session boundary %s: %s",
                session_key,
                e,
            )

    def _begin_session_run_generation(self, session_key: str) -> int:
        """Claim a fresh run generation token for ``session_key``.

        Every top-level gateway turn gets a monotonically increasing token.
        If a later command like /stop or /new invalidates that token while the
        old worker is still unwinding, the late result can be recognized and
        dropped instead of bleeding into the fresh session.
        """
        if not session_key:
            return 0
        generations = self.__dict__.get("_session_run_generation")
        if generations is None:
            generations = {}
            self._session_run_generation = generations
        next_generation = int(generations.get(session_key, 0)) + 1
        generations[session_key] = next_generation
        return next_generation

    def _invalidate_session_run_generation(self, session_key: str, *, reason: str = "") -> int:
        """Invalidate any in-flight run token for ``session_key``."""
        generation = self._begin_session_run_generation(session_key)
        if reason:
            logger.info(
                "Invalidated run generation for %s → %d (%s)",
                session_key,
                generation,
                reason,
            )
        return generation

    def _is_session_run_current(self, session_key: str, generation: int) -> bool:
        """Return True when ``generation`` is still current for ``session_key``."""
        if not session_key or generation is None:
            return True
        generations = self.__dict__.get("_session_run_generation") or {}
        return int(generations.get(session_key, 0)) == int(generation)

    def _bind_adapter_run_generation(
        self,
        adapter: Any,
        session_key: str,
        generation: int | None,
    ) -> None:
        """Bind a gateway run generation to the adapter's active-session event."""
        if not adapter or not session_key or generation is None:
            return
        try:
            interrupt_event = getattr(adapter, "_active_sessions", {}).get(session_key)
            if interrupt_event is not None:
                setattr(interrupt_event, "_hermes_run_generation", int(generation))
        except Exception:
            pass

    async def _interrupt_and_clear_session(
        self,
        session_key: str,
        source: SessionSource,
        *,
        interrupt_reason: str,
        invalidation_reason: str,
        release_running_state: bool = True,
    ) -> None:
        """Interrupt the current run and clear queued session state consistently."""
        if not session_key:
            return
        running_agent = self._running_agents.get(session_key)
        if running_agent and running_agent is not _AGENT_PENDING_SENTINEL:
            running_agent.interrupt(interrupt_reason)
        self._invalidate_session_run_generation(session_key, reason=invalidation_reason)
        adapter = self.adapters.get(source.platform)
        if adapter and hasattr(adapter, "interrupt_session_activity"):
            await adapter.interrupt_session_activity(session_key, source.chat_id)
        if adapter and hasattr(adapter, "get_pending_message"):
            adapter.get_pending_message(session_key)  # consume and discard
        self._pending_messages.pop(session_key, None)
        if release_running_state:
            self._release_running_agent_state(session_key)

    def _evict_cached_agent(self, session_key: str) -> None:
        """Remove a cached agent for a session (called on /new, /model, etc)."""
        _lock = getattr(self, "_agent_cache_lock", None)
        if _lock:
            with _lock:
                self._agent_cache.pop(session_key, None)

    @staticmethod
    def _init_cached_agent_for_turn(agent: Any, interrupt_depth: int) -> None:
        """Reset per-turn state on a cached agent before a new turn starts.

        Both _last_activity_ts and _last_activity_desc are only reset for
        fresh external turns (depth 0); they are semantically paired —
        desc describes the activity *at* ts, so updating one without the
        other would make get_activity_summary() misleading.
        For interrupt-recursive turns both are preserved so the inactivity
        watchdog can accumulate stuck-turn idle time and fire the 30-min
        timeout (#15654).  The depth-0 reset is still needed: a session
        idle for 29 min would otherwise trip the watchdog before the new
        turn makes its first API call (#9051).
        """
        if interrupt_depth == 0:
            agent._last_activity_ts = time.time()
            agent._last_activity_desc = "starting new turn (cached)"
        agent._api_call_count = 0

    def _release_evicted_agent_soft(self, agent: Any) -> None:
        """Soft cleanup for cache-evicted agents — preserves session tool state.

        Called from _enforce_agent_cache_cap and _sweep_idle_cached_agents.
        Distinct from _cleanup_agent_resources (full teardown) because a
        cache-evicted session may resume at any time — its terminal
        sandbox, browser daemon, and tracked bg processes must outlive
        the Python AIAgent instance so the next agent built for the
        same task_id inherits them.
        """
        if agent is None:
            return
        try:
            if hasattr(agent, "release_clients"):
                agent.release_clients()
            else:
                # Older agent instance (shouldn't happen in practice) —
                # fall back to the legacy full-close path.
                self._cleanup_agent_resources(agent)
        except Exception:
            pass

    def _enforce_agent_cache_cap(self) -> None:
        """Evict oldest cached agents when cache exceeds _AGENT_CACHE_MAX_SIZE.

        Must be called with _agent_cache_lock held.  Resource cleanup
        (memory provider shutdown, tool resource close) is scheduled
        on a daemon thread so the caller doesn't block on slow teardown
        while holding the cache lock.

        Agents currently in _running_agents are SKIPPED — their clients,
        terminal sandboxes, background processes, and child subagents
        are all in active use by the running turn.  Evicting them would
        tear down those resources mid-turn and crash the request.  If
        every candidate in the LRU order is active, we simply leave the
        cache over the cap; it will be re-checked on the next insert.
        """
        _cache = getattr(self, "_agent_cache", None)
        if _cache is None:
            return
        # OrderedDict.popitem(last=False) pops oldest; plain dict lacks the
        # arg so skip enforcement if a test fixture swapped the cache type.
        if not hasattr(_cache, "move_to_end"):
            return

        # Snapshot of agent instances that are actively mid-turn.  Use id()
        # so the lookup is O(1) and doesn't depend on AIAgent.__eq__ (which
        # MagicMock overrides in tests).
        running_ids = {
            id(a)
            for a in getattr(self, "_running_agents", {}).values()
            if a is not None and a is not _AGENT_PENDING_SENTINEL
        }

        # Walk LRU → MRU and evict excess-LRU entries that aren't mid-turn.
        # We only consider entries in the first (size - cap) LRU positions
        # as eviction candidates.  If one of those slots is held by an
        # active agent, we SKIP it without compensating by evicting a
        # newer entry — that would penalise a freshly-inserted session
        # (which has no cache history to retain) while protecting an
        # already-cached long-running one.  The cache may therefore stay
        # temporarily over cap; it will re-check on the next insert,
        # after active turns have finished.
        excess = max(0, len(_cache) - _AGENT_CACHE_MAX_SIZE)
        evict_plan: List[tuple] = []  # [(key, agent), ...]
        if excess > 0:
            ordered_keys = list(_cache.keys())
            for key in ordered_keys[:excess]:
                entry = _cache.get(key)
                agent = entry[0] if isinstance(entry, tuple) and entry else None
                if agent is not None and id(agent) in running_ids:
                    continue  # active mid-turn; don't evict, don't substitute
                evict_plan.append((key, agent))

        for key, _ in evict_plan:
            _cache.pop(key, None)

        remaining_over_cap = len(_cache) - _AGENT_CACHE_MAX_SIZE
        if remaining_over_cap > 0:
            logger.warning(
                "Agent cache over cap (%d > %d); %d excess slot(s) held by "
                "mid-turn agents — will re-check on next insert.",
                len(_cache), _AGENT_CACHE_MAX_SIZE, remaining_over_cap,
            )

        for key, agent in evict_plan:
            logger.info(
                "Agent cache at cap; evicting LRU session=%s (cache_size=%d)",
                key, len(_cache),
            )
            if agent is not None:
                threading.Thread(
                    target=self._release_evicted_agent_soft,
                    args=(agent,),
                    daemon=True,
                    name=f"agent-cache-evict-{key[:24]}",
                ).start()

    def _sweep_idle_cached_agents(self) -> int:
        """Evict cached agents whose AIAgent has been idle > _AGENT_CACHE_IDLE_TTL_SECS.

        Safe to call from the session expiry watcher without holding the
        cache lock — acquires it internally.  Returns the number of entries
        evicted.  Resource cleanup is scheduled on daemon threads.

        Agents currently in _running_agents are SKIPPED for the same reason
        as _enforce_agent_cache_cap: tearing down an active turn's clients
        mid-flight would crash the request.
        """
        _cache = getattr(self, "_agent_cache", None)
        _lock = getattr(self, "_agent_cache_lock", None)
        if _cache is None or _lock is None:
            return 0
        now = time.time()
        to_evict: List[tuple] = []
        running_ids = {
            id(a)
            for a in getattr(self, "_running_agents", {}).values()
            if a is not None and a is not _AGENT_PENDING_SENTINEL
        }
        with _lock:
            for key, entry in list(_cache.items()):
                agent = entry[0] if isinstance(entry, tuple) and entry else None
                if agent is None:
                    continue
                if id(agent) in running_ids:
                    continue  # mid-turn — don't tear it down
                last_activity = getattr(agent, "_last_activity_ts", None)
                if last_activity is None:
                    continue
                if (now - last_activity) > _AGENT_CACHE_IDLE_TTL_SECS:
                    to_evict.append((key, agent))
            for key, _ in to_evict:
                _cache.pop(key, None)
        for key, agent in to_evict:
            logger.info(
                "Agent cache idle-TTL evict: session=%s (idle=%.0fs)",
                key, now - getattr(agent, "_last_activity_ts", now),
            )
            threading.Thread(
                target=self._release_evicted_agent_soft,
                args=(agent,),
                daemon=True,
                name=f"agent-cache-idle-{key[:24]}",
            ).start()
        return len(to_evict)

    # ------------------------------------------------------------------
    # Proxy mode: forward messages to a remote Hermes API server
    # ------------------------------------------------------------------

    def _get_proxy_url(self) -> Optional[str]:
        """Return the proxy URL if proxy mode is configured, else None.

        Checks GATEWAY_PROXY_URL env var first (convenient for Docker),
        then ``gateway.proxy_url`` in config.yaml.
        """
        url = os.getenv("GATEWAY_PROXY_URL", "").strip()
        if url:
            return url.rstrip("/")
        cfg = _load_gateway_config()
        url = (cfg.get("gateway") or {}).get("proxy_url", "").strip()
        if url:
            return url.rstrip("/")
        return None

    def _evict_cached_agent(self, session_key: str) -> None:
        """Remove a cached agent for a session (called on /new, /model, etc)."""
        _lock = getattr(self, "_agent_cache_lock", None)
        if _lock:
            with _lock:
                self._agent_cache.pop(session_key, None)

    @staticmethod
    def _init_cached_agent_for_turn(agent: Any, interrupt_depth: int) -> None:
        """Reset per-turn state on a cached agent before a new turn starts."""
        if interrupt_depth == 0:
            agent._last_activity_ts = time.time()
            agent._last_activity_desc = "starting new turn (cached)"
        agent._api_call_count = 0

    async def _run_agent_via_proxy(
        self,
        message: str,
        context_prompt: str,
        history: List[Dict[str, Any]],
        source: "SessionSource",
        session_id: str,
        session_key: str = None,
        run_generation: Optional[int] = None,
        event_message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await agent_execution.run_agent_via_proxy(
            runner=self,
            message=message,
            source=source,
            session_key=session_key,
            session_id=session_id,
        )

        """Forward the message to a remote Hermes API server instead of
        running a local AIAgent.

        When ``GATEWAY_PROXY_URL`` (or ``gateway.proxy_url`` in config.yaml)
        is set, the gateway becomes a thin relay: it handles platform I/O
        (encryption, threading, media) and delegates all agent work to the
        remote server via ``POST /v1/chat/completions`` with SSE streaming.

        This lets a Docker container handle Matrix E2EE while the actual
        agent runs on the host with full access to local files, memory,
        skills, and a unified session store.
        """
        try:
            from aiohttp import ClientSession as _AioClientSession, ClientTimeout
        except ImportError:
            return {
                "final_response": "⚠️ Proxy mode requires aiohttp. Install with: pip install aiohttp",
                "messages": [],
                "api_calls": 0,
                "tools": [],
            }

        proxy_url = self._get_proxy_url()
        if not proxy_url:
            return {
                "final_response": "⚠️ Proxy URL not configured (GATEWAY_PROXY_URL or gateway.proxy_url)",
                "messages": [],
                "api_calls": 0,
                "tools": [],
            }

        proxy_key = os.getenv("GATEWAY_PROXY_KEY", "").strip()

        def _run_still_current() -> bool:
            if run_generation is None or not session_key:
                return True
            return self._is_session_run_current(session_key, run_generation)

        # Build messages in OpenAI chat format --------------------------
        #
        # The remote api_server can maintain session continuity via
        # X-Hermes-Session-Id, so it loads its own history.  We only
        # need to send the current user message.  If the remote has
        # no history for this session yet, include what we have locally
        # so the first exchange has context.
        #
        # We always include the current message.  For history, send a
        # compact version (text-only user/assistant turns) — the remote
        # handles tool replay and system prompts.
        api_messages: List[Dict[str, str]] = []

        if context_prompt:
            api_messages.append({"role": "system", "content": context_prompt})

        for msg in history:
            role = msg.get("role")
            content = msg.get("content")
            if role in {"user", "assistant"} and content:
                api_messages.append({"role": role, "content": content})

        api_messages.append({"role": "user", "content": message})

        # HTTP headers ---------------------------------------------------
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if proxy_key:
            headers["Authorization"] = f"Bearer {proxy_key}"
        if session_id:
            headers["X-Hermes-Session-Id"] = session_id

        body = {
            "model": "hermes-agent",
            "messages": api_messages,
            "stream": True,
        }

        # Set up platform streaming if available -------------------------
        _stream_consumer = None
        _scfg = getattr(getattr(self, "config", None), "streaming", None)
        if _scfg is None:
            from gateway.config import StreamingConfig
            _scfg = StreamingConfig()

        platform_key = _platform_config_key(source.platform)
        user_config = _load_gateway_config()
        from gateway.display_config import resolve_display_setting
        _plat_streaming = resolve_display_setting(
            user_config, platform_key, "streaming"
        )
        _streaming_enabled = (
            _scfg.enabled and _scfg.transport != "off"
            if _plat_streaming is None
            else bool(_plat_streaming)
        )

        _thread_metadata: Optional[Dict[str, Any]] = self._thread_metadata_for_source(source, event_message_id)

        if _streaming_enabled:
            try:
                from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
                _adapter = self.adapters.get(source.platform)
                if _adapter:
                    _adapter_supports_edit = getattr(_adapter, "SUPPORTS_MESSAGE_EDITING", True)
                    _effective_cursor = _scfg.cursor if _adapter_supports_edit else ""
                    _buffer_only = False
                    if source.platform == Platform.MATRIX:
                        _effective_cursor = ""
                        _buffer_only = True
                    # Fresh-final applies to Telegram only — other
                    # platforms either edit in place cheaply (Discord,
                    # Slack) or don't have the timestamp-on-edit
                    # problem.  (Ported from openclaw/openclaw#72038.)
                    _fresh_final_secs = (
                        float(getattr(_scfg, "fresh_final_after_seconds", 0.0) or 0.0)
                        if source.platform == Platform.TELEGRAM
                        else 0.0
                    )
                    _consumer_cfg = StreamConsumerConfig(
                        edit_interval=_scfg.edit_interval,
                        buffer_threshold=_scfg.buffer_threshold,
                        cursor=_effective_cursor,
                        buffer_only=_buffer_only,
                        fresh_final_after_seconds=_fresh_final_secs,
                        transport=_scfg.transport or "edit",
                        chat_type=getattr(source, "chat_type", "") or "",
                    )
                    _stream_consumer = GatewayStreamConsumer(
                        adapter=_adapter,
                        chat_id=source.chat_id,
                        config=_consumer_cfg,
                        metadata=_thread_metadata,
                        initial_reply_to_id=event_message_id,
                    )
            except Exception as _sc_err:
                logger.debug("Proxy: could not set up stream consumer: %s", _sc_err)

        # Run the stream consumer task in the background
        stream_task = None
        if _stream_consumer:
            stream_task = asyncio.create_task(_stream_consumer.run())

        # Send typing indicator
        _adapter = self.adapters.get(source.platform)
        if _adapter:
            try:
                await _adapter.send_typing(source.chat_id, metadata=_thread_metadata)
            except Exception:
                pass

        # Make the HTTP request with SSE streaming -----------------------
        full_response = ""
        _start = time.time()

        try:
            _timeout = ClientTimeout(total=0, sock_read=1800)
            async with _AioClientSession(timeout=_timeout) as session:
                async with session.post(
                    f"{proxy_url}/v1/chat/completions",
                    json=body,
                    headers=headers,
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.warning(
                            "Proxy error (%d) from %s: %s",
                            resp.status, proxy_url, error_text[:500],
                        )
                        return {
                            "final_response": f"⚠️ Proxy error ({resp.status}): {error_text[:300]}",
                            "messages": [],
                            "api_calls": 0,
                            "tools": [],
                        }

                    # Parse SSE stream
                    buffer = ""
                    async for chunk in resp.content.iter_any():
                        if not _run_still_current():
                            logger.info(
                                "Discarding stale proxy stream for %s — generation %d is no longer current",
                                session_key or "?",
                                run_generation or 0,
                            )
                            return {
                                "final_response": "",
                                "messages": [],
                                "api_calls": 0,
                                "tools": [],
                                "history_offset": len(history),
                                "session_id": session_id,
                                "response_previewed": False,
                            }
                        text = chunk.decode("utf-8", errors="replace")
                        buffer += text

                        # Process complete SSE lines
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            if line.startswith("data: "):
                                data = line[6:]
                                if data.strip() == "[DONE]":
                                    break
                                try:
                                    obj = json.loads(data)
                                    choices = obj.get("choices", [])
                                    if choices:
                                        delta = choices[0].get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            full_response += content
                                            if _stream_consumer:
                                                _stream_consumer.on_delta(content)
                                except json.JSONDecodeError:
                                    pass

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Proxy connection error to %s: %s", proxy_url, e)
            if not full_response:
                return {
                    "final_response": f"⚠️ Proxy connection error: {e}",
                    "messages": [],
                    "api_calls": 0,
                    "tools": [],
                }
            # Partial response — return what we got
        finally:
            # Finalize stream consumer
            if _stream_consumer:
                _stream_consumer.finish()
            if stream_task:
                try:
                    await asyncio.wait_for(stream_task, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    stream_task.cancel()

        _elapsed = time.time() - _start
        if not _run_still_current():
            # Preserve partial proxy response instead of discarding
            if full_response:
                logger.info(
                    "Preserving stale proxy result for %s — generation %d superseded, "
                    "partial response had %d chars after %.1fs",
                    session_key or "?",
                    run_generation or 0,
                    len(full_response),
                    _elapsed,
                )
                # Return the partial response so the caller can deliver it
            else:
                logger.info(
                    "Discarding stale proxy result for %s — generation %d is no longer current "
                    "(empty response)",
                    session_key or "?",
                    run_generation or 0,
                )
            return {
                "final_response": full_response or "",
                "messages": [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": full_response},
                ] if full_response else [],
                "api_calls": 1,
                "tools": [],
                "history_offset": len(history),
                "session_id": session_id,
                "response_previewed": False,
                "_stale_preserved": bool(full_response),
            }
        logger.info(
            "proxy response: url=%s session=%s time=%.1fs response=%d chars",
            proxy_url, (session_id or "")[:20], _elapsed, len(full_response),
        )

        return {
            "final_response": full_response or "(No response from remote agent)",
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": full_response},
            ],
            "api_calls": 1,
            "tools": [],
            "history_offset": len(history),
            "session_id": session_id,
            "response_previewed": _stream_consumer is not None and bool(full_response),
        }

    # ------------------------------------------------------------------

    async def _run_agent(
        self,
        message: str,
        context_prompt: str,
        history: List[Dict[str, Any]],
        source: SessionSource,
        session_id: str,
        session_key: str = None,
        run_generation: Optional[int] = None,
        _interrupt_depth: int = 0,
        event_message_id: Optional[str] = None,
        channel_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await agent_execution.run_agent(
            runner=self,
            message=message,
            context_prompt=context_prompt,
            history=history,
            source=source,
            session_id=session_id,
            session_key=session_key,
            run_generation=run_generation,
            _interrupt_depth=_interrupt_depth,
            event_message_id=event_message_id,
            channel_prompt=channel_prompt,
        )

        """
        Run the agent with the given message and context.

        Returns the full result dict from run_conversation, including:
          - "final_response": str (the text to send back)
          - "messages": list (full conversation including tool calls)
          - "api_calls": int
          - "completed": bool

        This is run in a thread pool to not block the event loop.
        Supports interruption via new messages.
        """
        if run_generation is None:
            run_generation = self._begin_session_run_generation(session_key) if session_key else None
        # ---- Proxy mode: delegate to remote API server ----
        if self._get_proxy_url():
            return await self._run_agent_via_proxy(
                message=message,
                context_prompt=context_prompt,
                history=history,
                source=source,
                session_id=session_id,
                session_key=session_key,
                run_generation=run_generation,
                event_message_id=event_message_id,
            )

        from run_agent import AIAgent
        import queue

        def _run_still_current() -> bool:
            if run_generation is None or not session_key:
                return True
            return self._is_session_run_current(session_key, run_generation)
        
        user_config = _load_gateway_config()
        platform_key = _platform_config_key(source.platform)

        from hermes_cli.tools_config import _get_platform_tools
        enabled_toolsets = sorted(_get_platform_tools(user_config, platform_key))
        agent_cfg_local = user_config.get("agent") or {}
        disabled_toolsets = agent_cfg_local.get("disabled_toolsets") or None

        display_config = user_config.get("display", {})
        if not isinstance(display_config, dict):
            display_config = {}

        # Per-platform display settings — resolve via display_config module
        # which checks display.platforms.<platform>.<key> first, then
        # display.<key> global, then built-in platform defaults.
        from gateway.display_config import resolve_display_setting

        # Apply tool preview length config (0 = no limit)
        try:
            from agent.display import set_tool_preview_max_len
            _tpl = resolve_display_setting(user_config, platform_key, "tool_preview_length", 0)
            set_tool_preview_max_len(int(_tpl) if _tpl else 0)
        except Exception:
            pass

        # Tool progress mode — resolved per-platform with env var fallback
        _resolved_tp = resolve_display_setting(user_config, platform_key, "tool_progress")
        _env_tp = os.getenv("HERMES_TOOL_PROGRESS_MODE")
        _display_cfg = display_config if isinstance(display_config, dict) else {}
        _platforms_cfg = _display_cfg.get("platforms") or {}
        _platform_cfg = _platforms_cfg.get(platform_key) or {}
        _legacy_tp_overrides = _display_cfg.get("tool_progress_overrides") or {}
        _tool_progress_configured = (
            "tool_progress" in _display_cfg
            or (
                isinstance(_platform_cfg, dict)
                and "tool_progress" in _platform_cfg
            )
            or (
                isinstance(_legacy_tp_overrides, dict)
                and platform_key in _legacy_tp_overrides
            )
        )
        progress_mode = (
            _env_tp
            if _env_tp and not _tool_progress_configured
            else (_resolved_tp or _env_tp or "all")
        )
        # Disable tool progress for webhooks - they don't support message editing,
        # so each progress line would be sent as a separate message.
        from gateway.config import Platform
        tool_progress_enabled = progress_mode != "off" and source.platform != Platform.WEBHOOK
        # Natural assistant status messages are intentionally independent from
        # tool progress and token streaming. Users can keep tool_progress quiet
        # in chat platforms while opting into concise mid-turn updates.
        interim_assistant_messages_enabled = (
            source.platform != Platform.WEBHOOK
            and bool(
                resolve_display_setting(
                    user_config,
                    platform_key,
                    "interim_assistant_messages",
                    True,
                )
            )
        )
        
        # Queue for progress messages (thread-safe)
        progress_queue = queue.Queue() if tool_progress_enabled else None
        last_tool = [None]  # Mutable container for tracking in closure
        last_progress_msg = [None]  # Track last message for dedup
        repeat_count = [0]  # How many times the same message repeated

        # Auto-cleanup of temporary progress bubbles (Telegram + any adapter
        # that implements ``delete_message``). When enabled via
        # ``display.platforms.<platform>.cleanup_progress: true``, message IDs
        # from the tool-progress / "⏳ Working — N min" / status-callback bubbles
        # are collected here and deleted after the final response lands.
        # Failed runs skip cleanup so the bubbles remain as breadcrumbs.
        _cleanup_progress = bool(
            resolve_display_setting(user_config, platform_key, "cleanup_progress")
        )
        _cleanup_adapter = self.adapters.get(source.platform) if _cleanup_progress else None
        if _cleanup_adapter is not None and (
            type(_cleanup_adapter).delete_message is BasePlatformAdapter.delete_message
        ):
            # Adapter doesn't support deletion — silently disable.
            _cleanup_progress = False
            _cleanup_adapter = None
        _cleanup_msg_ids: List[str] = []
        # First-touch onboarding latch: fires at most once per run, even if
        # several tools exceed the threshold.
        long_tool_hint_fired = [False]
        _LONG_TOOL_THRESHOLD_S = 30.0

        def progress_callback(event_type: str, tool_name: str = None, preview: str = None, args: dict = None, **kwargs):
            """Callback invoked by agent on tool lifecycle events."""
            if not progress_queue or not _run_still_current():
                return

            # First-touch onboarding: the first time a tool takes longer than
            # _LONG_TOOL_THRESHOLD_S during a run that's streaming every tool
            # (progress_mode == "all"), append a one-time hint suggesting
            # /verbose.  We only fire when (a) the user hasn't seen the hint
            # before and (b) /verbose is actually usable on this platform
            # (gateway gate must be open).  The CLI has its own trigger.
            if event_type == "tool.completed" and not long_tool_hint_fired[0]:
                try:
                    duration = kwargs.get("duration") or 0
                    if duration >= _LONG_TOOL_THRESHOLD_S and progress_mode == "all":
                        from agent.onboarding import (
                            TOOL_PROGRESS_FLAG,
                            is_seen,
                            mark_seen,
                            tool_progress_hint_gateway,
                        )
                        _cfg = _load_gateway_config()
                        gate_on = is_truthy_value(
                            cfg_get(_cfg, "display", "tool_progress_command"),
                            default=False,
                        )
                        if gate_on and not is_seen(_cfg, TOOL_PROGRESS_FLAG):
                            long_tool_hint_fired[0] = True
                            progress_queue.put(tool_progress_hint_gateway())
                            mark_seen(_hermes_home / "config.yaml", TOOL_PROGRESS_FLAG)
                except Exception as _hint_err:
                    logger.debug("tool-progress onboarding hint failed: %s", _hint_err)
                return


            # Only act on tool.started events (ignore tool.completed, reasoning.available, etc.)
            if event_type not in {"tool.started",}:
                return

            # Suppress tool-progress bubbles once the user has sent `stop`.
            # When the LLM response carries N parallel tool calls, the agent
            # fires N "tool.started" events back-to-back before checking for
            # interrupts — without this guard, a late `stop` still renders
            # all N as 🔍 bubbles, making the interrupt feel ignored.
            # (agent lives in run_sync's scope; agent_holder[0] is the shared
            # handle across nested scopes — see line ~9607.)
            try:
                _agent_for_interrupt = agent_holder[0] if agent_holder else None
                if _agent_for_interrupt is not None and getattr(
                    _agent_for_interrupt, "is_interrupted", False
                ):
                    return
            except Exception:
                pass

            # "new" mode: only report when tool changes
            if progress_mode == "new" and tool_name == last_tool[0]:
                return
            last_tool[0] = tool_name
            
            # Build progress message with primary argument preview
            from agent.display import get_tool_emoji
            emoji = get_tool_emoji(tool_name, default="⚙️")
            
            # Verbose mode: show detailed arguments, respects tool_preview_length
            if progress_mode == "verbose":
                if args:
                    from agent.display import get_tool_preview_max_len
                    _pl = get_tool_preview_max_len()
                    args_str = json.dumps(args, ensure_ascii=False, default=str)
                    # When tool_preview_length is 0 (default), don't truncate
                    # in verbose mode — the user explicitly asked for full
                    # detail.  Platform message-length limits handle the rest.
                    if _pl > 0 and len(args_str) > _pl:
                        args_str = args_str[:_pl - 3] + "..."
                    msg = f"{emoji} {tool_name}({list(args.keys())})\n{args_str}"
                elif preview:
                    msg = f"{emoji} {tool_name}: \"{preview}\""
                else:
                    msg = f"{emoji} {tool_name}..."
                progress_queue.put(msg)
                return
            
            # "all" / "new" modes: short preview, respects tool_preview_length
            # config (defaults to 40 chars when unset to keep gateway messages
            # compact — unlike CLI spinners, these persist as permanent messages).
            if preview:
                from agent.display import get_tool_preview_max_len
                _pl = get_tool_preview_max_len()
                _cap = _pl if _pl > 0 else 40
                if len(preview) > _cap:
                    preview = preview[:_cap - 3] + "..."
                msg = f"{emoji} {tool_name}: \"{preview}\""
            else:
                msg = f"{emoji} {tool_name}..."
            
            # Dedup: collapse consecutive identical progress messages.
            # Common with execute_code where models iterate with the same
            # code (same boilerplate imports → identical previews).
            if msg == last_progress_msg[0]:
                repeat_count[0] += 1
                # Update the last line in progress_lines with a counter
                # via a special "dedup" queue message.
                progress_queue.put(("__dedup__", msg, repeat_count[0]))
                return
            last_progress_msg[0] = msg
            repeat_count[0] = 0
            
            progress_queue.put(msg)
        
        # Background task to send progress messages
        # Accumulates tool lines into a single message that gets edited.
        #
        # Threading metadata is platform-specific:
        # - Slack DM threading needs event_message_id fallback (reply thread)
        # - Telegram forum topics use message_thread_id; Hermes-created private
        #   DM topic lanes require both thread metadata and a reply anchor
        # - Feishu only honors reply_in_thread when sending a reply, so topic
        #   progress uses the triggering event message as the reply target
        # - Other platforms should use explicit source.thread_id only
        if source.platform == Platform.SLACK:
            _progress_thread_id = source.thread_id or event_message_id
        else:
            _progress_thread_id = source.thread_id
        _progress_metadata = (
            self._thread_metadata_for_source(source, event_message_id)
            if _progress_thread_id == source.thread_id
            else {"thread_id": _progress_thread_id}
        ) if _progress_thread_id else None
        _progress_reply_to = (
            event_message_id
            if source.platform in (Platform.FEISHU, Platform.MATTERMOST) and source.thread_id and event_message_id
            else None
        )

        async def send_progress_messages():
            if not progress_queue:
                return

            adapter = self.adapters.get(source.platform)
            if not adapter:
                return

            # Skip tool progress for platforms that don't support message
            # editing (e.g. iMessage/BlueBubbles) — each progress update
            # would become a separate message bubble, which is noisy.
            if type(adapter).edit_message is BasePlatformAdapter.edit_message:
                while not progress_queue.empty():
                    try:
                        progress_queue.get_nowait()
                    except Exception:
                        break
                return

            progress_lines = []      # Accumulated tool lines for the CURRENT editable bubble
            progress_msg_id = None   # ID of the current progress message to edit
            can_edit = True          # False once an edit fails (platform doesn't support it)
            _last_edit_ts = 0.0      # Throttle edits to avoid Telegram flood control
            _PROGRESS_EDIT_INTERVAL = 1.5  # Minimum seconds between edits

            _progress_len_fn = (
                adapter.message_len_fn
                if isinstance(adapter, BasePlatformAdapter)
                else len
            )
            try:
                _raw_progress_limit = int(getattr(adapter, "MAX_MESSAGE_LENGTH", 4000) or 4000)
            except Exception:
                _raw_progress_limit = 4000
            # Leave a little room for platform quirks / formatting.  For tiny
            # test adapters keep the limit usable instead of clamping to 500+.
            _PROGRESS_TEXT_LIMIT = max(
                1,
                _raw_progress_limit - (64 if _raw_progress_limit > 128 else 0),
            )

            # Detect whether the adapter's edit_message accepts metadata so
            # overflow edits preserve Telegram topic/thread routing (#27487).
            _edit_accepts_metadata = False
            if _progress_metadata:
                try:
                    _edit_params = inspect.signature(adapter.edit_message).parameters
                    _edit_accepts_metadata = (
                        "metadata" in _edit_params
                        or any(
                            param.kind is inspect.Parameter.VAR_KEYWORD
                            for param in _edit_params.values()
                        )
                    )
                except (TypeError, ValueError):
                    _edit_accepts_metadata = False

            async def _edit_progress_message(message_id: str, content: str):
                kwargs = {
                    "chat_id": source.chat_id,
                    "message_id": message_id,
                    "content": content,
                }
                if _edit_accepts_metadata:
                    kwargs["metadata"] = _progress_metadata
                return await adapter.edit_message(**kwargs)

            def _progress_text(lines: list) -> str:
                return "\n".join(str(line) for line in lines)

            def _split_progress_groups(lines: list) -> list[list]:
                """Partition progress lines into platform-sized editable bubbles."""
                groups: list[list] = []
                current: list = []
                for line in lines:
                    candidate = current + [line]
                    if current and _progress_len_fn(_progress_text(candidate)) > _PROGRESS_TEXT_LIMIT:
                        groups.append(current)
                        current = [line]
                    else:
                        current = candidate
                if current:
                    groups.append(current)
                return groups

            def _track_progress_result(result) -> None:
                if (
                    _cleanup_progress
                    and getattr(result, "success", False)
                    and getattr(result, "message_id", None)
                ):
                    _cleanup_msg_ids.append(str(result.message_id))

            async def _send_progress_text(text: str):
                result = await adapter.send(
                    chat_id=source.chat_id,
                    content=text,
                    reply_to=_progress_reply_to,
                    metadata=_progress_metadata,
                )
                _track_progress_result(result)
                return result

            async def _roll_progress_overflow_if_needed() -> bool:
                """Start fresh editable progress bubbles before a bubble exceeds limit.

                Returns True when it delivered/split the current buffer and the
                caller should skip the normal send/edit path for this tick.
                """
                nonlocal progress_msg_id, progress_lines, can_edit
                if not progress_lines or not can_edit:
                    return False
                groups = _split_progress_groups(progress_lines)
                if len(groups) <= 1:
                    return False

                first_text = _progress_text(groups[0])
                if progress_msg_id is not None:
                    result = await _edit_progress_message(progress_msg_id, first_text)
                    if not result.success:
                        can_edit = False
                        # Fall back to the existing non-edit behavior below.
                        return False
                else:
                    result = await _send_progress_text(first_text)
                    if result.success and result.message_id:
                        progress_msg_id = result.message_id

                for group in groups[1:]:
                    result = await _send_progress_text(_progress_text(group))
                    if result.success and result.message_id:
                        progress_msg_id = result.message_id

                # The newest continuation is now the only mutable bubble.  Keep
                # just its lines so subsequent edits update it instead of
                # replaying the full historical transcript into new messages.
                progress_lines = groups[-1]
                return True

            while True:
                try:
                    if not _run_still_current():
                        while not progress_queue.empty():
                            try:
                                progress_queue.get_nowait()
                            except Exception:
                                break
                        return

                    raw = progress_queue.get_nowait()

                    # Drain silently when interrupted: events queued in the
                    # window between tool parse and interrupt processing
                    # should not render as bubbles.  The "⚡ Interrupting
                    # current task" message is sent separately and is the
                    # last progress-flavored bubble the user should see.
                    try:
                        _agent_for_interrupt = agent_holder[0] if agent_holder else None
                        if _agent_for_interrupt is not None and getattr(
                            _agent_for_interrupt, "is_interrupted", False
                        ):
                            # Drop this event and continue draining.
                            await asyncio.sleep(0)
                            continue
                    except Exception:
                        pass

                    # Handle dedup messages: update last line with repeat counter
                    if isinstance(raw, tuple) and len(raw) == 3 and raw[0] == "__dedup__":
                        _, base_msg, count = raw
                        if progress_lines:
                            progress_lines[-1] = f"{base_msg} (×{count + 1})"
                        msg = progress_lines[-1] if progress_lines else base_msg
                    elif isinstance(raw, tuple) and len(raw) >= 1 and raw[0] == "__reset__":
                        # Content bubble just landed on the platform — close off
                        # the current tool-progress bubble so the next tool
                        # starts a fresh bubble below the content. Without this,
                        # tool lines keep editing the ORIGINAL progress message
                        # above the new content, making the chat appear out of
                        # order. Mirrors GatewayStreamConsumer.on_segment_break
                        # on the content side. (Issue: tool + content
                        # linearization regression after PR #7885.)
                        progress_msg_id = None
                        progress_lines = []
                        last_progress_msg[0] = None
                        repeat_count[0] = 0
                        continue
                    else:
                        msg = raw
                        progress_lines.append(msg)

                    if await _roll_progress_overflow_if_needed():
                        _last_edit_ts = time.monotonic()
                        await asyncio.sleep(0.3)
                        if _run_still_current():
                            await adapter.send_typing(source.chat_id, metadata=_progress_metadata)
                        continue

                    # Throttle edits: batch rapid tool updates into fewer
                    # API calls to avoid hitting Telegram flood control.
                    # (grammY auto-retry pattern: proactively rate-limit
                    # instead of reacting to 429s.)
                    _now = time.monotonic()
                    _remaining = _PROGRESS_EDIT_INTERVAL - (_now - _last_edit_ts)
                    if _remaining > 0:
                        # Wait out the throttle interval, then loop back to
                        # drain any additional queued messages before sending
                        # a single batched edit.
                        await asyncio.sleep(_remaining)
                        continue

                    if not _run_still_current():
                        return

                    if can_edit and progress_msg_id is not None:
                        # Try to edit the existing progress message
                        full_text = "\n".join(progress_lines)
                        result = await _edit_progress_message(progress_msg_id, full_text)
                        if not result.success:
                            _err = (getattr(result, "error", "") or "").lower()
                            # Transient network errors (ConnectError, timeouts)
                            # must not permanently disable progress-message
                            # editing — the next cycle can catch up.  Only
                            # permanent failures (flood control, message not
                            # found, permissions) should set can_edit = False.
                            if getattr(result, "retryable", False):
                                logger.debug(
                                    "[%s] Transient edit failure — keeping can_edit=True",
                                    adapter.name,
                                )
                                continue
                            if "flood" in _err or "retry after" in _err:
                                # Flood control hit — backoff but keep editing.
                                # Only disable edits for non-recoverable errors.
                                logger.info(
                                    "[%s] Progress edit flood control, backing off",
                                    adapter.name,
                                )
                                _last_edit_ts = time.monotonic()
                            else:
                                can_edit = False
                            _flood_result = await adapter.send(
                                chat_id=source.chat_id,
                                content=msg,
                                reply_to=_progress_reply_to,
                                metadata=_progress_metadata,
                            )
                            if (
                                _cleanup_progress
                                and getattr(_flood_result, "success", False)
                                and getattr(_flood_result, "message_id", None)
                            ):
                                _cleanup_msg_ids.append(str(_flood_result.message_id))
                    else:
                        if can_edit:
                            # First tool: send all accumulated text as new message
                            full_text = "\n".join(progress_lines)
                            result = await adapter.send(
                                chat_id=source.chat_id,
                                content=full_text,
                                reply_to=_progress_reply_to,
                                metadata=_progress_metadata,
                            )
                        else:
                            # Editing unsupported: send just this line
                            result = await adapter.send(
                                chat_id=source.chat_id,
                                content=msg,
                                reply_to=_progress_reply_to,
                                metadata=_progress_metadata,
                            )
                        if result.success and result.message_id:
                            progress_msg_id = result.message_id
                            if _cleanup_progress:
                                _cleanup_msg_ids.append(str(result.message_id))

                    _last_edit_ts = time.monotonic()

                    # Restore typing indicator
                    await asyncio.sleep(0.3)
                    if _run_still_current():
                        await adapter.send_typing(source.chat_id, metadata=_progress_metadata)

                except queue.Empty:
                    await asyncio.sleep(0.3)
                except asyncio.CancelledError:
                    # Drain remaining queued messages
                    while not progress_queue.empty():
                        try:
                            raw = progress_queue.get_nowait()
                            if isinstance(raw, tuple) and len(raw) == 3 and raw[0] == "__dedup__":
                                _, base_msg, count = raw
                                if progress_lines:
                                    progress_lines[-1] = f"{base_msg} (×{count + 1})"
                                    await _roll_progress_overflow_if_needed()
                            elif isinstance(raw, tuple) and len(raw) >= 1 and raw[0] == "__reset__":
                                # Content-bubble marker during drain: close off
                                # the current progress bubble and start a fresh
                                # one for any tool lines that arrived after.
                                await _roll_progress_overflow_if_needed()
                                if can_edit and progress_lines and progress_msg_id:
                                    _pending_text = _progress_text(progress_lines)
                                    try:
                                        await _edit_progress_message(progress_msg_id, _pending_text)
                                    except Exception:
                                        pass
                                progress_msg_id = None
                                progress_lines = []
                                last_progress_msg[0] = None
                                repeat_count[0] = 0
                            else:
                                progress_lines.append(raw)
                                await _roll_progress_overflow_if_needed()
                        except Exception:
                            break
                    # Final edit with all remaining tools (only if editing works)
                    if can_edit and progress_lines and progress_msg_id:
                        await _roll_progress_overflow_if_needed()
                    if can_edit and progress_lines and progress_msg_id:
                        full_text = _progress_text(progress_lines)
                        try:
                            await _edit_progress_message(progress_msg_id, full_text)
                        except Exception:
                            pass
                    # Stop typing indicator — the progress streamer re-triggers
                    # send_typing() after each edit (line 14647), so a new loop
                    # may have been spawned after the main stop_typing at line 7533
                    # cancelled the previous one.  Without this cleanup the
                    # typing indicator persists indefinitely.
                    try:
                        await adapter.stop_typing(source.chat_id)
                    except Exception:
                        pass
                    return
                except Exception as e:
                    logger.error("Progress message error: %s", e)
                    await asyncio.sleep(1)
        
        # We need to share the agent instance for interrupt support
        agent_holder = [None]  # Mutable container for the agent instance
        result_holder = [None]  # Mutable container for the result
        tools_holder = [None]   # Mutable container for the tool definitions
        stream_consumer_holder = [None]  # Mutable container for stream consumer
        
        # Bridge sync step_callback → async hooks.emit for agent:step events
        _loop_for_step = asyncio.get_running_loop()
        _hooks_ref = self.hooks

        def _step_callback_sync(iteration: int, prev_tools: list) -> None:
            if not _run_still_current():
                return
            # prev_tools may be list[str] or list[dict] with "name"/"result"
            # keys.  Normalise to keep "tool_names" backward-compatible for
            # user-authored hooks that do ', '.join(tool_names)'.
            _names: list[str] = []
            for _t in (prev_tools or []):
                if isinstance(_t, dict):
                    _names.append(_t.get("name") or "")
                else:
                    _names.append(str(_t))
            safe_schedule_threadsafe(
                _hooks_ref.emit("agent:step", {
                    "platform": source.platform.value if source.platform else "",
                    "user_id": source.user_id,
                    "session_id": session_id,
                    "iteration": iteration,
                    "tool_names": _names,
                    "tools": prev_tools,
                }),
                _loop_for_step,
                logger=logger,
                log_message="agent:step hook scheduling error",
            )

        # Bridge sync status_callback → async adapter.send for context pressure
        _status_adapter = self.adapters.get(source.platform)
        _status_chat_id = source.chat_id
        if source.platform == Platform.FEISHU and source.thread_id and event_message_id:
            # Feishu topics only keep messages inside the topic when they are
            # sent via the reply API with reply_in_thread=true. Status/interim,
            # approval, and stream-consumer paths usually only receive metadata,
            # so carry the triggering message id as a Feishu-specific fallback.
            _status_thread_metadata: Optional[Dict[str, Any]] = {
                "thread_id": _progress_thread_id,
                "reply_to_message_id": event_message_id,
            }
        else:
            _status_thread_metadata = self._thread_metadata_for_source(source, event_message_id) if _progress_thread_id else None

        def _status_callback_sync(event_type: str, message: str) -> None:
            if not _status_adapter or not _run_still_current():
                return
            prepared_message = _prepare_gateway_status_message(
                source.platform,
                event_type,
                message,
            )
            if prepared_message is None:
                logger.debug(
                    "status_callback suppressed for %s/%s: %s",
                    source.platform.value if source.platform else "unknown",
                    event_type,
                    _redact_gateway_user_facing_secrets(str(message or ""))[:160],
                )
                return
            _fut = safe_schedule_threadsafe(
                _send_or_update_status_coro(_status_adapter, _status_chat_id, event_type, prepared_message, _status_thread_metadata),
                _loop_for_step,
                logger=logger,
                log_message=f"status_callback ({event_type}) scheduling error",
            )
            if _fut is None:
                return
            if _cleanup_progress:
                def _track_status_id(fut) -> None:
                    try:
                        res = fut.result()
                    except Exception:
                        return
                    mid = getattr(res, "message_id", None)
                    if getattr(res, "success", False) and mid:
                        _cleanup_msg_ids.append(str(mid))
                _fut.add_done_callback(_track_status_id)

        def run_sync():
            # The conditional re-assignment of `message` further below
            # (prepending model-switch notes) makes Python treat it as a
            # local variable in the entire function.  `nonlocal` lets us
            # read *and* reassign the outer `_run_agent` parameter without
            # triggering an UnboundLocalError on the earlier read at
            # `_resolve_turn_agent_config(message, …)`.
            nonlocal message

            # session_key is now set via contextvars in _set_session_env()
            # (concurrency-safe). Keep os.environ as fallback for CLI/cron.
            os.environ["HERMES_SESSION_KEY"] = session_key or ""

            # Read from env var or use default (same as CLI)
            max_iterations = int(os.getenv("HERMES_MAX_ITERATIONS", "90"))
            
            # Map platform enum to the platform hint key the agent understands.
            # Platform.LOCAL ("local") maps to "cli"; others pass through as-is.
            platform_key = "cli" if source.platform == Platform.LOCAL else source.platform.value
            
            # Combine platform context, per-channel context, and the user-configured
            # ephemeral system prompt.
            combined_ephemeral = context_prompt or ""
            event_channel_prompt = (channel_prompt or "").strip()
            if event_channel_prompt:
                combined_ephemeral = (combined_ephemeral + "\n\n" + event_channel_prompt).strip()
            if self._ephemeral_system_prompt:
                combined_ephemeral = (combined_ephemeral + "\n\n" + self._ephemeral_system_prompt).strip()

            # Re-read .env and config for fresh credentials (gateway is long-lived,
            # keys may change without restart). Keep config.yaml authoritative for
            # runtime budget settings bridged into env vars.
            _reload_runtime_env_preserving_config_authority()

            try:
                model, runtime_kwargs = self._resolve_session_agent_runtime(
                    source=source,
                    session_key=session_key,
                    user_config=user_config,
                )
                logger.debug(
                    "run_agent resolved: model=%s provider=%s session=%s",
                    model, runtime_kwargs.get("provider"), session_key or "",
                )
            except Exception as exc:
                return {
                    "final_response": f"⚠️ Provider authentication failed: {exc}",
                    "messages": [],
                    "api_calls": 0,
                    "tools": [],
                }

            pr = self._provider_routing
            reasoning_config = self._resolve_session_reasoning_config(
                source=source,
                session_key=session_key,
            )
            self._reasoning_config = reasoning_config
            self._service_tier = self._load_service_tier()
            # Set up stream consumer for token streaming or interim commentary.
            _stream_consumer = None
            _stream_delta_cb = None
            _scfg = getattr(getattr(self, 'config', None), 'streaming', None)
            if _scfg is None:
                from gateway.config import StreamingConfig
                _scfg = StreamingConfig()

            # Per-platform streaming gate: display.platforms.<plat>.streaming
            # can disable streaming for specific platforms even when the global
            # streaming config is enabled.
            _plat_streaming = resolve_display_setting(
                user_config, platform_key, "streaming"
            )
            # None = no per-platform override → follow global config
            _streaming_enabled = (
                _scfg.enabled and _scfg.transport != "off"
                if _plat_streaming is None
                else bool(_plat_streaming)
            )
            _want_stream_deltas = _streaming_enabled
            _want_interim_messages = interim_assistant_messages_enabled
            _want_interim_consumer = _want_interim_messages
            if _want_stream_deltas or _want_interim_consumer:
                try:
                    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
                    _adapter = self.adapters.get(source.platform)
                    if _adapter:
                        # Platforms that don't support editing sent messages
                        # (e.g. QQ, WeChat) should skip streaming entirely —
                        # without edit support, the consumer sends a partial
                        # first message that can never be updated, resulting in
                        # duplicate messages (partial + final).
                        _adapter_supports_edit = getattr(_adapter, "SUPPORTS_MESSAGE_EDITING", True)
                        if not _adapter_supports_edit:
                            raise RuntimeError("skip streaming for non-editable platform")
                        _effective_cursor = _scfg.cursor
                        # Some Matrix clients render the streaming cursor
                        # as a visible tofu/white-box artifact.  Keep
                        # streaming text on Matrix, but suppress the cursor.
                        _buffer_only = False
                        if source.platform == Platform.MATRIX:
                            _effective_cursor = ""
                            _buffer_only = True
                        # Fresh-final applies to Telegram only — other
                        # platforms either edit in place cheaply or don't
                        # have the edit-timestamp-stays-stale problem.
                        # (Ported from openclaw/openclaw#72038.)
                        _fresh_final_secs = (
                            float(getattr(_scfg, "fresh_final_after_seconds", 0.0) or 0.0)
                            if source.platform == Platform.TELEGRAM
                            else 0.0
                        )
                        _consumer_cfg = StreamConsumerConfig(
                            edit_interval=_scfg.edit_interval,
                            buffer_threshold=_scfg.buffer_threshold,
                            cursor=_effective_cursor,
                            buffer_only=_buffer_only,
                            fresh_final_after_seconds=_fresh_final_secs,
                            transport=_scfg.transport or "edit",
                            chat_type=getattr(source, "chat_type", "") or "",
                        )
                        _stream_consumer = GatewayStreamConsumer(
                            adapter=_adapter,
                            chat_id=source.chat_id,
                            config=_consumer_cfg,
                            metadata=_status_thread_metadata,
                            on_new_message=(
                                (lambda: progress_queue.put(("__reset__",)))
                                if progress_queue is not None
                                else None
                            ),
                            initial_reply_to_id=event_message_id,
                        )
                        if _want_stream_deltas:
                            def _stream_delta_cb(text: str) -> None:
                                if _run_still_current():
                                    _stream_consumer.on_delta(text)
                        stream_consumer_holder[0] = _stream_consumer
                except Exception as _sc_err:
                    logger.debug("Could not set up stream consumer: %s", _sc_err)

            def _interim_assistant_cb(text: str, *, already_streamed: bool = False) -> None:
                if not _run_still_current():
                    return
                if _stream_consumer is not None:
                    if already_streamed:
                        _stream_consumer.on_segment_break()
                    else:
                        _stream_consumer.on_commentary(text)
                    return
                if already_streamed or not _status_adapter or not str(text or "").strip():
                    return
                safe_schedule_threadsafe(
                    _status_adapter.send(
                        _status_chat_id,
                        text,
                        metadata=_status_thread_metadata,
                    ),
                    _loop_for_step,
                    logger=logger,
                    log_message="interim_assistant_callback scheduling error",
                )

            turn_route = self._resolve_turn_agent_config(message, model, runtime_kwargs)

            # Wire circuit breaker: track provider for failure/success callbacks
            _route_provider = turn_route.get("runtime", {}).get("provider")
            self._last_route_provider = _route_provider

            # Check agent cache — reuse the AIAgent from the previous message
            # in this session to preserve the frozen system prompt and tool
            # schemas for prompt cache hits.
            _sig = self._agent_config_signature(
                turn_route["model"],
                turn_route["runtime"],
                enabled_toolsets,
                combined_ephemeral,
                cache_keys=self._extract_cache_busting_config(user_config),
                user_id=getattr(source, "user_id", None),
                user_id_alt=getattr(source, "user_id_alt", None),
            )
            agent = None
            _cache_lock = getattr(self, "_agent_cache_lock", None)
            _cache = getattr(self, "_agent_cache", None)
            if _cache_lock and _cache is not None:
                with _cache_lock:
                    cached = _cache.get(session_key)
                    if cached and cached[1] == _sig:
                        agent = cached[0]
                        # Refresh LRU order so the cap enforcement evicts
                        # truly-oldest entries, not the one we just used.
                        if hasattr(_cache, "move_to_end"):
                            try:
                                _cache.move_to_end(session_key)
                            except KeyError:
                                pass
                        self._init_cached_agent_for_turn(agent, _interrupt_depth)
                        logger.debug("Reusing cached agent for session %s", session_key)

            if agent is None:
                # Config changed or first message — create fresh agent
                agent = AIAgent(
                    model=turn_route["model"],
                    **turn_route["runtime"],
                    max_iterations=max_iterations,
                    quiet_mode=True,
                    verbose_logging=False,
                    enabled_toolsets=enabled_toolsets,
                    disabled_toolsets=disabled_toolsets,
                    ephemeral_system_prompt=combined_ephemeral or None,
                    prefill_messages=self._prefill_messages or None,
                    reasoning_config=reasoning_config,
                    service_tier=self._service_tier,
                    request_overrides=turn_route.get("request_overrides"),
                    providers_allowed=pr.get("only"),
                    providers_ignored=pr.get("ignore"),
                    providers_order=pr.get("order"),
                    provider_sort=pr.get("sort"),
                    provider_require_parameters=pr.get("require_parameters", False),
                    provider_data_collection=pr.get("data_collection"),
                    session_id=session_id,
                    platform=platform_key,
                    user_id=source.user_id,
                    user_id_alt=source.user_id_alt,
                    user_name=source.user_name,
                    chat_id=source.chat_id,
                    chat_name=source.chat_name,
                    chat_type=source.chat_type,
                    thread_id=source.thread_id,
                    gateway_session_key=session_key,
                    session_db=self._session_db,
                    fallback_model=self._fallback_model,
                )
                if _cache_lock and _cache is not None:
                    with _cache_lock:
                        _cache[session_key] = (agent, _sig)
                        self._enforce_agent_cache_cap()
                logger.debug("Created new agent for session %s (sig=%s)", session_key, _sig)

            # Per-message state — callbacks and reasoning config change every
            # turn and must not be baked into the cached agent constructor.
            agent.tool_progress_callback = progress_callback if tool_progress_enabled else None
            agent.step_callback = _step_callback_sync if _hooks_ref.loaded_hooks else None
            agent.stream_delta_callback = _stream_delta_cb
            agent.interim_assistant_callback = _interim_assistant_cb if _want_interim_messages else None
            agent.status_callback = _status_callback_sync
            agent.reasoning_config = reasoning_config
            agent.service_tier = self._service_tier
            agent.request_overrides = turn_route.get("request_overrides") or {}

            _bg_review_release = threading.Event()
            _bg_review_pending: list[str] = []
            _bg_review_pending_lock = threading.Lock()

            def _deliver_bg_review_message(message: str) -> None:
                if not _status_adapter or not _run_still_current():
                    return
                safe_schedule_threadsafe(
                    _status_adapter.send(
                        _status_chat_id,
                        message,
                        metadata=_status_thread_metadata,
                    ),
                    _loop_for_step,
                    logger=logger,
                    log_message="background_review_callback scheduling error",
                )

            def _release_bg_review_messages() -> None:
                _bg_review_release.set()
                with _bg_review_pending_lock:
                    pending = list(_bg_review_pending)
                    _bg_review_pending.clear()
                for queued in pending:
                    _deliver_bg_review_message(queued)

            # Background review delivery — send "💾 Memory updated" etc. to user
            def _bg_review_send(message: str) -> None:
                if not _status_adapter or not _run_still_current():
                    return
                if not _bg_review_release.is_set():
                    with _bg_review_pending_lock:
                        if not _bg_review_release.is_set():
                            _bg_review_pending.append(message)
                            return
                _deliver_bg_review_message(message)

            agent.background_review_callback = _bg_review_send
            # Register the release hook on the adapter so base.py's finally
            # block can fire it after delivering the main response.
            if _status_adapter and session_key:
                if getattr(type(_status_adapter), "register_post_delivery_callback", None) is not None:
                    _status_adapter.register_post_delivery_callback(
                        session_key,
                        _release_bg_review_messages,
                        generation=run_generation,
                    )
                else:
                    _pdc = getattr(_status_adapter, "_post_delivery_callbacks", None)
                    if _pdc is not None:
                        _pdc[session_key] = _release_bg_review_messages

            # ------------------------------------------------------------------
            # Clarify callback: present a clarify prompt and block on a response.
            #
            # Runs on the agent's worker thread (see clarify_tool's synchronous
            # callback contract).  Bridges sync→async by scheduling the
            # adapter's send_clarify on the gateway event loop, then blocks on
            # the clarify primitive's threading.Event with a configurable
            # timeout.  Returns the user's response string, or a sentinel
            # explaining that no response arrived (so the agent can adapt
            # rather than hang forever).
            # ------------------------------------------------------------------
            def _clarify_callback_sync(question: str, choices) -> str:
                from tools import clarify_gateway as _clarify_mod
                import uuid as _uuid

                if not _status_adapter:
                    return ""

                clarify_id = _uuid.uuid4().hex[:10]
                _clarify_mod.register(
                    clarify_id=clarify_id,
                    session_key=session_key or "",
                    question=question,
                    choices=list(choices) if choices else None,
                )

                # Pause typing — like approval, we don't want a "thinking..."
                # status to obscure the prompt or block the user from typing
                # an "Other" response on platforms that disable input while
                # typing is active (Slack Assistant API).
                try:
                    _status_adapter.pause_typing_for_chat(_status_chat_id)
                except Exception:
                    pass

                send_ok = False
                fut = safe_schedule_threadsafe(
                    _status_adapter.send_clarify(
                        chat_id=_status_chat_id,
                        question=question,
                        choices=list(choices) if choices else None,
                        clarify_id=clarify_id,
                        session_key=session_key or "",
                        metadata=_status_thread_metadata,
                    ),
                    _loop_for_step,
                    logger=logger,
                    log_message="Clarify send failed to schedule",
                )
                if fut is None:
                    send_ok = False
                else:
                    try:
                        result = fut.result(timeout=15)
                        send_ok = bool(getattr(result, "success", False))
                    except Exception as exc:
                        logger.warning("Clarify send failed: %s", exc)
                        send_ok = False

                if not send_ok:
                    # Couldn't deliver the prompt — clean up and return
                    # sentinel so the agent can fall back to a sensible
                    # default rather than hanging.
                    _clarify_mod.clear_session(session_key or "")
                    return "[clarify prompt could not be delivered]"

                timeout = _clarify_mod.get_clarify_timeout()
                response = _clarify_mod.wait_for_response(clarify_id, timeout=float(timeout))
                if response is None or response == "":
                    # Timeout or session-boundary cancellation
                    return f"[user did not respond within {int(timeout / 60)}m]"
                return response

            agent.clarify_callback = _clarify_callback_sync

            # Store agent reference for interrupt support
            agent_holder[0] = agent
            # Capture the full tool definitions for transcript logging
            tools_holder[0] = agent.tools if hasattr(agent, 'tools') else None
            
            # Convert history to agent format.
            # Two cases:
            #   1. Normal path (from transcript): simple {role, content, timestamp} dicts
            #      - Strip timestamps, keep role+content
            #   2. Interrupt path (from agent result["messages"]): full agent messages
            #      that may include tool_calls, tool_call_id, reasoning, etc.
            #      - These must be passed through intact so the API sees valid
            #        assistant→tool sequences (dropping tool_calls causes 500 errors)
            #
            # Telegram observed group context is handled structurally here:
            # observed=True transcript rows are withheld from replayable
            # history and attached to the current addressed message as
            # API-only context, so persisted history stores only the real
            # addressed user turn.
            agent_history, observed_group_context = _build_gateway_agent_history(
                history,
                channel_prompt=channel_prompt,
            )
            
            # Collect MEDIA paths already in history so we can exclude them
            # from the current turn's extraction. This is compression-safe:
            # even if the message list shrinks, we know which paths are old.
            _history_media_paths: set = set()
            for _hm in agent_history:
                if _hm.get("role") in {"tool", "function"}:
                    _hc = _hm.get("content", "")
                    if "MEDIA:" in _hc:
                        _TOOL_MEDIA_RE = re.compile(
                            r'MEDIA:((?:[A-Za-z]:[/\\]|/|~\/)\S+\.(?:png|jpe?g|gif|webp|'
                            r'mp4|mov|avi|mkv|webm|ogg|opus|mp3|wav|m4a|'
                            r'flac|epub|pdf|zip|rar|7z|docx?|xlsx?|pptx?|'
                            r'txt|csv|apk|ipa))',
                            re.IGNORECASE
                        )
                        for _match in _TOOL_MEDIA_RE.finditer(_hc):
                            _p = _match.group(1).strip().rstrip('",}')
                            if _p:
                                _history_media_paths.add(_p)
            
            # Register per-session gateway approval callback so dangerous
            # command approval blocks the agent thread (mirrors CLI input()).
            # The callback bridges sync→async to send the approval request
            # to the user immediately.
            from tools.approval import (
                register_gateway_notify,
                reset_current_session_key,
                set_current_session_key,
                unregister_gateway_notify,
            )

            def _approval_notify_sync(approval_data: dict) -> None:
                """Send the approval request to the user from the agent thread.

                If the adapter supports interactive button-based approvals
                (e.g. Discord's ``send_exec_approval``), use that for a richer
                UX.  Otherwise fall back to a plain text message with
                ``/approve`` instructions.
                """
                # Pause the typing indicator while the agent waits for
                # user approval.  Critical for Slack's Assistant API where
                # assistant_threads_setStatus disables the compose box — the
                # user literally cannot type /approve while "is thinking..."
                # is active.  The approval message send auto-clears the Slack
                # status; pausing prevents _keep_typing from re-setting it.
                # Typing resumes in _handle_approve_command/_handle_deny_command.
                _status_adapter.pause_typing_for_chat(_status_chat_id)

                cmd = approval_data.get("command", "")
                desc = approval_data.get("description", "dangerous command")

                # Prefer button-based approval when the adapter supports it.
                # Check the *class* for the method, not the instance — avoids
                # false positives from MagicMock auto-attribute creation in tests.
                if getattr(type(_status_adapter), "send_exec_approval", None) is not None:
                    try:
                        _approval_fut = safe_schedule_threadsafe(
                            _status_adapter.send_exec_approval(
                                chat_id=_status_chat_id,
                                command=cmd,
                                session_key=_approval_session_key,
                                description=desc,
                                metadata=_status_thread_metadata,
                            ),
                            _loop_for_step,
                            logger=logger,
                            log_message="send_exec_approval scheduling error",
                        )
                        if _approval_fut is None:
                            raise RuntimeError("send_exec_approval: loop unavailable")
                        _approval_result = _approval_fut.result(timeout=15)
                        if _approval_result.success:
                            return
                        logger.warning(
                            "Button-based approval failed (send returned error), falling back to text: %s",
                            _approval_result.error,
                        )
                    except Exception as _e:
                        logger.warning(
                            "Button-based approval failed, falling back to text: %s", _e
                        )

                # Fallback: plain text approval prompt
                cmd_preview = cmd[:200] + "..." if len(cmd) > 200 else cmd
                msg = (
                    f"⚠️ **Dangerous command requires approval:**\n"
                    f"```\n{cmd_preview}\n```\n"
                    f"Reason: {desc}\n\n"
                    f"Reply `/approve` to execute, `/approve session` to approve this pattern "
                    f"for the session, `/approve always` to approve permanently, or `/deny` to cancel."
                )
                try:
                    _approval_send_fut = safe_schedule_threadsafe(
                        _status_adapter.send(
                            _status_chat_id,
                            msg,
                            metadata=_status_thread_metadata,
                        ),
                        _loop_for_step,
                        logger=logger,
                        log_message="Approval text-send scheduling error",
                    )
                    if _approval_send_fut is not None:
                        _approval_send_fut.result(timeout=15)
                except Exception as _e:
                    logger.error("Failed to send approval request: %s", _e)

            # Prepend pending model switch note so the model knows about the switch
            _pending_notes = getattr(self, '_pending_model_notes', {})
            _msn = _pending_notes.pop(session_key, None) if session_key else None
            if _msn:
                message = _msn + "\n\n" + message

            # Auto-continue: if the loaded history ends with a tool result,
            # the previous agent turn was interrupted mid-work (gateway
            # restart, crash, SIGTERM).  Prepend a system note so the model
            # finishes processing the pending tool results before addressing
            # the user's new message.  (#4493)
            #
            # Session-level resume_pending (set on drain-timeout shutdown)
            # escalates the wording — the transcript's last role may be
            # anything (tool, assistant with unfinished work, etc.), so we
            # give a stronger, reason-aware instruction that subsumes the
            # tool-tail case.
            #
            # Freshness gate (#16802): both branches are gated on the age
            # of the last persisted transcript row.  That is the correct
            # "when did we last do anything here" signal for both the
            # resume_pending path (restart watchdog) and the tool-tail
            # path (in-flight tool loop killed).  We read ``history[-1]``
            # here because ``agent_history`` has already stripped the
            # ``timestamp`` field off tool/tool_call rows for API purity
            # (see the `k != "timestamp"` filter above).  Rows without a
            # timestamp (legacy transcripts) are treated as fresh so the
            # historical auto-continue behaviour is preserved.
            _freshness_window = _auto_continue_freshness_window()
            _interruption_is_fresh = _is_fresh_gateway_interruption(
                _last_transcript_timestamp(history),
                window_secs=_freshness_window,
            )

            _resume_entry = None
            if session_key:
                try:
                    _resume_entry = self.session_store._entries.get(session_key)
                except Exception:
                    _resume_entry = None
            _is_resume_pending = bool(
                _resume_entry is not None
                and getattr(_resume_entry, "resume_pending", False)
                and _interruption_is_fresh
            )
            _has_fresh_tool_tail = bool(
                agent_history
                and agent_history[-1].get("role") == "tool"
                and _interruption_is_fresh
            )

            if _is_resume_pending:
                _reason = getattr(_resume_entry, "resume_reason", None) or "restart_timeout"
                _reason_phrase = (
                    "a gateway restart"
                    if _reason == "restart_timeout"
                    else "a gateway shutdown"
                    if _reason == "shutdown_timeout"
                    else "a gateway interruption"
                )
                message = (
                    f"[System note: Your previous turn in this session was interrupted "
                    f"by {_reason_phrase}. The conversation history below is intact. "
                    f"If it contains unfinished tool result(s), process them first and "
                    f"summarize what was accomplished, then address the user's new "
                    f"message below.]\n\n"
                    + message
                )
            elif _has_fresh_tool_tail:
                message = (
                    "[System note: Your previous turn was interrupted before you could "
                    "process the last tool result(s). The conversation history contains "
                    "tool outputs you haven't responded to yet. Please finish processing "
                    "those results and summarize what was accomplished, then address the "
                    "user's new message below.]\n\n"
                    + message
                )

            # Consume one-shot /reload-skills note (if the user ran
            # /reload-skills since their last turn in this session). Same
            # queue pattern as CLI: prepend to the NEXT user message, then
            # clear. Nothing was written to the transcript out-of-band, so
            # message alternation stays intact.
            _pending_notes = getattr(self, "_pending_skills_reload_notes", None)
            if _pending_notes and session_key and session_key in _pending_notes:
                _srn = _pending_notes.pop(session_key, None)
                if _srn:
                    message = _srn + "\n\n" + message

            _approval_session_key = session_key or ""
            _approval_session_token = set_current_session_key(_approval_session_key)
            register_gateway_notify(_approval_session_key, _approval_notify_sync)
            try:
                # If _prepare_inbound_message_text buffered image paths for native
                # attachment, wrap the user turn as an OpenAI-style multimodal
                # content list. Consume-and-clear so subsequent turns on the same
                # runner instance don't re-attach stale images.
                _native_imgs = self._consume_pending_native_image_paths(session_key)
                if _native_imgs:
                    try:
                        from agent.image_routing import build_native_content_parts
                        _parts, _skipped = build_native_content_parts(
                            message,
                            _native_imgs,
                        )
                        if _skipped:
                            logger.warning(
                                "Native image attachment: skipped %d unreadable path(s): %s",
                                len(_skipped), _skipped,
                            )
                        if any(p.get("type") == "image_url" for p in _parts):
                            _run_message: Any = _parts
                        else:
                            # All images failed to read — fall back to plain text.
                            _run_message = message
                    except Exception as _img_exc:
                        logger.warning(
                            "Native image attachment failed, falling back to text: %s",
                            _img_exc,
                        )
                        _run_message = message
                else:
                    _run_message = message

                _api_run_message = _wrap_current_message_with_observed_context(
                    _run_message,
                    observed_group_context,
                )
                _conversation_kwargs = {
                    "conversation_history": agent_history,
                    "task_id": session_id,
                }
                if observed_group_context:
                    _conversation_kwargs["persist_user_message"] = message
                result = agent.run_conversation(_api_run_message, **_conversation_kwargs)
            finally:
                unregister_gateway_notify(_approval_session_key)
                # Cancel any pending clarify entries so blocked agent
                # threads don't hang past the end of the run (interrupt,
                # completion, gateway shutdown).  Idempotent.
                try:
                    from tools.clarify_gateway import clear_session as _clear_clarify_session
                    _clear_clarify_session(_approval_session_key)
                except Exception:
                    pass
                reset_current_session_key(_approval_session_token)
            result_holder[0] = result

            # Signal the stream consumer that the agent is done
            if _stream_consumer is not None:
                _stream_consumer.finish()
            
            # Return final response, or a message if something went wrong
            final_response = result.get("final_response")

            # Extract actual token counts from the agent instance used for this run
            _last_prompt_toks = 0
            _input_toks = 0
            _output_toks = 0
            _context_length = 0
            _agent = agent_holder[0]
            if _agent and hasattr(_agent, "context_compressor"):
                _last_prompt_toks = getattr(_agent.context_compressor, "last_prompt_tokens", 0)
                _input_toks = getattr(_agent, "session_prompt_tokens", 0)
                _output_toks = getattr(_agent, "session_completion_tokens", 0)
                _context_length = getattr(_agent.context_compressor, "context_length", 0) or 0
            _resolved_model = getattr(_agent, "model", None) if _agent else None

            if not final_response:
                error_msg = f"⚠️ {result['error']}" if result.get("error") else ""
                return {
                    "final_response": error_msg,
                    "messages": result.get("messages", []),
                    "api_calls": result.get("api_calls", 0),
                    "failed": result.get("failed", False),
                    "partial": result.get("partial", False),
                    "completed": result.get("completed"),
                    "interrupted": result.get("interrupted", False),
                    "interrupt_message": result.get("interrupt_message"),
                    "error": result.get("error"),
                    "compression_exhausted": result.get("compression_exhausted", False),
                    "tools": tools_holder[0] or [],
                    "history_offset": len(agent_history),
                    "last_prompt_tokens": _last_prompt_toks,
                    "input_tokens": _input_toks,
                    "output_tokens": _output_toks,
                    "model": _resolved_model,
                    "context_length": _context_length,
                    "provider": _route_provider,

                }
            
            # Scan tool results for MEDIA:<path> tags that need to be delivered
            # as native audio/file attachments.  The TTS tool embeds MEDIA: tags
            # in its JSON response, but the model's final text reply usually
            # doesn't include them.  We collect unique tags from tool results and
            # append any that aren't already present in the final response, so the
            # adapter's extract_media() can find and deliver the files exactly once.
            #
            # Scope the scan to THIS turn's tool results only. ``agent_history``
            # was passed into run_conversation as ``conversation_history``, so the
            # agent's returned ``messages`` list is ``agent_history`` followed by
            # the messages produced this turn. Slicing at ``len(agent_history)``
            # isolates the current turn precisely, so a stale MEDIA: path emitted
            # by a tool several turns earlier (still present in the full message
            # list) can never leak onto a later text-only reply. (Fixes #34608)
            #
            # Path-based deduplication against _history_media_paths (collected
            # before run_conversation) is retained as a secondary guard. It is
            # also the sole guard on the fallback branch taken when mid-run
            # context compression shrinks the message list below the original
            # history length, preserving the compression-safe behaviour of #160.
            if "MEDIA:" not in final_response:
                media_tags, has_voice_directive = _collect_auto_append_media_tags(
                    result.get("messages", []),
                    history_offset=len(agent_history),
                    history_media_paths=_history_media_paths,
                )

                if media_tags:
                    seen = set()
                    unique_tags = []
                    for tag in media_tags:
                        if tag not in seen:
                            seen.add(tag)
                            unique_tags.append(tag)
                    if has_voice_directive:
                        unique_tags.insert(0, "[[audio_as_voice]]")
                    final_response = final_response + "\n" + "\n".join(unique_tags)
            
            # Sync session_id: the agent may have created a new session during
            # mid-run context compression (_compress_context splits sessions).
            # If so, update the session store entry so the NEXT message loads
            # the compressed transcript, not the stale pre-compression one.
            agent = agent_holder[0]
            _session_was_split = False
            if agent and session_key and hasattr(agent, 'session_id') and agent.session_id != session_id:
                _session_was_split = True
                logger.info(
                    "Session split detected: %s → %s (compression)",
                    session_id, agent.session_id,
                )
                entry = self.session_store._entries.get(session_key)
                if entry:
                    entry.session_id = agent.session_id
                    self.session_store._save()

                # If this is a Telegram DM and source.thread_id was lost during
                # the session split (synthetic / recovered event), restore it
                # from the binding so _thread_metadata_for_source produces the
                # correct message_thread_id instead of routing to the General
                # thread.  Failure here is non-fatal — we log and continue;
                # worst case the message lands in General, which is the
                # pre-fix behaviour.
                if (
                    getattr(source, "platform", None) == Platform.TELEGRAM
                    and getattr(source, "chat_type", None) == "dm"
                    and getattr(source, "thread_id", None) is None
                    and self._session_db is not None
                ):
                    try:
                        _binding = self._session_db.get_telegram_topic_binding_by_session(
                            session_id=agent.session_id,
                        )
                        if _binding and _binding.get("thread_id"):
                            source.thread_id = str(_binding["thread_id"])
                            logger.debug(
                                "Restored source.thread_id=%s from binding after session split %s → %s",
                                source.thread_id,
                                session_id,
                                agent.session_id,
                            )
                    except Exception:
                        logger.debug(
                            "Failed to restore thread_id from binding after session split",
                            exc_info=True,
                        )

            effective_session_id = getattr(agent, 'session_id', session_id) if agent else session_id

            # When compression created a new session, the messages list was
            # shortened.  Using the original history offset would produce an
            # empty new_messages slice, causing the gateway to write only a
            # user/assistant pair — losing the compressed summary and tail.
            # Reset to 0 so the gateway writes ALL compressed messages.
            _effective_history_offset = 0 if _session_was_split else len(agent_history)

            # Auto-generate session title after first exchange (non-blocking)
            if final_response and self._session_db:
                try:
                    from agent.title_generator import maybe_auto_title
                    all_msgs = result_holder[0].get("messages", []) if result_holder[0] else []
                    # In Gateway mode, auto-title failures must NOT be
                    # surfaced as user-visible messages (fixes #23246).
                    # Log them at debug level only — they are not actionable
                    # to the end user. CLI mode keeps the existing behaviour
                    # via the agent's _emit_auxiliary_failure path.
                    def _title_failure_cb(task: str, exc: BaseException) -> None:
                        logger.debug(
                            "Gateway auto-title failure suppressed (not user-visible): %s: %s",
                            task, exc,
                        )
                    maybe_auto_title_kwargs = {
                        "failure_callback": _title_failure_cb,
                        "main_runtime": {
                            "model": getattr(agent, "model", None),
                            "provider": getattr(agent, "provider", None),
                            "base_url": getattr(agent, "base_url", None),
                            "api_key": getattr(agent, "api_key", None),
                            "api_mode": getattr(agent, "api_mode", None),
                        } if agent else None,
                    }
                    if self._is_telegram_topic_lane(source):
                        maybe_auto_title_kwargs["title_callback"] = lambda title: self._schedule_telegram_topic_title_rename(
                            source,
                            effective_session_id,
                            title,
                        )
                    maybe_auto_title(
                        self._session_db,
                        effective_session_id,
                        message,
                        final_response,
                        all_msgs,
                        **maybe_auto_title_kwargs,
                    )
                except Exception:
                    pass

            return {
                "final_response": final_response,
                "last_reasoning": result.get("last_reasoning"),
                "messages": result_holder[0].get("messages", []) if result_holder[0] else [],
                "api_calls": result_holder[0].get("api_calls", 0) if result_holder[0] else 0,
                "completed": result_holder[0].get("completed") if result_holder[0] else None,
                "interrupted": result_holder[0].get("interrupted", False) if result_holder[0] else False,
                "partial": result_holder[0].get("partial", False) if result_holder[0] else False,
                "error": result_holder[0].get("error") if result_holder[0] else None,
                "interrupt_message": result_holder[0].get("interrupt_message") if result_holder[0] else None,
                "tools": tools_holder[0] or [],
                "history_offset": _effective_history_offset,
                "last_prompt_tokens": _last_prompt_toks,
                "input_tokens": _input_toks,
                "output_tokens": _output_toks,
                "model": _resolved_model,
                "context_length": _context_length,
                "session_id": effective_session_id,
                "response_previewed": result.get("response_previewed", False),
                "response_transformed": result.get("response_transformed", False),
            }
        
        # Start progress message sender if enabled
        progress_task = None
        if tool_progress_enabled:
            progress_task = asyncio.create_task(send_progress_messages())

        # Start stream consumer task — polls for consumer creation since it
        # happens inside run_sync (thread pool) after the agent is constructed.
        stream_task = None

        async def _start_stream_consumer():
            """Wait for the stream consumer to be created, then run it."""
            for _ in range(200):  # Up to 10s wait
                if stream_consumer_holder[0] is not None:
                    await stream_consumer_holder[0].run()
                    return
                await asyncio.sleep(0.05)

        stream_task = asyncio.create_task(_start_stream_consumer())
        
        # Track this agent as running for this session (for interrupt support)
        # We do this in a callback after the agent is created
        async def track_agent():
            # Wait for agent to be created
            while agent_holder[0] is None:
                await asyncio.sleep(0.05)
            if not session_key:
                return
            # Only promote the sentinel to the real agent if this run is still
            # current.  If /stop or /new bumped the generation while we were
            # spinning up, leave the newer run's slot alone — we'll be
            # discarded by the stale-result check in _handle_message_with_agent.
            if run_generation is not None and not self._is_session_run_current(
                session_key, run_generation
            ):
                logger.info(
                    "Skipping stale agent promotion for %s — generation %s is no longer current",
                    session_key or "",
                    run_generation,
                )
                return
            self._running_agents[session_key] = agent_holder[0]
            if self._draining:
                self._update_runtime_status("draining")
        
        tracking_task = asyncio.create_task(track_agent())
        
        # Monitor for interrupts from the adapter (new messages arriving).
        # This is the PRIMARY interrupt path for regular text messages —
        # Level 1 (base.py) catches them before _handle_message() is reached,
        # so the Level 2 running_agent.interrupt() path never fires.
        # The inactivity poll loop below has a BACKUP check in case this
        # task dies (no error handling = silent death = lost interrupts).
        _interrupt_detected = asyncio.Event()  # shared with backup check

        async def monitor_for_interrupt():
            if not session_key:
                return

            while True:
                await asyncio.sleep(0.2)  # Check every 200ms
                try:
                    # Re-resolve adapter each iteration so reconnects don't
                    # leave us holding a stale reference.
                    _adapter = self.adapters.get(source.platform)
                    if not _adapter:
                        continue
                    # Check if adapter has a pending interrupt for this session.
                    # Must use session_key (build_session_key output) — NOT
                    # source.chat_id — because the adapter stores interrupt events
                    # under the full session key.
                    if hasattr(_adapter, 'has_pending_interrupt') and _adapter.has_pending_interrupt(session_key):
                        agent = agent_holder[0]
                        if agent:
                            # Peek at the pending message text WITHOUT consuming it.
                            # The message must remain in _pending_messages so the
                            # post-run dequeue at _dequeue_pending_event() can
                            # retrieve the full MessageEvent (with media metadata).
                            # If we pop here, a race exists: the agent may finish
                            # before checking _interrupt_requested, and the message
                            # is lost — neither the interrupt path nor the dequeue
                            # path finds it.
                            _peek_event = _adapter._pending_messages.get(session_key)
                            pending_text = _peek_event.text if _peek_event else None
                            logger.debug("Interrupt detected from adapter, signaling agent...")
                            agent.interrupt(pending_text)
                            _interrupt_detected.set()
                            break
                except asyncio.CancelledError:
                    raise
                except Exception as _mon_err:
                    logger.debug("monitor_for_interrupt error (will retry): %s", _mon_err)
        
        interrupt_monitor = asyncio.create_task(monitor_for_interrupt())

        # Periodic "still working" notifications for long-running tasks.
        # Fires every N seconds so the user knows the agent hasn't died.
        # Config: agent.gateway_notify_interval in config.yaml, or
        # HERMES_AGENT_NOTIFY_INTERVAL env var.  Default 180s (3 min).
        # 0 = disable notifications.
        _NOTIFY_INTERVAL_RAW = _float_env("HERMES_AGENT_NOTIFY_INTERVAL", 180)
        _NOTIFY_INTERVAL = _NOTIFY_INTERVAL_RAW if _NOTIFY_INTERVAL_RAW > 0 else None
        if not bool(
            resolve_display_setting(
                user_config,
                platform_key,
                "long_running_notifications",
                True,
            )
        ):
            _NOTIFY_INTERVAL = None
        _notify_start = time.time()

        async def _notify_long_running():
            if _NOTIFY_INTERVAL is None:
                return  # Notifications disabled (gateway_notify_interval: 0)
            _notify_adapter = self.adapters.get(source.platform)
            if not _notify_adapter:
                return
            # Track the heartbeat message id so we can edit-in-place on
            # platforms that support it (Telegram, Discord, Slack, etc.)
            # instead of spamming a new "Still working" bubble every
            # interval. Falls back to send-new when edit fails or isn't
            # supported by the adapter.
            _heartbeat_msg_id: Optional[str] = None
            while True:
                await asyncio.sleep(_NOTIFY_INTERVAL)
                _elapsed_mins = int((time.time() - _notify_start) // 60)
                # Include agent activity context if available. Default
                # heartbeat is terse: elapsed + current tool. Verbose
                # iteration counter is gated on busy_ack_detail so users
                # who want it can opt in per platform.
                _agent_ref = agent_holder[0]
                _status_detail = ""
                _want_iteration_detail = bool(
                    resolve_display_setting(
                        user_config,
                        platform_key,
                        "busy_ack_detail",
                        True,
                    )
                )
                if _agent_ref and hasattr(_agent_ref, "get_activity_summary"):
                    try:
                        _a = _agent_ref.get_activity_summary()
                        _parts = []
                        if _want_iteration_detail:
                            _parts.append(
                                f"iteration {_a['api_call_count']}/{_a['max_iterations']}"
                            )
                        _action = _a.get("current_tool") or _a.get("last_activity_desc")
                        if _action:
                            _parts.append(str(_action))
                        if _parts:
                            _status_detail = " — " + ", ".join(_parts)
                    except Exception:
                        pass
                _heartbeat_text = f"⏳ Working — {_elapsed_mins} min{_status_detail}"
                try:
                    _notify_res = None
                    if _heartbeat_msg_id:
                        try:
                            _notify_res = await _notify_adapter.edit_message(
                                source.chat_id,
                                _heartbeat_msg_id,
                                _heartbeat_text,
                            )
                        except Exception as _ee:
                            logger.debug("Heartbeat edit failed: %s", _ee)
                            _notify_res = None
                    if not (_notify_res and getattr(_notify_res, "success", False)):
                        _notify_res = await _notify_adapter.send(
                            source.chat_id,
                            _heartbeat_text,
                            metadata=_status_thread_metadata,
                        )
                        if getattr(_notify_res, "success", False) and getattr(
                            _notify_res, "message_id", None
                        ):
                            _heartbeat_msg_id = str(_notify_res.message_id)
                            if _cleanup_progress:
                                _cleanup_msg_ids.append(_heartbeat_msg_id)
                except Exception as _ne:
                    logger.debug("Long-running notification error: %s", _ne)

        _notify_task = asyncio.create_task(_notify_long_running())

        try:
            # Run in thread pool to not block.  Use an *inactivity*-based
            # timeout instead of a wall-clock limit: the agent can run for
            # hours if it's actively calling tools / receiving stream tokens,
            # but a hung API call or stuck tool with no activity for the
            # configured duration is caught and killed.  (#4815)
            #
            # Config: agent.gateway_timeout in config.yaml, or
            # HERMES_AGENT_TIMEOUT env var (env var takes precedence).
            # Default 1800s (30 min inactivity).  0 = unlimited.
            _agent_timeout_raw = _float_env("HERMES_AGENT_TIMEOUT", 1800)
            _agent_timeout = _agent_timeout_raw if _agent_timeout_raw > 0 else None
            _agent_warning_raw = _float_env("HERMES_AGENT_TIMEOUT_WARNING", 900)
            _agent_warning = _agent_warning_raw if _agent_warning_raw > 0 else None
            _warning_fired = False
            _executor_task = asyncio.ensure_future(
                self._run_in_executor_with_context(run_sync)
            )

            _inactivity_timeout = False
            _POLL_INTERVAL = 5.0

            if _agent_timeout is None:
                # Unlimited — still poll periodically for backup interrupt
                # detection in case monitor_for_interrupt() silently died.
                response = None
                while True:
                    done, _ = await asyncio.wait(
                        {_executor_task}, timeout=_POLL_INTERVAL
                    )
                    if done:
                        response = _executor_task.result()
                        break
                    # Backup interrupt check: if the monitor task died or
                    # missed the interrupt, catch it here.
                    if not _interrupt_detected.is_set() and session_key:
                        _backup_adapter = self.adapters.get(source.platform)
                        _backup_agent = agent_holder[0]
                        if (_backup_adapter and _backup_agent
                                and hasattr(_backup_adapter, 'has_pending_interrupt')
                                and _backup_adapter.has_pending_interrupt(session_key)):
                            _bp_event = _backup_adapter._pending_messages.get(session_key)
                            _bp_text = _bp_event.text if _bp_event else None
                            logger.info(
                                "Backup interrupt detected for session %s "
                                "(monitor task state: %s)",
                                session_key,
                                "done" if interrupt_monitor.done() else "running",
                            )
                            _backup_agent.interrupt(_bp_text)
                            _interrupt_detected.set()
            else:
                # Poll loop: check the agent's built-in activity tracker
                # (updated by _touch_activity() on every tool call, API
                # call, and stream delta) every few seconds.
                response = None
                while True:
                    done, _ = await asyncio.wait(
                        {_executor_task}, timeout=_POLL_INTERVAL
                    )
                    if done:
                        response = _executor_task.result()
                        break
                    # Agent still running — check inactivity.
                    _agent_ref = agent_holder[0]
                    _idle_secs = 0.0
                    if _agent_ref and hasattr(_agent_ref, "get_activity_summary"):
                        try:
                            _act = _agent_ref.get_activity_summary()
                            _idle_secs = _act.get("seconds_since_activity", 0.0)
                        except Exception:
                            pass
                    # Staged warning: fire once before escalating to full timeout.
                    if (not _warning_fired and _agent_warning is not None
                            and _idle_secs >= _agent_warning):
                        _warning_fired = True
                        _warn_adapter = self.adapters.get(source.platform)
                        if _warn_adapter:
                            _elapsed_warn = int(_agent_warning // 60) or 1
                            _remaining_mins = int((_agent_timeout - _agent_warning) // 60) or 1
                            try:
                                await _warn_adapter.send(
                                    source.chat_id,
                                    f"⚠️ No activity for {_elapsed_warn} min. "
                                    f"If the agent does not respond soon, it will "
                                    f"be timed out in {_remaining_mins} min. "
                                    f"You can continue waiting or use /reset.",
                                    metadata=_status_thread_metadata,
                                )
                            except Exception as _warn_err:
                                logger.debug("Inactivity warning send error: %s", _warn_err)
                    if _idle_secs >= _agent_timeout:
                        _inactivity_timeout = True
                        break
                    # Backup interrupt check (same as unlimited path).
                    if not _interrupt_detected.is_set() and session_key:
                        _backup_adapter = self.adapters.get(source.platform)
                        _backup_agent = agent_holder[0]
                        if (_backup_adapter and _backup_agent
                                and hasattr(_backup_adapter, 'has_pending_interrupt')
                                and _backup_adapter.has_pending_interrupt(session_key)):
                            _bp_event = _backup_adapter._pending_messages.get(session_key)
                            _bp_text = _bp_event.text if _bp_event else None
                            logger.info(
                                "Backup interrupt detected for session %s "
                                "(monitor task state: %s)",
                                session_key,
                                "done" if interrupt_monitor.done() else "running",
                            )
                            _backup_agent.interrupt(_bp_text)
                            _interrupt_detected.set()

            if _inactivity_timeout:
                # Build a diagnostic summary from the agent's activity tracker.
                _timed_out_agent = agent_holder[0]
                _activity = {}
                if _timed_out_agent and hasattr(_timed_out_agent, "get_activity_summary"):
                    try:
                        _activity = _timed_out_agent.get_activity_summary()
                    except Exception:
                        pass

                _last_desc = _activity.get("last_activity_desc", "unknown")
                _secs_ago = _activity.get("seconds_since_activity", 0)
                _cur_tool = _activity.get("current_tool")
                _iter_n = _activity.get("api_call_count", 0)
                _iter_max = _activity.get("max_iterations", 0)

                logger.error(
                    "Agent idle for %.0fs (timeout %.0fs) in session %s "
                    "| last_activity=%s | iteration=%s/%s | tool=%s",
                    _secs_ago, _agent_timeout, session_key,
                    _last_desc, _iter_n, _iter_max,
                    _cur_tool or "none",
                )

                # Interrupt the agent if it's still running so the thread
                # pool worker is freed.
                if _timed_out_agent and hasattr(_timed_out_agent, "interrupt"):
                    _timed_out_agent.interrupt(_INTERRUPT_REASON_TIMEOUT)

                _timeout_mins = int(_agent_timeout // 60) or 1

                # Construct a user-facing message with diagnostic context.
                _diag_lines = [
                    f"⏱️ Agent inactive for {_timeout_mins} min — no tool calls "
                    f"or API responses."
                ]
                if _cur_tool:
                    _diag_lines.append(
                        f"The agent appears stuck on tool `{_cur_tool}` "
                        f"({_secs_ago:.0f}s since last activity, "
                        f"iteration {_iter_n}/{_iter_max})."
                    )
                else:
                    _diag_lines.append(
                        f"Last activity: {_last_desc} ({_secs_ago:.0f}s ago, "
                        f"iteration {_iter_n}/{_iter_max}). "
                        "The agent may have been waiting on an API response."
                    )
                _diag_lines.append(
                    "To increase the limit, set agent.gateway_timeout in config.yaml "
                    "(value in seconds, 0 = no limit) and restart the gateway.\n"
                    "Try again, or use /reset to start fresh."
                )

                response = {
                    "final_response": "\n".join(_diag_lines),
                    "messages": result_holder[0].get("messages", []) if result_holder[0] else [],
                    "api_calls": _iter_n,
                    "tools": tools_holder[0] or [],
                    "history_offset": 0,
                    "failed": True,
                    "provider": _route_provider,
                }

            # Track fallback model state: if the agent switched to a
            # fallback model during this run, persist it so /model shows
            # the actually-active model instead of the config default.
            # Skip eviction when the run failed — evicting a failed agent
            # forces MCP reinit on the next message for no benefit (the
            # same error will recur).  This was the root cause of #7130:
            # a bad model ID triggered fallback → eviction → recreation →
            # MCP reinit → same 400 → loop, burning 91% CPU for hours.
            _agent = agent_holder[0]
            _result_for_fb = result_holder[0]
            _run_failed = _result_for_fb.get("failed") if _result_for_fb else False
            if _agent is not None and hasattr(_agent, 'model') and not _run_failed:
                _cfg_model = _resolve_gateway_model()
                if _agent.model != _cfg_model and not self._is_intentional_model_switch(session_key, _agent.model):
                    # Fallback activated on a successful run — evict cached
                    # agent so the next message retries the primary model.
                    self._evict_cached_agent(session_key)

            # Check if we were interrupted OR have a queued message (/queue).
            result = result_holder[0]
            adapter = self.adapters.get(source.platform)
            
            # Get pending message from adapter.
            # Use session_key (not source.chat_id) to match adapter's storage keys.
            pending_event = None
            pending = None
            if result and adapter and session_key:
                pending_event = _dequeue_pending_event(adapter, session_key)
                # /queue overflow: after consuming the adapter's "next-up"
                # slot, promote the next queued event into it so the
                # recursive run's drain will see it.  This keeps the slot
                # occupied for the full FIFO chain, which (a) preserves
                # order, and (b) causes any mid-chain /queue to correctly
                # route to overflow rather than jumping the queue.
                pending_event = self._promote_queued_event(session_key, adapter, pending_event)
                if result.get("interrupted") and not pending_event and result.get("interrupt_message"):
                    interrupt_message = result.get("interrupt_message")
                    if _is_control_interrupt_message(interrupt_message):
                        logger.info(
                            "Ignoring control interrupt message for session %s: %s",
                            session_key or "?",
                            interrupt_message,
                        )
                    else:
                        pending = interrupt_message
                elif pending_event:
                    pending = pending_event.text or _build_media_placeholder(pending_event)
                    logger.debug("Processing queued message after agent completion: '%s...'", pending[:40])

            # Leftover /steer: if a steer arrived after the last tool batch
            # (e.g. during the final API call), the agent couldn't inject it
            # and returned it in result["pending_steer"]. Deliver it as the
            # next user turn so it isn't silently dropped.
            if result and not pending and not pending_event:
                _leftover_steer = result.get("pending_steer")
                if _leftover_steer:
                    pending = _leftover_steer
                    logger.debug("Delivering leftover /steer as next turn: '%s...'", pending[:40])

            # Safety net: if the pending text is a slash command (e.g. "/stop",
            # "/new"), discard it — commands should never be passed to the agent
            # as user input.  The primary fix is in base.py (commands bypass the
            # active-session guard), but this catches edge cases where command
            # text leaks through the interrupt_message fallback.
            if pending and pending.strip().startswith("/"):
                _pending_parts = pending.strip().split(None, 1)
                _pending_cmd_word = _pending_parts[0][1:].lower() if _pending_parts else ""
                if _pending_cmd_word:
                    try:
                        from hermes_cli.commands import resolve_command as _rc_pending
                        if _rc_pending(_pending_cmd_word):
                            logger.info(
                                "Discarding command '/%s' from pending queue — "
                                "commands must not be passed as agent input",
                                _pending_cmd_word,
                            )
                            pending_event = None
                            pending = None
                    except Exception:
                        pass

            if self._draining and (pending_event or pending):
                logger.info(
                    "Discarding pending follow-up for session %s during gateway %s",
                    session_key or "?",
                    self._status_action_label(),
                )
                pending_event = None
                pending = None

            if pending_event or pending:
                logger.debug("Processing pending message: '%s...'", pending[:40])

                # Clear the adapter's interrupt event so the next _run_agent call
                # doesn't immediately re-trigger the interrupt before the new agent
                # even makes its first API call (this was causing an infinite loop).
                if adapter and hasattr(adapter, '_active_sessions') and session_key and session_key in adapter._active_sessions:
                    adapter._active_sessions[session_key].clear()

                # Cap recursion depth to prevent resource exhaustion when the
                # user sends multiple messages while the agent keeps failing. (#816)
                if _interrupt_depth >= self._MAX_INTERRUPT_DEPTH:
                    logger.warning(
                        "Interrupt recursion depth %d reached for session %s — "
                        "queueing message instead of recursing.",
                        _interrupt_depth, session_key,
                    )
                    adapter = self.adapters.get(source.platform)
                    if adapter and pending_event:
                        merge_pending_message_event(adapter._pending_messages, session_key, pending_event)
                    elif adapter and hasattr(adapter, 'queue_message'):
                        adapter.queue_message(session_key, pending)
                    return result_holder[0] or {"final_response": response, "messages": history}

                was_interrupted = result.get("interrupted")
                if not was_interrupted:
                    # Queued message after normal completion — deliver the first
                    # response before processing the queued follow-up.
                    # Skip if streaming already delivered it.
                    _sc = stream_consumer_holder[0]
                    if _sc and stream_task:
                        try:
                            await asyncio.wait_for(stream_task, timeout=5.0)
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            stream_task.cancel()
                            try:
                                await stream_task
                            except asyncio.CancelledError:
                                pass
                        except Exception as e:
                            logger.debug("Stream consumer wait before queued message failed: %s", e)
                    _previewed = bool(result.get("response_previewed"))
                    _already_streamed = bool(
                        (_sc and getattr(_sc, "final_response_sent", False))
                        or _previewed
                        or (_sc and getattr(_sc, "final_content_delivered", False))
                    )
                    first_response = result.get("final_response", "")
                    if first_response and not _already_streamed:
                        try:
                            logger.info(
                                "Queued follow-up for session %s: final stream delivery not confirmed; sending first response before continuing.",
                                session_key or "?",
                            )
                            await adapter.send(
                                source.chat_id,
                                first_response,
                                metadata=_status_thread_metadata,
                            )
                        except Exception as e:
                            logger.warning("Failed to send first response before queued message: %s", e)
                    elif first_response:
                        logger.info(
                            "Queued follow-up for session %s: skipping resend because final streamed delivery was confirmed.",
                            session_key or "?",
                        )
                    # Release deferred bg-review notifications now that the
                    # first response has been delivered.  Pop from the
                    # adapter's callback dict (prevents double-fire in
                    # base.py's finally block) and call it.
                    if getattr(type(adapter), "pop_post_delivery_callback", None) is not None:
                        _bg_cb = adapter.pop_post_delivery_callback(
                            session_key,
                            generation=run_generation,
                        )
                        if callable(_bg_cb):
                            try:
                                _bg_result = _bg_cb()
                                if inspect.isawaitable(_bg_result):
                                    await _bg_result
                            except Exception:
                                pass
                    elif adapter and hasattr(adapter, "_post_delivery_callbacks"):
                        _bg_cb = adapter._post_delivery_callbacks.pop(session_key, None)
                        if callable(_bg_cb):
                            try:
                                _bg_result = _bg_cb()
                                if inspect.isawaitable(_bg_result):
                                    await _bg_result
                            except Exception:
                                pass
                # else: interrupted — discard the interrupted response ("Operation
                # interrupted." is just noise; the user already knows they sent a
                # new message).

                updated_history = result.get("messages", history)
                next_source = source
                next_message = pending
                next_message_id = None
                next_channel_prompt = None
                if pending_event is not None:
                    next_source = getattr(pending_event, "source", None) or source
                    if self._is_goal_continuation_event(pending_event) and not self._goal_still_active_for_session(session_id):
                        logger.info(
                            "Discarding stale goal continuation for session %s — goal is no longer active",
                            session_key or "?",
                        )
                        return result
                    next_message = await self._prepare_inbound_message_text(
                        event=pending_event,
                        source=next_source,
                        history=updated_history,
                    )
                    if next_message is None:
                        return result
                    next_message_id = self._reply_anchor_for_event(pending_event)
                    next_channel_prompt = getattr(pending_event, "channel_prompt", None)

                # Restart typing indicator so the user sees activity while
                # the follow-up turn runs.  The outer _process_message_background
                # typing task is still alive but may be stale.
                _followup_adapter = self.adapters.get(source.platform)
                if _followup_adapter:
                    try:
                        await _followup_adapter.send_typing(
                            source.chat_id,
                            metadata=_status_thread_metadata,
                        )
                    except Exception:
                        pass

                followup_result = await self._run_agent(
                    message=next_message,
                    context_prompt=context_prompt,
                    history=updated_history,
                    source=next_source,
                    session_id=session_id,
                    session_key=session_key,
                    run_generation=run_generation,
                    _interrupt_depth=_interrupt_depth + 1,
                    event_message_id=next_message_id,
                    channel_prompt=next_channel_prompt,
                )
                return _preserve_queued_followup_history_offset(result, followup_result)
        finally:
            # Stop progress sender, interrupt monitor, and notification task
            if progress_task:
                progress_task.cancel()
            interrupt_monitor.cancel()
            _notify_task.cancel()

            # Wait for stream consumer to finish its final edit
            if stream_task:
                # If the agent never created a stream consumer (e.g. non-
                # streaming code path, or a test stub returning synchronously)
                # there is nothing to flush — cancel immediately instead of
                # waiting out the 5s timeout on a task that's just polling for
                # a consumer that will never arrive.  This was a 5-second
                # cost per non-streaming test run.
                _has_stream_consumer = (
                    stream_consumer_holder
                    and stream_consumer_holder[0] is not None
                )
                if not _has_stream_consumer:
                    stream_task.cancel()
                    try:
                        await stream_task
                    except asyncio.CancelledError:
                        pass
                else:
                    try:
                        await asyncio.wait_for(stream_task, timeout=5.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        stream_task.cancel()
                        try:
                            await stream_task
                        except asyncio.CancelledError:
                            pass
            
            # Clean up tracking
            tracking_task.cancel()
            if session_key:
                # Only release the slot if this run's generation still owns
                # it.  A /stop or /new that bumped the generation while we
                # were unwinding has already installed its own state; this
                # guard prevents an old run from clobbering it on the way
                # out.
                self._release_running_agent_state(
                    session_key, run_generation=run_generation
                )
            if self._draining:
                self._update_runtime_status("draining")
            
            # Wait for cancelled tasks
            for task in [progress_task, interrupt_monitor, tracking_task, _notify_task]:
                if task:
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        # If streaming already delivered the response, mark it so the
        # caller's send() is skipped (avoiding duplicate messages).
        # BUT: never suppress delivery when the agent failed — the error
        # message is new content the user hasn't seen, and it must reach
        # them even if streaming had sent earlier partial output.
        #
        # Also never suppress when the final response is "(empty)" — this
        # means the model failed to produce content after tool calls (common
        # with mimo-v2-pro, GLM-5, etc.).  The stream consumer may have
        # sent intermediate text ("Let me search for that…") alongside the
        # tool call, setting already_sent=True, but that text is NOT the
        # final answer.  Suppressing delivery here leaves the user staring
        # at silence.  (#10xxx — "agent stops after web search")
        _sc = stream_consumer_holder[0]
        if isinstance(response, dict) and not response.get("failed"):
            _final = response.get("final_response") or ""
            _is_empty_sentinel = not _final or _final == "(empty)"
            _streamed = bool(
                _sc and getattr(_sc, "final_response_sent", False)
            )
            # response_previewed means the interim_assistant_callback already
            # sent the final text via the adapter (non-streaming path).
            _previewed = bool(response.get("response_previewed"))
            _content_delivered = bool(
                _sc and getattr(_sc, "final_content_delivered", False)
            )
            # Plugin hooks (e.g. transform_llm_output) may have appended content
            # after streaming finished — when the response was transformed, always
            # send the final version so the appended content reaches the client.
            _transformed = bool(response.get("response_transformed"))
            if not _is_empty_sentinel and not _transformed and (_streamed or _previewed or _content_delivered):
                logger.info(
                    "Suppressing normal final send for session %s: final delivery already confirmed (streamed=%s previewed=%s content_delivered=%s).",
                    session_key or "?",
                    _streamed,
                    _previewed,
                    _content_delivered,
                )
                response["already_sent"] = True
            elif not _is_empty_sentinel and _transformed and _sc is not None:
                # Plugin hooks transformed the response after streaming — edit the
                # existing streamed message instead of sending a duplicate.
                _sc_msg_id = _sc.message_id
                if _sc_msg_id:
                    try:
                        await _sc.adapter.edit_message(
                            chat_id=source.chat_id,
                            message_id=_sc_msg_id,
                            content=response["final_response"],
                            finalize=True,
                        )
                        response["already_sent"] = True
                        logger.info(
                            "Edited streamed message %s for session %s to include plugin-transformed content.",
                            _sc_msg_id, session_key or "?",
                        )
                    except Exception as _edit_err:
                        logger.warning(
                            "Failed to edit streamed message for session %s: %s",
                            session_key or "?", _edit_err,
                        )

        # Schedule deletion of tracked temporary progress bubbles after the
        # final response lands. Failed runs skip this so bubbles remain as
        # breadcrumbs for the user to see what work happened. Only fires on
        # adapters that support ``delete_message`` (see init above); failures
        # are swallowed — deletion is best-effort.
        if (
            _cleanup_progress
            and _cleanup_adapter is not None
            and _cleanup_msg_ids
            and session_key
            and isinstance(response, dict)
            and not response.get("failed")
            and hasattr(_cleanup_adapter, "register_post_delivery_callback")
        ):
            _ids_snapshot = list(_cleanup_msg_ids)
            _chat_id_snapshot = source.chat_id
            _adapter_snapshot = _cleanup_adapter
            _loop_snapshot = asyncio.get_running_loop()

            def _cleanup_temp_bubbles() -> None:
                async def _delete_all() -> None:
                    for _mid in _ids_snapshot:
                        try:
                            await _adapter_snapshot.delete_message(
                                _chat_id_snapshot, _mid
                            )
                        except Exception:
                            pass
                try:
                    safe_schedule_threadsafe(
                        _delete_all(), _loop_snapshot,
                        logger=logger,
                        log_message="Temp bubble cleanup scheduling error",
                    )
                except Exception:
                    pass

            try:
                _cleanup_adapter.register_post_delivery_callback(
                    session_key,
                    _cleanup_temp_bubbles,
                    generation=run_generation,
                )
            except Exception as _rpe:
                logger.debug("Post-delivery cleanup registration failed: %s", _rpe)

        return response


def _run_planned_stop_watcher(
    stop_event: threading.Event,
    runner,
    loop: asyncio.AbstractEventLoop,
    shutdown_handler,
    *,
    poll_interval: float = 0.5,
) -> None:
    """Poll for the planned-stop marker and trigger graceful shutdown.

    On Windows, ``asyncio.add_signal_handler`` raises NotImplementedError
    for SIGTERM/SIGINT, so the standard signal-driven shutdown path
    never runs when ``hermes gateway stop`` signals the gateway. The
    consequence is that the drain loop is skipped — in-flight agent
    sessions are killed mid-turn and ``resume_pending`` is never set,
    so the next gateway boot has no idea those sessions need to be
    auto-resumed (issue #33778, v0.13.0 session-resume feature broken
    on native Windows).

    This watcher runs on every platform (cheap, defensive) and bridges
    the gap on Windows by translating a filesystem marker into the
    same shutdown-handler invocation a real SIGTERM would have produced
    on POSIX. The CLI's ``hermes_cli.gateway_windows.stop()`` writes
    the marker via ``write_planned_stop_marker(pid)`` and then waits
    for the gateway PID to exit; this watcher is what makes that
    exit happen cleanly.

    On POSIX this is a no-op safety net — the signal handler always
    races us to consuming the marker file because it fires synchronously
    from the kernel's signal delivery.

    Args:
        stop_event: cleared by start_gateway() during normal shutdown
            to tell the watcher to exit.
        runner: the GatewayRunner instance; we check ``_running`` and
            ``_draining`` to avoid triggering shutdown if the gateway
            is already in one of those states.
        loop: the asyncio event loop the shutdown handler must run on.
        shutdown_handler: same callable that's wired to SIGTERM —
            tolerates a ``None`` signal argument (planned stop case)
            and consumes the marker via
            ``consume_planned_stop_marker_for_self()``.
        poll_interval: seconds between marker checks. 0.5s gives a
            responsive shutdown without burning CPU.
    """
    from gateway.status import (
        _get_planned_stop_marker_path,
        planned_stop_marker_targets_self,
    )
    marker_path = _get_planned_stop_marker_path()
    while not stop_event.is_set():
        try:
            if (
                marker_path.exists()
                and not getattr(runner, "_draining", False)
                and getattr(runner, "_running", False)
            ):
                # A marker existing is NOT sufficient — it may have been
                # written for a PREVIOUS gateway instance (different PID)
                # and left behind because that process exited before the
                # CLI's stop() could clean it up. Firing the handler on a
                # stale/foreign marker drives the gateway into shutdown,
                # then consume_planned_stop_marker_for_self() correctly
                # reports a PID mismatch — but by then we're already
                # stopping, so it's logged as an unexpected "UNKNOWN" exit
                # and the watchdog crash-loops the gateway (issue #34597,
                # a regression from PR #33798 which added this watcher
                # without the PID check).
                #
                # Only fire when the marker actually targets us. The probe
                # is non-destructive on a match (the handler does the
                # authoritative consume on the loop thread) and self-heals
                # by unlinking stale/malformed markers so they cannot wedge
                # a freshly booted gateway.
                if not planned_stop_marker_targets_self():
                    stop_event.wait(poll_interval)
                    continue
                # Drive the same path as a real signal handler.
                # Pass signal=None — the handler tolerates that and consumes
                # the marker via consume_planned_stop_marker_for_self,
                # which also validates target_pid + start_time match us.
                loop.call_soon_threadsafe(shutdown_handler, None)
                # Done — the handler will set _draining; we exit on next tick.
                break
        except Exception as _e:
            logger.debug("Planned-stop watcher tick error: %s", _e)
        stop_event.wait(poll_interval)


    logger.info("Cron ticker stopped")# Re-export start_gateway for external imports (e.g., hermes_cli.main)
start_gateway = gateway_startup.start_gateway


def main():
    """CLI entry point for the gateway."""
    # Force UTF-8 stdio on Windows — gateway logs and startup banner would
    # otherwise UnicodeEncodeError on cp1252 consoles.  No-op on POSIX.
    try:
        from hermes_cli.stdio import configure_windows_stdio
        configure_windows_stdio()
    except Exception:
        pass

    import argparse
    
    parser = argparse.ArgumentParser(description="Hermes Gateway - Multi-platform messaging")
    parser.add_argument("--config", "-c", help="Path to gateway config file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    config = None
    if args.config:
        import yaml
        with open(args.config, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            config = GatewayConfig.from_dict(data)
    
    # Run the gateway - exit with code 1 if no platforms connected,
    # so systemd Restart=on-failure will retry on transient errors (e.g. DNS)
    success = asyncio.run(start_gateway(config))
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()