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
        return await command_handlers._handle_adapter_fatal_error(
            runner=self,
            adapter=adapter,
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
            runner=self,
            event=event,
            source=source,
            history=history,
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
        return await command_handlers.handle_status_command(
            runner=self,
            event=event,
        )

    async def _handle_agents_command(self, event: MessageEvent) -> str:
        return await command_handlers.handle_agents_command(
            runner=self,
            event=event,
        )

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
        return await command_handlers.handle_stop_command(
            runner=self,
            event=event,
        )

    async def _handle_platform_command(self, event: MessageEvent) -> str:
        return await command_handlers.handle_platform_command(
            runner=self,
            event=event,
        )

    async def _handle_restart_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        return await command_handlers.handle_restart_command(
            runner=self,
            event=event,
        )

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
        return await command_handlers.handle_voice_channel_join(
            runner=self,
            event=event,
        )

    async def _handle_voice_channel_leave(self, event: MessageEvent) -> str:
        return await command_handlers.handle_voice_channel_leave(
            runner=self,
            event=event,
        )

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
        return await command_handlers._handle_voice_channel_input(
            runner=self,
            guild_id=guild_id,
            user_id=user_id,
            transcript=transcript,
        )
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
        return command_handlers._should_send_voice_reply(
            runner=self,
            event=event,
            response=response,
            agent_messages=agent_messages,
            already_sent=already_sent,
        )
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
            runner=self,
            response=response,
            event=event,
            adapter=adapter,
        )

    async def _handle_rollback_command(self, event: MessageEvent) -> str:
        return await command_handlers.handle_rollback_command(
            runner=self,
            event=event,
        )

    async def _handle_background_command(self, event: MessageEvent) -> str:
        """Handle /background <prompt> — run a prompt in a separate background session.

        Spawns a new AIAgent in a background thread with its own session.
        When it completes, sends the result back to the same chat without
        modifying the active session's conversation history.
        """
        return await command_handlers.handle_background_command(
            runner=self,
            event=event,
        )

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
        return await command_handlers.handle_fast_command(
            runner=self,
            event=event,
        )

    async def _handle_yolo_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        return await command_handlers.handle_yolo_command(
            runner=self,
            event=event,
        )

    async def _handle_verbose_command(self, event: MessageEvent) -> str:
        return await command_handlers.handle_verbose_command(
            runner=self,
            event=event,
        )

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
        return await command_handlers._get_telegram_topic_capabilities(
            runner=self,
            source=source,
        )

    async def _ensure_telegram_system_topic(self, source: SessionSource) -> None:
        """Create/pin the managed System topic after /topic activation when possible."""
        return await command_handlers._ensure_telegram_system_topic(
            runner=self,
            source=source,
        )

    async def _send_telegram_topic_setup_image(self, source: SessionSource) -> None:
        """Send the bundled BotFather Threads Settings screenshot when available."""
        return await command_handlers._send_telegram_topic_setup_image(
            runner=self,
            source=source,
        )

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
        return command_handlers._should_send_telegram_capability_hint(
            runner=self,
            source=source,
        )

    def _telegram_topic_help_text(self) -> str:
        return command_handlers._telegram_topic_help_text()

    def _disable_telegram_topic_mode_for_chat(self, source: SessionSource) -> str:
        """Cleanly disable topic mode for a chat via /topic off."""
        return command_handlers._disable_telegram_topic_mode_for_chat(
            runner=self,
            source=source,
        )

    async def _handle_topic_command(self, event: MessageEvent, args: str = "") -> str:
        """Handle /topic for Telegram DM user-managed topic sessions."""
        return await command_handlers.handle_topic_command(
            runner=self,
            event=event,
            args=args,
        )

    def _telegram_topic_root_status_message(self, source: SessionSource) -> str:
        return command_handlers._telegram_topic_root_status_message(
            runner=self,
            source=source,
        )

    async def _restore_telegram_topic_session(self, event: MessageEvent, raw_session_id: str) -> str:
        """Restore an existing Telegram-owned Hermes session into this topic."""
        return await command_handlers._restore_telegram_topic_session(
            runner=self,
            event=event,
            raw_session_id=raw_session_id,
        )

    async def _handle_title_command(self, event: MessageEvent) -> str:
        """Handle /title command — set or show the current session's title."""
        return await command_handlers.handle_title_command(
            runner=self,
            event=event,
        )

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
        return await command_handlers.handle_branch_command(
            runner=self,
            event=event,
        )

    async def _handle_usage_command(self, event: MessageEvent) -> str:
        """Handle /usage command -- show token usage for the current session.

        Delegates to the extracted handler in gateway/command_handlers.py.
        """
        return await command_handlers.handle_usage_command(self, event)
    async def _handle_insights_command(self, event: MessageEvent) -> str:
        """Handle /insights command -- show usage insights and analytics."""
        return await command_handlers.handle_insights_command(
            runner=self,
            event=event,
        )

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
        return await command_handlers.handle_bundles_command(
            runner=self,
            event=event,
        )

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
        return await command_handlers.handle_approve_command(
            runner=self,
            event=event,
        )

    async def _handle_deny_command(self, event: MessageEvent) -> str:
        """Handle /deny command — reject pending dangerous command(s).

        Signals blocked agent thread(s) with a 'deny' result so they receive
        a definitive BLOCKED message, same as the CLI deny flow.

        ``/deny`` denies the oldest; ``/deny all`` denies everything.
        """
        return await command_handlers.handle_deny_command(
            runner=self,
            event=event,
        )

    async def _handle_debug_command(self, event: MessageEvent) -> str:
        """Handle /debug — upload debug report (summary only) and return paste URLs.

        Gateway uploads ONLY the summary report (system info + log tails),
        NOT full log files, to protect conversation privacy.  Users who need
        full log uploads should use ``hermes debug share`` from the CLI.
        """
        return await command_handlers.handle_debug_command(
            runner=self,
            event=event,
        )

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

        return agent_cache.enforce_agent_cache_cap(
            agent_cache=getattr(self, "_agent_cache", None),
            agent_cache_lock=getattr(self, "_agent_cache_lock", None),
            running_agents=getattr(self, "_running_agents", {}),
            agent_pending_sentinel=_AGENT_PENDING_SENTINEL,
            release_fn=self._release_evicted_agent_soft,
        )

    def _sweep_idle_cached_agents(self) -> int:
        """Evict cached agents whose AIAgent has been idle > _AGENT_CACHE_IDLE_TTL_SECS.

        Safe to call from the session expiry watcher without holding the
        cache lock — acquires it internally.  Returns the number of entries
        evicted.  Resource cleanup is scheduled on daemon threads.

        Agents currently in _running_agents are SKIPPED for the same reason
        as _enforce_agent_cache_cap: tearing down an active turn's clients
        mid-flight would crash the request.
        """

        return agent_cache.sweep_idle_cached_agents(
            agent_cache=getattr(self, "_agent_cache", None),
            agent_cache_lock=getattr(self, "_agent_cache_lock", None),
            running_agents=getattr(self, "_running_agents", {}),
            agent_pending_sentinel=_AGENT_PENDING_SENTINEL,
            release_fn=self._release_evicted_agent_soft,
        )

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