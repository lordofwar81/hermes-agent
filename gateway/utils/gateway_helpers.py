"""
Gateway utility functions.

This module contains pure utility functions extracted from gateway/run.py
to improve code organization and maintainability. These functions have no
dependencies on gateway state and can be safely imported and used independently.
"""

import asyncio
import logging
import os
import re
import wave
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# --- Regex patterns for gateway functionality ---

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

_GATEWAY_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._\-]{20,}\b"),
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

# --- Control interrupt message constants ---

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


# --- Platform and validation utilities ---

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


# --- Secret and error handling utilities ---

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


# --- Telegram-specific utilities ---

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


# --- Timestamp and duration utilities ---

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


def _format_duration(seconds: float) -> str:
    """Format duration in seconds as HH:MM:SS or MM:SS."""
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


# --- Control message utilities ---

def _is_control_interrupt_message(message: Optional[str]) -> bool:
    """Return True when an interrupt message is internal control flow."""
    if not message:
        return False
    normalized = " ".join(str(message).strip().split()).lower()
    return normalized in _CONTROL_INTERRUPT_MESSAGES


# --- Process notification utilities ---

def _format_gateway_process_notification(evt: dict) -> "str | None":
    """Format a process event for gateway logging."""
    event_type = evt.get("type") or evt.get("event_type", "")
    message = evt.get("message", "")
    if event_type == "process_start":
        return f"🔧 Process started: {message}"
    if event_type == "process_complete":
        return f"✅ Process completed: {message}"
    if event_type == "process_error":
        return f"❌ Process error: {message}"
    if event_type == "process_timeout":
        return f"⏱️ Process timeout: {message}"
    return message or None
