"""Transcript/replay cleanup helpers extracted from gateway/run.py.

Round 3 of gateway decomposition. These functions normalize agent replay
history: strip interrupted tool sequences and persisted auto-continue notes.
Pure (no gateway instance state); only stdlib + a module logger.
Names kept identical to originals so call sites are unchanged.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


_AUTO_CONTINUE_NOTE_PREFIX = "[System note: Your previous turn"
_AUTO_CONTINUE_FALLBACK_PREFIX = "[System note: A new message"


def _is_interrupted_tool_result(content: Any) -> bool:
    """Return True if a tool result indicates the tool was interrupted."""
    if not isinstance(content, str):
        return False
    lowered = content.lower()
    if "[command interrupted]" in lowered:
        return True
    if "exit_code" in lowered and ("130" in lowered or "-1" in lowered):
        return "interrupt" in lowered
    return False


def _strip_interrupted_tool_tails(
    agent_history: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Strip interrupted assistant→tool sequences from replay history.

    Older interrupted gateway turns can be followed by a queued real user
    message, so the interrupted assistant/tool block is not necessarily the
    final tail by the time we rebuild replay history.  Remove any contiguous
    assistant(tool_calls) + tool-result block that contains an interrupted tool
    result, while preserving successful tool-call sequences intact.
    """
    if not agent_history:
        return agent_history

    cleaned: List[Dict[str, Any]] = []
    i = 0
    n = len(agent_history)
    while i < n:
        msg = agent_history[i]
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            j = i + 1
            tool_results: List[Dict[str, Any]] = []
            while j < n and agent_history[j].get("role") == "tool":
                tool_results.append(agent_history[j])
                j += 1
            if tool_results and any(
                _is_interrupted_tool_result(m.get("content", ""))
                for m in tool_results
            ):
                logger.debug(
                    "Stripping interrupted assistant→tool replay block "
                    "(indices %d–%d, tool_results=%d)",
                    i, j - 1, len(tool_results),
                )
                i = j
                continue
        if msg.get("role") == "tool" and _is_interrupted_tool_result(msg.get("content", "")):
            logger.debug("Stripping orphan interrupted tool result from replay history")
            i += 1
            continue
        cleaned.append(msg)
        i += 1

    return cleaned


def _is_auto_continue_noise(content: Any) -> bool:
    """Return True if this user-message content is a gateway-injected
    auto-continue note that should NOT be replayed as a real user turn."""
    if not isinstance(content, str):
        return False
    return (
        content.startswith(_AUTO_CONTINUE_NOTE_PREFIX)
        or content.startswith(_AUTO_CONTINUE_FALLBACK_PREFIX)
    )


def _strip_auto_continue_noise(content: Any) -> Any:
    """Remove persisted gateway auto-continue note prefix from user text.

    Older gateway builds prepended the recovery note directly to the user
    message, so the transcript row can contain both the synthetic note and
    the user's real question.  Strip one or more leading synthetic notes while
    preserving any real text that follows.
    """
    if not _is_auto_continue_noise(content):
        return content
    text = str(content)
    while _is_auto_continue_noise(text):
        end = text.find("]")
        if end < 0:
            return ""
        text = text[end + 1 :].lstrip()
    return text
