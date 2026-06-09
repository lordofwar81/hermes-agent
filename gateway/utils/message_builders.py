"""
Gateway message construction utilities.

This module contains functions for building and constructing messages
in the gateway, particularly for agent history and reply formatting.
"""

import json
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _build_replay_entry(
    role: str,
    content: Optional[str],
    tool_name: Optional[str] = None,
    tool_args: Optional[Union[str, Dict]] = None,
    timestamp: Optional[float] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a standardized replay entry for gateway agent history.

    Args:
        role: Message role (user, assistant, tool, system)
        content: Message content text
        tool_name: Tool name for tool role messages
        tool_args: Tool arguments (JSON string or dict)
        timestamp: Unix timestamp for the message
        run_id: Optional run/session identifier

    Returns:
        Dictionary with standardized replay entry structure
    """
    entry: Dict[str, Any] = {
        "role": role,
        "content": content or "",
    }

    if tool_name:
        entry["tool_name"] = tool_name

    if tool_args is not None:
        if isinstance(tool_args, str):
            entry["tool_args"] = tool_args
        elif isinstance(tool_args, dict):
            entry["tool_args"] = json.dumps(tool_args)

    if timestamp is not None:
        entry["timestamp"] = timestamp

    if run_id is not None:
        entry["run_id"] = run_id

    return entry


def _build_gateway_agent_history(
    messages: List[Dict[str, Any]],
    include_system: bool = True,
    max_messages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build agent history from gateway messages for agent processing.

    Args:
        messages: Raw gateway messages from state
        include_system: Whether to include system messages
        max_messages: Maximum number of messages to include (None for unlimited)

    Returns:
        List of formatted message dictionaries ready for agent consumption
    """
    history = []

    for msg in messages:
        role = msg.get("role", "")
        if not include_system and role == "system":
            continue

        entry = _build_replay_entry(
            role=role,
            content=msg.get("content"),
            tool_name=msg.get("tool_name"),
            tool_args=msg.get("tool_args"),
            timestamp=msg.get("timestamp"),
            run_id=msg.get("run_id"),
        )

        history.append(entry)

        if max_messages and len(history) >= max_messages:
            break

    return history


def _wrap_current_message_with_observed_context(
    current_message: str,
    observed_context: Optional[str],
) -> str:
    """Wrap the current message with observed context if available.

    Args:
        current_message: The current message text
        observed_context: Optional observed context to prepend

    Returns:
        Message with observed context prepended if available
    """
    if not observed_context or not observed_context.strip():
        return current_message

    context = observed_context.strip()
    if not current_message:
        return context

    return f"{context}\n\n{current_message}"


def _build_media_placeholder(url: str, media_type: str = "file") -> str:
    """Build a placeholder string for media attachments.

    Args:
        url: URL or path to the media file
        media_type: Type of media (image, video, audio, file)

    Returns:
        Placeholder string for the media
    """
    if media_type == "image":
        return f"[User sent an image: {url}]"
    elif media_type == "video":
        return f"[User sent a video: {url}]"
    elif media_type.startswith("audio"):
        return f"[User sent audio: {url}]"
    else:
        return f"[User sent a file: {url}]"


def _build_media_collection_placeholders(media_items: List[Dict[str, Any]]) -> str:
    """Build placeholder text for multiple media items.

    Args:
        media_items: List of media item dicts with 'url' and 'type' keys

    Returns:
        Multi-line placeholder text for all media items
    """
    if not media_items:
        return ""

    parts = []
    for item in media_items:
        url = item.get("url", "")
        media_type = item.get("type", "file")
        if url:
            parts.append(_build_media_placeholder(url, media_type))

    return "\n".join(parts)


def _normalize_empty_agent_response(response: Optional[str]) -> str:
    """Normalize empty or None responses to empty string.

    Args:
        response: Agent response that may be None or empty

    Returns:
        Normalized response string (empty string if None/whitespace-only)
    """
    if not response:
        return ""
    return str(response).strip()


def _format_gateway_process_notification(evt: Dict[str, Any]) -> Optional[str]:
    """Format a process event for gateway display.

    Args:
        evt: Event dictionary with 'type' and 'message' keys

    Returns:
        Formatted notification string or None if type unrecognized
    """
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


def _skill_slug_from_frontmatter(skill_md: Path) -> tuple[Optional[str], Optional[str]]:
    """Derive the /command slug and declared frontmatter name from a SKILL.md.

    Args:
        skill_md: Path to the SKILL.md file

    Returns:
        Tuple of (slug, name) where either may be None if not found
    """
    try:
        content = skill_md.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None, None

    # Extract frontmatter between --- markers
    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not frontmatter_match:
        return None, None

    frontmatter = frontmatter_match.group(1)

    # Extract command slug
    slug_match = re.search(r"command:\s*/(\w+)", frontmatter)
    slug = slug_match.group(1) if slug_match else None

    # Extract skill name
    name_match = re.search(r"name:\s*[\"']([^\"']+)[\"']", frontmatter)
    name = name_match.group(1) if name_match else None

    return slug, name
