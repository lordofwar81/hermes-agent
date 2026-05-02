#!/usr/bin/env python3
"""
Session Manager for Hermes Agent.

Handles all session persistence and trajectory management functionality,
extracted from the AIAgent class for better separation of concerns.

This module provides:
- Session state persistence to JSON logs and SQLite
- Trajectory saving in JSONL format for training data
- Content cleaning and sanitization
- User message override handling for API-facing variants
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.trajectory import convert_scratchpad_to_think
from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

# Think tag patterns for regex
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


class SessionManager:
    """
    Manages session persistence and trajectory saving for Hermes Agent.

    This class handles all aspects of session state management including
    saving conversation history to both JSON logs and SQLite databases,
    converting conversations to trajectory format for training data,
    and cleaning sensitive content from session logs.

    Args:
        session_db: HermesState database instance for SQLite persistence
        session_id: Unique identifier for this session
        save_trajectories: Whether to save trajectories to JSONL files
        persist_session: Whether to persist session state (ephemeral flows set False)
        log_prefix: Prefix for log messages (useful for parallel processing)
        platform: Platform identifier (cli, telegram, discord, etc.) for log context

    Attributes:
        session_db: Reference to the SQLite session database
        session_id: Unique session identifier
        save_trajectories: Flag for trajectory saving
        persist_session: Flag for session persistence
        log_prefix: Prefix for log messages
        platform: Platform identifier
        logs_dir: Directory path for session logs
        session_log_file: Path to the JSON session log file
        _session_messages: Cached list of session messages
        _last_flushed_db_idx: Tracks which messages have been written to SQLite
        _persist_user_message_idx: Index of user message to override
        _persist_user_message_override: Override content for user message
    """

    def __init__(
        self,
        session_db: Any,
        session_id: str,
        save_trajectories: bool = False,
        persist_session: bool = True,
        log_prefix: str = "",
        platform: str = "cli",
    ):
        self.session_db = session_db
        self.session_id = session_id
        self.save_trajectories = save_trajectories
        self.persist_session = persist_session
        self.log_prefix = log_prefix
        self.platform = platform

        # Initialize logging directories
        hermes_home = get_hermes_home()
        self.logs_dir = hermes_home / "sessions"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.session_log_file = self.logs_dir / f"session_{self.session_id}.json"

        # Track conversation messages for session logging
        self._session_messages: List[Dict[str, Any]] = []

        # Track which messages have been flushed to SQLite to prevent duplicates
        self._last_flushed_db_idx = 0

        # User message override state for API-facing variants
        self._persist_user_message_idx: Optional[int] = None
        self._persist_user_message_override: Optional[str] = None

        # Cached agent state (set by the agent after initialization)
        self.model: Optional[str] = None
        self.base_url: Optional[str] = None
        self.session_start: Optional[datetime] = None
        self._cached_system_prompt: Optional[str] = None
        self.tools: Optional[List[Dict[str, Any]]] = None
        self.verbose_logging: bool = False

    def apply_persist_user_message_override(self, messages: List[Dict]) -> None:
        """
        Rewrite the current-turn user message before persistence/return.

        Some call paths need an API-only user-message variant without letting
        that synthetic text leak into persisted transcripts or resumed session
        history. When an override is configured for the active turn, mutate the
        in-memory messages list in place so both persistence and returned
        history stay clean.
        """
        idx = self._persist_user_message_idx
        override = self._persist_user_message_override
        if override is None or idx is None:
            return
        if 0 <= idx < len(messages):
            msg = messages[idx]
            if isinstance(msg, dict) and msg.get("role") == "user":
                msg["content"] = override

    def persist_session(
        self,
        messages: List[Dict],
        conversation_history: Optional[List[Dict]] = None,
    ) -> None:
        """
        Save session state to both JSON log and SQLite on any exit path.

        Ensures conversations are never lost, even on errors or early returns.
        Skipped when persist_session=False (ephemeral helper flows).

        Args:
            messages: Current message list to persist
            conversation_history: Optional initial conversation history for offset calculation
        """
        if not self.persist_session:
            return
        self.apply_persist_user_message_override(messages)
        self._session_messages = messages
        self.save_session_log(messages)
        self.flush_messages_to_session_db(messages, conversation_history)

    def flush_messages_to_session_db(
        self, messages: List[Dict], conversation_history: Optional[List[Dict]] = None
    ) -> None:
        """
        Persist any un-flushed messages to the SQLite session store.

        Uses _last_flushed_db_idx to track which messages have already been
        written, so repeated calls (from multiple exit paths) only write
        truly new messages — preventing the duplicate-write bug (#860).

        Args:
            messages: Full message list to flush
            conversation_history: Optional initial conversation history for offset calculation
        """
        if not self.session_db:
            return
        self.apply_persist_user_message_override(messages)
        try:
            # If create_session() failed at startup (e.g. transient lock), the
            # session row may not exist yet.  ensure_session() uses INSERT OR
            # IGNORE so it is a no-op when the row is already there.
            self.session_db.ensure_session(
                self.session_id,
                source=self.platform or "cli",
                model=self.model,
            )
            start_idx = len(conversation_history) if conversation_history else 0
            flush_from = max(start_idx, self._last_flushed_db_idx)
            for msg in messages[flush_from:]:
                role = msg.get("role", "unknown")
                content = msg.get("content")
                tool_calls_data = None
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls_data = [
                        {"name": tc.function.name, "arguments": tc.function.arguments}
                        for tc in msg.tool_calls
                    ]
                elif isinstance(msg.get("tool_calls"), list):
                    tool_calls_data = msg["tool_calls"]
                self.session_db.append_message(
                    session_id=self.session_id,
                    role=role,
                    content=content,
                    tool_name=msg.get("tool_name"),
                    tool_calls=tool_calls_data,
                    tool_call_id=msg.get("tool_call_id"),
                    finish_reason=msg.get("finish_reason"),
                    reasoning=msg.get("reasoning") if role == "assistant" else None,
                    reasoning_details=msg.get("reasoning_details")
                    if role == "assistant"
                    else None,
                    codex_reasoning_items=msg.get("codex_reasoning_items")
                    if role == "assistant"
                    else None,
                )
            self._last_flushed_db_idx = len(messages)
        except Exception as e:
            logger.warning("Session DB append_message failed: %s", e)

    def get_messages_up_to_last_assistant(self, messages: List[Dict]) -> List[Dict]:
        """
        Get messages up to (but not including) the last assistant turn.

        This is used when we need to "roll back" to the last successful point
        in the conversation, typically when the final assistant message is
        incomplete or malformed.

        Args:
            messages: Full message list

        Returns:
            Messages up to the last complete assistant turn (ending with user/tool message)
        """
        if not messages:
            return []

        # Find the index of the last assistant message
        last_assistant_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break

        if last_assistant_idx is None:
            # No assistant message found, return all messages
            return messages.copy()

        # Return everything up to (not including) the last assistant message
        return messages[:last_assistant_idx]

    def _format_tools_for_system_message(self) -> str:
        """
        Format tool definitions for the system message in the trajectory format.

        Returns:
            str: JSON string representation of tool definitions
        """
        if not self.tools:
            return "[]"

        # Convert tool definitions to the format expected in trajectories
        formatted_tools = []
        for tool in self.tools:
            func = tool["function"]
            formatted_tool = {
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
                "required": None,  # Match the format in the example
            }
            formatted_tools.append(formatted_tool)

        return json.dumps(formatted_tools, ensure_ascii=False)

    def convert_to_trajectory_format(
        self, messages: List[Dict[str, Any]], user_query: str, completed: bool
    ) -> List[Dict[str, Any]]:
        """
        Convert internal message format to trajectory format for saving.

        Args:
            messages (List[Dict]): Internal message history
            user_query (str): Original user query
            completed (bool): Whether the conversation completed successfully

        Returns:
            List[Dict]: Messages in trajectory format
        """
        trajectory = []

        # Add system message with tool definitions
        system_msg = (
            "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. "
            "You may call one or more functions to assist with the user query. If available tools are not relevant in assisting "
            "with user query, just respond in natural conversational language. Don't make assumptions about what values to plug "
            "into functions. After calling & executing the functions, you will be provided with function results within "
            "<functions_results> </functions_results> XML tags. "
            "Here are the available tools:\n"
            f"<tools>\n{self._format_tools_for_system_message()}\n</tools>\n"
            "For each function call return a JSON object, with the following pydantic model json schema for each:\n"
            "{'title': 'FunctionCall', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, "
            "'arguments': {'title': 'Arguments', 'type': 'object'}}, 'required': ['name', 'arguments']}\n"
            "Each function call should be enclosed within <functions_results> </functions_results> XML tags.\n"
            "Example:\n<functions_results>\n{'name': <function-name>,'arguments': <args-dict>}\n</functions_results>\n"
        )

        trajectory.append({"from": "system", "value": system_msg})

        # Add the actual user prompt (from the dataset) as the first human message
        trajectory.append({"from": "human", "value": user_query})

        # Skip the first message (the user query) since we already added it above.
        # Prefill messages are injected at API-call time only (not in the messages
        # list), so no offset adjustment is needed here.
        i = 1

        while i < len(messages):
            msg = messages[i]

            if msg["role"] == "assistant":
                # Check if this message has tool calls
                if "tool_calls" in msg and msg["tool_calls"]:
                    # Format assistant message with tool calls
                    # Add <think> tags around reasoning for trajectory storage
                    content = ""

                    # Prepend reasoning in <think> tags if available (native thinking tokens)
                    if msg.get("reasoning") and msg["reasoning"].strip():
                        content = f"{THINK_OPEN}\n{msg['reasoning']}\n{THINK_CLOSE}\n"

                    if msg.get("content") and msg["content"].strip():
                        # Convert any <REASONING_SCRATCHPAD> tags to <think> tags
                        # (used when native thinking is disabled and model reasons via XML)
                        content += convert_scratchpad_to_think(msg["content"]) + "\n"

                    # Add tool calls wrapped in XML tags
                    for tool_call in msg["tool_calls"]:
                        if not tool_call or not isinstance(tool_call, dict):
                            continue
                        # Parse arguments - should always succeed since we validate during conversation
                        # but keep try-except as safety net
                        try:
                            arguments = (
                                json.loads(tool_call["function"]["arguments"])
                                if isinstance(tool_call["function"]["arguments"], str)
                                else tool_call["function"]["arguments"]
                            )
                        except json.JSONDecodeError:
                            # This shouldn't happen since we validate and retry during conversation,
                            # but if it does, log warning and use empty dict
                            logging.warning(
                                f"Unexpected invalid JSON in trajectory conversion: {tool_call['function']['arguments'][:100]}"
                            )
                            arguments = {}

                        tool_call_json = {
                            "name": tool_call["function"]["name"],
                            "arguments": arguments,
                        }
                        content += f"<functions_results>\n{json.dumps(tool_call_json, ensure_ascii=False)}\n</functions_results>\n"

                    # Ensure every gpt turn has a <think> block (empty if no reasoning)
                    # so the format is consistent for training data
                    if THINK_OPEN not in content:
                        content = f"{THINK_OPEN}\n{THINK_CLOSE}\n" + content

                    trajectory.append({"from": "gpt", "value": content.rstrip()})

                    # Collect all subsequent tool responses
                    tool_responses: list[str] = []
                    j = i + 1
                    while j < len(messages) and messages[j]["role"] == "tool":
                        tool_msg = messages[j]
                        # Format tool response with XML tags
                        tool_response = "<function_results>\n"

                        # Try to parse tool content as JSON if it looks like JSON
                        tool_content = tool_msg["content"]
                        try:
                            if tool_content.strip().startswith(("{", "[")):
                                tool_content = json.loads(tool_content)
                        except (json.JSONDecodeError, AttributeError):
                            pass  # Keep as string if not valid JSON

                        tool_index = len(tool_responses)
                        tool_name = (
                            msg["tool_calls"][tool_index]["function"]["name"]
                            if tool_index < len(msg["tool_calls"])
                            else "unknown"
                        )
                        tool_response += json.dumps(
                            {
                                "tool_call_id": tool_msg.get("tool_call_id", ""),
                                "name": tool_name,
                                "content": tool_content,
                            },
                            ensure_ascii=False,
                        )
                        tool_response += "\n</function_results>"
                        tool_responses.append(tool_response)
                        j += 1

                    # Add all tool responses as a single message
                    if tool_responses:
                        trajectory.append(
                            {"from": "tool", "value": "\n".join(tool_responses)}
                        )
                        i = j - 1  # Skip the tool messages we just processed

                else:
                    # Regular assistant message without tool calls
                    # Add <think> tags around reasoning for trajectory storage
                    content = ""

                    # Prepend reasoning in <think> tags if available (native thinking tokens)
                    if msg.get("reasoning") and msg["reasoning"].strip():
                        content = f"{THINK_OPEN}\n{msg['reasoning']}\n{THINK_CLOSE}\n"

                    # Convert any <REASONING_SCRATCHPAD> tags to <think> tags
                    # (used when native thinking is disabled and model reasons via XML)
                    raw_content = msg["content"] or ""
                    content += convert_scratchpad_to_think(raw_content)

                    # Ensure every gpt turn has a <think> block (empty if no reasoning)
                    if THINK_OPEN not in content:
                        content = f"{THINK_OPEN}\n{THINK_CLOSE}\n" + content

                    trajectory.append({"from": "gpt", "value": content.strip()})

            elif msg["role"] == "user":
                trajectory.append({"from": "human", "value": msg["content"]})

            i += 1

        return trajectory

    def save_trajectory(
        self, messages: List[Dict[str, Any]], user_query: str, completed: bool
    ) -> None:
        """
        Save conversation trajectory to JSONL file.

        Args:
            messages (List[Dict]): Complete message history
            user_query (str): Original user query
            completed (bool): Whether the conversation completed successfully
        """
        if not self.save_trajectories:
            return

        # Import here to avoid circular dependency
        from agent.trajectory import save_trajectory as _save_trajectory_to_file

        trajectory = self.convert_to_trajectory_format(messages, user_query, completed)
        _save_trajectory_to_file(trajectory, self.model, completed)

    @staticmethod
    def clean_session_content(content: str) -> str:
        """
        Convert REASONING_SCRATCHPAD to think tags and clean up whitespace.

        Args:
            content: Raw content string from a message

        Returns:
            Cleaned content with think tags and normalized whitespace
        """
        if not content:
            return content
        content = convert_scratchpad_to_think(content)
        # Clean up extra newlines around think tags
        content = re.sub(r"\n+(" + re.escape(THINK_OPEN) + r")", r"\n\1", content)
        content = re.sub(r"(" + re.escape(THINK_CLOSE) + r")\n+", r"\1\n", content)
        return content.strip()

    def save_session_log(self, messages: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Save the full raw session to a JSON file.

        Stores every message exactly as the agent sees it: user messages,
        assistant messages (with reasoning, finish_reason, tool_calls),
        tool responses (with tool_call_id, tool_name), and injected system
        messages (compression summaries, todo snapshots, etc.).

        REASONING_SCRATCHPAD tags are converted to <think> blocks for consistency.
        Overwritten after each turn so it always reflects the latest state.

        Args:
            messages: Message list to save (uses cached _session_messages if None)
        """
        messages = messages or self._session_messages
        if not messages:
            return

        try:
            # Clean assistant content for session logs
            cleaned = []
            for msg in messages:
                if msg.get("role") == "assistant" and msg.get("content"):
                    msg = dict(msg)
                    msg["content"] = self.clean_session_content(msg["content"])
                cleaned.append(msg)

            # Guard: never overwrite a larger session log with fewer messages.
            # This protects against data loss when --resume loads a session whose
            # messages weren't fully written to SQLite — the resumed agent starts
            # with partial history and would otherwise clobber the full JSON log.
            if self.session_log_file.exists():
                try:
                    existing = json.loads(
                        self.session_log_file.read_text(encoding="utf-8")
                    )
                    existing_count = existing.get(
                        "message_count", len(existing.get("messages", []))
                    )
                    if existing_count > len(cleaned):
                        logging.debug(
                            "Skipping session log overwrite: existing has %d messages, current has %d",
                            existing_count,
                            len(cleaned),
                        )
                        return
                except Exception:
                    pass  # corrupted existing file — allow the overwrite

            entry = {
                "session_id": self.session_id,
                "model": self.model,
                "base_url": self.base_url,
                "platform": self.platform,
                "session_start": self.session_start.isoformat() if self.session_start else None,
                "last_updated": datetime.now().isoformat(),
                "system_prompt": self._cached_system_prompt or "",
                "tools": self.tools or [],
                "message_count": len(cleaned),
                "messages": cleaned,
            }

            atomic_json_write(
                self.session_log_file,
                entry,
                indent=2,
                default=str,
            )

        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to save session log: {e}")

    def set_user_message_override(self, idx: int, override: str) -> None:
        """
        Set a user message override for the current turn.

        This allows the API-facing user message to differ from the persisted
        transcript (e.g., CLI voice mode adds a temporary prefix for the
        live call only).

        Args:
            idx: Index of the user message in the messages list
            override: Override content to use for persistence
        """
        self._persist_user_message_idx = idx
        self._persist_user_message_override = override

    def clear_user_message_override(self) -> None:
        """Clear any active user message override."""
        self._persist_user_message_idx = None
        self._persist_user_message_override = None

    def reset_flush_cursor(self) -> None:
        """Reset the SQLite flush cursor (e.g., after session compression split)."""
        self._last_flushed_db_idx = 0

    @property
    def last_flushed_db_idx(self) -> int:
        """Get the last flushed database index."""
        return self._last_flushed_db_idx

    def update_session_log_path(self, new_session_id: str) -> None:
        """
        Update the session log file path after a session ID change.

        This is used during session compression splits when a new session
        ID is generated.

        Args:
            new_session_id: New session identifier
        """
        self.session_id = new_session_id
        self.session_log_file = self.logs_dir / f"session_{self.session_id}.json"
