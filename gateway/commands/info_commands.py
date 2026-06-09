"""
Info command handler mixin.

This mixin contains command handlers for information-related commands:
- /status - Show gateway and agent status
- /agents - Show active agents
- /kanban - Show kanban board status
- /retry - Retry failed operations
"""

import logging
from datetime import datetime
from typing import Optional

from agent.i18n import t as _t
from gateway.platforms.base import MessageEvent

logger = logging.getLogger(__name__)


class InfoCommandMixin:
    """Information command handlers.

    This mixin provides handlers for commands that show information about
    the gateway, agents, sessions, and system state.

    Relies on GatewayRunner state accessed via self.
    """

    async def _handle_status_command(self, event: MessageEvent) -> str:
        """Handle /status command — show gateway and agent status."""
        source = event.source
        session_key = self._session_key_for_source(source)

        lines = ["**Gateway Status**"]

        # Gateway info
        lines.append(f"\n**Platform:** {source.platform.value if source.platform else 'CLI'}")
        lines.append(f"**User:** {source.user_id}")

        # Session info
        session_entry = self.session_store.get_or_create_session(source)
        lines.append(f"**Session ID:** {session_entry.session_id}")
        lines.append(f"**Session Key:** {session_key}")

        # Agent status
        agent = self._running_agents.get(session_key)
        if agent:
            lines.append(f"**Agent:** Running")
        else:
            lines.append(f"**Agent:** Idle")

        # Running agents count
        if hasattr(self, "_running_agent_count"):
            count = self._running_agent_count()
            lines.append(f"**Active Agents:** {count}")

        # Cache status
        if hasattr(self, "_agent_cache"):
            cache_size = len(self._agent_cache)
            lines.append(f"**Cached Agents:** {cache_size}")

        # Queue status
        if hasattr(self, "_queued_events"):
            queue_size = len(self._queued_events)
            if queue_size > 0:
                lines.append(f"**Queued Events:** {queue_size}")

        # Model info
        override = self._session_model_overrides.get(session_key, {})
        if override:
            model = override.get("model", "")
            provider = override.get("provider", "")
            lines.append(f"\n**Model Override:** {model}")
            lines.append(f"**Provider:** {provider}")

        # Uptime
        if hasattr(self, "_start_time"):
            uptime = datetime.now().timestamp() - self._start_time
            from gateway.utils.gateway_helpers import _format_duration
            lines.append(f"**Uptime:** {_format_duration(uptime)}")

        return "\n".join(lines)

    async def _handle_agents_command(self, event: MessageEvent) -> str:
        """Handle /agents command — show active agents."""
        lines = ["**Active Agents**"]

        # Get all running agents
        if hasattr(self, "_running_agents"):
            running = self._running_agents
            if running:
                for session_key, agent in running.items():
                    if agent is getattr(self, "_AGENT_PENDING_SENTINEL", None):
                        lines.append(f"  {session_key} — Pending")
                    else:
                        # Get agent info
                        try:
                            agent_id = getattr(agent, "agent_id", "unknown")
                            model = getattr(agent, "model", "unknown")
                            lines.append(f"  {session_key} — {model} (ID: {agent_id})")
                        except Exception:
                            lines.append(f"  {session_key} — Running")
            else:
                lines.append("  No active agents")
        else:
            lines.append("  No agent tracking available")

        # Cached agents
        if hasattr(self, "_agent_cache"):
            cache = self._agent_cache
            if cache:
                lines.append(f"\n**Cached Agents:** {len(cache)}")
            else:
                lines.append(f"\n**Cached Agents:** 0")

        return "\n".join(lines)

    async def _handle_kanban_command(self, event: MessageEvent) -> str:
        """Handle /kanban command — show kanban board status."""
        args = event.get_command_args().strip().lower()

        # Check if kanban is enabled
        try:
            from hermes_cli.kanban import list_boards, get_current_board
        except ImportError:
            return "⚠️ Kanban feature not available"

        if not args or args == "list":
            # List kanban boards
            boards = list_boards(include_archived=False)
            if boards:
                lines = ["**Kanban Boards**"]
                current = get_current_board()
                for board in boards:
                    is_current = "✓ " if board.get("slug") == current else "  "
                    lines.append(f"{is_current}{board.get('slug')} — {board.get('display_name', '')}")
                return "\n".join(lines)
            else:
                return "No kanban boards found"

        if args == "current" or args == "show":
            # Show current board
            current = get_current_board()
            if current:
                try:
                    from hermes_cli.kanban import get_board
                    board = get_board(current)
                    if board:
                        lines = [f"**Board:** {board.get('display_name', current)}"]
                        lines.append(f"**Slug:** {current}")
                        # Show task counts by status
                        try:
                            from hermes_cli.kanban import _board_task_counts
                            counts = _board_task_counts(current)
                            if counts:
                                lines.append("\n**Task Counts:**")
                                for status, count in counts.items():
                                    lines.append(f"  {status}: {count}")
                        except Exception:
                            pass
                        return "\n".join(lines)
                except Exception as e:
                    logger.error("Failed to get kanban board: %s", e, exc_info=True)
                    return f"⚠️ Failed to get board info: {e}"
            else:
                return "No current board set"

        # Show usage
        return "Usage: /kanban <list|current|show>"

    async def _handle_retry_command(self, event: MessageEvent) -> str:
        """Handle /retry command — retry failed operations.

        Usage:
            /retry — Retry last failed operation
            /retry list — Show failed operations
            /retry clear — Clear failed operation log
        """
        source = event.source
        session_key = self._session_key_for_source(source)
        args = event.get_command_args().strip().lower()

        if not args or args == "list":
            # Show failed operations
            if hasattr(self, "_failed_operations"):
                failed = self._failed_operations.get(session_key, [])
                if failed:
                    lines = [f"**Failed Operations ({len(failed)}):**"]
                    for i, op in enumerate(failed[-10:], 1):  # Last 10
                        op_type = op.get("type", "unknown")
                        error = op.get("error", "no error message")[:50]
                        timestamp = op.get("timestamp", 0)
                        from gateway.utils.gateway_helpers import _format_duration
                        time_str = _format_duration(timestamp) if timestamp else "unknown"
                        lines.append(f"  {i}. {op_type} — {error} ({time_str})")
                    if len(failed) > 10:
                        lines.append(f"  ... and {len(failed) - 10} more")
                    return "\n".join(lines)
            return "No failed operations"

        if args == "clear":
            # Clear failed operations log
            if hasattr(self, "_failed_operations"):
                self._failed_operations.pop(session_key, None)
            return "✓ Failed operations cleared"

        # Retry last failed operation
        if hasattr(self, "_failed_operations"):
            failed = self._failed_operations.get(session_key, [])
            if failed:
                last_op = failed[-1]
                op_type = last_op.get("type", "unknown")

                # Attempt retry based on operation type
                try:
                    if op_type == "tool_call":
                        tool_name = last_op.get("tool_name", "")
                        tool_args = last_op.get("tool_args", {})
                        # Re-execute tool call
                        result = await self._execute_tool(tool_name, tool_args, session_key)
                        return f"✓ Retried {tool_name}: {result}"
                    elif op_type == "message":
                        content = last_op.get("content", "")
                        # Re-send message
                        result = await self._send_message(content, session_key)
                        return f"✓ Retried message send"
                    else:
                        return f"⚠️ Cannot retry operation type: {op_type}"
                except Exception as e:
                    logger.error("Retry failed: %s", e, exc_info=True)
                    return f"⚠️ Retry failed: {e}"
            else:
                return "No failed operations to retry"
        else:
            return "No failed operations to retry"
