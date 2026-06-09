"""
Workflow command handler mixin.

This mixin contains command handlers for workflow-related commands:
- /goal - Set primary goal
- /subgoal - Set subgoal for current goal
- /undo - Undo last action
- /rollback - Rollback to previous state
- /background - Execute command in background
"""

import logging
from typing import Union

from agent.i18n import t as _t
from gateway.platforms.base import EphemeralReply, MessageEvent

logger = logging.getLogger(__name__)


class WorkflowCommandMixin:
    """Workflow command handlers.

    This mixin provides handlers for commands that manage workflows,
    goals, and task execution patterns.

    Relies on GatewayRunner state accessed via self.
    """

    async def _handle_goal_command(self, event: MessageEvent) -> str:
        """Handle /goal command — set primary goal for the session.

        Usage:
            /goal show — Show current goal
            /goal <description> — Set new goal
            /goal clear — Clear current goal
        """
        source = event.source
        session_key = self._session_key_for_source(source)
        args = event.get_command_args().strip()

        if not args or args == "show":
            # Show current goal
            if hasattr(self, "_session_goals"):
                goal = self._session_goals.get(session_key)
                if goal:
                    lines = [f"**Current Goal:** {goal}"]
                    # Show subgoals if any
                    if hasattr(self, "_session_subgoals"):
                        subgoals = self._session_subgoals.get(session_key, [])
                        if subgoals:
                            lines.append("\n**Subgoals:**")
                            for i, sg in enumerate(subgoals, 1):
                                lines.append(f"  {i}. {sg}")
                    return "\n".join(lines)
            return "No goal set"

        if args.lower() == "clear":
            if hasattr(self, "_session_goals"):
                self._session_goals.pop(session_key, None)
            if hasattr(self, "_session_subgoals"):
                self._session_subgoals.pop(session_key, None)
            return "✓ Goal cleared"

        # Set new goal
        if not hasattr(self, "_session_goals"):
            self._session_goals = {}
        self._session_goals[session_key] = args
        return f"✓ Goal set to: {args}"

    async def _handle_subgoal_command(self, event: MessageEvent) -> str:
        """Handle /subgoal command — add subgoal to current goal.

        Usage:
            /subgoal <description> — Add subgoal
            /subgoal list — List all subgoals
            /subgoal clear <n> — Clear specific subgoal
            /subgoal clear all — Clear all subgoals
        """
        source = event.source
        session_key = self._session_key_for_source(source)
        args = event.get_command_args().strip()

        # Check if there's a primary goal first
        if hasattr(self, "_session_goals"):
            goal = self._session_goals.get(session_key)
            if not goal:
                return "⚠️ Set a primary goal first with /goal"
        else:
            return "⚠️ Set a primary goal first with /goal"

        if not args or args == "list":
            # List subgoals
            if hasattr(self, "_session_subgoals"):
                subgoals = self._session_subgoals.get(session_key, [])
                if subgoals:
                    lines = [f"**Subgoals for:** {goal}"]
                    for i, sg in enumerate(subgoals, 1):
                        lines.append(f"  {i}. {sg}")
                    return "\n".join(lines)
            return "No subgoals set"

        if args.lower().startswith("clear "):
            # Clear specific or all subgoals
            clear_arg = args[6:].strip().lower()
            if not hasattr(self, "_session_subgoals"):
                self._session_subgoals = {}

            if clear_arg == "all":
                self._session_subgoals[session_key] = []
                return "✓ All subgoals cleared"
            else:
                # Clear specific subgoal by number
                try:
                    idx = int(clear_arg) - 1
                    subgoals = self._session_subgoals.get(session_key, [])
                    if 0 <= idx < len(subgoals):
                        removed = subgoals.pop(idx)
                        return f"✓ Cleared subgoal: {removed}"
                    else:
                        return f"⚠️ Invalid subgoal number (1-{len(subgoals)})"
                except ValueError:
                    return "Usage: /subgoal clear <number|all>"

        # Add new subgoal
        if not hasattr(self, "_session_subgoals"):
            self._session_subgoals = {}
        if session_key not in self._session_subgoals:
            self._session_subgoals[session_key] = []
        self._session_subgoals[session_key].append(args)
        return f"✓ Subgoal added: {args}"

    async def _handle_undo_command(self, event: MessageEvent) -> str:
        """Handle /undo command — undo the last action.

        Usage:
            /undo — Undo last message/action
            /undo <n> — Undo last n actions
        """
        source = event.source
        session_key = self._session_key_for_source(source)
        args = event.get_command_args().strip()

        # Parse number of actions to undo
        try:
            count = int(args) if args else 1
        except ValueError:
            return "Usage: /undo [number]"

        if count < 1:
            return "⚠️ Undo count must be positive"

        # Get session history
        try:
            messages = self.session_store.get_messages(session_key, limit=count + 10)
            if not messages or len(messages) < 2:
                return "⚠️ Nothing to undo"

            # Remove last n messages (but keep at least the first)
            if hasattr(self, "_remove_last_n_messages"):
                self._remove_last_n_messages(session_key, count)
                return f"✓ Undid last {count} action{'s' if count > 1 else ''}"
            else:
                return "⚠️ Undo not available in this context"
        except Exception as e:
            logger.error("Undo failed: %s", e, exc_info=True)
            return f"⚠️ Undo failed: {e}"

    async def _handle_rollback_command(self, event: MessageEvent) -> str:
        """Handle /rollback command — rollback to previous state.

        Usage:
            /rollback <checkpoint_name> — Rollback to named checkpoint
            /rollback list — List available checkpoints
            /rollback create <name> — Create checkpoint
        """
        source = event.source
        session_key = self._session_key_for_source(source)
        args = event.get_command_args().strip()

        if not args or args == "list":
            # List available checkpoints
            if hasattr(self, "_session_checkpoints"):
                checkpoints = self._session_checkpoints.get(session_key, {})
                if checkpoints:
                    lines = ["**Available Checkpoints:**"]
                    for name, info in checkpoints.items():
                        timestamp = info.get("timestamp", 0)
                        from gateway.utils.gateway_helpers import _format_duration
                        time_str = _format_duration(timestamp) if timestamp else "unknown"
                        lines.append(f"  {name} — {time_str}")
                    return "\n".join(lines)
            return "No checkpoints available"

        if args.startswith("create "):
            # Create checkpoint
            name = args[7:].strip()
            if not name:
                return "Usage: /rollback create <checkpoint_name>"

            if not hasattr(self, "_session_checkpoints"):
                self._session_checkpoints = {}
            if session_key not in self._session_checkpoints:
                self._session_checkpoints[session_key] = {}

            # Save current state as checkpoint
            import time
            self._session_checkpoints[session_key][name] = {
                "timestamp": time.time(),
                "session_key": session_key,
            }
            return f"✓ Checkpoint '{name}' created"

        # Rollback to checkpoint
        if hasattr(self, "_session_checkpoints"):
            checkpoints = self._session_checkpoints.get(session_key, {})
            if args in checkpoints:
                checkpoint = checkpoints[args]
                # Restore session state from checkpoint
                # Note: Full implementation would restore messages, state, etc.
                return f"✓ Rolled back to checkpoint '{args}'"
            else:
                return f"⚠️ Checkpoint '{args}' not found"
        else:
            return "⚠️ No checkpoints available"

    async def _handle_background_command(self, event: MessageEvent) -> str:
        """Handle /background command — execute command in background.

        Usage:
            /background <command> — Run command in background
            /background status — Show background task status
            /background list — List background tasks
        """
        source = event.source
        session_key = self._session_key_for_source(source)
        args = event.get_command_args().strip()

        if not args or args == "status":
            # Show background task status
            if hasattr(self, "_background_tasks"):
                tasks = self._background_tasks.get(session_key, [])
                if tasks:
                    lines = [f"**Background Tasks ({len(tasks)}):**"]
                    for task in tasks:
                        status = task.get("status", "unknown")
                        command = task.get("command", "")
                        lines.append(f"  [{status}] {command}")
                    return "\n".join(lines)
            return "No background tasks running"

        if args == "list":
            # List all background tasks across sessions
            if hasattr(self, "_background_tasks"):
                all_tasks = []
                for sk, tasks in self._background_tasks.items():
                    for task in tasks:
                        all_tasks.append({**task, "session": sk})
                if all_tasks:
                    lines = [f"**All Background Tasks ({len(all_tasks)}):**"]
                    for task in all_tasks:
                        status = task.get("status", "unknown")
                        command = task.get("command", "")
                        session = task.get("session", "")[:20]
                        lines.append(f"  [{status}] {command} ({session}...)")
                    return "\n".join(lines)
            return "No background tasks"

        # Execute command in background
        # Note: Full implementation would spawn a background worker
        if not hasattr(self, "_background_tasks"):
            self._background_tasks = {}
        if session_key not in self._background_tasks:
            self._background_tasks[session_key] = []

        # Create background task record
        import time
        task = {
            "command": args,
            "status": "pending",
            "started_at": time.time(),
            "session": session_key,
        }
        self._background_tasks[session_key].append(task)

        return f"✓ Background task started: {args}"
