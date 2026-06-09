"""
Admin command handler mixin.

This mixin contains command handlers for administrative commands:
- /update - Update and restart gateway
- /reload mcp - Reload MCP tools
- /reload skills - Reload skills
- /debug - Toggle debug mode
- /approve - Approve dangerous command
- /deny - Deny dangerous command
"""

import logging
from typing import Union

from agent.i18n import t as _t
from gateway.platforms.base import EphemeralReply, MessageEvent

logger = logging.getLogger(__name__)


class AdminCommandMixin:
    """Administrative command handlers.

    This mixin provides handlers for commands that perform administrative
    or gateway-level operations.

    Relies on GatewayRunner state accessed via self.
    """

    async def _handle_update_command(self, event: MessageEvent) -> str:
        """Handle /update command — update gateway and restart.

        Usage:
            /update — Check for updates
            /update now — Apply updates and restart
            /update cancel — Cancel scheduled update
        """
        args = event.get_command_args().strip().lower()

        if not args or args == "check":
            # Check for updates
            try:
                from hermes_cli.update import check_for_updates
                update_info = await check_for_updates()
                if update_info:
                    lines = ["**Update Available**"]
                    lines.append(f"Version: {update_info.get('version', 'unknown')}")
                    lines.append(f"\n{update_info.get('description', '')}")
                    lines.append("\nRun `/update now` to apply and restart")
                    return "\n".join(lines)
                else:
                    return "✓ Already up to date"
            except Exception as e:
                logger.error("Update check failed: %s", e, exc_info=True)
                return f"⚠️ Update check failed: {e}"

        if args == "now":
            # Schedule update and restart
            try:
                from hermes_cli.update import schedule_update
                result = await schedule_update()
                if result:
                    self._restart_requested = True
                    self._draining = True
                    return "✓ Update scheduled. Gateway will restart shortly."
                else:
                    return "⚠️ No updates available"
            except Exception as e:
                logger.error("Update scheduling failed: %s", e, exc_info=True)
                return f"⚠️ Update failed: {e}"

        if args == "cancel":
            # Cancel scheduled update
            if hasattr(self, "_update_scheduled"):
                if self._update_scheduled:
                    self._update_scheduled = False
                    return "✓ Update cancelled"
                else:
                    return "No update scheduled"
            else:
                return "No update scheduled"

        return "Usage: /update <check|now|cancel>"

    async def _handle_reload_mcp_command(self, event: MessageEvent) -> str:
        """Handle /reload mcp command — reload MCP tools.

        Usage:
            /reload mcp — Reload all MCP tools
            /reload mcp <server> — Reload specific MCP server
        """
        args = event.get_command_args().strip()
        source = event.source
        session_key = self._session_key_for_source(source)

        try:
            from tools.mcp_tool import reload_mcp_tools, discover_mcp_tools

            if args:
                # Reload specific MCP server
                result = await reload_mcp_tools(args)
                if result:
                    return f"✓ Reloaded MCP server: {args}"
                else:
                    return f"⚠️ MCP server '{args}' not found or reload failed"
            else:
                # Reload all MCP tools
                count = await reload_mcp_tools()
                return f"✓ Reloaded {count} MCP tools"

        except Exception as e:
            logger.error("MCP reload failed: %s", e, exc_info=True)
            return f"⚠️ MCP reload failed: {e}"

    async def _handle_reload_skills_command(self, event: MessageEvent) -> str:
        """Handle /reload skills command — reload skills.

        Usage:
            /reload skills — Reload all skills
            /reload skills <skill_name> — Reload specific skill
        """
        args = event.get_command_args().strip()
        source = event.source
        session_key = self._session_key_for_source(source)

        try:
            from tools.skill_usage import reload_skills

            if args:
                # Reload specific skill
                result = await reload_skills(args)
                if result:
                    return f"✓ Reloaded skill: {args}"
                else:
                    return f"⚠️ Skill '{args}' not found or reload failed"
            else:
                # Reload all skills
                count = await reload_skills()
                return f"✓ Reloaded {count} skills"

        except Exception as e:
            logger.error("Skills reload failed: %s", e, exc_info=True)
            return f"⚠️ Skills reload failed: {e}"

    async def _handle_debug_command(self, event: MessageEvent) -> str:
        """Handle /debug command — toggle debug mode.

        Usage:
            /debug on — Enable debug mode
            /debug off — Disable debug mode
            /debug status — Show debug mode status
        """
        args = event.get_command_args().strip().lower()
        source = event.source
        session_key = self._session_key_for_source(source)

        if not args or args == "status":
            # Show current debug status
            if hasattr(self, "_debug_modes"):
                debug = self._debug_modes.get(session_key, False)
                return f"Debug mode: {'enabled' if debug else 'disabled'}"
            return "Debug mode: disabled"

        if args in ("on", "true", "1", "enable", "enabled"):
            if not hasattr(self, "_debug_modes"):
                self._debug_modes = {}
            self._debug_modes[session_key] = True
            return "✓ Debug mode enabled for this session"
        elif args in ("off", "false", "0", "disable", "disabled"):
            if hasattr(self, "_debug_modes"):
                self._debug_modes[session_key] = False
            return "✓ Debug mode disabled for this session"
        else:
            return "Usage: /debug <on|off|status>"

    async def _handle_approve_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        """Handle /approve command — approve pending dangerous command.

        Usage:
            /approve — Approve pending command
            /approve <id> — Approve specific pending command
        """
        args = event.get_command_args().strip()
        source = event.source
        session_key = self._session_key_for_source(source)

        # Check for pending approvals
        if hasattr(self, "_pending_approvals"):
            pending = self._pending_approvals.get(session_key)
            if not pending:
                return "⚠️ No pending commands to approve"

            if args:
                # Approve specific command by ID
                for approval in pending:
                    if approval.get("id") == args:
                        # Execute the approved command
                        command = approval.get("command", "")
                        command_args = approval.get("args", {})
                        try:
                            result = await self._execute_command(command, command_args, source)
                            # Remove from pending
                            pending.remove(approval)
                            return f"✓ Executed: {command}"
                        except Exception as e:
                            logger.error("Approved command failed: %s", e, exc_info=True)
                            return f"⚠️ Command failed: {e}"
                return f"⚠️ No pending command with ID: {args}"
            else:
                # Approve first pending command
                approval = pending[0]
                command = approval.get("command", "")
                command_args = approval.get("args", {})
                try:
                    result = await self._execute_command(command, command_args, source)
                    pending.remove(approval)
                    return EphemeralReply(f"✓ Executed: {command}")
                except Exception as e:
                    logger.error("Approved command failed: %s", e, exc_info=True)
                    return f"⚠️ Command failed: {e}"
        else:
            return "⚠️ No pending commands to approve"

    async def _handle_deny_command(self, event: MessageEvent) -> str:
        """Handle /deny command — deny pending dangerous command.

        Usage:
            /deny — Deny pending command
            /deny <id> — Deny specific pending command
        """
        args = event.get_command_args().strip()
        source = event.source
        session_key = self._session_key_for_source(source)

        # Check for pending approvals
        if hasattr(self, "_pending_approvals"):
            pending = self._pending_approvals.get(session_key)
            if not pending:
                return "⚠️ No pending commands to deny"

            if args:
                # Deny specific command by ID
                for approval in pending:
                    if approval.get("id") == args:
                        command = approval.get("command", "")
                        pending.remove(approval)
                        return f"✓ Denied: {command}"
                return f"⚠️ No pending command with ID: {args}"
            else:
                # Deny first pending command
                approval = pending[0]
                command = approval.get("command", "")
                pending.remove(approval)
                return f"✓ Denied: {command}"
        else:
            return "⚠️ No pending commands to deny"
