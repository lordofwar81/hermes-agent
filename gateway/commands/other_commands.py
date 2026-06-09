"""
Miscellaneous command handler mixin.

This mixin contains command handlers that don't fit into the main
command categories:

- /footer - Show footer/version info
- /resume - Resume pending sessions
- /branch - Git branch operations
- /usage - Usage/account information
- /insights - Insights/analytics
- /bundles - Skill bundle management
"""

import logging
from typing import Optional

from agent.i18n import t as _t
from gateway.platforms.base import EphemeralReply, MessageEvent

logger = logging.getLogger(__name__)


class OtherCommandMixin:
    """Miscellaneous command handlers.

    This mixin provides handlers for miscellaneous commands that don't
    fit into the main command categories.

    Relies on GatewayRunner state accessed via self.
    """

    async def _handle_footer_command(self, event: MessageEvent) -> str:
        """Handle /footer command - show version/footer info."""
        # Get version from config
        try:
            from hermes_cli.version import __version__
            version = __version__
        except Exception:
            version = "unknown"

        lines = ["**Hermes Gateway**"]
        lines.append(f"Version: {version}")
        lines.append("")
        lines.append("Available commands:")
        lines.append("  /help - Show help")
        lines.append("  /status - Show gateway status")
        lines.append("  /agents - Show active agents")
        lines.append("  /model - Switch model")
        return "\n".join(lines)

    async def _handle_resume_command(self, event: MessageEvent) -> str:
        """Handle /resume command - resume pending sessions."""
        source = event.source
        session_key = self._session_key_for_source(source)

        # Check for pending sessions
        if hasattr(self, "_pending_resumes"):
            pending = self._pending_resumes.get(session_key)
            if pending:
                # Resume logic here
                return f"✓ Resuming session: {pending}"
            else:
                return "No pending sessions to resume"
        return "No pending sessions"

    async def _handle_branch_command(self, event: MessageEvent) -> str:
        """Handle /branch command - git branch operations."""
        args = event.get_command_args().strip().lower()

        if not args or args == "status":
            # Show current branch
            try:
                import subprocess
                result = subprocess.run(
                    ["git", "branch", "--show-current"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                branch = result.stdout.strip()
                return f"Current branch: {branch}"
            except Exception as e:
                logger.debug("Git branch command failed: %s", e)
                return "⚠️ Could not determine current branch"

        if args == "list":
            # List all branches
            try:
                import subprocess
                result = subprocess.run(
                    ["git", "branch", "--format=%(refname:short)"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                branches = result.stdout.strip().split("\n")
                lines = ["**Git Branches:**"]
                lines.extend(branches)
                return "\n".join(lines)
            except Exception as e:
                logger.debug("Git branch list failed: %s", e)
                return "⚠️ Could not list branches"

        return "Usage: /branch <status|list>"

    async def _handle_usage_command(self, event: MessageEvent) -> str:
        """Handle /usage command - show account usage."""
        try:
            from agent.account_usage import render_account_usage_lines
            lines = render_account_usage_lines()
            return "\n".join(lines)
        except Exception as e:
            logger.debug("Usage command failed: %s", e)
            return "⚠️ Could not fetch usage information"

    async def _handle_insights_command(self, event: MessageEvent) -> str:
        """Handle /insights command - show analytics insights."""
        args = event.get_command_args().strip().lower()

        if args == "clear":
            # Clear insights cache
            if hasattr(self, "_insights_cache"):
                self._insights_cache.clear()
            return "✓ Insights cache cleared"

        # Show insights summary
        lines = ["**Insights**"]
        lines.append("Usage: /insights <clear>")
        return "\n".join(lines)

    async def _handle_bundles_command(self, event: MessageEvent) -> str:
        """Handle /bundles command - skill bundle management."""
        args = event.get_command_args().strip().lower()

        if not args or args == "list":
            # List available bundles
            try:
                from tools.skill_usage import list_bundles
                bundles = list_bundles()
                if bundles:
                    lines = ["**Skill Bundles:**"]
                    for bundle in bundles:
                        lines.append(f"  • {bundle}")
                    return "\n".join(lines)
                else:
                    return "No skill bundles found"
            except Exception as e:
                logger.debug("Bundles command failed: %s", e)
                return "⚠️ Could not list bundles"

        return "Usage: /bundles <list>"
