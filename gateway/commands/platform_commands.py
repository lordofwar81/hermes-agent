"""
Platform command handler mixin.

This mixin contains command handlers for platform-related commands:
- /platform - List/pause/resume platform adapters
- /topic - Set Telegram topic mode
- /title - Set group chat title
- /voice - Configure voice mode
- /sethome - Set home location
"""

import logging
from typing import Union

from agent.i18n import t as _t
from gateway.platforms.base import EphemeralReply, MessageEvent, Platform

logger = logging.getLogger(__name__)


class PlatformCommandMixin:
    """Platform command handlers.

    This mixin provides handlers for commands that interact with or configure
    platform adapters and platform-specific features.

    Relies on GatewayRunner state accessed via self.
    """

    async def _handle_platform_command(self, event: MessageEvent) -> str:
        """Handle ``/platform list|pause|resume [name]`` — manage platform adapters.

        Examples:
            ``/platform list``           — show connected + failed/paused platforms
            ``/platform pause whatsapp`` — stop reconnecting to whatsapp
            ``/platform resume whatsapp`` — resume reconnection attempts
        """
        text = (getattr(event, "content", "") or "").strip()
        # Strip the leading "/platform" token if present
        parts = text.split(maxsplit=2)
        if parts and parts[0].lower().lstrip("/").startswith("platform"):
            parts = parts[1:]
        action = (parts[0] if parts else "list").lower()
        target = parts[1].lower() if len(parts) > 1 else ""

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
                        lines.append(f"  · {p.value} — retrying (attempt {attempts})")
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

    async def _handle_topic_command(self, event: MessageEvent, args: str = "") -> str:
        """Handle /topic command — configure Telegram topic mode.

        Usage:
            /topic on — Enable topic mode (create new topic per conversation)
            /topic off — Disable topic mode (use main chat)
            /topic status — Show current topic mode status
        """
        source = event.source
        session_key = self._session_key_for_source(source)

        if not args or args == "status":
            # Show current topic mode status
            if hasattr(self, "_topic_mode_states"):
                topic_state = self._topic_mode_states.get(session_key)
                if topic_state:
                    return f"Topic mode: enabled (session: {session_key})"
            return "Topic mode: disabled"

        if args.lower() in ("on", "true", "1", "enable", "enabled"):
            if not hasattr(self, "_topic_mode_states"):
                self._topic_mode_states = {}
            self._topic_mode_states[session_key] = True
            return "✓ Topic mode enabled for this session"
        elif args.lower() in ("off", "false", "0", "disable", "disabled"):
            if hasattr(self, "_topic_mode_states"):
                self._topic_mode_states.pop(session_key, None)
            return "✓ Topic mode disabled for this session"
        else:
            return "Usage: /topic <on|off|status>"

    async def _handle_title_command(self, event: MessageEvent) -> str:
        """Handle /title command — set group chat title.

        Usage:
            /title show — Show current title
            /title <text> — Set new title
        """
        source = event.source
        args = event.get_command_args().strip()

        if not args or args == "show":
            # Show current title
            title = getattr(source, "title", None)
            if title:
                return f"Chat title: {title}"
            return "No title set"

        # Set new title
        if hasattr(source, "set_title"):
            try:
                await source.set_title(args)
                return f"✓ Title set to: {args}"
            except Exception as e:
                logger.error("Failed to set title: %s", e, exc_info=True)
                return f"⚠️ Failed to set title: {e}"
        else:
            return "⚠️ This platform doesn't support title changes"

    async def _handle_voice_command(self, event: MessageEvent) -> str:
        """Handle /voice command — configure voice mode.

        Usage:
            /voice on — Enable voice input/output
            /voice off — Disable voice mode
            /voice input — Enable voice input only
            /voice output — Enable voice output only
            /voice status — Show current voice mode status
        """
        source = event.source
        session_key = self._session_key_for_source(source)
        args = event.get_command_args().strip().lower()

        if not args or args == "status":
            # Show current voice mode
            if hasattr(self, "_voice_mode_states"):
                voice_state = self._voice_mode_states.get(session_key, {})
                if voice_state:
                    parts = [f"Voice mode: {voice_state.get('mode', 'off')}"]
                    if voice_state.get('input'):
                        parts.append("  Input: enabled")
                    if voice_state.get('output'):
                        parts.append("  Output: enabled")
                    return "\n".join(parts)
            return "Voice mode: disabled"

        if args == "on":
            if not hasattr(self, "_voice_mode_states"):
                self._voice_mode_states = {}
            self._voice_mode_states[session_key] = {"mode": "full", "input": True, "output": True}
            return "✓ Voice mode enabled (input + output)"
        elif args == "off":
            if hasattr(self, "_voice_mode_states"):
                self._voice_mode_states.pop(session_key, None)
            return "✓ Voice mode disabled"
        elif args == "input":
            if not hasattr(self, "_voice_mode_states"):
                self._voice_mode_states = {}
            self._voice_mode_states[session_key] = {"mode": "input", "input": True, "output": False}
            return "✓ Voice input enabled"
        elif args == "output":
            if not hasattr(self, "_voice_mode_states"):
                self._voice_mode_states = {}
            self._voice_mode_states[session_key] = {"mode": "output", "input": False, "output": True}
            return "✓ Voice output enabled"
        else:
            return "Usage: /voice <on|off|input|output|status>"

    async def _handle_set_home_command(self, event: MessageEvent) -> str:
        """Handle /sethome command — set home location for location-aware features.

        Usage:
            /sethome <location> — Set home location
            /sethome clear — Clear home location
            /sethome show — Show current home location
        """
        source = event.source
        session_key = self._session_key_for_source(source)
        args = event.get_command_args().strip()

        if not args or args == "show":
            # Show current home location
            if hasattr(self, "_home_locations"):
                home = self._home_locations.get(session_key)
                if home:
                    return f"Home location: {home}"
            return "No home location set"

        if args.lower() == "clear":
            if hasattr(self, "_home_locations"):
                self._home_locations.pop(session_key, None)
            return "✓ Home location cleared"

        # Set home location
        if not hasattr(self, "_home_locations"):
            self._home_locations = {}
        self._home_locations[session_key] = args
        return f"✓ Home location set to: {args}"
