"""MCP tool reload for ``GatewayRunner``.

Round 43 of the god-file decomposition. Lifted verbatim from
gateway/run.py into gateway/mcp_reload_mixin.py.

``_execute_mcp_reload`` actually disconnects, reconnects, and notifies MCP
tool changes. Split out from ``_handle_reload_mcp_command`` so the
confirmation wrapper can invoke the same path whether the user confirmed
via button, text reply, or has the confirm gate disabled. Captures old
server names, shuts down existing connections, re-discovers tools (reading
config.yaml fresh), computes added/removed/reconnected servers, refreshes
cached agents so existing sessions see new MCP tools on their next turn,
and appends a tool-change notice to the session transcript.

``self.*`` references resolve unchanged via the MRO. Behavior-neutral
lift matching the existing mixin pattern.

``gateway.run`` module-level runtime global (``logger``) is lazy-imported
at the top of the method body to avoid the circular import (``gateway.run``
imports this mixin at module top). Stdlib (``asyncio``), the non-circular
translator symbol ``t`` (from agent.i18n), and ``MessageEvent`` (from
gateway.platforms.base) are imported at module top. (Note: the body uses
``t`` both as the module translator and as an unrelated comprehension
target in ``{t["function"]["name"] for t in new_defs}``; the latter is a
local scope and does not shadow the module import outside that
comprehension — analyze.py flags the comprehension target as a body-level
Store, which would otherwise hide the real translator import.)
``shutdown_mcp_servers``, ``discover_mcp_tools``, ``_servers``, and
``_lock`` are imported in-body together at the top of the try block
(already lazy in source), and ``get_tool_definitions`` is imported
in-body within the cached-agent refresh try/except (already lazy in
source) — both kept verbatim.
"""

from __future__ import annotations

import asyncio

from agent.i18n import t
from gateway.platforms.base import MessageEvent


class McpReloadMixin:
    async def _execute_mcp_reload(self, event: MessageEvent) -> str:
        """Actually disconnect, reconnect, and notify MCP tool changes.

        Split out from ``_handle_reload_mcp_command`` so the confirmation
        wrapper can invoke the same path whether the user confirmed via
        button, text reply, or has the confirm gate disabled.
        """
        from gateway.run import logger

        loop = asyncio.get_running_loop()
        try:
            from tools.mcp_tool import shutdown_mcp_servers, discover_mcp_tools, _servers, _lock

            # Capture old server names before shutdown
            with _lock:
                old_servers = set(_servers.keys())

            # Read new config before shutting down, so we know what will be added/removed
            # Shutdown existing connections
            await loop.run_in_executor(None, shutdown_mcp_servers)

            # Reconnect by discovering tools (reads config.yaml fresh)
            new_tools = await loop.run_in_executor(None, discover_mcp_tools)

            # Compute what changed
            with _lock:
                connected_servers = set(_servers.keys())

            added = connected_servers - old_servers
            removed = old_servers - connected_servers
            reconnected = connected_servers & old_servers

            lines = [t("gateway.reload_mcp.header")]
            if reconnected:
                lines.append(t("gateway.reload_mcp.reconnected", names=", ".join(sorted(reconnected))))
            if added:
                lines.append(t("gateway.reload_mcp.added", names=", ".join(sorted(added))))
            if removed:
                lines.append(t("gateway.reload_mcp.removed", names=", ".join(sorted(removed))))
            if not connected_servers:
                lines.append(t("gateway.reload_mcp.none_connected"))
            else:
                lines.append(t("gateway.reload_mcp.tools_available", tools=len(new_tools), servers=len(connected_servers)))

            # Refresh cached agents so existing sessions see new MCP tools on
            # their next turn — without this, the user has to `/new` (which
            # discards conversation history) to pick up tools from a server
            # that was just added or reconnected. The user has already
            # consented to the prompt-cache invalidation via the slash-confirm
            # gate in _handle_reload_mcp_command before we reach this point.
            try:
                from model_tools import get_tool_definitions
                _cache = getattr(self, "_agent_cache", None)
                _cache_lock = getattr(self, "_agent_cache_lock", None)
                if _cache_lock is not None and _cache:
                    with _cache_lock:
                        for _sess_key, _entry in list(_cache.items()):
                            try:
                                _agent = _entry[0] if isinstance(_entry, tuple) else _entry
                            except Exception:
                                continue
                            if _agent is None:
                                continue
                            new_defs = get_tool_definitions(
                                enabled_toolsets=getattr(_agent, "enabled_toolsets", None),
                                disabled_toolsets=getattr(_agent, "disabled_toolsets", None),
                                quiet_mode=True,
                            )
                            _agent.tools = new_defs
                            _agent.valid_tool_names = {
                                t["function"]["name"] for t in new_defs
                            } if new_defs else set()
            except Exception as _exc:
                logger.debug(
                    "Failed to update cached agent tools after MCP reload: %s",
                    _exc,
                )

            # Inject a message at the END of the session history so the
            # model knows tools changed on its next turn.  Appended after
            # all existing messages to preserve prompt-cache for the prefix.
            change_parts = []
            if added:
                change_parts.append(f"Added servers: {', '.join(sorted(added))}")
            if removed:
                change_parts.append(f"Removed servers: {', '.join(sorted(removed))}")
            if reconnected:
                change_parts.append(f"Reconnected servers: {', '.join(sorted(reconnected))}")
            tool_summary = f"{len(new_tools)} MCP tool(s) now available" if new_tools else "No MCP tools available"
            change_detail = ". ".join(change_parts) + ". " if change_parts else ""
            reload_msg = {
                "role": "user",
                "content": f"[IMPORTANT: MCP servers have been reloaded. {change_detail}{tool_summary}. The tool list for this conversation has been updated accordingly.]",
            }
            try:
                session_entry = self.session_store.get_or_create_session(event.source)
                self.session_store.append_to_transcript(
                    session_entry.session_id, reload_msg
                )
            except Exception:
                pass  # Best-effort; don't fail the reload over a transcript write

            return "\n".join(lines)

        except Exception as e:
            logger.warning("MCP reload failed: %s", e)
            return t("gateway.reload_mcp.failed", error=e)
