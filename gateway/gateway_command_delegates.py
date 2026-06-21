"""Gateway slash-command delegation shims extracted from GatewayRunner.

Round 12 of gateway decomposition. /suggestions and /blueprint are thin
async shims: they build an ``origin`` dict from the message event and
delegate to the shared hermes_cli handler so CLI/TUI/gateway never drift.
Neither touches instance state — ``self`` was unused in both.
"""

from gateway.platforms.base import MessageEvent

import logging
logger = logging.getLogger("gateway.run")


async def _handle_suggestions_command(event: MessageEvent) -> str:
    """Handle /suggestions in the gateway.

    Delegates to the shared handler so CLI and gateway never drift. The
    origin is built from the event source so an accepted suggestion's job
    delivers back to this chat/thread.
    """
    args = (event.get_command_args() or "").strip()
    source = event.source
    origin = None
    try:
        platform = getattr(source.platform, "value", None) or str(getattr(source, "platform", "") or "")
        chat_id = getattr(source, "chat_id", None)
        if platform and chat_id:
            origin = {
                "platform": platform,
                "chat_id": str(chat_id),
                "chat_name": getattr(source, "chat_name", None),
                "thread_id": getattr(source, "thread_id", None),
            }
    except Exception:
        origin = None
    try:
        from hermes_cli.suggestions_cmd import handle_suggestions_command

        return handle_suggestions_command(args, origin=origin, surface="gateway")
    except Exception as e:
        logger.debug("suggestions command failed: %s", e)
        return f"Suggestions command failed: {e}"


async def _handle_blueprint_command(event: MessageEvent):
    """Handle /blueprint in the gateway.

    Delegates to the shared handler so CLI, TUI, and gateway never drift.
    Returns a BlueprintCommandResult: ``text`` is shown to the user, and if
    ``agent_seed`` is set the dispatch site rewrites ``event.text`` to the
    seed and falls through to the agent (the ``/steer`` pattern) so the
    agent gathers the slot values conversationally. Origin is built from the
    event source so a directly created blueprint job delivers back to this chat.
    """
    args = (event.get_command_args() or "").strip()
    source = event.source
    origin = None
    try:
        platform = getattr(source.platform, "value", None) or str(getattr(source, "platform", "") or "")
        chat_id = getattr(source, "chat_id", None)
        if platform and chat_id:
            origin = {
                "platform": platform,
                "chat_id": str(chat_id),
                "chat_name": getattr(source, "chat_name", None),
                "thread_id": getattr(source, "thread_id", None),
            }
    except Exception:
        origin = None
    try:
        from hermes_cli.blueprint_cmd import handle_blueprint_command

        return handle_blueprint_command(args, origin=origin, surface="gateway")
    except Exception as e:
        logger.debug("blueprint command failed: %s", e)
        from hermes_cli.blueprint_cmd import BlueprintCommandResult

        return BlueprintCommandResult(f"Cron blueprint command failed: {e}")
