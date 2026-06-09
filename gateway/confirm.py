"""
Slash confirmation utilities for GatewayRunner.

Extracted from gateway/run.py to reduce the God file size.
Provides functions for handling destructive slash command confirmations.
"""

import logging
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from gateway.run import GatewayRunner
    from gateway.message_event import MessageEvent

logger = logging.getLogger(__name__)


async def maybe_confirm_destructive_slash(
    runner,  # GatewayRunner instance
    *,
    event: "MessageEvent",
    command: str,
    title: str,
    detail: str,
    execute,
) -> Union[str, "EphemeralReply", None]:
    """Gate a destructive session slash command (/new, /reset, /undo).

    ``execute`` is an async callable ``execute() -> str | EphemeralReply``
    that performs the destructive action.  If the
    ``approvals.destructive_slash_confirm`` config gate is off, ``execute``
    runs immediately (returning its result).  Otherwise this routes
    through ``request_slash_confirm`` — native yes/no buttons on
    Telegram/Discord/Slack, text fallback elsewhere.

    Three-option resolution:

      - ``once``  — run ``execute`` and return its result
      - ``always`` — persist ``approvals.destructive_slash_confirm: false``,
                    then run ``execute``
      - ``cancel`` — return a "cancelled" message; do not run ``execute``
    """
    # Gate check.
    confirm_required = True
    try:
        cfg = runner._read_user_config()
        approvals = cfg.get("approvals") if isinstance(cfg, dict) else None
        if isinstance(approvals, dict):
            confirm_required = bool(approvals.get("destructive_slash_confirm", True))
    except Exception:
        pass

    if not confirm_required:
        return await execute()

    session_key = runner._session_key_for_source(event.source)

    async def _on_confirm(choice: str):
        if choice == "cancel":
            return f"🟡 /{command} cancelled. Conversation unchanged."
        if choice == "always":
            try:
                from cli import save_config_value
                save_config_value("approvals.destructive_slash_confirm", False)
                logger.info(
                    "User opted out of destructive slash confirm (session=%s)",
                    session_key,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to persist destructive_slash_confirm=false: %s", exc,
                )
        result = await execute()
        if choice == "always":
            note = (
                "\n\nℹ️ Future /clear, /new, /reset, and /undo will run "
                "without confirmation. Re-enable via "
                "`approvals.destructive_slash_confirm: true` in config.yaml."
            )
            if isinstance(result, str):
                return result + note
            # EphemeralReply or other — leave untouched; the opt-out note
            # would otherwise mangle structured replies.  The persist itself
            # already happened above; user gets the same UX next time.
            return result
        return result

    prompt_message = (
        f"⚠️ **Confirm /{command}**\n\n"
        f"{detail}\n\n"
        "Choose:\n"
        "• **Approve Once** — proceed this time only\n"
        "• **Always Approve** — proceed and silence this prompt permanently\n"
        "• **Cancel** — keep current conversation\n\n"
        "_Text fallback: reply `/approve`, `/always`, or `/cancel`._"
    )
    return await runner._request_slash_confirm(
        event=event,
        command=command,
        title=title,
        message=prompt_message,
        handler=_on_confirm,
    )


async def request_slash_confirm(
    runner,  # GatewayRunner instance
    *,
    event: "MessageEvent",
    command: str,
    title: str,
    message: str,
    handler,
) -> Optional[str]:
    """Ask the user to confirm an expensive slash command.

    ``handler`` is an async callable ``handler(choice: str) -> str``
    where ``choice`` is ``"once"``, ``"always"``, or ``"cancel"``.
    The handler runs on the event loop when the user responds; its
    return value is sent back as a gateway message.

    Returns a short acknowledgment string to send immediately (before
    the user's response).  If buttons rendered successfully the ack
    is ``None`` (buttons are self-explanatory); if we fell back to
    text the message itself IS the ack.
    """
    from tools import slash_confirm as _slash_confirm_mod

    source = event.source
    session_key = runner._session_key_for_source(source)
    # Bare-runner test harnesses (object.__new__(GatewayRunner)) skip
    # __init__ and don't have the counter attribute — fall back to a
    # local counter so tests don't AttributeError.  Real runs always
    # have the instance attribute.
    counter = getattr(runner, "_slash_confirm_counter", None)
    if counter is None:
        import itertools as _itertools
        counter = _itertools.count(1)
        runner._slash_confirm_counter = counter
    confirm_id = f"{next(counter)}"

    # Register the pending confirm FIRST so a super-fast button click
    # cannot race the send_slash_confirm return.
    _slash_confirm_mod.register(session_key, confirm_id, command, handler)

    adapter = runner.adapters.get(source.platform)
    metadata = runner._thread_metadata_for_source(source, runner._reply_anchor_for_event(event))

    used_buttons = False
    if adapter is not None:
        try:
            button_result = await adapter.send_slash_confirm(
                chat_id=source.chat_id,
                title=title,
                message=message,
                session_key=session_key,
                confirm_id=confirm_id,
                metadata=metadata,
            )
            if button_result and getattr(button_result, "success", False):
                used_buttons = True
        except Exception as exc:
            logger.debug(
                "send_slash_confirm failed for %s on %s: %s",
                command, source.platform, exc,
            )

    if used_buttons:
        # Buttons rendered — no redundant text ack.
        return None
    # Text fallback — return the prompt message as the direct reply.
    return message
