"""Destructive-slash confirmation gate for ``GatewayRunner``.

Round 42 of the god-file decomposition. Lifted verbatim from
gateway/run.py into gateway/slash_confirm_mixin.py.

``_maybe_confirm_destructive_slash`` gates a destructive session slash
command (``/new``, ``/reset``, ``/undo``). ``execute`` is an async
callable that performs the destructive action. If the
``approvals.destructive_slash_confirm`` config gate is off, ``execute``
runs immediately. Otherwise this routes through
``_request_slash_confirm`` — native yes/no buttons on
Telegram/Discord/Slack, text fallback elsewhere. Three-option
resolution: ``once`` runs ``execute``; ``always`` persists the opt-out
then runs ``execute``; ``cancel`` returns a cancelled message without
running ``execute``.

``self.*`` references resolve unchanged via the MRO. Behavior-neutral
lift matching the existing mixin pattern.

``gateway.run`` module-level runtime global (``logger``) is
lazy-imported at the top of the method body to avoid the circular import
(``gateway.run`` imports this mixin at module top). Stdlib typing symbol
(``Union``) and ``MessageEvent`` are imported at module top.
``_read_user_config`` is reached via ``self._read_user_config`` (the R48
staticmethod binding on GatewayRunner) so tests can override the config by
patching the instance; ``slash_commands.py`` uses the same pattern.
``save_config_value`` is imported in-body within the ``always``
branch (already lazy in source, inside a try/except) and kept verbatim.
"""

from __future__ import annotations

from typing import Union

from gateway.platforms.base import MessageEvent


class SlashConfirmMixin:
    async def _maybe_confirm_destructive_slash(
        self,
        *,
        event: MessageEvent,
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
        through ``_request_slash_confirm`` — native yes/no buttons on
        Telegram/Discord/Slack, text fallback elsewhere.

        Three-option resolution:

          - ``once``  — run ``execute`` and return its result
          - ``always`` — persist ``approvals.destructive_slash_confirm: false``,
                        then run ``execute``
          - ``cancel`` — return a "cancelled" message; do not run ``execute``
        """
        from gateway.run import logger

        # Gate check. Read via self._read_user_config (the R48 staticmethod
        # binding on GatewayRunner) so tests can override the config by
        # patching the instance; slash_commands.py uses the same pattern.
        confirm_required = True
        try:
            cfg = self._read_user_config()
            approvals = cfg.get("approvals") if isinstance(cfg, dict) else None
            if isinstance(approvals, dict):
                confirm_required = bool(approvals.get("destructive_slash_confirm", True))
        except Exception:
            pass

        if not confirm_required:
            return await execute()

        session_key = self._session_key_for_source(event.source)

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

        _p = self._typed_command_prefix_for(event.source.platform)
        prompt_message = (
            f"⚠️ **Confirm /{command}**\n\n"
            f"{detail}\n\n"
            "Choose:\n"
            "• **Approve Once** — proceed this time only\n"
            "• **Always Approve** — proceed and silence this prompt permanently\n"
            "• **Cancel** — keep current conversation\n\n"
            f"_Text fallback: reply `{_p}approve`, `{_p}always`, or `{_p}cancel`._"
        )
        return await self._request_slash_confirm(
            event=event,
            command=command,
            title=title,
            message=prompt_message,
            handler=_on_confirm,
        )
