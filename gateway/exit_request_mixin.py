"""Clean-exit / restart-request / slash-confirm methods for ``GatewayRunner``.

Round 23 of the god-file decomposition. Extracted verbatim into a mixin.
Three related request-flow subsystems lifted together:

1. **Exit-state accessors** — ``should_exit_cleanly`` (plain method) and the
   ``should_exit_with_failure`` / ``exit_reason`` / ``exit_code``
   ``@property`` accessors (``self._exit_cleanly``, ``self._exit_with_failure``,
   ``self._exit_reason``, ``self._exit_code``).
2. **Clean-exit / restart requests** — ``_request_clean_exit``,
   ``request_restart`` (``self._shutdown_event``, ``self._restart_*``,
   ``self._background_tasks``, ``self.stop``).
3. **Slash-command confirmation** — ``_request_slash_confirm``
   (``self._session_key_for_source``, ``self._reply_anchor_for_event``,
   ``self._slash_confirm_counter``, ``self.adapters``).

``self.*`` references resolve unchanged via the MRO. The three ``@property``
decorators are preserved verbatim. Behavior-neutral lift matching the
existing mixin pattern.

``logger`` is lazy-imported inside the method that uses it to avoid a
circular import (gateway.run imports this mixin at module top).
``MessageEvent``, ``Optional``, ``asyncio`` and
``_thread_metadata_for_source`` import at module top from modules with no
circular dependency on gateway.run.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from gateway.gateway_message_pipeline import _thread_metadata_for_source
from gateway.platforms.base import MessageEvent


class GatewayExitRequestMixin:
    """Clean-exit / restart-request / slash-confirm methods for ``GatewayRunner``."""

    @property
    def should_exit_cleanly(self) -> bool:
        return self._exit_cleanly

    @property
    def should_exit_with_failure(self) -> bool:
        return self._exit_with_failure

    @property
    def exit_reason(self) -> Optional[str]:
        return self._exit_reason

    @property
    def exit_code(self) -> Optional[int]:
        return self._exit_code

    def _request_clean_exit(self, reason: str) -> None:
        self._exit_cleanly = True
        self._exit_reason = reason
        self._shutdown_event.set()

    def request_restart(self, *, detached: bool = False, via_service: bool = False) -> bool:
        if self._restart_task_started:
            return False
        self._restart_requested = True
        self._restart_detached = detached
        self._restart_via_service = via_service
        self._restart_task_started = True

        async def _run_restart() -> None:
            await asyncio.sleep(0.05)
            await self.stop(restart=True, detached_restart=detached, service_restart=via_service)

        task = asyncio.create_task(_run_restart())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return True

    async def _request_slash_confirm(
        self,
        *,
        event: MessageEvent,
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
        session_key = self._session_key_for_source(source)
        # Bare-runner test harnesses (object.__new__(GatewayRunner)) skip
        # __init__ and don't have the counter attribute — fall back to a
        # local counter so tests don't AttributeError.  Real runs always
        # have the instance attribute.
        counter = getattr(self, "_slash_confirm_counter", None)
        if counter is None:
            import itertools as _itertools
            counter = _itertools.count(1)
            self._slash_confirm_counter = counter
        confirm_id = f"{next(counter)}"

        # Register the pending confirm FIRST so a super-fast button click
        # cannot race the send_slash_confirm return.
        _slash_confirm_mod.register(session_key, confirm_id, command, handler)

        adapter = self.adapters.get(source.platform)
        metadata = _thread_metadata_for_source(source, self._reply_anchor_for_event(event))

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
                from gateway.run import logger
                logger.debug(
                    "send_slash_confirm failed for %s on %s: %s",
                    command, source.platform, exc,
                )

        if used_buttons:
            # Buttons rendered — no redundant text ack.
            return None
        # Text fallback — return the prompt message as the direct reply.
        return message
