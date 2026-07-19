"""Async-delegation watcher for ``GatewayRunner``.

Round 41 of the god-file decomposition. Lifted verbatim from
gateway/run.py into gateway/async_delegation_mixin.py.

``_async_delegation_watcher`` drains async-delegation completions and
injects them as new turns. Background subagents
(``delegate_task(background=true)``) run on the async-delegation daemon
executor — they have no per-process watcher task, so their completion
events would only be seen by the post-turn queue drain. This watcher
covers the IDLE case: when a background subagent finishes while no agent
turn is running, its result still re-enters the originating session
promptly.

``self.*`` references resolve unchanged via the MRO. Behavior-neutral
lift matching the existing mixin pattern.

``gateway.run`` module-level runtime symbols (``logger``, and the
run.py-defined free function ``_format_gateway_process_notification``)
are lazy-imported at the top of the method body to avoid the circular
import (``gateway.run`` imports this mixin at module top). Stdlib
(``asyncio``) and the non-circular module symbol
``_enrich_async_delegation_routing`` (from
gateway.gateway_message_pipeline) are imported at module top.
``process_registry`` (as ``_pr``) is imported in-body (already lazy in
source, between the initial sleep and the loop) and kept verbatim.
"""

from __future__ import annotations

import asyncio

from gateway.gateway_message_pipeline import _enrich_async_delegation_routing


class AsyncDelegationMixin:
    async def _async_delegation_watcher(self, interval: float = 2.0) -> None:
        """Drain async-delegation completions and inject them as new turns.

        Background subagents (``delegate_task(background=true)``) run on the
        async-delegation daemon executor — they have no per-process watcher
        task, so their completion events would only be seen by the post-turn
        queue drain. This watcher covers the IDLE case: when a background
        subagent finishes while no agent turn is running, its result still
        re-enters the originating session promptly.

        Mirrors the CLI's idle ``process_loop`` drain. Stays silent when the
        queue has nothing for us; ignores non-async event types (those are
        handled by ``_run_process_watcher`` / the post-turn drain).
        """
        from gateway.run import _format_gateway_process_notification, logger

        await asyncio.sleep(3)  # let platforms finish connecting
        from tools.process_registry import process_registry as _pr
        while self._running:
            try:
                # Peek the queue for async-delegation events. We must NOT
                # consume watch/completion events here (other drains own them),
                # so requeue anything that isn't ours.
                requeue = []
                async_events = []
                while not _pr.completion_queue.empty():
                    try:
                        evt = _pr.completion_queue.get_nowait()
                    except Exception:
                        break
                    if evt.get("type") == "async_delegation":
                        async_events.append(evt)
                    else:
                        requeue.append(evt)
                for evt in requeue:
                    _pr.completion_queue.put(evt)
                for evt in async_events:
                    _enrich_async_delegation_routing(evt)
                    synth_text = _format_gateway_process_notification(evt)
                    if not synth_text:
                        continue
                    try:
                        await self._inject_watch_notification(synth_text, evt)
                    except Exception as e:
                        logger.error("Async delegation injection error: %s", e)
            except Exception as e:
                logger.debug("Async delegation watcher error: %s", e)
            await asyncio.sleep(interval)
