"""Integration tests for the busy-agent dispatch extraction (round 50).

Exercises ``BusyAgentDispatchMixin._dispatch_busy_agent_message`` end-to-end
via ``GatewayRunner._handle_message`` — the verbatim-lifted PRIORITY region
that routes a message arriving while an agent is already running. Uses the
shared ``_runner_harness.build_runner`` to construct a runner and seeds
``_running_agents[session_key]`` with a fake agent so the busy-branch guard
fires.

Two contract guarantees of the extraction:

1. **MRO resolution** —
   ``BusyAgentDispatchMixin._dispatch_busy_agent_message
      is GatewayRunner._dispatch_busy_agent_message``.
   Without this, the method wouldn't resolve to the mixin (mixin missing
   from bases, or a stale inline copy shadowing it) and the extraction is
   a no-op.
2. **Branch canaries** — every distinct behavior in the lifted region must
   still fire when dispatched through ``_handle_message``:

     * ``/model`` while busy -> rejected with the "Agent is running" string
       (the catch-all reject path for commands that can't run mid-turn).
     * ``/queue <text>`` -> queued via ``_enqueue_fifo`` (depth-1 reply).
     * non-command text follow-up in ``interrupt`` mode -> agent.interrupt()
       called and ``None`` returned.
     * non-command text follow-up in ``queue`` mode -> event queued, no
       interrupt (``_busy_input_mode`` policy respected).
     * subagent-active demotion (#30170) -> interrupt demoted to queue even
       in ``interrupt`` mode.

A dropped branch (e.g. the /model reject skipped during the lift) would fail
its canary.
"""

from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType, SessionSource
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL
from gateway.busy_agent_dispatch_mixin import BusyAgentDispatchMixin

from tests.gateway._runner_harness import (
    RecordingAdapter,
    build_event,
    build_runner,
    build_source,
)


def test_dispatch_lives_on_mixin_and_resolves_via_mro():
    """``_dispatch_busy_agent_message`` must resolve to the mixin through the MRO.

    The single most important assertion: proves the method actually lives on
    ``BusyAgentDispatchMixin`` and that mixin is in ``GatewayRunner``'s bases.
    If the inline region were still in ``_handle_message`` (no delegate) or
    the mixin wasn't appended to the bases, this identity check fails.
    """
    assert (
        BusyAgentDispatchMixin._dispatch_busy_agent_message
        is GatewayRunner._dispatch_busy_agent_message
    )
    assert BusyAgentDispatchMixin in GatewayRunner.__mro__


def _seed_running_agent(runner, session_key, *, agent=None, ts=0.0):
    """Plant a fake running agent under session_key so the busy guard fires.

    Also stubs the access-control gate (``_check_slash_access``) to "allowed"
    (None) so command-routing canaries reach their target branch instead of
    being short-circuited by a deny. ``_check_slash_access`` lives on
    ``authz_mixin`` and resolves via MRO in production, but the harness-built
    runner doesn't carry real config-backed allowlists, so we stub it to keep
    the canary focused on dispatch routing.
    """
    runner._running_agents[session_key] = agent if agent is not None else MagicMock()
    runner._running_agents_ts[session_key] = ts
    runner._check_slash_access = lambda _source, _name: None


@pytest.mark.asyncio
async def test_model_command_rejected_while_busy(tmp_path):
    """``/model`` while an agent is running must hit the catch-all reject path.

    This exercises the region's last-resort branch: a recognized slash command
    that can't run mid-turn returns the "Agent is running" string rather than
    interrupting or queuing. If the reject branch was dropped during the lift,
    the message would fall through to ``_handle_model_command`` (wrong path).
    """
    adapter = RecordingAdapter(platform=Platform.TELEGRAM)
    runner = build_runner(tmp_path=tmp_path, adapters={Platform.TELEGRAM: adapter})
    event = build_event(text="/model openai/gpt-5", platform=Platform.TELEGRAM)
    session_key = runner._session_key_for_source(event.source)
    _seed_running_agent(runner, session_key)

    result = await runner._handle_message(event)

    assert isinstance(result, str)
    assert "running" in result.lower(), f"expected busy-reject, got: {result!r}"


@pytest.mark.asyncio
async def test_queue_command_queues_followup(tmp_path):
    """``/queue <text>`` while busy must enqueue via _enqueue_fifo.

    Exercises the dedicated /queue branch: it builds a queued MessageEvent and
    calls ``_enqueue_fifo``, returning the depth-1 "Queued for the next turn."
    reply. A dropped branch would either interrupt the agent or fall through
    to the catch-all reject.
    """
    adapter = RecordingAdapter(platform=Platform.TELEGRAM)
    runner = build_runner(tmp_path=tmp_path, adapters={Platform.TELEGRAM: adapter})
    event = build_event(text="/queue do the thing", platform=Platform.TELEGRAM)
    session_key = runner._session_key_for_source(event.source)
    _seed_running_agent(runner, session_key)

    result = await runner._handle_message(event)

    assert result == "Queued for the next turn."
    # The event must have landed in the FIFO for this session.
    assert runner._queue_depth(session_key, adapter=adapter) == 1


@pytest.mark.asyncio
async def test_text_followup_in_interrupt_mode_calls_interrupt(tmp_path):
    """Non-command text in ``interrupt`` mode must call agent.interrupt().

    Exercises the terminal branch of the region: a plain-text follow-up with
    no special-case match calls ``running_agent.interrupt(event.text)`` and
    returns ``None``. If the interrupt branch was dropped, the message would
    either be queued or fall through to a new agent run (wrong).
    """
    adapter = RecordingAdapter(platform=Platform.TELEGRAM)
    runner = build_runner(tmp_path=tmp_path, adapters={Platform.TELEGRAM: adapter})
    runner._busy_input_mode = "interrupt"
    event = build_event(text="wait, also do X", platform=Platform.TELEGRAM)
    session_key = runner._session_key_for_source(event.source)
    fake_agent = MagicMock()
    _seed_running_agent(runner, session_key, agent=fake_agent)

    result = await runner._handle_message(event)

    assert result is None
    fake_agent.interrupt.assert_called_once_with("wait, also do X")


@pytest.mark.asyncio
async def test_text_followup_in_queue_mode_does_not_interrupt(tmp_path):
    """Non-command text in ``queue`` mode must queue, NOT interrupt.

    Exercises the ``_busy_input_mode == "queue"`` branch: the event is routed
    to ``_queue_or_replace_pending_event`` and the agent's ``interrupt()`` is
    never called. If the queue-mode branch was dropped, the message would hit
    the interrupt path below it (wrong).
    """
    adapter = RecordingAdapter(platform=Platform.TELEGRAM)
    runner = build_runner(tmp_path=tmp_path, adapters={Platform.TELEGRAM: adapter})
    runner._busy_input_mode = "queue"
    event = build_event(text="another followup", platform=Platform.TELEGRAM)
    session_key = runner._session_key_for_source(event.source)
    fake_agent = MagicMock()
    _seed_running_agent(runner, session_key, agent=fake_agent)

    result = await runner._handle_message(event)

    assert result is None
    fake_agent.interrupt.assert_not_called()
    # And the event was queued for the session.
    assert runner._queue_depth(session_key, adapter=adapter) >= 1


@pytest.mark.asyncio
async def test_subagent_active_demotes_interrupt_to_queue(tmp_path):
    """#30170 — active subagents must demote an interrupt to queue semantics.

    Exercises the subagent-protection branch: even in ``interrupt`` mode, if
    ``_agent_has_active_subagents(running_agent)`` is truthy, the message is
    queued instead of interrupting (which would cascade-abort delegate_task
    work). A dropped branch would call ``interrupt()`` and destroy subagent
    progress.
    """
    adapter = RecordingAdapter(platform=Platform.TELEGRAM)
    runner = build_runner(tmp_path=tmp_path, adapters={Platform.TELEGRAM: adapter})
    runner._busy_input_mode = "interrupt"
    event = build_event(text="casual followup", platform=Platform.TELEGRAM)
    session_key = runner._session_key_for_source(event.source)
    fake_agent = MagicMock()
    _seed_running_agent(runner, session_key, agent=fake_agent)

    # Force the subagent-active check to True for this agent. The mixin
    # imports _agent_has_active_subagents into its OWN module namespace at
    # import time (from gateway.gateway_events), so the patch must target
    # the mixin module — patching gateway.run has no effect.
    import gateway.busy_agent_dispatch_mixin as _bam
    original = _bam._agent_has_active_subagents
    _bam._agent_has_active_subagents = lambda _a: True
    try:
        result = await runner._handle_message(event)
    finally:
        _bam._agent_has_active_subagents = original

    assert result is None
    fake_agent.interrupt.assert_not_called()
    assert runner._queue_depth(session_key, adapter=adapter) >= 1
