"""Integration tests for the ``stop()`` extraction (round 49).

Exercises ``GatewayRunner.stop()`` end-to-end with the method moved onto
``GatewayStopMixin``. Uses the shared ``_runner_harness.build_runner`` to
construct a runner with the full set of attrs ``stop()`` touches, wires a
``RecordingAdapter`` so the adapter-teardown phase runs against a real
``BasePlatformAdapter`` subclass, and asserts side-effect canaries for each
teardown phase — proving the verbatim lift didn't drop a phase.

Two contract guarantees of the extraction:

1. **MRO resolution** — ``GatewayStopMixin.stop is GatewayRunner.stop``.
   Without this, the old inline ``stop`` would still be shadowing the mixin
   (or the mixin wasn't added to the bases) and the extraction is a no-op.
2. **Phase canaries** — every distinct teardown phase must leave an
   observable side effect: adapters disconnected, ``_shutdown_event`` set,
   ``_background_tasks`` / ``adapters`` cleared, and the ``.clean_shutdown``
   marker written. A dropped phase (e.g. adapter disconnect skipped during
   the lift) would fail its canary.
"""

from pathlib import Path

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.stop_mixin import GatewayStopMixin

from tests.gateway._runner_harness import RecordingAdapter, build_runner


def test_stop_lives_on_mixin_and_resolves_via_mro():
    """``stop`` must resolve to ``GatewayStopMixin.stop`` through the MRO.

    This is the single most important assertion: it proves the inline
    method was actually removed from run.py AND the mixin was added to
    the bases. If the inline method survived, ``GatewayRunner.stop`` would
    be a different function object than ``GatewayStopMixin.stop``.
    """
    assert GatewayStopMixin.stop is GatewayRunner.stop
    # And GatewayStopMixin is in the MRO.
    assert GatewayStopMixin in GatewayRunner.__mro__


@pytest.mark.asyncio
async def test_stop_runs_all_teardown_phases(monkeypatch, tmp_path):
    """``stop()`` must execute every teardown phase and leave observable
    side effects. Each canary maps to a distinct phase in ``_stop_impl``:

      * adapter disconnect        -> ``adapter._running`` flips to False
      * background-task cancel    -> ``_background_tasks`` cleared
      * state-dict teardown       -> ``adapters`` / ``_pending_*`` cleared
      * shutdown event            -> ``_shutdown_event.is_set()``
      * clean-shutdown marker     -> ``_hermes_home / '.clean_shutdown'``
    """
    # Route the module-global _hermes_home (lazy-imported by stop()) at the
    # tmp_path so the .clean_shutdown marker lands where we can assert it.
    monkeypatch.setattr("gateway.run._hermes_home", Path(tmp_path))

    adapter = RecordingAdapter(platform=Platform.TELEGRAM)
    adapter._running = True  # precondition for the disconnect canary
    runner = build_runner(
        tmp_path=tmp_path,
        adapters={Platform.TELEGRAM: adapter},
    )
    # Zero drain timeout so _drain_active_agents returns immediately
    # (no running agents to wait on) — exercises the graceful path that
    # writes the .clean_shutdown marker.
    runner._restart_drain_timeout = 0.0
    # Seed a background task so the cancel phase has something to clear.
    import asyncio as _asyncio

    async def _noop():
        return None

    runner._background_tasks = {_asyncio.ensure_future(_noop())}

    await runner.stop()

    # --- phase canaries ---
    # Adapter teardown: disconnect() flips _running to False via
    # _mark_disconnected().
    assert adapter._running is False, "adapter.disconnect() phase did not run"
    # State-dict teardown: adapters + pending dicts cleared.
    assert runner.adapters == {}, "adapters.clear() phase did not run"
    assert runner._background_tasks == set() or not runner._background_tasks, (
        "_background_tasks.clear() phase did not run"
    )
    assert runner._pending_messages == {}
    assert runner._pending_approvals == {}
    # Shutdown event signaled.
    assert runner._shutdown_event.is_set(), "_shutdown_event.set() phase did not run"
    # Draining flag reset at the very end of _stop_impl.
    assert runner._draining is False
    # Clean-shutdown marker written (graceful path, no drain timeout).
    assert (Path(tmp_path) / ".clean_shutdown").exists(), (
        ".clean_shutdown marker was not written (graceful-path phase dropped?)"
    )
    # _stop_task was created and awaited.
    assert runner._stop_task is not None


@pytest.mark.asyncio
async def test_stop_restart_branch_sets_flags(monkeypatch, tmp_path):
    """``stop(restart=True)`` must set the restart flags before teardown.

    Exercises the ``if restart:`` branch at the top of ``stop()`` — the
    extraction must preserve the early flag mutation so downstream restart
    logic (detached watcher, systemd shortcut, notification marker) fires.
    """
    monkeypatch.setattr("gateway.run._hermes_home", Path(tmp_path))

    adapter = RecordingAdapter(platform=Platform.TELEGRAM)
    adapter._running = True
    runner = build_runner(
        tmp_path=tmp_path,
        adapters={Platform.TELEGRAM: adapter},
    )
    runner._restart_drain_timeout = 0.0

    await runner.stop(restart=True, detached_restart=True, service_restart=False)

    assert runner._restart_requested is True
    assert runner._restart_detached is True
    assert runner._restart_via_service is False
    # Teardown still ran (restart doesn't skip adapter disconnect).
    assert adapter._running is False
    assert runner._shutdown_event.is_set()
