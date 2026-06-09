"""
Planned stop watcher service.

Polls for the planned-stop marker file and triggers graceful shutdown.
Bridges the gap on Windows where signal handlers don't work for SIGTERM/SIGINT.

The CLI's `hermes_cli.gateway_windows.stop()` writes the marker via
`write_planned_stop_marker(pid)` and then waits for the gateway PID to exit;
this watcher is what makes that exit happen cleanly.

On POSIX this is a no-op safety net — the signal handler always races us
to consuming the marker file because it fires synchronously from the kernel's
signal delivery.

Usage:
    stop_event = threading.Event()
    thread = run_planned_stop_watcher(
        stop_event, runner, loop, shutdown_handler, poll_interval=0.5
    )
    # Later:
    stop_event.set()
    thread.join()
"""

import logging
import threading
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    import asyncio

logger = logging.getLogger(__name__)


def run_planned_stop_watcher(
    stop_event: threading.Event,
    runner,
    loop: "asyncio.AbstractEventLoop",
    shutdown_handler: Callable[[Optional[int]], None],
    *,
    poll_interval: float = 0.5,
) -> threading.Thread:
    """Start the planned stop watcher background thread.

    Args:
        stop_event: Threading event that signals shutdown when set.
        runner: The GatewayRunner instance (checked for _running and _draining state).
        loop: The asyncio event loop for running shutdown_handler.
        shutdown_handler: Callable wired to SIGTERM — tolerates None signal argument.
        poll_interval: Seconds between marker checks (default: 0.5).

    Returns:
        The started thread object.
    """
    thread = threading.Thread(
        target=_watcher_loop,
        args=(stop_event, runner, loop, shutdown_handler, poll_interval),
        name="PlannedStopWatcher",
        daemon=True,
    )
    thread.start()
    return thread


def _watcher_loop(
    stop_event: threading.Event,
    runner,
    loop: "asyncio.AbstractEventLoop",
    shutdown_handler: Callable[[Optional[int]], None],
    poll_interval: float,
) -> None:
    """Main planned stop watcher loop running in background thread.

    Polls for the planned-stop marker and triggers graceful shutdown.
    """
    from gateway.status import (
        _get_planned_stop_marker_path,
        planned_stop_marker_targets_self,
    )

    marker_path = _get_planned_stop_marker_path()

    while not stop_event.is_set():
        try:
            if (
                marker_path.exists()
                and not getattr(runner, "_draining", False)
                and getattr(runner, "_running", False)
            ):
                # A marker existing is NOT sufficient — it may have been
                # written for a PREVIOUS gateway instance (different PID)
                # and left behind because that process exited before the
                # CLI's stop() could clean it up. Firing the handler on a
                # stale/foreign marker drives the gateway into shutdown,
                # then consume_planned_stop_marker_for_self() correctly
                # reports a PID mismatch — but by then we're already
                # stopping, so it's logged as an unexpected "UNKNOWN" exit
                # and the watchdog crash-loops the gateway (issue #34597,
                # a regression from PR #33798 which added this watcher
                # without the PID check).
                #
                # Only fire when the marker actually targets us. The probe
                # is non-destructive on a match (the handler does the
                # authoritative consume on the loop thread) and self-heals
                # by unlinking stale/malformed markers so they cannot wedge
                # a freshly booted gateway.
                if not planned_stop_marker_targets_self():
                    stop_event.wait(poll_interval)
                    continue

                # Drive the same path as a real signal handler.
                # Pass signal=None — the handler tolerates that and consumes
                # the marker via consume_planned_stop_marker_for_self,
                # which also validates target_pid + start_time match us.
                loop.call_soon_threadsafe(shutdown_handler, None)
                # Done — the handler will set _draining; we exit on next tick.
                break
        except Exception as e:
            logger.debug("Planned-stop watcher tick error: %s", e)

        stop_event.wait(poll_interval)


class PlannedStopWatcher:
    """Wrapper class for planned stop watcher service.

    Provides an object-oriented interface for starting and managing
    the planned stop watcher background thread.
    """

    def __init__(self, poll_interval: float = 0.5):
        """Initialize planned stop watcher.

        Args:
            poll_interval: Seconds between marker checks (default: 0.5).
        """
        self._poll_interval = poll_interval
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None

    def start(
        self,
        runner,
        loop: "asyncio.AbstractEventLoop",
        shutdown_handler: Callable[[Optional[int]], None],
    ) -> threading.Thread:
        """Start the planned stop watcher thread.

        Args:
            runner: The GatewayRunner instance.
            loop: The asyncio event loop.
            shutdown_handler: Callable for shutdown handling.

        Returns:
            The started thread object.
        """
        self._stop_event = threading.Event()
        self._thread = run_planned_stop_watcher(
            self._stop_event,
            runner,
            loop,
            shutdown_handler,
            poll_interval=self._poll_interval,
        )
        return self._thread

    def stop(self) -> None:
        """Signal the watcher thread to stop."""
        if self._stop_event:
            self._stop_event.set()

    def join(self, timeout: Optional[float] = None) -> None:
        """Wait for the watcher thread to exit.

        Args:
            timeout: Optional timeout in seconds.
        """
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def is_running(self) -> bool:
        """Check if the watcher thread is currently running.

        Returns:
            True if the thread is alive, False otherwise.
        """
        return self._thread is not None and self._thread.is_alive()
