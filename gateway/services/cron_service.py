"""
Cron ticker service.

Background thread that ticks the cron scheduler at a regular interval.
Runs inside the gateway process so cronjobs fire automatically without
needing a separate `hermes cron daemon` or system cron entry.

Usage:
    stop_event = threading.Event()
    thread = start_cron_ticker(stop_event, adapters, loop, interval=60)
    # Later:
    stop_event.set()
    thread.join()
"""

import logging
import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import asyncio

logger = logging.getLogger(__name__)

# Interval constants (in ticks, where 1 tick = interval seconds)
IMAGE_CACHE_EVERY = 60  # ticks — once per hour at default 60s interval
CHANNEL_DIR_EVERY = 5  # ticks — every 5 minutes
PASTE_SWEEP_EVERY = 60  # ticks — once per hour
CURATOR_EVERY = 60  # ticks — poll hourly (inner gate handles the real cadence)


def start_cron_ticker(
    stop_event: threading.Event,
    adapters=None,
    loop: Optional["asyncio.AbstractEventLoop"] = None,
    interval: int = 60,
) -> threading.Thread:
    """Start the cron ticker background thread.

    Args:
        stop_event: Threading event that signals shutdown when set.
        adapters: Optional platform adapters dict for E2EE room support.
        loop: Optional asyncio event loop for async operations.
        interval: Seconds between cron ticks (default: 60).

    Returns:
        The started thread object.
    """
    thread = threading.Thread(
        target=_cron_ticker_loop,
        args=(stop_event, adapters, loop, interval),
        name="CronTicker",
        daemon=True,
    )
    thread.start()
    return thread


def _cron_ticker_loop(
    stop_event: threading.Event,
    adapters,
    loop: Optional["asyncio.AbstractEventLoop"],
    interval: int,
) -> None:
    """Main cron ticker loop running in background thread."""
    from agent.async_utils import safe_schedule_threadsafe
    from cron.scheduler import tick as cron_tick
    from gateway.platforms.base import cleanup_document_cache, cleanup_image_cache
    from hermes_cli.debug import _sweep_expired_pastes

    logger.info("Cron ticker started (interval=%ds)", interval)
    tick_count = 0

    while not stop_event.is_set():
        try:
            cron_tick(verbose=False, adapters=adapters, loop=loop)
        except Exception as e:
            logger.debug("Cron tick error: %s", e)

        tick_count += 1

        # Refresh channel directory every 5 minutes
        if tick_count % CHANNEL_DIR_EVERY == 0 and adapters:
            try:
                from gateway.channel_directory import build_channel_directory

                if loop is not None:
                    # build_channel_directory is async (Slack web calls), and
                    # this ticker runs in a background thread. Schedule onto
                    # the gateway event loop and wait briefly for completion
                    # so refresh failures are still logged via the except.
                    fut = safe_schedule_threadsafe(
                        build_channel_directory(adapters),
                        loop,
                        logger=logger,
                        log_message="Channel directory refresh scheduling error",
                    )
                    if fut is not None:
                        fut.result(timeout=30)
            except Exception as e:
                logger.debug("Channel directory refresh error: %s", e)

        # Cleanup image/document cache once per hour
        if tick_count % IMAGE_CACHE_EVERY == 0:
            try:
                removed = cleanup_image_cache(max_age_hours=24)
                if removed:
                    logger.info("Image cache cleanup: removed %d stale file(s)", removed)
            except Exception as e:
                logger.debug("Image cache cleanup error: %s", e)
            try:
                removed = cleanup_document_cache(max_age_hours=24)
                if removed:
                    logger.info("Document cache cleanup: removed %d stale file(s)", removed)
            except Exception as e:
                logger.debug("Document cache cleanup error: %s", e)

        # Sweep expired debug paste shares once per hour
        if tick_count % PASTE_SWEEP_EVERY == 0:
            try:
                deleted, remaining = _sweep_expired_pastes()
                if deleted:
                    logger.info(
                        "Paste sweep: deleted %d expired paste(s), %d pending",
                        deleted,
                        remaining,
                    )
            except Exception as e:
                logger.debug("Paste sweep error: %s", e)

        # Curator — piggy-back on the existing cron ticker so long-running
        # gateways get weekly skill maintenance without needing restarts.
        # maybe_run_curator() is internally gated by config.interval_hours
        # (7 days by default), so CURATOR_EVERY is just the poll rate — the
        # real work only fires once per config interval.
        if tick_count % CURATOR_EVERY == 0:
            try:
                from agent.curator import maybe_run_curator

                maybe_run_curator(
                    idle_for_seconds=float("inf"),
                    on_summary=lambda msg: logger.info("curator: %s", msg),
                )
            except Exception as e:
                logger.debug("Curator tick error: %s", e)

        stop_event.wait(timeout=interval)

    logger.info("Cron ticker stopped")


class CronService:
    """Wrapper class for cron ticker service.

    Provides an object-oriented interface for starting and managing
    the cron ticker background thread.
    """

    def __init__(self, interval: int = 60):
        """Initialize cron service.

        Args:
            interval: Seconds between cron ticks (default: 60).
        """
        self._interval = interval
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None

    def start(
        self,
        adapters=None,
        loop: Optional["asyncio.AbstractEventLoop"] = None,
    ) -> threading.Thread:
        """Start the cron ticker thread.

        Args:
            adapters: Optional platform adapters dict.
            loop: Optional asyncio event loop.

        Returns:
            The started thread object.
        """
        self._stop_event = threading.Event()
        self._thread = start_cron_ticker(self._stop_event, adapters, loop, self._interval)
        return self._thread

    def stop(self) -> None:
        """Signal the cron ticker thread to stop."""
        if self._stop_event:
            self._stop_event.set()

    def join(self, timeout: Optional[float] = None) -> None:
        """Wait for the cron ticker thread to exit.

        Args:
            timeout: Optional timeout in seconds.
        """
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def is_running(self) -> bool:
        """Check if the cron ticker thread is currently running.

        Returns:
            True if the thread is alive, False otherwise.
        """
        return self._thread is not None and self._thread.is_alive()
