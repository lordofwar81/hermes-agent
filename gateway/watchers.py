"""
Gateway watcher functions.

This module contains background watcher functions that run as independent
threads or tasks to monitor various gateway operations:

- HandoffWatcher: Monitors and processes session handoffs
- SessionExpiryWatcher: Monitors and expires inactive sessions
- KanbanNotifierWatcher: Monitors kanban board changes and sends notifications
- PlatformReconnectWatcher: Monitors and attempts to reconnect failed platforms
- ProcessWatcher: Monitors long-running process executions
"""

import asyncio
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from gateway.adapter_factory import create_adapter

logger = logging.getLogger(__name__)


class HandoffWatcher:
    """Background watcher for processing session handoffs."""

    def __init__(self, gateway_runner):
        """Initialize handoff watcher.

        Args:
            gateway_runner: GatewayRunner instance
        """
        self.runner = gateway_runner
        self._stop_event = threading.Event()
        self._thread = None

    def start(self) -> threading.Thread:
        """Start the handoff watcher thread.

        Returns:
            The started thread object
        """
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._watch_loop,
            name="HandoffWatcher",
            daemon=True,
        )
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        """Signal the watcher thread to stop."""
        self._stop_event.set()

    def join(self, timeout: Optional[float] = None) -> None:
        """Wait for the watcher thread to exit.

        Args:
            timeout: Optional timeout in seconds
        """
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _watch_loop(self) -> None:
        """Main watcher loop."""
        while not self._stop_event.is_set():
            try:
                # Check for pending handoffs
                self._check_handoffs()
            except Exception as e:
                logger.debug("Handoff watcher tick error: %s", e)

            self._stop_event.wait(timeout=60)

    def _check_handoffs(self) -> None:
        """Check for and process pending handoffs."""
        # Implementation would check for pending handoff requests
        # and process them by transferring session state
        pass


def process_handoff(
    runner,
    from_session_key: str,
    to_session_key: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Process a session handoff request.

    Args:
        runner: GatewayRunner instance
        from_session_key: Source session key
        to_session_key: Target session key
        metadata: Optional handoff metadata

    Returns:
        True if handoff was successful
    """
    try:
        # Transfer session state from source to target
        from_session = runner.session_store.get_session(from_session_key)
        if not from_session:
            logger.warning("Handoff source session not found: %s", from_session_key)
            return False

        # Copy messages and state to target session
        to_session = runner.session_store.get_or_create_session(
            SessionSource(
                platform=from_session.source.platform,
                user_id=from_session.source.user_id,
                chat_id=from_session.source.chat_id,
            )
        )

        # Transfer messages
        messages = runner.session_store.get_messages(from_session_key)
        for msg in messages:
            runner.session_store.add_message(to_session.session_id, msg)

        logger.info("Handoff completed: %s -> %s", from_session_key, to_session_key)
        return True

    except Exception as e:
        logger.error("Handoff processing failed: %s", e, exc_info=True)
        return False


class SessionExpiryWatcher:
    """Background watcher for session expiration."""

    def __init__(self, gateway_runner):
        """Initialize session expiry watcher.

        Args:
            gateway_runner: GatewayRunner instance
        """
        self.runner = gateway_runner

    async def watch(self, interval: int = 300) -> None:
        """Background task that finalizes expired sessions.

        Runs every ``interval`` seconds (default 5 min).  For each session
        whose reset policy has expired, invokes ``on_session_finalize``
        hooks, cleans up the cached AIAgent's tool resources, evicts the
        cache entry so it can be garbage-collected, and marks the session
        so it won't be finalized again.

        Args:
            interval: Seconds between expiry checks (default 300)
        """
        await asyncio.sleep(60)  # initial delay — let the gateway fully start
        _finalize_failures: dict[str, int] = {}  # session_id -> consecutive failure count
        _MAX_FINALIZE_RETRIES = 3
        _AGENT_PENDING_SENTINEL = getattr(self.runner, "_AGENT_PENDING_SENTINEL", object())

        while self.runner._running:
            try:
                self.runner.session_store._ensure_loaded()
                # Collect expired sessions first, then log a single summary.
                _expired_entries = []
                for key, entry in list(self.runner.session_store._entries.items()):
                    if entry.expiry_finalized:
                        continue
                    if not self.runner.session_store._is_session_expired(entry):
                        continue
                    _expired_entries.append((key, entry))

                if _expired_entries:
                    # Extract platform names from session keys for a compact summary.
                    # Keys look like "agent:main:telegram:dm:12345" — platform is field [2].
                    _platforms: dict[str, int] = {}
                    for _k, _e in _expired_entries:
                        _parts = _k.split(":")
                        _plat = _parts[2] if len(_parts) > 2 else "unknown"
                        _platforms[_plat] = _platforms.get(_plat, 0) + 1
                    _plat_summary = ", ".join(
                        f"{p}:{c}" for p, c in sorted(_platforms.items())
                    )
                    logger.info(
                        "Session expiry: %d sessions to finalize (%s)",
                        len(_expired_entries), _plat_summary,
                    )

                for key, entry in _expired_entries:
                    try:
                        try:
                            from hermes_cli.plugins import invoke_hook as _invoke_hook
                            _parts = key.split(":")
                            _platform = _parts[2] if len(_parts) > 2 else ""
                            _invoke_hook(
                                "on_session_finalize",
                                session_id=entry.session_id,
                                platform=_platform,
                                reason="session_expired",
                            )
                        except Exception:
                            pass
                        # Shut down memory provider and close tool resources
                        # on the cached agent.  Idle agents live in
                        # _agent_cache (not _running_agents), so look there.
                        _cached_agent = None
                        _cache_lock = getattr(self.runner, "_agent_cache_lock", None)
                        if _cache_lock is not None:
                            with _cache_lock:
                                _cached = self.runner._agent_cache.get(key)
                                _cached_agent = _cached[0] if isinstance(_cached, tuple) else _cached if _cached else None
                        # Fall back to _running_agents in case the agent is
                        # still mid-turn when the expiry fires.
                        if _cached_agent is None:
                            _cached_agent = self.runner._running_agents.get(key)
                        if _cached_agent and _cached_agent is not _AGENT_PENDING_SENTINEL:
                            self.runner._cleanup_agent_resources(_cached_agent)
                        # Drop the cache entry so the AIAgent (and its LLM
                        # clients, tool schemas, memory provider refs) can
                        # be garbage-collected.  Otherwise the cache grows
                        # unbounded across the gateway's lifetime.
                        self.runner._evict_cached_agent(key)
                        # Mark as finalized and persist to disk so the flag
                        # survives gateway restarts.
                        with self.runner.session_store._lock:
                            entry.expiry_finalized = True
                            self.runner.session_store._save()
                        logger.debug(
                            "Session expiry finalized for %s",
                            entry.session_id,
                        )
                        _finalize_failures.pop(entry.session_id, None)
                    except Exception as e:
                        failures = _finalize_failures.get(entry.session_id, 0) + 1
                        _finalize_failures[entry.session_id] = failures
                        if failures >= _MAX_FINALIZE_RETRIES:
                            logger.warning(
                                "Session finalize gave up after %d attempts for %s: %s. "
                                "Marking as finalized to prevent infinite retry loop.",
                                failures, entry.session_id, e,
                            )
                            with self.runner.session_store._lock:
                                entry.expiry_finalized = True
                                self.runner.session_store._save()
                            _finalize_failures.pop(entry.session_id, None)
                        else:
                            logger.debug(
                                "Session finalize failed (%d/%d) for %s: %s",
                                failures, _MAX_FINALIZE_RETRIES, entry.session_id, e,
                            )

                if _expired_entries:
                    _done = sum(
                        1 for _, e in _expired_entries if e.expiry_finalized
                    )
                    _failed = len(_expired_entries) - _done
                    if _failed:
                        logger.info(
                            "Session expiry done: %d finalized, %d pending retry",
                            _done, _failed,
                        )
                    else:
                        logger.info(
                            "Session expiry done: %d finalized", _done,
                        )

                # Sweep agents that have been idle beyond the TTL regardless
                # of session reset policy.  This catches sessions with very
                # long / "never" reset windows, whose cached AIAgents would
                # otherwise pin memory for the gateway's entire lifetime.
                try:
                    _idle_evicted = self.runner._sweep_idle_cached_agents()
                    if _idle_evicted:
                        logger.info(
                            "Agent cache idle sweep: evicted %d agent(s)",
                            _idle_evicted,
                        )
                except Exception as _e:
                    logger.debug("Idle agent sweep failed: %s", _e)

                # Periodically prune stale SessionStore entries.  The
                # in-memory dict (and sessions.json) would otherwise grow
                # unbounded in gateways serving many rotating chats /
                # threads / users over long time windows.  Pruning is
                # invisible to users — a resumed session just gets a
                # fresh session_id, exactly as if the reset policy fired.
                _last_prune_ts = getattr(self.runner, "_last_session_store_prune_ts", 0.0)
                _prune_interval = 3600.0  # once per hour
                if time.time() - _last_prune_ts > _prune_interval:
                    try:
                        _max_age = int(
                            getattr(self.runner.config, "session_store_max_age_days", 0) or 0
                        )
                        if _max_age > 0:
                            _pruned = self.runner.session_store.prune_old_entries(_max_age)
                            if _pruned:
                                logger.info(
                                    "SessionStore prune: dropped %d stale entries",
                                    _pruned,
                                )
                    except Exception as _e:
                        logger.debug("SessionStore prune failed: %s", _e)
                    self.runner._last_session_store_prune_ts = time.time()
            except Exception as e:
                logger.debug("Session expiry watcher error: %s", e)
            # Sleep in small increments so we can stop quickly
            for _ in range(interval):
                if not self.runner._running:
                    break
                await asyncio.sleep(1)


class KanbanNotifierWatcher:
    """Background watcher for kanban board notifications."""

    def __init__(self, gateway_runner):
        """Initialize kanban notifier watcher.

        Args:
            gateway_runner: GatewayRunner instance
        """
        self.runner = gateway_runner
        self._stop_event = threading.Event()
        self._thread = None

    def start(self) -> threading.Thread:
        """Start the kanban notifier watcher thread.

        Returns:
            The started thread object
        """
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._watch_loop,
            name="KanbanNotifierWatcher",
            daemon=True,
        )
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        """Signal the watcher thread to stop."""
        self._stop_event.set()

    def join(self, timeout: Optional[float] = None) -> None:
        """Wait for the watcher thread to exit.

        Args:
            timeout: Optional timeout in seconds
        """
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _watch_loop(self) -> None:
        """Main watcher loop."""
        while not self._stop_event.is_set():
            try:
                self._check_kanban_updates()
            except Exception as e:
                logger.debug("Kanban notifier watcher tick error: %s", e)

            self._stop_event.wait(timeout=60)

    def _check_kanban_updates(self) -> None:
        """Check for kanban board updates and notify."""
        # Implementation would poll kanban board for changes
        # and send notifications to relevant channels
        pass


class PlatformReconnectWatcher:
    """Background watcher for platform reconnection with exponential backoff."""

    def __init__(self, gateway_runner):
        """Initialize platform reconnect watcher.

        Args:
            gateway_runner: GatewayRunner instance
        """
        self.runner = gateway_runner

    async def watch(self) -> None:
        """Background task that periodically retries connecting failed platforms.

        Uses exponential backoff: 30s -> 60s -> 120s -> 240s -> 300s (cap).
        Retryable failures (network/DNS blips) keep retrying at the backoff
        cap indefinitely - they self-heal once connectivity returns, so a
        transient outage never requires manual intervention. Non-retryable
        failures (bad auth, etc.) drop out of the queue immediately. The
        circuit breaker (_pause_failed_platform / /platform pause)
        remains available for manual operator control via /platform list
        and /platform resume <name>, but is no longer triggered
        automatically - auto-pausing a recovered platform was the cause of
        bots silently staying dead after a transient DNS failure.
        """
        _BACKOFF_CAP = 300  # 5 minutes max between retries

        await asyncio.sleep(10)  # initial delay - let startup finish
        while self.runner._running:
            if not self.runner._failed_platforms:
                # Nothing to reconnect - sleep and check again
                for _ in range(30):
                    if not self.runner._running:
                        return
                    await asyncio.sleep(1)
                continue

            now = time.monotonic()
            for platform in list(self.runner._failed_platforms.keys()):
                if not self.runner._running:
                    return
                info = self.runner._failed_platforms[platform]
                # Skip paused platforms entirely - they need explicit
                # /platform resume to come back.
                if info.get("paused"):
                    continue
                if now < info["next_retry"]:
                    continue  # not time yet

                platform_config = info["config"]
                attempt = info["attempts"] + 1
                logger.info(
                    "Reconnecting %s (attempt %d)...",
                    platform.value, attempt,
                )

                adapter = None
                try:
                    adapter = self.runner._create_adapter(platform, platform_config)
                    if not adapter:
                        logger.warning(
                            "Reconnect %s: adapter creation returned None, removing from retry queue",
                            platform.value,
                        )
                        del self.runner._failed_platforms[platform]
                        continue

                    adapter.set_message_handler(self.runner._handle_message)
                    adapter.set_fatal_error_handler(self.runner._handle_adapter_fatal_error)
                    adapter.set_session_store(self.runner.session_store)
                    adapter.set_busy_session_handler(self.runner._handle_active_session_busy_message)
                    adapter.set_topic_recovery_fn(self.runner._recover_telegram_topic_thread_id)
                    adapter._busy_text_mode = self.runner._busy_text_mode

                    success = await self.runner._connect_adapter_with_timeout(adapter, platform)
                    if success:
                        self.runner.adapters[platform] = adapter
                        self.runner._sync_voice_mode_state_to_adapter(adapter)
                        self.runner.delivery_router.adapters = self.runner.adapters
                        del self.runner._failed_platforms[platform]
                        self.runner._update_platform_runtime_status(
                            platform.value,
                            platform_state="connected",
                            error_code=None,
                            error_message=None,
                        )
                        logger.info("✓ %s reconnected successfully", platform.value)

                        # Rebuild channel directory with the new adapter
                        try:
                            from gateway.channel_directory import build_channel_directory
                            await build_channel_directory(self.runner.adapters)
                        except Exception:
                            pass
                    # Check if the failure is non-retryable
                    elif adapter.has_fatal_error and not adapter.fatal_error_retryable:
                        self.runner._update_platform_runtime_status(
                            platform.value,
                            platform_state="fatal",
                            error_code=adapter.fatal_error_code,
                            error_message=adapter.fatal_error_message,
                        )
                        logger.warning(
                            "Reconnect %s: non-retryable error (%s), removing from retry queue",
                            platform.value, adapter.fatal_error_message,
                        )
                        # The adapter is about to be dropped from the queue
                        # without ever being installed on self.adapters, so
                        # nothing else will call disconnect() on it. We must
                        # dispose it here, otherwise the resource owners it
                        # constructed in __init__ (ResponseStore for
                        # APIServerAdapter, etc.) leak 2 fds each. The
                        # gateway hits the 2560-fd limit after ~12h of
                        # failed reconnects at the 300s backoff cap (#37011).
                        await _dispose_unused_adapter(adapter)
                        del self.runner._failed_platforms[platform]
                    else:
                        self.runner._update_platform_runtime_status(
                            platform.value,
                            platform_state="retrying",
                            error_code=adapter.fatal_error_code,
                            error_message=adapter.fatal_error_message or "failed to reconnect",
                        )
                        backoff = min(30 * (2 ** (attempt - 1)), _BACKOFF_CAP)
                        info["attempts"] = attempt
                        info["next_retry"] = time.monotonic() + backoff
                        logger.info(
                            "Reconnect %s failed, next retry in %ds",
                            platform.value, backoff,
                        )
                        # Same fd-leak concern as the non-retryable branch
                        # above: the adapter failed to connect and is being
                        # thrown away. Without an explicit dispose call, the
                        # resources it opened in __init__ stay open until
                        # the next GC pass - and aiohttp/SQLite handles
                        # don't get GC'd promptly, so 2 fds/retry leak at
                        # 300s backoff cap = ~12 fds/hour (#37011).
                        await _dispose_unused_adapter(adapter)
                        # Retryable failures (network/DNS blips) keep retrying
                        # at the backoff cap indefinitely - they self-heal once
                        # connectivity returns. We do NOT auto-pause them: a
                        # transient outage must never require manual `/platform
                        # resume` to recover. Non-retryable failures (bad auth,
                        # etc.) already drop out of the queue via the
                        # `not fatal_error_retryable` branch above, so anything
                        # reaching here is by definition retryable.
                except Exception as e:
                    if adapter is not None:
                        # An exception escaping the connect call path
                        # (DNS timeout, aiohttp server.start() crash, etc.)
                        # leaves the adapter in the same unowned state as
                        # the two branches above. Dispose so __init__
                        # resources don't accumulate while the watcher
                        # keeps retrying.
                        await _dispose_unused_adapter(adapter)
                    self.runner._update_platform_runtime_status(
                        platform.value,
                        platform_state="retrying",
                        error_code=None,
                        error_message=str(e),
                    )
                    backoff = min(30 * (2 ** (attempt - 1)), _BACKOFF_CAP)
                    info["attempts"] = attempt
                    info["next_retry"] = time.monotonic() + backoff
                    logger.warning(
                        "Reconnect %s error: %s, next retry in %ds",
                        platform.value, e, backoff,
                    )
                    # A raised exception during reconnect (connect timeout, DNS
                    # resolution failure, etc.) is inherently transient - keep
                    # retrying at the backoff cap rather than auto-pausing.

            # Check every 10 seconds for platforms that need reconnection
            for _ in range(10):
                if not self.runner._running:
                    return
                await asyncio.sleep(1)


async def _dispose_unused_adapter(adapter: "BasePlatformAdapter | None") -> None:
    """Best-effort dispose for an adapter that never made it onto self.adapters.

    The reconnect watcher constructs a fresh adapter on every retry attempt.
    When the connect call fails - for any of the three reasons (non-retryable
    error, retryable error, exception during connect) - the adapter is dropped
    without ever being installed, so nothing else will call its disconnect().
    Any resources the adapter opened in __init__ (e.g. APIServerAdapter opens
    a SQLite ResponseStore that holds 2 fds - the db file and its WAL sidecar)
    stay open until garbage collection sweeps the unreachable object, which
    Python's cyclic GC does not do promptly for asyncio-bound objects with
    native handles. The cumulative leak is 2 fds × every retry at the 300s
    backoff cap ≈ 12 fds/hour, and the default 2560-fd ulimit is exhausted in
    ~12h of continuous failure, after which every open() call on the gateway
    raises OSError: [Errno 24] Too many open files and the gateway becomes a
    zombie (#37011).

    This helper centralises the dispose-with-suppression so the three failure
    paths in the reconnect watcher can all call it without each one having to
    know that disconnect() may itself raise on a half-constructed adapter.

    adapter may be None: the reconnect watcher initialises adapter = None
    before the try so the except Exception arm can dispose a half-constructed
    object, and also early-returns here when _create_adapter() returned None.
    """
    if adapter is None:
        return
    try:
        await adapter.disconnect()
    except Exception:
        # Half-constructed adapters (e.g. APIServerAdapter that
        # crashed during aiohttp app setup) can raise from
        # disconnect() on objects that never finished initializing.
        # We must not let that escape and abort the watcher loop.
        #
        # On Python 3.8+, asyncio.CancelledError inherits from
        # BaseException (not Exception), so this except Exception
        # does not swallow task cancellation. We don't re-raise
        # explicitly because the watcher loop intentionally treats
        # dispose failures as best-effort: a failed disconnect call
        # should not take down the reconnect watcher that itself
        # is what's keeping the gateway alive during a partial outage.
        logger.debug(
            "Adapter dispose raised on unowned adapter %r",
            getattr(adapter, "name", type(adapter).__name__),
            exc_info=True,
        )


class ProcessWatcher:
    """Background watcher for long-running process executions."""

    def __init__(self, gateway_runner):
        """Initialize process watcher.

        Args:
            gateway_runner: GatewayRunner instance
        """
        self.runner = gateway_runner
        self._stop_event = asyncio.Event()
        self._task = None

    async def start(self, loop: asyncio.AbstractEventLoop) -> asyncio.Task:
        """Start the process watcher task.

        Args:
            loop: asyncio event loop

        Returns:
            The started task object
        """
        self._stop_event.clear()
        self._task = loop.create_task(self._watch_loop())
        return self._task

    async def stop(self) -> None:
        """Signal the watcher task to stop."""
        self._stop_event.set()
        if self._task:
            await self._task

    async def _watch_loop(self) -> None:
        """Main watcher loop."""
        while not self._stop_event.is_set():
            try:
                self._check_running_processes()
            except Exception as e:
                logger.debug("Process watcher tick error: %s", e)

            await asyncio.sleep(30)

    async def _check_running_processes(self) -> None:
        """Check for and update running process statuses."""
        # Implementation would check process registry
        # and update statuses for long-running processes
        pass



async def run_process_watcher(
    runner,  # GatewayRunner instance
    watcher: dict,
) -> None:
    """
    Periodically check a background process and push updates to the user.

    Runs as an asyncio task. Stays silent when nothing changed.
    Auto-removes when the process exits or is killed.

    Notification mode (from ``display.background_process_notifications``):
      - ``all``    — running-output updates + final message
      - ``result`` — final completion message only
      - ``error``  — final message only when exit code != 0
      - ``off``    — no messages at all
    """
    from tools.process_registry import process_registry
    from gateway.platforms.base import MessageEvent, MessageType

    session_id = watcher["session_id"]
    interval = watcher["check_interval"]
    session_key = watcher.get("session_key", "")
    platform_name = watcher.get("platform", "")
    chat_id = watcher.get("chat_id", "")
    thread_id = watcher.get("thread_id", "")
    user_id = watcher.get("user_id", "")
    user_name = watcher.get("user_name", "")
    message_id = str(watcher.get("message_id") or "").strip() or None
    agent_notify = watcher.get("notify_on_complete", False)
    notify_mode = runner._load_background_notifications_mode()

    logger.debug("Process watcher started: %s (every %ss, notify=%s, agent_notify=%s)",
                  session_id, interval, notify_mode, agent_notify)

    if notify_mode == "off" and not agent_notify:
        # Still wait for the process to exit so we can log it, but don't
        # push any messages to the user.
        while True:
            await asyncio.sleep(interval)
            session = process_registry.get(session_id)
            if session is None or session.exited:
                break
        logger.debug("Process watcher ended (silent): %s", session_id)
        return

    last_output_len = 0
    while True:
        await asyncio.sleep(interval)

        session = process_registry.get(session_id)
        if session is None:
            break

        current_output_len = len(session.output_buffer)
        has_new_output = current_output_len > last_output_len
        last_output_len = current_output_len

        if session.exited:
            # --- Agent-triggered completion: inject synthetic message ---
            # Skip if the agent already consumed the result via wait/poll/log
            from tools.process_registry import process_registry as _pr_check
            if agent_notify and not _pr_check.is_completion_consumed(session_id):
                from tools.ansi_strip import strip_ansi
                _raw = strip_ansi(session.output_buffer) if session.output_buffer else ""
                # Truncate at line boundaries so notifications never start
                # mid-line (fixes #23284). Keep the last ~2000 chars but
                # snap to the nearest preceding newline, then prepend a
                # truncation marker when output was cut.
                _LIMIT = 2000
                if len(_raw) > _LIMIT:
                    _tail = _raw[-_LIMIT:]
                    _nl = _tail.find("\n")
                    _tail = _tail[_nl + 1:] if _nl != -1 else _tail
                    _out = f"[… output truncated — showing last {len(_tail)} chars]\n{_tail}"
                else:
                    _out = _raw
                synth_text = (
                    f"[IMPORTANT: Background process {session_id} completed "
                    f"(exit code {session.exit_code}).\n"
                    f"Command: {session.command}\n"
                    f"Output:\n{_out}]"
                )
                source = runner._build_process_event_source({
                    "session_id": session_id,
                    "session_key": session_key,
                    "platform": platform_name,
                    "chat_id": chat_id,
                    "thread_id": thread_id,
                    "user_id": user_id,
                    "user_name": user_name,
                })
                if not source:
                    logger.warning(
                        "Dropping completion notification with no routing metadata for process %s",
                        session_id,
                    )
                    break

                adapter = None
                for p, a in runner.adapters.items():
                    if p == source.platform:
                        adapter = a
                        break
                if adapter and source.chat_id:
                    try:
                        synth_event = MessageEvent(
                            text=synth_text,
                            message_type=MessageType.TEXT,
                            source=source,
                            internal=True,
                            message_id=message_id,
                        )
                        logger.info(
                            "Process %s finished — injecting agent notification for session %s chat=%s thread=%s",
                            session_id,
                            session_key,
                            source.chat_id,
                            source.thread_id,
                        )
                        await adapter.handle_message(synth_event)
                    except Exception as e:
                        logger.error("Agent notify injection error: %s", e)
                break

            # --- Normal text-only notification ---
            # Decide whether to notify based on mode
            should_notify = (
                notify_mode in {"all", "result"}
                or (notify_mode == "error" and session.exit_code not in {0, None})
            )
            if should_notify:
                new_output = session.output_buffer[-1000:] if session.output_buffer else ""
                message_text = (
                    f"[Background process {session_id} finished with exit code {session.exit_code}~ "
                    f"Here's the final output:\n{new_output}]"
                )
                adapter = None
                for p, a in runner.adapters.items():
                    if p.value == platform_name:
                        adapter = a
                        break
                if adapter and chat_id:
                    try:
                        send_meta = {"thread_id": thread_id} if thread_id else None
                        await adapter.send(chat_id, message_text, metadata=send_meta)
                    except Exception as e:
                        logger.error("Watcher delivery error: %s", e)
                break

        elif has_new_output and notify_mode == "all" and not agent_notify:
            # New output available -- deliver status update (only in "all" mode)
            # Skip periodic updates for agent_notify watchers (they only care about completion)
            new_output = session.output_buffer[-500:] if session.output_buffer else ""
            message_text = (
                f"[Background process {session_id} is still running~ "
                f"New output:\n{new_output}]"
            )
            adapter = None
            for p, a in runner.adapters.items():
                if p.value == platform_name:
                    adapter = a
                    break
            if adapter and chat_id:
                try:
                    send_meta = {"thread_id": thread_id} if thread_id else None
                    await adapter.send(chat_id, message_text, metadata=send_meta)
                except Exception as e:
                    logger.error("Watcher delivery error: %s", e)

    logger.debug("Process watcher ended: %s", session_id)

async def kanban_dispatcher_watcher(
    runner,  # GatewayRunner instance
) -> None:
    """Embedded kanban dispatcher — one tick every `dispatch_interval_seconds`.

    Gated by `kanban.dispatch_in_gateway` in config.yaml (default True).
    When true, the gateway hosts the single dispatcher for this profile:
    no separate `hermes kanban daemon` process needed. When false, the
    loop exits immediately and an external daemon is expected.

    Each tick calls :func:`kanban_db.dispatch_once` inside
    ``asyncio.to_thread`` so the SQLite WAL lock never blocks the
    event loop. Failures in one tick don't stop subsequent ticks —
    same pattern as `_kanban_notifier_watcher`.

    Shutdown: the loop checks ``runner._running`` between ticks; gateway
    stop() flips it to False and cancels pending tasks, and the
    in-flight ``to_thread`` returns on its own after the current
    ``dispatch_once`` call finishes (typically <1ms on an idle board).
    """
    # Read config once at boot. If the user flips the flag later, they
    # restart the gateway; same pattern as every other background
    # watcher here. Honours HERMES_KANBAN_DISPATCH_IN_GATEWAY env var
    # as an escape hatch (false-y value disables without editing YAML).
    try:
        from hermes_cli.config import load_config as _load_config
    except Exception:
        logger.warning("kanban dispatcher: config loader unavailable; disabled")
        return
    env_override = os.environ.get("HERMES_KANBAN_DISPATCH_IN_GATEWAY", "").strip().lower()
    if env_override in {"0", "false", "no", "off"}:
        logger.info("kanban dispatcher: disabled via HERMES_KANBAN_DISPATCH_IN_GATEWAY env")
        return

    try:
        cfg = _load_config()
    except Exception as exc:
        logger.warning("kanban dispatcher: cannot load config (%s); disabled", exc)
        return
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    if not kanban_cfg.get("dispatch_in_gateway", True):
        logger.info(
            "kanban dispatcher: disabled via config kanban.dispatch_in_gateway=false"
        )
        return

    try:
        from hermes_cli import kanban_db as _kb
    except Exception:
        logger.warning("kanban dispatcher: kanban_db not importable; dispatcher disabled")
        return

    interval = float(kanban_cfg.get("dispatch_interval_seconds", 60) or 60)
    interval = max(interval, 1.0)  # sanity floor — tighter than this is a footgun

    # Read max_spawn config to limit concurrent kanban tasks
    max_spawn = kanban_cfg.get("max_spawn", None)
    if max_spawn is not None:
        logger.info(f"kanban dispatcher: max_spawn={max_spawn}")

    # Cap the number of simultaneously running tasks so slow workers
    # (local LLMs, resource-constrained hosts) don't pile up and time
    # out. When set, the dispatcher skips spawning when the board
    # already has this many tasks in 'running' status.
    raw_max_in_progress = kanban_cfg.get("max_in_progress", None)
    max_in_progress = None
    if raw_max_in_progress is not None:
        try:
            max_in_progress = int(raw_max_in_progress)
        except (TypeError, ValueError):
            logger.warning(
                "kanban dispatcher: invalid kanban.max_in_progress=%r; ignoring",
                raw_max_in_progress,
            )
            max_in_progress = None
        else:
            if max_in_progress < 1:
                logger.warning(
                    "kanban dispatcher: kanban.max_in_progress=%r is below 1; ignoring",
                    raw_max_in_progress,
                )
                max_in_progress = None
            else:
                logger.info(f"kanban dispatcher: max_in_progress={max_in_progress}")

    raw_failure_limit = kanban_cfg.get("failure_limit", _kb.DEFAULT_FAILURE_LIMIT)
    try:
        failure_limit = int(raw_failure_limit)
    except (TypeError, ValueError):
        logger.warning(
            "kanban dispatcher: invalid kanban.failure_limit=%r; using default %d",
            raw_failure_limit,
            _kb.DEFAULT_FAILURE_LIMIT,
        )
        failure_limit = _kb.DEFAULT_FAILURE_LIMIT
    if failure_limit < 1:
        logger.warning(
            "kanban dispatcher: kanban.failure_limit=%r is below 1; using default %d",
            raw_failure_limit,
            _kb.DEFAULT_FAILURE_LIMIT,
        )
        failure_limit = _kb.DEFAULT_FAILURE_LIMIT

    # Read stale_timeout_seconds — 0 disables stale detection.
    raw_stale = kanban_cfg.get("dispatch_stale_timeout_seconds", 0)
    try:
        stale_timeout_seconds = int(raw_stale or 0)
    except (TypeError, ValueError):
        logger.warning(
            "kanban dispatcher: invalid kanban.dispatch_stale_timeout_seconds=%r; "
            "disabling stale detection",
            raw_stale,
        )
        stale_timeout_seconds = 0

    # Read kanban.default_assignee — fallback profile for tasks
    # created without an explicit assignee (e.g. via the dashboard).
    # When set, the dispatcher applies it to unassigned ready tasks
    # instead of skipping them indefinitely (#27145). Empty string
    # (the schema default) means "no fallback, keep skipping" —
    # backward-compatible with existing installs.
    default_assignee = (kanban_cfg.get("default_assignee") or "").strip() or None
    if default_assignee:
        logger.info(
            "kanban dispatcher: default_assignee=%r (unassigned ready tasks "
            "will route to this profile)",
            default_assignee,
        )

    # Read kanban.max_in_progress_per_profile — per-profile concurrency
    # cap (#21582). When set, no single profile gets more than N
    # workers running at once, even if the global max_in_progress
    # would allow it. Prevents one profile's local model / API quota
    # / browser pool from being overwhelmed by a fan-out.
    raw_per_profile = kanban_cfg.get("max_in_progress_per_profile", None)
    max_in_progress_per_profile = None
    if raw_per_profile is not None:
        try:
            max_in_progress_per_profile = int(raw_per_profile)
        except (TypeError, ValueError):
            logger.warning(
                "kanban dispatcher: invalid kanban.max_in_progress_per_profile=%r; ignoring",
                raw_per_profile,
            )
            max_in_progress_per_profile = None
        else:
            if max_in_progress_per_profile < 1:
                logger.warning(
                    "kanban dispatcher: kanban.max_in_progress_per_profile=%r is below 1; ignoring",
                    raw_per_profile,
                )
                max_in_progress_per_profile = None
            else:
                logger.info(
                    "kanban dispatcher: max_in_progress_per_profile=%d",
                    max_in_progress_per_profile,
                )

    # Initial delay so the gateway finishes wiring adapters before the
    # dispatcher spawns workers (those workers may hit gateway notify
    # subscriptions etc.). Matches the notifier watcher's delay.
    await asyncio.sleep(5)

    # Health telemetry mirrored from `_cmd_daemon`: warn when ready
    # queue is non-empty but spawns are 0 for N consecutive ticks —
    # usually means broken PATH, missing venv, or credential loss.
    HEALTH_WINDOW = 6
    bad_ticks = 0
    last_warn_at = 0
    # Avoid hot-looping corrupt-looking board DBs, but do not suppress
    # same-fingerprint retries forever: transient WAL/open races can
    # surface as "database disk image is malformed" for one tick.
    CORRUPT_BOARD_RETRY_AFTER_SECONDS = 300
    disabled_corrupt_boards: dict[
        str, tuple[tuple[str, int | None, int | None], float]
    ] = {}

    def _board_db_fingerprint(slug: str) -> tuple[str, int | None, int | None]:
        path = _kb.kanban_db_path(slug)
        try:
            resolved = str(path.expanduser().resolve())
        except Exception:
            resolved = str(path)
        try:
            stat = path.stat()
        except OSError:
            return (resolved, None, None)
        return (resolved, stat.st_mtime_ns, stat.st_size)

    def _is_corrupt_board_db_error(exc: Exception) -> bool:
        corrupt_guard_error = getattr(_kb, "KanbanDbCorruptError", None)
        if corrupt_guard_error is not None and isinstance(exc, corrupt_guard_error):
            return True
        if not isinstance(exc, sqlite3.DatabaseError):
            return False
        msg = str(exc).lower()
        return (
            "file is not a database" in msg
            or "database disk image is malformed" in msg
        )

    def _tick_once_for_board(slug: str) -> "Optional[object]":
        """Run one dispatch_once for a specific board.

        Runs in a worker thread via `asyncio.to_thread`. `board=slug`
        is passed through `dispatch_once` so `resolve_workspace` and
        `_default_spawn` see the right paths. The per-board DB is
        opened explicitly so concurrent boards never share a
        connection handle or accidentally claim across each other.
        """
        conn = None
        fingerprint = _board_db_fingerprint(slug)
        disabled_entry = disabled_corrupt_boards.get(slug)
        if disabled_entry is not None:
            disabled_fingerprint, disabled_at = disabled_entry
            age = time.monotonic() - disabled_at
            if (
                disabled_fingerprint == fingerprint
                and age < CORRUPT_BOARD_RETRY_AFTER_SECONDS
            ):
                return None
            if disabled_fingerprint == fingerprint:
                logger.info(
                    "kanban dispatcher: board %s database fingerprint unchanged "
                    "after %.0fs quarantine; retrying dispatch",
                    slug,
                    age,
                )
            else:
                logger.info(
                    "kanban dispatcher: board %s database changed; retrying dispatch",
                    slug,
                )
            disabled_corrupt_boards.pop(slug, None)
        try:
            conn = _kb.connect(board=slug)
            # `connect()` runs the schema + idempotent migration on
            # first open per process; the previous explicit
            # `init_db()` call here busted the per-process cache and
            # re-ran the migration on a second connection, racing
            # the first. See the matching comment in
            # `_kanban_notifier_watcher` and issue #21378.
            return _kb.dispatch_once(
                conn,
                board=slug,
                max_spawn=max_spawn,
                max_in_progress=max_in_progress,
                failure_limit=failure_limit,
                stale_timeout_seconds=stale_timeout_seconds,
                default_assignee=default_assignee,
                max_in_progress_per_profile=max_in_progress_per_profile,
            )
        except sqlite3.DatabaseError as exc:
            if _is_corrupt_board_db_error(exc):
                disabled_corrupt_boards[slug] = (fingerprint, time.monotonic())
                logger.error(
                    "kanban dispatcher: board %s database %s is not a valid "
                    "SQLite database; pausing dispatch for this board until "
                    "the file changes, the gateway restarts, or the "
                    "quarantine timer expires. Move or restore the file, "
                    "then run `hermes kanban init` if you need a fresh board.",
                    slug,
                    fingerprint[0],
                )
                return None
            logger.exception("kanban dispatcher: tick failed on board %s", slug)
            return None
        except Exception as exc:
            if _is_corrupt_board_db_error(exc):
                disabled_corrupt_boards[slug] = (fingerprint, time.monotonic())
                logger.error(
                    "kanban dispatcher: board %s database %s is not a valid "
                    "SQLite database; pausing dispatch for this board until "
                    "the file changes, the gateway restarts, or the "
                    "quarantine timer expires. Move or restore the file, "
                    "then run `hermes kanban init` if you need a fresh board.",
                    slug,
                    fingerprint[0],
                )
                return None
            logger.exception("kanban dispatcher: tick failed on board %s", slug)
            return None
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def _tick_once() -> "list[tuple[str, Optional[object]]]":
        """Run one dispatch_once per board. Returns (slug, result) pairs.

        Enumerating boards on every tick keeps the dispatcher honest
        when users create a new board mid-run: no restart required,
        the next tick picks it up automatically.
        """
        try:
            boards = _kb.list_boards(include_archived=False)
        except Exception:
            boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
        out: list[tuple[str, "Optional[object]"]] = []
        for b in boards:
            slug = b.get("slug") or _kb.DEFAULT_BOARD
            out.append((slug, _tick_once_for_board(slug)))
        return out

    def _ready_nonempty() -> bool:
        """Cheap probe: is there at least one ready+assigned+unclaimed
        task on ANY board whose assignee maps to a real Hermes profile
        (i.e. one the dispatcher would actually spawn for)?

        Tasks assigned to control-plane lanes (e.g. ``orion-cc``,
        ``orion-research``) are pulled by terminals via
        ``claim_task`` directly and never spawnable, so a queue full
        of those is "correctly idle", not "stuck". Filtering them out
        here keeps the stuck-warn fire only on real failures (broken
        PATH, missing venv, credential loss for a real Hermes profile).
        """
        try:
            boards = _kb.list_boards(include_archived=False)
        except Exception:
            boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
        for b in boards:
            slug = b.get("slug") or _kb.DEFAULT_BOARD
            conn = None
            try:
                conn = _kb.connect(board=slug)
                if _kb.has_spawnable_ready(conn):
                    return True
                if _kb.has_spawnable_review(conn):
                    return True
            except Exception:
                continue
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
        return False

    # Auto-decompose: turn fresh triage tasks into ready workgraphs
    # before the dispatcher fans out workers. Gated by
    # ``kanban.auto_decompose`` (default True). Capped by
    # ``kanban.auto_decompose_per_tick`` (default 3) so a bulk-load
    # of triage tasks doesn't burst-spend the aux LLM in one tick;
    # remainder defers to subsequent ticks.
    auto_decompose_enabled = bool(kanban_cfg.get("auto_decompose", True))
    try:
        auto_decompose_per_tick = int(
            kanban_cfg.get("auto_decompose_per_tick", 3) or 3
        )
    except (TypeError, ValueError):
        auto_decompose_per_tick = 3
    if auto_decompose_per_tick < 1:
        auto_decompose_per_tick = 1

    def _auto_decompose_tick() -> int:
        """Run the auto-decomposer for up to N triage tasks across all
        boards. Returns the number of triage tasks that were
        successfully decomposed or specified this tick.
        """
        try:
            from hermes_cli import kanban_decompose as _decomp
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "kanban auto-decompose: import failed (%s); skipping", exc,
            )
            return 0
        try:
            boards = _kb.list_boards(include_archived=False)
        except Exception:
            boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
        attempted = 0
        successes = 0
        for b in boards:
            slug = b.get("slug") or _kb.DEFAULT_BOARD
            if attempted >= auto_decompose_per_tick:
                break
            # Pin this board for the duration of the call — same
            # pattern as the dashboard specify endpoint. The
            # decomposer module connects with no board kwarg and
            # relies on the env var.
            prev_env = os.environ.get("HERMES_KANBAN_BOARD")
            try:
                os.environ["HERMES_KANBAN_BOARD"] = slug
                try:
                    triage_ids = _decomp.list_triage_ids()
                except Exception as exc:
                    logger.debug(
                        "kanban auto-decompose: list_triage_ids failed on board %s (%s)",
                        slug, exc,
                    )
                    triage_ids = []
                for tid in triage_ids:
                    if attempted >= auto_decompose_per_tick:
                        break
                    attempted += 1
                    try:
                        outcome = _decomp.decompose_task(
                            tid, author="auto-decomposer",
                        )
                    except Exception:
                        logger.exception(
                            "kanban auto-decompose: decompose_task crashed on %s",
                            tid,
                        )
                        continue
                    if outcome.ok:
                        successes += 1
                        if outcome.fanout and outcome.child_ids:
                            logger.info(
                                "kanban auto-decompose [%s]: %s → %d children",
                                slug, tid, len(outcome.child_ids),
                            )
                        else:
                            logger.info(
                                "kanban auto-decompose [%s]: %s → single task (no fanout)",
                                slug, tid,
                            )
                    else:
                        # Common no-op reasons (no aux client configured) shouldn't
                        # spam logs every tick. Log at debug.
                        logger.debug(
                            "kanban auto-decompose [%s]: %s skipped: %s",
                            slug, tid, outcome.reason,
                        )
            finally:
                if prev_env is None:
                    os.environ.pop("HERMES_KANBAN_BOARD", None)
                else:
                    os.environ["HERMES_KANBAN_BOARD"] = prev_env
        return successes

    logger.info(
        "kanban dispatcher: embedded in gateway (interval=%.1fs)", interval
    )
    while runner._running:
        try:
            # Reap zombie children before per-board work so a board DB
            # failure cannot block cleanup of unrelated workers.
            pids = await asyncio.to_thread(_kb.reap_worker_zombies)
            if pids:
                logger.info(
                    "kanban dispatcher: reaped %d zombie worker(s), pids=%s",
                    len(pids),
                    pids,
                )
        except Exception:
            logger.exception("kanban dispatcher: zombie reaper failed")

        try:
            if auto_decompose_enabled:
                await asyncio.to_thread(_auto_decompose_tick)
            results = await asyncio.to_thread(_tick_once)
            any_spawned = False
            for slug, res in (results or []):
                if res is not None and getattr(res, "spawned", None):
                    any_spawned = True
                    # Quiet by default — only log when something actually
                    # happened, so an idle gateway stays silent.
                    logger.info(
                        "kanban dispatcher [%s]: spawned=%d reclaimed=%d "
                        "crashed=%d timed_out=%d promoted=%d auto_blocked=%d",
                        slug,
                        len(res.spawned),
                        res.reclaimed,
                        len(res.crashed) if hasattr(res.crashed, "__len__") else 0,
                        len(res.timed_out) if hasattr(res.timed_out, "__len__") else 0,
                        res.promoted,
                        len(res.auto_blocked) if hasattr(res.auto_blocked, "__len__") else 0,
                    )
            # Health telemetry (aggregate across boards)
            ready_pending = await asyncio.to_thread(_ready_nonempty)
            if ready_pending and not any_spawned:
                bad_ticks += 1
            else:
                bad_ticks = 0
            if bad_ticks >= HEALTH_WINDOW:
                now = int(time.time())
                if now - last_warn_at >= 300:
                    logger.warning(
                        "kanban dispatcher stuck: ready queue non-empty for "
                        "%d consecutive ticks but 0 workers spawned. Check "
                        "profile health (venv, PATH, credentials) and "
                        "`hermes kanban list --status ready`.",
                        bad_ticks,
                    )
                    last_warn_at = now
        except asyncio.CancelledError:
            logger.debug("kanban dispatcher: cancelled")
            raise
        except Exception:
            logger.exception("kanban dispatcher: unexpected watcher error")

        # Sleep in 1s slices so shutdown is snappy — otherwise a stop()
        # waits up to `interval` seconds for the current sleep to finish.
        slept = 0.0
        while slept < interval and runner._running:
            await asyncio.sleep(min(1.0, interval - slept))
            slept += 1.0



def _kanban_advance(
    runner, sub: dict, cursor: int, board: Optional[str] = None,
) -> None:
    """Sync helper: advance a subscription's cursor. Runs in to_thread.

    ``board`` scopes the DB connection to the board that owns this
    subscription. Unsub cursors in one board can't touch another's.
    """
    from hermes_cli import kanban_db as _kb
    conn = _kb.connect(board=board)
    try:
        _kb.advance_notify_cursor(
            conn,
            task_id=sub["task_id"],
            platform=sub["platform"],
            chat_id=sub["chat_id"],
            thread_id=sub.get("thread_id") or "",
            new_cursor=cursor,
        )
    finally:
        conn.close()


def _kanban_unsub(runner, sub: dict, board: Optional[str] = None) -> None:
    """Sync helper: remove a kanban notification subscription."""
    from hermes_cli import kanban_db as _kb
    conn = _kb.connect(board=board)
    try:
        _kb.remove_notify_sub(
            conn,
            task_id=sub["task_id"],
            platform=sub["platform"],
            chat_id=sub["chat_id"],
            thread_id=sub.get("thread_id") or "",
        )
    finally:
        conn.close()


def _kanban_rewind(
    runner,
    sub: dict,
    claimed_cursor: int,
    old_cursor: int,
    board: Optional[str] = None,
) -> None:
    """Sync helper: undo a claimed notification cursor after send failure."""
    from hermes_cli import kanban_db as _kb
    conn = _kb.connect(board=board)
    try:
        _kb.rewind_notify_cursor(
            conn,
            task_id=sub["task_id"],
            platform=sub["platform"],
            chat_id=sub["chat_id"],
            thread_id=sub.get("thread_id") or "",
            claimed_cursor=claimed_cursor,
            old_cursor=old_cursor,
        )
    finally:
        conn.close()


async def _deliver_kanban_artifacts(
    runner,
    *,
    adapter,
    chat_id: str,
    metadata: dict,
    event_payload: Optional[dict],
    task,
) -> None:
    """Upload artifact files referenced by a completed kanban task.

    Workers passing ``kanban_complete(artifacts=[...])`` ship absolute
    file paths through the completion event so downstream humans get
    the deliverable as a native upload instead of a path printed in
    chat.

    Sources scanned, in priority order:
      1. ``event_payload['artifacts']`` (explicit list — preferred)
      2. ``event_payload['summary']`` (truncated first line)
      3. ``task.result`` (legacy fallback)

    Files are deduplicated, missing files are silently skipped (the
    path may have been mentioned for reference only), and delivery
    errors are logged but do not break the notifier loop.
    """
    from gateway import kanban_helpers
    return await kanban_helpers.deliver_kanban_artifacts(
        adapter=adapter,
        chat_id=chat_id,
        metadata=metadata,
        event_payload=event_payload,
        task=task,
    )


async def kanban_notifier_watcher(
    runner,  # GatewayRunner instance
    interval: float = 5.0,
) -> None:
    """Poll ``kanban_notify_subs`` and deliver terminal events to users.

    For each subscription row, fetches ``task_events`` newer than the
    stored cursor with kind in the terminal set (``completed``,
    ``blocked``, ``gave_up``, ``crashed``, ``timed_out``). Sends one
    message per new event to ``(platform, chat_id, thread_id)``,
    then advances the cursor. When a task reaches a terminal state
    (``completed`` / ``archived``), the subscription is removed.

    Runs in the gateway event loop; all SQLite work is pushed to a
    thread via ``asyncio.to_thread`` so the loop never blocks on the
    WAL lock. Failures in one tick don't stop subsequent ticks.

    **Multi-board:** iterates every board discovered on disk per
    tick. Subscriptions live inside each board's own DB and cannot
    cross boards, so delivery semantics are unchanged — this is
    purely a fan-out of the single-DB poll.
    """
    # Gate: only the dispatch-owning gateway opens kanban DBs for notifier polling.
    # Non-dispatch gateways have no subscriptions to deliver — all kanban state lives
    # in the dispatch owner's per-board DBs. This prevents N-gateway -shm contention.
    # TODO: gate per-board when per-board dispatcher_owner tracking lands.
    try:
        from hermes_cli.config import load_config as _load_config
    except Exception:
        logger.warning("kanban notifier: config loader unavailable; disabled")
        return
    env_override = os.environ.get("HERMES_KANBAN_DISPATCH_IN_GATEWAY", "").strip().lower()
    if env_override in {"0", "false", "no", "off"}:
        logger.info("kanban notifier: disabled via HERMES_KANBAN_DISPATCH_IN_GATEWAY env")
        return
    try:
        cfg = _load_config()
    except Exception as exc:
        logger.warning("kanban notifier: cannot load config (%s); disabled", exc)
        return
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    if not kanban_cfg.get("dispatch_in_gateway", True):
        logger.info(
            "kanban notifier: disabled via config kanban.dispatch_in_gateway=false"
        )
        return
    from gateway.config import Platform as _Platform
    try:
        from hermes_cli import kanban_db as _kb
    except Exception:
        logger.warning("kanban notifier: kanban_db not importable; notifier disabled")
        return

    TERMINAL_KINDS = ("completed", "blocked", "gave_up", "crashed", "timed_out")
    # Subscriptions are removed only when the task reaches a truly final
    # status (done / archived). We used to also unsub on any terminal
    # event kind (gave_up / crashed / timed_out / blocked), but that
    # silently dropped the user out of the loop whenever the dispatcher
    # respawned the task: a worker that crashes, gets reclaimed, runs
    # again, and crashes a second time would only notify on the first
    # crash because the subscription was deleted after the first event.
    # Same shape as the reblock-after-unblock cycle that PR #22941
    # fixed for `blocked`. Keeping the subscription alive until the
    # task is genuinely done lets the cursor (advanced atomically by
    # claim_unseen_events_for_sub) handle dedup, and any retry-loop
    # event reaches the user.
    # Per-subscription send-failure counter. Adapter.send raising
    # means the chat is dead (deleted, bot kicked, etc.) — after N
    # consecutive send failures the sub is dropped so we don't spin
    # against a dead chat every 5 seconds forever.
    MAX_SEND_FAILURES = 3
    sub_fail_counts: dict[tuple, int] = getattr(
        runner, "_kanban_sub_fail_counts", {}
    )
    runner._kanban_sub_fail_counts = sub_fail_counts
    notifier_profile = getattr(runner, "_kanban_notifier_profile", None)
    if not notifier_profile:
        notifier_profile = runner._active_profile_name()
        runner._kanban_notifier_profile = notifier_profile

    # Initial delay so the gateway can finish wiring adapters.
    await asyncio.sleep(5)

    while runner._running:
        try:
            def _collect():
                deliveries: list[dict] = []
                active_platforms = {
                    getattr(platform, "value", str(platform)).lower()
                    for platform in runner.adapters.keys()
                }
                if not active_platforms:
                    logger.debug("kanban notifier: no connected adapters; skipping tick")
                    return deliveries

                # Enumerate every board on disk, but poll each resolved DB
                # path once. Multiple slugs can point at the same DB when
                # HERMES_KANBAN_DB pins the board path; without this guard
                # one gateway could collect the same subscription/event
                # more than once before advancing the cursor.
                try:
                    boards = _kb.list_boards(include_archived=False)
                except Exception:
                    boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
                seen_db_paths: set[str] = set()
                for board_meta in boards:
                    slug = board_meta.get("slug") or _kb.DEFAULT_BOARD
                    db_path = board_meta.get("db_path")
                    try:
                        resolved_db_path = str(Path(db_path).expanduser().resolve()) if db_path else str(_kb.kanban_db_path(slug).resolve())
                    except Exception:
                        resolved_db_path = f"slug:{slug}"
                    if resolved_db_path in seen_db_paths:
                        logger.debug(
                            "kanban notifier: skipping duplicate board slug %s for DB %s",
                            slug, resolved_db_path,
                        )
                        continue
                    seen_db_paths.add(resolved_db_path)
                    try:
                        conn = _kb.connect(board=slug)
                    except Exception as exc:
                        logger.debug("kanban notifier: cannot open board %s: %s", slug, exc)
                        continue
                    try:
                        # `connect()` runs the schema + idempotent migration
                        # on first open per process, so an explicit
                        # `init_db()` here would be redundant. Worse:
                        # `init_db()` deliberately busts the per-process
                        # cache and re-runs the migration on a *second*
                        # connection, which races the first and used to
                        # log a benign but noisy `duplicate column name`
                        # traceback (and intermittent "database is locked"
                        # — issue #21378) on every gateway start against
                        # a legacy DB. `_add_column_if_missing` now
                        # tolerates that race, but we still skip the
                        # redundant call to avoid the wasted work.
                        subs = _kb.list_notify_subs(conn)
                        if not subs:
                            logger.debug("kanban notifier: board %s has no subscriptions", slug)
                        for sub in subs:
                            owner_profile = sub.get("notifier_profile") or None
                            if owner_profile and owner_profile != notifier_profile:
                                logger.debug(
                                    "kanban notifier: subscription for %s owned by profile %s; current profile %s skipping",
                                    sub.get("task_id"), owner_profile, notifier_profile,
                                )
                                continue
                            platform = (sub.get("platform") or "").lower()
                            if platform not in active_platforms:
                                logger.debug(
                                    "kanban notifier: subscription for %s on %s skipped; adapter not connected",
                                    sub.get("task_id"), platform or "<missing>",
                                )
                                continue
                            old_cursor, cursor, events = _kb.claim_unseen_events_for_sub(
                                conn,
                                task_id=sub["task_id"],
                                platform=sub["platform"],
                                chat_id=sub["chat_id"],
                                thread_id=sub.get("thread_id") or "",
                                kinds=TERMINAL_KINDS,
                            )
                            if not events:
                                continue
                            task = _kb.get_task(conn, sub["task_id"])
                            logger.debug(
                                "kanban notifier: claimed %d event(s) for %s on board %s cursor %s→%s",
                                len(events), sub["task_id"], slug, old_cursor, cursor,
                            )
                            deliveries.append({
                                "sub": sub,
                                "old_cursor": old_cursor,
                                "cursor": cursor,
                                "events": events,
                                "task": task,
                                "board": slug,
                            })
                    finally:
                        conn.close()
                return deliveries

            deliveries = await asyncio.to_thread(_collect)
            for d in deliveries:
                sub = d["sub"]
                task = d["task"]
                board_slug = d.get("board")
                platform_str = (sub["platform"] or "").lower()
                try:
                    plat = _Platform(platform_str)
                except ValueError:
                    # Unknown platform string; skip and advance cursor so
                    # we don't replay forever.
                    await asyncio.to_thread(
                        _kanban_advance, runner, sub, d["cursor"], board_slug,
                    )
                    continue
                adapter = runner.adapters.get(plat)
                if adapter is None:
                    logger.debug(
                        "kanban notifier: adapter %s disconnected before delivery for %s; rewinding claim",
                        platform_str, sub["task_id"],
                    )
                    await asyncio.to_thread(
                        _kanban_rewind,
                        runner,
                        sub,
                        d["cursor"],
                        d.get("old_cursor", 0),
                        board_slug,
                    )
                    continue
                title = (task.title if task else sub["task_id"])[:120]
                for ev in d["events"]:
                    kind = ev.kind
                    # Identity prefix: attribute terminal pings to the
                    # worker that did the work. Makes fleets (where one
                    # chat subscribes to many tasks) legible at a glance.
                    who = (task.assignee if task and task.assignee else None)
                    tag = f"@{who} " if who else ""
                    if kind == "completed":
                        # Prefer the run's summary (the worker's
                        # intentional human-facing handoff, carried
                        # in the event payload), then fall back to
                        # task.result for legacy rows written before
                        # runs shipped.
                        handoff = ""
                        payload_summary = None
                        if ev.payload and ev.payload.get("summary"):
                            payload_summary = str(ev.payload["summary"])
                        if payload_summary:
                            h = payload_summary.strip().splitlines()[0][:200]
                            handoff = f"\n{h}"
                        elif task and task.result:
                            r = task.result.strip().splitlines()[0][:160]
                            handoff = f"\n{r}"
                        msg = (
                            f"✔ {tag}Kanban {sub['task_id']} done"
                            f" — {title}{handoff}"
                        )
                    elif kind == "blocked":
                        reason = ""
                        if ev.payload and ev.payload.get("reason"):
                            reason = f": {str(ev.payload['reason'])[:160]}"
                        msg = f"⏸ {tag}Kanban {sub['task_id']} blocked{reason}"
                    elif kind == "gave_up":
                        err = ""
                        if ev.payload and ev.payload.get("error"):
                            err = f"\n{str(ev.payload['error'])[:200]}"
                        msg = (
                            f"✖ {tag}Kanban {sub['task_id']} gave up "
                            f"after repeated spawn failures{err}"
                        )
                    elif kind == "crashed":
                        msg = (
                            f"✖ {tag}Kanban {sub['task_id']} worker crashed "
                            f"(pid gone); dispatcher will retry"
                        )
                    elif kind == "timed_out":
                        limit = 0
                        if ev.payload and ev.payload.get("limit_seconds"):
                            limit = int(ev.payload["limit_seconds"])
                        msg = (
                            f"⏱ {tag}Kanban {sub['task_id']} timed out "
                            f"(max_runtime={limit}s); will retry"
                        )
                    else:
                        continue
                    metadata: dict[str, Any] = {}
                    if sub.get("thread_id"):
                        metadata["thread_id"] = sub["thread_id"]
                    sub_key = (
                        sub["task_id"], sub["platform"],
                        sub["chat_id"], sub.get("thread_id") or "",
                    )
                    try:
                        await adapter.send(
                            sub["chat_id"], msg, metadata=metadata,
                        )
                        logger.debug(
                            "kanban notifier: delivered %s event for %s to %s/%s on board %s",
                            kind, sub["task_id"], platform_str, sub["chat_id"], board_slug,
                        )
                        # After delivering the text notification, surface
                        # any artifact paths the worker referenced in
                        # ``kanban_complete(summary=..., artifacts=[...])``
                        # (or the legacy ``result`` field) as native
                        # uploads. ``extract_local_files`` finds bare
                        # absolute paths in the summary;
                        # ``send_document`` / ``send_image_file`` uploads
                        # them. Only fires on the ``completed`` event so
                        # we never spam attachments on retries.
                        if kind == "completed":
                            try:
                                await _deliver_kanban_artifacts(
                                    runner,
                                    adapter=adapter,
                                    chat_id=sub["chat_id"],
                                    metadata=metadata,
                                    event_payload=getattr(ev, "payload", None),
                                    task=task,
                                )
                            except Exception as art_exc:
                                logger.debug(
                                    "kanban notifier: artifact delivery for %s failed: %s",
                                    sub["task_id"], art_exc,
                                )
                        # Reset the failure counter on success.
                        sub_fail_counts.pop(sub_key, None)
                    except Exception as exc:
                        fails = sub_fail_counts.get(sub_key, 0) + 1
                        sub_fail_counts[sub_key] = fails
                        logger.warning(
                            "kanban notifier: send failed for %s on %s "
                            "(attempt %d/%d): %s",
                            sub["task_id"], platform_str, fails,
                            MAX_SEND_FAILURES, exc,
                        )
                        if fails >= MAX_SEND_FAILURES:
                            logger.warning(
                                "kanban notifier: dropping subscription "
                                "%s on %s after %d consecutive send failures",
                                sub["task_id"], platform_str, fails,
                            )
                            await asyncio.to_thread(_kanban_unsub, runner, sub, board_slug)
                            sub_fail_counts.pop(sub_key, None)
                        else:
                            await asyncio.to_thread(
                                _kanban_rewind,
                                runner,
                                sub,
                                d["cursor"],
                                d.get("old_cursor", 0),
                                board_slug,
                            )
                        # Rewind the pre-send claim on transient failure so
                        # a later tick can retry. After too many failures,
                        # dropping the subscription is the terminal action.
                        break
                else:
                    # All events delivered; advance cursor. The cursor
                    # is the dedup mechanism — it prevents re-delivery
                    # of the same event on subsequent ticks.
                    await asyncio.to_thread(
                        _kanban_advance, runner, sub, d["cursor"], board_slug,
                    )
                    # Unsubscribe only when the task has reached a truly
                    # final status (done / archived). For blocked /
                    # gave_up / crashed / timed_out the subscription is
                    # kept alive so the user gets notified again if the
                    # dispatcher respawns the task and it cycles into the
                    # same state. See the longer comment on TERMINAL_KINDS
                    # above for the failure mode this prevents.
                    task_terminal = task and task.status in {"done", "archived"}
                    if task_terminal:
                        await asyncio.to_thread(
                            _kanban_unsub, runner, sub, board_slug,
                        )
        except Exception as exc:
            logger.warning("kanban notifier tick failed: %s", exc)
        # Sleep with cancellation checks.
        for _ in range(int(max(1, interval))):
            if not runner._running:
                return
            await asyncio.sleep(1)
