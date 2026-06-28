"""Gateway startup orchestrator — round 54 of the gateway god-file decomposition.

Extracts ``GatewayRunner.start`` (run.py L2082-2409, 328ln) into
``GatewayStartMixin.start``. Whole-method verbatim lift (the R49/R51/R52/R53
protocol). The method is self-contained — no nested closures, no global
declarations, a linear flow with a single try/except inside the platform
connect loop.

It is the gateway startup sequence:

  * capture the running event loop + run startup preflight checks
  * serialize startup-restore against inbound dispatch (queue inbound messages
    until the resume pass completes)
  * initialize + connect each configured platform adapter (with defensive
    cleanup on failure, retryable/non-retryable error classification, and
    failed-platform queueing for the reconnect watcher)
  * handle the all-platforms-failed branches (non-retryable -> clean exit;
    retryable -> degrade and keep cron alive; no adapter -> fleet-graceful)
  * wire the delivery router + teams pipeline runtime
  * emit the gateway:startup hook
  * build the channel directory for send_message name resolution
  * /update notification check + watch scheduling
  * planned-restart notification (chat-originated vs home-channel broadcast)
  * auto-resume interrupted sessions + finish startup restore
  * recover crash-recovery process watchers (batched with event-loop yields)
  * launch background watchers: session expiry, kanban notifier/dispatcher,
    platform reconnect, handoff, async delegation

``gateway.run`` module-level runtime symbols (``logger``, ``_hermes_home``,
``_planned_restart_notification_pending``,
``_clear_planned_restart_notification``) are lazy-imported at the top of the
method body to avoid the circular import (``gateway.run`` imports this mixin
at module top). Stdlib and third-party top-level imports are at module top.
Every other name in the body is either an in-body lazy import (kept verbatim
from source) or a ``self.*`` reference that resolves unchanged through the
MRO. Behavior-neutral extraction matching the existing mixin pattern
(rounds 42-53).
"""

from __future__ import annotations

import asyncio
import time

from gateway.config import Platform
from gateway.gateway_agent_mgmt import _update_platform_runtime_status
from gateway.gateway_async_utils import (
    _connect_adapter_with_timeout,
    _safe_adapter_disconnect,
)


class GatewayStartMixin:
    """Provides ``start`` — the gateway startup orchestrator.

    Mixed into ``GatewayRunner`` as the last base (tail of MRO). No
    ``__init__``; all state lives on ``GatewayRunner`` and is touched via
    ``self.*``. See module docstring for the lift rationale.
    """

    async def start(self) -> bool:
        """
        Start the gateway and all configured platform adapters.
        
        Returns True if at least one adapter connected successfully.
        """
        # gateway.run module-level runtime symbols — lazy-imported here to
        # avoid the circular import (gateway.run imports this mixin at top).
        from gateway.run import (
            _clear_planned_restart_notification,
            _hermes_home,
            _planned_restart_notification_pending,
            logger,
        )
        logger.info("Starting Hermes Gateway...")
        try:
            self._gateway_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._gateway_loop = None
        logger.info("Session storage: %s", self.config.sessions_dir)

        self._run_startup_preflight_checks()

        # Serialize startup restore against inbound dispatch.  Platform
        # adapters can begin receiving messages as soon as they connect, but
        # restart-interrupted sessions are not auto-resumed until all startup
        # wiring below completes.  Queue inbound messages until the resume
        # pass runs and every synthetic resume turn has finished.
        self._startup_restore_in_progress = True
        self._startup_restore_queue = []
        self._startup_restore_tasks = []

        connected_count = 0
        enabled_platform_count = 0
        startup_nonretryable_errors: list[str] = []
        startup_retryable_errors: list[str] = []
        
        # Initialize and connect each configured platform
        for platform, platform_config in self.config.platforms.items():
            if not platform_config.enabled:
                continue
            enabled_platform_count += 1
            
            adapter = self._create_adapter(platform, platform_config)
            if not adapter:
                # Distinguish between missing builtin deps and missing plugin
                _pval = platform.value
                _builtin_names = {m.value for m in Platform.__members__.values()}
                if _pval not in _builtin_names:
                    logger.warning(
                        "No adapter for '%s' — is the plugin installed? "
                        "(platform is enabled in config.yaml but no plugin registered it)",
                        _pval,
                    )
                else:
                    logger.warning("No adapter available for %s", _pval)
                continue
            
            # Set up message + fatal error handlers
            adapter.set_message_handler(self._handle_message)
            adapter.set_fatal_error_handler(self._handle_adapter_fatal_error)
            adapter.set_session_store(self.session_store)
            adapter.set_busy_session_handler(self._handle_active_session_busy_message)
            adapter.set_topic_recovery_fn(self._recover_telegram_topic_thread_id)
            adapter._busy_text_mode = self._busy_text_mode
            
            # Try to connect
            logger.info("Connecting to %s...", platform.value)
            _update_platform_runtime_status(
                platform.value,
                platform_state="connecting",
                error_code=None,
                error_message=None,
            )
            try:
                success = await _connect_adapter_with_timeout(adapter, platform)
                if success:
                    self.adapters[platform] = adapter
                    self._sync_voice_mode_state_to_adapter(adapter)
                    connected_count += 1
                    _update_platform_runtime_status(
                        platform.value,
                        platform_state="connected",
                        error_code=None,
                        error_message=None,
                    )
                    logger.info("✓ %s connected", platform.value)
                else:
                    logger.warning("✗ %s failed to connect", platform.value)
                    # Defensive cleanup: a failed connect() may have
                    # allocated resources (aiohttp.ClientSession, poll
                    # tasks, bridge subprocesses) before giving up.
                    # Without this call, those resources are orphaned
                    # and Python logs "Unclosed client session" at
                    # process exit. Adapter disconnect() implementations
                    # are expected to be idempotent and tolerate
                    # partial-init state.
                    await _safe_adapter_disconnect(adapter, platform)
                    if adapter.has_fatal_error:
                        _update_platform_runtime_status(
                            platform.value,
                            platform_state="retrying" if adapter.fatal_error_retryable else "fatal",
                            error_code=adapter.fatal_error_code,
                            error_message=adapter.fatal_error_message,
                        )
                        target = (
                            startup_retryable_errors
                            if adapter.fatal_error_retryable
                            else startup_nonretryable_errors
                        )
                        target.append(
                            f"{platform.value}: {adapter.fatal_error_message}"
                        )
                        # Queue for reconnection if the error is retryable
                        if adapter.fatal_error_retryable:
                            self._failed_platforms[platform] = {
                                "config": platform_config,
                                "attempts": 1,
                                "next_retry": time.monotonic() + 30,
                            }
                    else:
                        _update_platform_runtime_status(
                            platform.value,
                            platform_state="retrying",
                            error_code=None,
                            error_message="failed to connect",
                        )
                        startup_retryable_errors.append(
                            f"{platform.value}: failed to connect"
                        )
                        # No fatal error info means likely a transient issue — queue for retry
                        self._failed_platforms[platform] = {
                            "config": platform_config,
                            "attempts": 1,
                            "next_retry": time.monotonic() + 30,
                        }
            except Exception as e:
                logger.error("✗ %s error: %s", platform.value, e)
                # Same defensive cleanup path for exceptions — an adapter
                # that raised mid-connect may still have a live
                # aiohttp.ClientSession or child subprocess.
                await _safe_adapter_disconnect(adapter, platform)
                _update_platform_runtime_status(
                    platform.value,
                    platform_state="retrying",
                    error_code=None,
                    error_message=str(e),
                )
                startup_retryable_errors.append(f"{platform.value}: {e}")
                # Unexpected exceptions are typically transient — queue for retry
                self._failed_platforms[platform] = {
                    "config": platform_config,
                    "attempts": 1,
                    "next_retry": time.monotonic() + 30,
                }
        
        if connected_count == 0:
            if startup_nonretryable_errors:
                reason = "; ".join(startup_nonretryable_errors)
                logger.error("Gateway hit a non-retryable startup conflict: %s", reason)
                try:
                    from gateway.status import write_runtime_status
                    write_runtime_status(gateway_state="startup_failed", exit_reason=reason)
                except Exception:
                    pass
                self._request_clean_exit(reason)
                self._startup_restore_in_progress = False
                return True
            if enabled_platform_count > 0:
                if startup_retryable_errors:
                    # All enabled platforms hit retryable failures (network
                    # blip, bridge not paired, npm install timeout, etc.).
                    # Keep the gateway alive so:
                    #   • cron jobs still run
                    #   • the reconnect watcher gets a chance to recover the
                    #     failing platforms once the underlying problem is
                    #     fixed (e.g. user runs `hermes whatsapp`, fixes
                    #     proxy, etc.)
                    # Exiting here used to convert a single misconfigured
                    # platform into an infinite systemd restart loop.
                    reason = "; ".join(startup_retryable_errors)
                    logger.warning(
                        "Gateway started with no connected platforms — "
                        "%d platform(s) queued for retry: %s",
                        len(self._failed_platforms), reason,
                    )
                    try:
                        from gateway.status import write_runtime_status
                        write_runtime_status(
                            gateway_state="degraded",
                            exit_reason=None,
                        )
                    except Exception:
                        pass
                    # Fall through to the normal "running" state — reconnect
                    # watcher takes it from here.
                # All enabled platforms had no adapter (missing library or credentials).
                # In fleet deployments the same config.yaml is shared across nodes that
                # may only have credentials for a subset of platforms.  Rather than
                # failing hard, degrade gracefully and allow cron jobs to run (#5196).
                logger.warning(
                    "No adapter could be created for any of the %d configured platform(s). "
                    "Check that required dependencies are installed and credentials are set. "
                    "Gateway will continue for cron job execution.",
                    enabled_platform_count,
                )
            else:
                logger.warning("No messaging platforms enabled.")
                logger.info("Gateway will continue running for cron job execution.")
        
        # Update delivery router with adapters
        self.delivery_router.adapters = self.adapters
        self._wire_teams_pipeline_runtime()

        self._running = True
        self._update_runtime_status("running")

        # Bring up adapters for every non-active profile this gateway serves
        # (gateway.multiplex_profiles). No-op unless the flag is on. Runs after
        # the primary startup loop so the active profile's adapters seed the
        # credential-conflict detection. A MultiplexConfigError (e.g. a
        # secondary profile binding a port) aborts startup cleanly.
        try:
            _secondary_connected = await self._start_secondary_profile_adapters()
            connected_count += _secondary_connected
        except Exception as _mux_exc:
            # Re-raise config errors so the operator fixes config.yaml instead
            # of running a half-wired multiplexer; log+continue on others.
            from gateway.run import MultiplexConfigError
            if isinstance(_mux_exc, MultiplexConfigError):
                raise
            logger.error("Secondary-profile adapter startup failed: %s", _mux_exc, exc_info=True)

        # Emit gateway:startup hook
        hook_count = len(self.hooks.loaded_hooks)
        if hook_count:
            logger.info("%s hook(s) loaded", hook_count)
        await self.hooks.emit("gateway:startup", {
            "platforms": [p.value for p in self.adapters.keys()],
        })
        
        if connected_count > 0:
            logger.info("Gateway running with %s platform(s)", connected_count)
        
        # Build initial channel directory for send_message name resolution
        try:
            from gateway.channel_directory import build_channel_directory
            directory = await build_channel_directory(self.adapters)
            ch_count = sum(len(chs) for chs in directory.get("platforms", {}).values())
            logger.info("Channel directory built: %d target(s)", ch_count)
        except Exception as e:
            logger.warning("Channel directory build failed: %s", e)
        
        # Check if we're restarting after a /update command. If the update is
        # still running, keep watching so we notify once it actually finishes.
        notified = await self._send_update_notification()
        if not notified and any(
            path.exists()
            for path in (
                _hermes_home / ".update_pending.json",
                _hermes_home / ".update_pending.claimed.json",
            )
        ):
            self._schedule_update_notification_watch()

        # Give freshly connected platform adapters a brief moment to settle
        # before sending restart/startup lifecycle messages. In practice this
        # helps Discord thread deliveries right after reconnect.
        if connected_count > 0:
            await asyncio.sleep(1.0)

        # Notify the chat that initiated /restart that the gateway is back.
        planned_restart_notification_pending = _planned_restart_notification_pending()
        await self._send_restart_notification()

        # Broadcast a lightweight "gateway is back" message to configured home
        # channels only for non-chat planned restarts (terminal/SIGUSR1/service
        # paths). Chat-originated /restart already has a precise reply target
        # in .restart_notify.json, so keep that lifecycle in the originating
        # chat/topic instead of also leaking it to the configured home channel.
        if planned_restart_notification_pending:
            try:
                await self._send_home_channel_startup_notifications(
                    skip_targets=None,
                )
            finally:
                _clear_planned_restart_notification()

        # Automatically continue fresh sessions that were interrupted by the
        # previous gateway restart/shutdown.  The resume_pending flag is cleared
        # by the normal successful-turn path, so a failed auto-resume remains
        # visible for manual recovery on the next user message.
        self._schedule_resume_pending_sessions()
        await self._finish_startup_restore()

        # Drain any recovered process watchers (from crash recovery checkpoint)
        try:
            from tools.process_registry import process_registry
            # Detach the current batch atomically: reassigning to a fresh list
            # takes ownership of exactly the watchers present now, so any watcher
            # appended concurrently during the yield below isn't silently dropped
            # by a clear() on the shared list.
            watchers = process_registry.pending_watchers
            process_registry.pending_watchers = []
            # Process in batches of 100 with event-loop yield points to avoid
            # O(n^2) event-loop blocking when recovering thousands of watchers.
            for i, watcher in enumerate(watchers):
                asyncio.create_task(self._run_process_watcher(watcher))
                logger.info("Resumed watcher for recovered process %s", watcher.get("session_id"))
                if i % 100 == 99:
                    await asyncio.sleep(0)
        except Exception as e:
            logger.error("Recovered watcher setup error: %s", e)

        # Start background session expiry watcher to finalize expired sessions
        asyncio.create_task(self._session_expiry_watcher())

        # Start background kanban notifier — delivers `completed`, `blocked`,
        # `spawn_auto_blocked`, and `crashed` events to gateway subscribers
        # so human-in-the-loop workflows hear back without polling.
        asyncio.create_task(self._kanban_notifier_watcher())

        # Start background kanban dispatcher — spawns workers for ready
        # tasks. Gated by `kanban.dispatch_in_gateway` (default True).
        # When false, users run `hermes kanban daemon` externally or
        # simply don't use kanban; this loop becomes a no-op.
        asyncio.create_task(self._kanban_dispatcher_watcher())

        # Start background reconnection watcher for platforms that failed at startup
        if self._failed_platforms:
            logger.info(
                "Starting reconnection watcher for %d failed platform(s): %s",
                len(self._failed_platforms),
                ", ".join(p.value for p in self._failed_platforms),
            )
        asyncio.create_task(self._platform_reconnect_watcher())

        # Start background handoff watcher — picks up CLI sessions marked
        # handoff_state='pending' in state.db and re-binds them to the
        # destination platform's home channel, then forges a synthetic user
        # turn so the agent kicks off the new chat.
        asyncio.create_task(self._handoff_watcher())

        # Start background async-delegation watcher — drains completion events
        # from delegate_task(background=true) subagents and injects each
        # result back into its originating session as a new turn, covering the
        # idle case where the subagent finishes with no agent turn running.
        asyncio.create_task(self._async_delegation_watcher())

        logger.info("Press Ctrl+C to stop")
        
        return True

