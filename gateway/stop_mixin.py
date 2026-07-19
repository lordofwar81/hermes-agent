"""Gateway shutdown orchestrator for ``GatewayRunner``.

Round 49 of the god-file decomposition. ``stop()`` is the linear lifecycle
teardown orchestrator — the last inline lifecycle method after rounds 8-28
extracted its helpers (``_notify_active_sessions_of_shutdown`` ->
ActiveSessionMixin, ``_drain_active_agents`` / ``_interrupt_running_agents``
-> DrainQueueMixin, ``_finalize_shutdown_agents`` /
``_cleanup_agent_resources`` / ``_launch_*_restart_*`` -> gateway_lifecycle).

This is a whole-method verbatim lift, not a sub-region extraction (cf.
round 46's startup-preflight region move): ``stop()`` is self-contained,
called only via ``self.stop(...)`` (2 production call sites in run.py plus
the test net), and resolves unchanged through the MRO once hosted on a
mixin base. The nested closures (``_stop_impl``, ``_kill_tool_subprocesses``,
``_phase_elapsed``) move with the method body verbatim.

``gateway.run`` module-level runtime symbols (``logger``, ``time``, ``sys``,
``os``, ``_hermes_home``, ``_AGENT_PENDING_SENTINEL``, ``_INTERRUPT_REASON_*``,
``_cleanup_agent_resources``, ``_finalize_shutdown_agents``,
``_launch_detached_restart_command``, ``_launch_systemd_restart_shortcut``,
``_planned_restart_notification_path``, ``atomic_json_write``,
``GATEWAY_SERVICE_RESTART_EXIT_CODE``) are lazy-imported at the top of the
method body to avoid the circular import (``gateway.run`` imports this mixin
at module top). ``asyncio`` is stdlib. All ``self.*`` references resolve
unchanged via the MRO. Behavior-neutral extraction matching the existing
mixin pattern (rounds 42-46).
"""

from __future__ import annotations

import asyncio


class GatewayStopMixin:
    """Provides ``stop()`` — full gateway teardown orchestrator.

    Mixed into ``GatewayRunner`` as the last base (tail of MRO). No
    ``__init__``; all state lives on ``GatewayRunner`` and is touched via
    ``self.*``. See module docstring for the lift rationale.
    """

    async def stop(
        self,
        *,
        restart: bool = False,
        detached_restart: bool = False,
        service_restart: bool = False,
    ) -> None:
        """Stop the gateway and disconnect all adapters."""
        # Lazy-import run.py module globals to avoid the circular import
        # (gateway.run imports GatewayStopMixin at module top).
        import os
        import sys
        import time

        from gateway.run import (
            GATEWAY_SERVICE_RESTART_EXIT_CODE,
            _AGENT_PENDING_SENTINEL,
            _INTERRUPT_REASON_GATEWAY_RESTART,
            _INTERRUPT_REASON_GATEWAY_SHUTDOWN,
            _cleanup_agent_resources,
            _finalize_shutdown_agents,
            _hermes_home,
            _launch_detached_restart_command,
            _launch_systemd_restart_shortcut,
            _planned_restart_notification_path,
            atomic_json_write,
            logger,
        )

        if restart:
            self._restart_requested = True
            self._restart_detached = detached_restart
            self._restart_via_service = service_restart
        if self._stop_task is not None:
            await self._stop_task
            return

        async def _stop_impl() -> None:
            def _kill_tool_subprocesses(phase: str) -> None:
                """Kill tool subprocesses + tear down terminal envs + browsers.

                Called twice in the shutdown path: once eagerly after a
                drain timeout forces agent interrupt (so we reclaim bash/
                sleep children before systemd TimeoutStopSec escalates to
                SIGKILL on the cgroup — #8202), and once as a final
                catch-all at the end of _stop_impl() for the graceful
                path or anything respawned mid-teardown.

                All steps are best-effort; exceptions are swallowed so
                one subsystem's failure doesn't block the rest.
                """
                try:
                    from tools.process_registry import process_registry
                    _killed = process_registry.kill_all()
                    if _killed:
                        logger.info(
                            "Shutdown (%s): killed %d tool subprocess(es)",
                            phase, _killed,
                        )
                except Exception as _e:
                    logger.debug("process_registry.kill_all (%s) error: %s", phase, _e)
                try:
                    from tools.async_delegation import interrupt_all as _interrupt_async
                    _async_n = _interrupt_async(reason=f"gateway shutdown ({phase})")
                    if _async_n:
                        logger.info(
                            "Shutdown (%s): interrupted %d background delegation(s)",
                            phase, _async_n,
                        )
                except Exception as _e:
                    logger.debug("async interrupt_all (%s) error: %s", phase, _e)
                try:
                    from tools.terminal_tool import cleanup_all_environments
                    cleanup_all_environments()
                except Exception as _e:
                    logger.debug("cleanup_all_environments (%s) error: %s", phase, _e)
                try:
                    from tools.browser_tool import cleanup_all_browsers
                    cleanup_all_browsers()
                except Exception as _e:
                    logger.debug("cleanup_all_browsers (%s) error: %s", phase, _e)

            logger.info(
                "Stopping gateway%s...",
                " for restart" if self._restart_requested else "",
            )
            _stop_started_at = time.monotonic()

            def _phase_elapsed() -> float:
                return time.monotonic() - _stop_started_at

            self._running = False
            self._draining = True

            # Notify all chats with active agents BEFORE draining.
            # Adapters are still connected here, so messages can be sent.
            await self._notify_active_sessions_of_shutdown()
            logger.info(
                "Shutdown phase: notify_active_sessions done at +%.2fs",
                _phase_elapsed(),
            )

            timeout = self._restart_drain_timeout

            # Pre-mark sessions as resume_pending BEFORE the drain wait.
            # If the process is killed by the service manager during the
            # drain, the durable marker is already written so the next
            # gateway boot can recover in-flight sessions (#27856).
            _pre_drain_keys: list[str] = []
            for _sk, _agent in list(self._running_agents.items()):
                if _agent is _AGENT_PENDING_SENTINEL:
                    continue
                try:
                    self.session_store.mark_resume_pending(
                        _sk,
                        "restart_timeout" if self._restart_requested else "shutdown_timeout",
                    )
                    _pre_drain_keys.append(_sk)
                except Exception as _e:
                    logger.debug("pre-drain mark_resume_pending failed for %s: %s", _sk, _e)

            _drain_started_at = time.monotonic()
            active_agents, timed_out = await self._drain_active_agents(timeout)
            logger.info(
                "Shutdown phase: drain done at +%.2fs (drain took %.2fs, "
                "timed_out=%s, active_at_start=%d, active_now=%d)",
                _phase_elapsed(),
                time.monotonic() - _drain_started_at,
                timed_out,
                len(active_agents),
                self._running_agent_count(),
            )

            if not timed_out:
                # Drain completed gracefully — all running sessions finished.
                # Clear the pre-drain resume_pending markers so sessions that
                # completed during the drain window don't carry a stale flag.
                for _sk in _pre_drain_keys:
                    if _sk not in self._running_agents:
                        try:
                            self.session_store.clear_resume_pending(_sk)
                        except Exception as _e:
                            logger.debug(
                                "clear_resume_pending after drain failed for %s: %s",
                                _sk, _e,
                            )

            if timed_out:
                logger.warning(
                    "Gateway drain timed out after %.1fs with %d active agent(s); interrupting remaining work.",
                    timeout,
                    self._running_agent_count(),
                )
                # Mark forcibly-interrupted sessions as resume_pending BEFORE
                # interrupting the agents.  This preserves each session's
                # session_id + transcript so the next message on the same
                # session_key auto-resumes from the existing conversation
                # instead of getting routed through suspend_recently_active()
                # and converted into a fresh session.  Terminal escalation
                # for genuinely stuck sessions still flows through the
                # existing ``.restart_failure_counts`` stuck-loop counter
                # (incremented below, threshold 3), which sets
                # ``suspended=True`` and overrides resume_pending.
                #
                # Iterate self._running_agents (current) rather than the
                # drain-start ``active_agents`` snapshot — the snapshot
                # may include sessions that finished gracefully during
                # the drain window, and marking those falsely would give
                # them a stray restart-interruption system note on their
                # next turn even though their previous turn completed
                # cleanly.  Skip pending sentinels for the same reason
                # _interrupt_running_agents() does: their agent hasn't
                # started yet, there's nothing to interrupt, and the
                # session shouldn't carry a misleading resume flag.
                _resume_reason = (
                    "restart_timeout" if self._restart_requested else "shutdown_timeout"
                )
                for _sk, _agent in list(self._running_agents.items()):
                    if _agent is _AGENT_PENDING_SENTINEL:
                        continue
                    try:
                        self.session_store.mark_resume_pending(_sk, _resume_reason)
                    except Exception as _e:
                        logger.debug(
                            "mark_resume_pending failed for %s: %s",
                            _sk, _e,
                        )
                self._interrupt_running_agents(
                    _INTERRUPT_REASON_GATEWAY_RESTART if self._restart_requested else _INTERRUPT_REASON_GATEWAY_SHUTDOWN
                )
                interrupt_deadline = asyncio.get_running_loop().time() + 5.0
                while self._running_agents and asyncio.get_running_loop().time() < interrupt_deadline:
                    self._update_runtime_status("draining")
                    await asyncio.sleep(0.1)

                # Kill lingering tool subprocesses NOW, before we spend more
                # budget on adapter disconnect / session DB close.  Under
                # systemd (TimeoutStopSec bounded by drain_timeout+headroom),
                # deferring this to the end of stop() risks systemd escalating
                # to SIGKILL on the cgroup first — at which point bash/sleep
                # children left behind by an interrupted terminal tool get
                # killed by systemd instead of us (issue #8202).  The final
                # catch-all cleanup below still runs for the graceful path.
                _kill_tool_subprocesses("post-interrupt")
                logger.info(
                    "Shutdown phase: post-interrupt tool kill done at +%.2fs",
                    _phase_elapsed(),
                )

            if self._restart_requested and self._restart_detached:
                try:
                    await _launch_detached_restart_command()
                except Exception as e:
                    logger.error("Failed to launch detached gateway restart: %s", e)

            _finalize_shutdown_agents(active_agents)

            # Also shut down memory providers on idle cached agents.
            # _finalize_shutdown_agents only handles agents that were
            # mid-turn at drain time; the _agent_cache may still hold
            # idle agents whose MemoryProviders never received
            # on_session_end().
            _cache_lock = getattr(self, "_agent_cache_lock", None)
            _cache = getattr(self, "_agent_cache", None)
            if _cache_lock is not None and _cache is not None:
                with _cache_lock:
                    _idle_agents = list(_cache.values())
                    _cache.clear()
                for _entry in _idle_agents:
                    _agent = (
                        _entry[0] if isinstance(_entry, tuple) else _entry
                    )
                    _cleanup_agent_resources(_agent)

            for platform, adapter in list(self.adapters.items()):
                _adapter_started_at = time.monotonic()
                try:
                    await adapter.cancel_background_tasks()
                except Exception as e:
                    logger.debug("✗ %s background-task cancel error: %s", platform.value, e)
                try:
                    await adapter.disconnect()
                    logger.info(
                        "✓ %s disconnected (%.2fs)",
                        platform.value,
                        time.monotonic() - _adapter_started_at,
                    )
                except Exception as e:
                    logger.error(
                        "✗ %s disconnect error after %.2fs: %s",
                        platform.value,
                        time.monotonic() - _adapter_started_at,
                        e,
                    )
            logger.info(
                "Shutdown phase: all adapters disconnected at +%.2fs",
                _phase_elapsed(),
            )

            for _task in list(self._background_tasks):
                if _task is self._stop_task:
                    continue
                _task.cancel()
            self._background_tasks.clear()

            self.adapters.clear()
            for _session_key in list(self._running_agents):
                self._release_running_agent_state(_session_key)
            self._running_agents.clear()
            self._running_agents_ts.clear()
            if hasattr(self, "_active_session_leases"):
                self._active_session_leases.clear()
            self._pending_messages.clear()
            self._pending_approvals.clear()
            if hasattr(self, '_busy_ack_ts'):
                self._busy_ack_ts.clear()
            self._shutdown_event.set()

            # Global cleanup: kill any remaining tool subprocesses not tied
            # to a specific agent (catch-all for zombie prevention). On the
            # drain-timeout path we already did this earlier after agent
            # interrupt — this second call catches (a) the graceful path
            # where drain succeeded without interrupt, and (b) anything
            # that got respawned between the earlier call and adapter
            # disconnect (defense in depth; safe to call repeatedly).
            _kill_tool_subprocesses("final-cleanup")
            logger.info(
                "Shutdown phase: final-cleanup tool kill done at +%.2fs",
                _phase_elapsed(),
            )

            # Reap the process-global auxiliary-client cache once at the very
            # end of teardown.  Per-turn cleanup runs in _cleanup_agent_resources
            # for each active agent, but clients bound to worker-thread loops
            # that died with their ThreadPoolExecutor (notably cron ticks) only
            # get swept here.  Without this, long-running gateways accumulate
            # async httpx transports until they hit EMFILE on macOS's default
            # RLIMIT_NOFILE=256.  See #14210.
            try:
                from agent.auxiliary_client import shutdown_cached_clients
                shutdown_cached_clients()
            except Exception as _e:
                logger.debug("shutdown_cached_clients error: %s", _e)

            # Close SQLite session DBs so the WAL write lock is released.
            # Without this, --replace and similar restart flows leave the
            # old gateway's connection holding the WAL lock until Python
            # actually exits — causing 'database is locked' errors when
            # the new gateway tries to open the same file.
            for _db_holder in (self, getattr(self, "session_store", None)):
                _db = getattr(_db_holder, "_db", None) if _db_holder else None
                if _db is None or not hasattr(_db, "close"):
                    continue
                try:
                    _db.close()
                except Exception as _e:
                    logger.debug("SessionDB close error: %s", _e)
            logger.info(
                "Shutdown phase: SessionDB close done at +%.2fs",
                _phase_elapsed(),
            )

            from gateway.status import remove_pid_file, release_gateway_runtime_lock
            remove_pid_file()
            release_gateway_runtime_lock()

            # Write a clean-shutdown marker so the next startup knows this
            # wasn't a crash.  suspend_recently_active() only needs to run
            # after unexpected exits.  However, if the drain timed out and
            # agents were force-interrupted, their sessions may be in an
            # incomplete state (trailing tool response, no final assistant
            # message).  Skip the marker in that case so the next startup
            # suspends those sessions — giving users a clean slate instead
            # of resuming a half-finished tool loop.
            if not timed_out:
                try:
                    (_hermes_home / ".clean_shutdown").touch()
                except Exception:
                    pass
            else:
                logger.info(
                    "Skipping .clean_shutdown marker — drain timed out with "
                    "interrupted agents; next startup will suspend recently "
                    "active sessions."
                )

            # Track sessions that were active at shutdown for stuck-loop
            # detection (#7536).  On each restart, the counter increments
            # for sessions that were running.  If a session hits the
            # threshold (3 consecutive restarts while active), the next
            # startup auto-suspends it — breaking the loop.
            if active_agents:
                self._increment_restart_failure_counts(set(active_agents.keys()))

            if self._restart_requested and self._restart_command_source is None:
                try:
                    atomic_json_write(
                        _planned_restart_notification_path(),
                        {
                            "requested_at": time.time(),
                            "via_service": bool(self._restart_via_service),
                            "detached": bool(self._restart_detached),
                        },
                        indent=None,
                    )
                except Exception as e:
                    logger.debug("Failed to write planned restart notification marker: %s", e)

            if self._restart_requested and self._restart_via_service:
                self._launch_systemd_restart_shortcut()
                # systemd units use Restart=always, so a planned restart should
                # exit cleanly and still be relaunched.  Using TEMPFAIL here
                # makes systemd treat the operator-requested restart as a
                # failure and can trip stepped restart backoff.  launchd's
                # KeepAlive.SuccessfulExit=false needs a non-zero exit to
                # relaunch, so keep the old code on macOS.
                self._exit_code = (
                    GATEWAY_SERVICE_RESTART_EXIT_CODE
                    if sys.platform == "darwin" or not os.environ.get("INVOCATION_ID")
                    else 0
                )
                self._exit_reason = self._exit_reason or "Gateway restart requested"

            self._draining = False
            # Persist the terminal gateway_state. The default is "stopped",
            # but when this teardown was triggered by an UNEXPECTED external
            # signal (container/s6 SIGTERM on `docker restart` or image
            # upgrade, OOM-killer, bare `kill`) we instead persist "running"
            # to preserve the operator's run-intent across the restart.
            #
            # On Docker (s6-overlay), container_boot.py reads gateway_state
            # on the next boot and only auto-starts gateways whose last
            # state was "running" (_AUTOSTART_STATES). Persisting "stopped"
            # — or leaving the mid-shutdown "draining" marker in place — for
            # a routine `docker compose up --force-recreate` permanently
            # suppresses auto-start, so the messaging channels silently stay
            # dark until the operator manually restarts (issue #42675).
            #
            # An operator-initiated stop (`hermes gateway stop`,
            # systemd/launchd ExecStop, the s6 stop path, Ctrl+C) writes a
            # planned-stop marker BEFORE signalling, so it is classified as
            # a planned stop (not signal-initiated) and correctly persists
            # "stopped" — respecting the explicit intent. A restart also
            # persists "stopped" here; the restarting process brings the
            # gateway back up itself.
            if getattr(self, "_signal_initiated_shutdown", False) and not self._restart_requested:
                logger.info(
                    "Gateway stopped by an unexpected signal — persisting "
                    "gateway_state=running so container_boot auto-starts on "
                    "the next boot (issue #42675)"
                )
                self._update_runtime_status("running", self._exit_reason)
            else:
                self._update_runtime_status("stopped", self._exit_reason)
            logger.info("Gateway stopped (total teardown %.2fs)", _phase_elapsed())

        self._stop_task = asyncio.create_task(_stop_impl())
        await self._stop_task
