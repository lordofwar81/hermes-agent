"""
Gateway lifecycle methods extracted from GatewayRunner.

This module contains gateway initialization and shutdown methods:
- start_gateway_runner: Initialize gateway and connect adapters
- stop_gateway_runner: Shutdown gateway and disconnect adapters

These methods handle the complete gateway lifecycle from startup to
graceful shutdown including adapter management and resource cleanup.

NOTE: Helper methods (_active_profile_name, _kanban_*, etc.) remain
in GatewayRunner and are called via runner._method_name().
"""

import asyncio
import logging
import os
import time

from gateway.watchers import kanban_dispatcher_watcher
from gateway.adapter_factory import create_adapter

logger = logging.getLogger(__name__)


async def start_gateway_runner(
    runner,  # GatewayRunner instance
) -> bool:
    """
    Start the gateway and all configured platform adapters.
    
    Returns True if at least one adapter connected successfully.
    """
    logger.info("Starting Hermes Gateway...")
    try:
        runner._gateway_loop = asyncio.get_running_loop()
    except RuntimeError:
        runner._gateway_loop = None
    logger.info("Session storage: %s", runner.config.sessions_dir)

    # Sanity-check that systemd's TimeoutStopSec covers our drain
    # window.  When the user upgraded hermes-agent without re-running
    # ``hermes setup``, their unit file may still encode the old
    # default - in which case SIGKILL hits mid-drain and looks like
    # a phantom kill in the journal.  Best-effort, never raises.
    try:
        from gateway.shutdown_forensics import check_systemd_timing_alignment
        _alignment = check_systemd_timing_alignment(runner._restart_drain_timeout)
        if _alignment is not None and _alignment.get("mismatch"):
            logger.warning(
                "Stale systemd unit detected: %s has TimeoutStopSec=%.0fs but "
                "drain_timeout=%.0fs (expected >=%.0fs). systemd may SIGKILL the "
                "gateway mid-drain. Run `hermes gateway service install --replace` "
                "to regenerate the unit, or shorten agent.restart_drain_timeout.",
                _alignment.get("unit", "(unknown)"),
                _alignment["timeout_stop_sec"],
                _alignment["drain_timeout"],
                _alignment["expected_min"],
            )
    except Exception as _e:
        logger.debug("check_systemd_timing_alignment failed: %s", _e)
    # Log the resolved max_iterations budget so operators can verify the
    # config.yaml → env bridge did the right thing at a glance (instead
    # of silently running at a stale .env value for weeks).
    try:
        _effective_max_iter = int(os.getenv("HERMES_MAX_ITERATIONS", "90"))
        logger.info(
            "Agent budget: max_iterations=%d (agent.max_turns from config.yaml, "
            "or HERMES_MAX_ITERATIONS from .env, or default 90)",
            _effective_max_iter,
        )
    except Exception:
        pass
    # Redaction status: ON by default (#17691). Surface a prominent
    # warning if an operator has explicitly opted out so they don't
    # forget the downgrade is active - the redactor snapshots its
    # state at import time, so this log line is the source of truth
    # for this process's lifetime.
    try:
        _redact_raw = os.getenv("HERMES_REDACT_SECRETS", "true")
        _redact_on = _redact_raw.lower() in {"1", "true", "yes", "on"}
        if _redact_on:
            logger.info(
                "Secret redaction: ENABLED (tool output, logs, and chat "
                "responses are scrubbed before delivery)"
            )
        else:
            logger.warning(
                "Secret redaction: DISABLED (HERMES_REDACT_SECRETS=%s). "
                "API keys and tokens may appear verbatim in chat output, "
                "session JSONs, and logs. Set security.redact_secrets: true "
                "in config.yaml to re-enable.",
                _redact_raw,
            )
    except Exception:
        pass
    try:
        from hermes_cli.profiles import get_active_profile_name
        _profile = get_active_profile_name()
        if _profile and _profile != "default":
            logger.info("Active profile: %s", _profile)
    except Exception:
        pass
    try:
        from gateway.status import write_runtime_status
        write_runtime_status(gateway_state="starting", exit_reason=None)
    except Exception:
        pass

    # Log any active supply-chain security advisories. Operators see this
    # in gateway.log and `hermes status` surfaces it; we do NOT block
    # startup or surface it inline to user messages, since the gateway
    # operator is the one who can act on it (uninstall the package,
    # rotate credentials).  See hermes_cli/security_advisories.py.
    try:
        from hermes_cli.security_advisories import (
            detect_compromised,
            gateway_log_message,
        )
        _adv_hits = detect_compromised()
        _adv_msg = gateway_log_message(_adv_hits)
        if _adv_msg:
            logger.warning("%s", _adv_msg)
            logger.warning(
                "Run `hermes doctor` on the gateway host for full "
                "remediation steps."
            )
    except Exception:
        logger.debug(
            "security advisory check failed at gateway startup",
            exc_info=True,
        )
    
    # Warn if no user allowlists are configured and open access is not opted in
    _builtin_allowed_vars = (
        "TELEGRAM_ALLOWED_USERS", "DISCORD_ALLOWED_USERS",
        "WHATSAPP_ALLOWED_USERS", "SLACK_ALLOWED_USERS",
        "SIGNAL_ALLOWED_USERS", "SIGNAL_GROUP_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_CHATS",
        "EMAIL_ALLOWED_USERS",
        "SMS_ALLOWED_USERS", "MATTERMOST_ALLOWED_USERS",
        "MATRIX_ALLOWED_USERS", "DINGTALK_ALLOWED_USERS",
        "FEISHU_ALLOWED_USERS",
        "WECOM_ALLOWED_USERS",
        "WECOM_CALLBACK_ALLOWED_USERS",
        "WEIXIN_ALLOWED_USERS",
        "BLUEBUBBLES_ALLOWED_USERS",
        "QQ_ALLOWED_USERS",
        "YUANBAO_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
    )
    _builtin_allow_all_vars = (
        "TELEGRAM_ALLOW_ALL_USERS", "DISCORD_ALLOW_ALL_USERS",
        "WHATSAPP_ALLOW_ALL_USERS", "SLACK_ALLOW_ALL_USERS",
        "SIGNAL_ALLOW_ALL_USERS", "EMAIL_ALLOW_ALL_USERS",
        "SMS_ALLOW_ALL_USERS", "MATTERMOST_ALLOW_ALL_USERS",
        "MATRIX_ALLOW_ALL_USERS", "DINGTALK_ALLOW_ALL_USERS",
        "FEISHU_ALLOW_ALL_USERS",
        "WECOM_ALLOW_ALL_USERS",
        "WECOM_CALLBACK_ALLOW_ALL_USERS",
        "WEIXIN_ALLOW_ALL_USERS",
        "BLUEBUBBLES_ALLOW_ALL_USERS",
        "QQ_ALLOW_ALL_USERS",
        "YUANBAO_ALLOW_ALL_USERS",
    )
    # Also pick up plugin-registered platforms - each entry can declare
    # its own allowed_users_env / allow_all_env, so the warning stays
    # accurate as plugins like IRC come online.
    _plugin_allowed_vars: tuple = ()
    _plugin_allow_all_vars: tuple = ()
    try:
        from gateway.platform_registry import platform_registry
        _plugin_allowed_vars = tuple(
            e.allowed_users_env for e in platform_registry.plugin_entries()
            if e.allowed_users_env
        )
        _plugin_allow_all_vars = tuple(
            e.allow_all_env for e in platform_registry.plugin_entries()
            if e.allow_all_env
        )
    except Exception:
        pass
    _any_allowlist = any(
        os.getenv(v) for v in _builtin_allowed_vars + _plugin_allowed_vars
    )
    _allow_all = os.getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in {"true", "1", "yes"} or any(
        os.getenv(v, "").lower() in {"true", "1", "yes"}
        for v in _builtin_allow_all_vars + _plugin_allow_all_vars
    )
    if not _any_allowlist and not _allow_all:
        logger.warning(
            "No user allowlists configured. All unauthorized users will be denied. "
            "Set GATEWAY_ALLOW_ALL_USERS=true in ~/.hermes/.env to allow open access, "
            "or configure platform allowlists (e.g., TELEGRAM_ALLOWED_USERS=your_id)."
        )
    
    # Discover Python plugins before shell hooks so plugin block
    # decisions take precedence in tie cases.  The CLI startup path
    # does this via an explicit call in hermes_cli/main.py; the
    # gateway lazily imports run_agent inside per-request handlers,
    # so the discover_plugins() side-effect in model_tools.py is NOT
    # guaranteed to have run by the time we reach this point.
    try:
        from hermes_cli.plugins import discover_plugins
        discover_plugins()
    except Exception:
        logger.warning(
            "plugin discovery failed at gateway startup", exc_info=True,
        )

    # Register declarative shell hooks from cli-config.yaml.  Gateway
    # has no TTY, so consent has to come from one of the three opt-in
    # channels (--accept-hooks on launch, HERMES_ACCEPT_HOOKS env var,
    # or hooks_auto_accept: true in config.yaml).  We pass
    # accept_hooks=False here and let register_from_config resolve
    # the effective value from env + config itself - the CLI-side
    # registration already honored --accept-hooks, and re-reading
    # hooks_auto_accept here would just duplicate that lookup.
    # Failures are logged but must never block gateway startup.
    try:
        from hermes_cli.config import load_config
        from agent.shell_hooks import register_from_config
        register_from_config(load_config(), accept_hooks=False)
    except Exception:
        logger.debug(
            "shell-hook registration failed at gateway startup",
            exc_info=True,
        )

    # Discover and load event hooks
    runner.hooks.discover_and_load()

    
    # Recover background processes from checkpoint (crash recovery)
    try:
        from tools.process_registry import process_registry
        recovered = process_registry.recover_from_checkpoint()
        if recovered:
            logger.info("Recovered %s background process(es) from previous run", recovered)
    except Exception as e:
        logger.warning("Process checkpoint recovery: %s", e)

    # Suspend sessions that were active when the gateway last exited.
    # This prevents stuck sessions from being blindly resumed on restart,
    # which can create an unrecoverable loop (#7536).  Suspended sessions
    # auto-reset on the next incoming message, giving the user a clean start.
    #
    # SKIP suspension after a clean (graceful) shutdown - the previous
    # process already drained active agents, so sessions aren't stuck.
    # This prevents unwanted auto-resets after `hermes update`,
    # `hermes gateway restart`, or `/restart`.
    _clean_marker = _hermes_home / ".clean_shutdown"
    if _clean_marker.exists():
        logger.info("Previous gateway exited cleanly - skipping session suspension")
        try:
            _clean_marker.unlink()
        except Exception:
            pass
    else:
        try:
            suspended = runner.session_store.suspend_recently_active()
            if suspended:
                logger.info("Marked %d in-flight session(s) as resumable from previous run", suspended)
        except Exception as e:
            logger.warning("Session suspension on startup failed: %s", e)

    # Stuck-loop detection (#7536): if a session has been active across
    # 3+ consecutive restarts, it's probably stuck in a loop (the same
    # history keeps causing the agent to hang).  Auto-suspend it so the
    # user gets a clean slate on the next message.
    try:
        stuck = runner._suspend_stuck_loop_sessions()
        if stuck:
            logger.warning("Auto-suspended %d stuck-loop session(s)", stuck)
    except Exception as e:
        logger.debug("Stuck-loop detection failed: %s", e)

    connected_count = 0
    enabled_platform_count = 0
    startup_nonretryable_errors: list[str] = []
    startup_retryable_errors: list[str] = []
    
    # Initialize and connect each configured platform
    for platform, platform_config in runner.config.platforms.items():
        if not platform_config.enabled:
            continue
        enabled_platform_count += 1

        adapter = create_adapter(runner, platform, platform_config)
        if not adapter:
            # Distinguish between missing builtin deps and missing plugin
            _pval = platform.value
            _builtin_names = {m.value for m in Platform.__members__.values()}
            if _pval not in _builtin_names:
                logger.warning(
                    "No adapter for '%s' - is the plugin installed? "
                    "(platform is enabled in config.yaml but no plugin registered it)",
                    _pval,
                )
            else:
                logger.warning("No adapter available for %s", _pval)
            continue
        
        # Set up message + fatal error handlers
        adapter.set_message_handler(lambda event: handle_message(self, event))
        adapter.set_fatal_error_handler(runner._handle_adapter_fatal_error)
        adapter.set_session_store(runner.session_store)
        adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
        adapter.set_topic_recovery_fn(runner._recover_telegram_topic_thread_id)
        adapter._busy_text_mode = runner._busy_text_mode
        
        # Try to connect
        logger.info("Connecting to %s...", platform.value)
        runner._update_platform_runtime_status(
            platform.value,
            platform_state="connecting",
            error_code=None,
            error_message=None,
        )
        try:
            success = await runner._connect_adapter_with_timeout(adapter, platform)
            if success:
                runner.adapters[platform] = adapter
                voice_mode.sync_voice_mode_state_to_adapter(adapter)
                connected_count += 1
                runner._update_platform_runtime_status(
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
                await runner._safe_adapter_disconnect(adapter, platform)
                if adapter.has_fatal_error:
                    runner._update_platform_runtime_status(
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
                        runner._failed_platforms[platform] = {
                            "config": platform_config,
                            "attempts": 1,
                            "next_retry": time.monotonic() + 30,
                        }
                else:
                    runner._update_platform_runtime_status(
                        platform.value,
                        platform_state="retrying",
                        error_code=None,
                        error_message="failed to connect",
                    )
                    startup_retryable_errors.append(
                        f"{platform.value}: failed to connect"
                    )
                    # No fatal error info means likely a transient issue - queue for retry
                    runner._failed_platforms[platform] = {
                        "config": platform_config,
                        "attempts": 1,
                        "next_retry": time.monotonic() + 30,
                    }
        except Exception as e:
            logger.error("✗ %s error: %s", platform.value, e)
            # Same defensive cleanup path for exceptions - an adapter
            # that raised mid-connect may still have a live
            # aiohttp.ClientSession or child subprocess.
            await runner._safe_adapter_disconnect(adapter, platform)
            runner._update_platform_runtime_status(
                platform.value,
                platform_state="retrying",
                error_code=None,
                error_message=str(e),
            )
            startup_retryable_errors.append(f"{platform.value}: {e}")
            # Unexpected exceptions are typically transient - queue for retry
            runner._failed_platforms[platform] = {
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
            runner._request_clean_exit(reason)
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
                    "Gateway started with no connected platforms - "
                    "%d platform(s) queued for retry: %s",
                    len(runner._failed_platforms), reason,
                )
                try:
                    from gateway.status import write_runtime_status
                    write_runtime_status(
                        gateway_state="degraded",
                        exit_reason=None,
                    )
                except Exception:
                    pass
                # Fall through to the normal "running" state - reconnect
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
    runner.delivery_router.adapters = runner.adapters
    runner._wire_teams_pipeline_runtime()

    runner._running = True
    runner._update_runtime_status("running")
    
    # Emit gateway:startup hook
    hook_count = len(runner.hooks.loaded_hooks)
    if hook_count:
        logger.info("%s hook(s) loaded", hook_count)
    await runner.hooks.emit("gateway:startup", {
        "platforms": [p.value for p in runner.adapters.keys()],
    })
    
    if connected_count > 0:
        logger.info("Gateway running with %s platform(s)", connected_count)
    
    # Build initial channel directory for send_message name resolution
    try:
        from gateway.channel_directory import build_channel_directory
        directory = await build_channel_directory(runner.adapters)
        ch_count = sum(len(chs) for chs in directory.get("platforms", {}).values())
        logger.info("Channel directory built: %d target(s)", ch_count)
    except Exception as e:
        logger.warning("Channel directory build failed: %s", e)
    
    # Check if we're restarting after a /update command. If the update is
    # still running, keep watching so we notify once it actually finishes.
    notified = await runner._send_update_notification()
    if not notified and any(
        path.exists()
        for path in (
            _hermes_home / ".update_pending.json",
            _hermes_home / ".update_pending.claimed.json",
        )
    ):
        runner._schedule_update_notification_watch()

    # Give freshly connected platform adapters a brief moment to settle
    # before sending restart/startup lifecycle messages. In practice this
    # helps Discord thread deliveries right after reconnect.
    if connected_count > 0:
        await asyncio.sleep(1.0)

    # Notify the chat that initiated /restart that the gateway is back.
    planned_restart_notification_pending = _planned_restart_notification_pending()
    await runner._send_restart_notification()

    # Broadcast a lightweight "gateway is back" message to configured home
    # channels only for non-chat planned restarts (terminal/SIGUSR1/service
    # paths). Chat-originated /restart already has a precise reply target
    # in .restart_notify.json, so keep that lifecycle in the originating
    # chat/topic instead of also leaking it to the configured home channel.
    if planned_restart_notification_pending:
        try:
            await runner._send_home_channel_startup_notifications(
                skip_targets=None,
            )
        finally:
            _clear_planned_restart_notification()

    # Automatically continue fresh sessions that were interrupted by the
    # previous gateway restart/shutdown.  The resume_pending flag is cleared
    # by the normal successful-turn path, so a failed auto-resume remains
    # visible for manual recovery on the next user message.
    runner._schedule_resume_pending_sessions()

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
            asyncio.create_task(runner._run_process_watcher(watcher))
            logger.info("Resumed watcher for recovered process %s", watcher.get("session_id"))
            if i % 100 == 99:
                await asyncio.sleep(0)
    except Exception as e:
        logger.error("Recovered watcher setup error: %s", e)

    # Start background session expiry watcher to finalize expired sessions
    asyncio.create_task(runner._session_expiry_watcher())

    # Start background kanban notifier - delivers `completed`, `blocked`,
    # `spawn_auto_blocked`, and `crashed` events to gateway subscribers
    # so human-in-the-loop workflows hear back without polling.
    asyncio.create_task(runner._kanban_notifier_watcher())

    # Start background kanban dispatcher - spawns workers for ready
    # tasks. Gated by `kanban.dispatch_in_gateway` (default True).
    # When false, users run `hermes kanban daemon` externally or
    # simply don't use kanban; this loop becomes a no-op.
    asyncio.create_task(kanban_dispatcher_watcher(runner))

    # Start background reconnection watcher for platforms that failed at startup
    if runner._failed_platforms:
        logger.info(
            "Starting reconnection watcher for %d failed platform(s): %s",
            len(runner._failed_platforms),
            ", ".join(p.value for p in runner._failed_platforms),
        )
    asyncio.create_task(runner._platform_reconnect_watcher())

    # Start background handoff watcher - picks up CLI sessions marked
    # handoff_state='pending' in state.db and re-binds them to the
    # destination platform's home channel, then forges a synthetic user
    # turn so the agent kicks off the new chat.
    asyncio.create_task(runner._handoff_watcher())

    logger.info("Press Ctrl+C to stop")
    
    return True



async def stop_gateway_runner(
    runner,  # GatewayRunner instance
    *,
    restart: bool = False,
    detached_restart: bool = False,
    service_restart: bool = False,
) -> None:
    """Stop the gateway and disconnect all adapters."""
    if restart:
        runner._restart_requested = True
        runner._restart_detached = detached_restart
        runner._restart_via_service = service_restart
    if runner._stop_task is not None:
        await runner._stop_task
        return

    async def _stop_impl() -> None:
        def _kill_tool_subprocesses(phase: str) -> None:
            """Kill tool subprocesses + tear down terminal envs + browsers.

            Called twice in the shutdown path: once eagerly after a
            drain timeout forces agent interrupt (so we reclaim bash/
            sleep children before systemd TimeoutStopSec escalates to
            SIGKILL on the cgroup - #8202), and once as a final
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
            " for restart" if runner._restart_requested else "",
        )
        _stop_started_at = time.monotonic()

        _shutdown_budget = 70.0  # Must finish well before systemd TimeoutStopSec

        def _phase_elapsed() -> float:
            return time.monotonic() - _stop_started_at

        def _budget_exhausted(phase: str) -> bool:
            if _phase_elapsed() >= _shutdown_budget:
                logger.warning(
                    "Shutdown budget exhausted at %s (+%.2fs); "
                    "skipping to final cleanup",
                    phase, _phase_elapsed(),
                )
                return True
            return False

        runner._running = False
        runner._draining = True

        # ── PHASE 0: Kill all tool subprocesses IMMEDIATELY ──────────
        # These are bash/python/sleep children that systemd won't signal
        # (KillMode=mixed only signals the main PID).  Killing them now
        # reclaims resources and prevents systemd SIGKILL escalation.
        _kill_tool_subprocesses("immediate")
        logger.info(
            "Shutdown phase: immediate tool kill done at +%.2fs",
            _phase_elapsed(),
        )

        # Kill untracked children (e.g. llama-server spawned by agent
        # tool calls) that process_registry doesn't know about.
        try:
            import psutil as _psutil
            _me = _psutil.Process(os.getpid())
            for _child in _me.children(recursive=False):
                try:
                    _cmd = " ".join(_child.cmdline()).lower()
                except (_psutil.NoSuchProcess, _psutil.AccessDenied):
                    continue
                if "llama" in _cmd:
                    _child.terminate()
                    logger.info(
                        "Shutdown: terminated llama-server child "
                        "(pid=%d)", _child.pid,
                    )
        except ImportError:
            pass
        except Exception as _e:
            logger.debug("Untracked child cleanup: %s", _e)
        logger.info(
            "Shutdown phase: untracked child cleanup done at +%.2fs",
            _phase_elapsed(),
        )

        # ── PHASE 1: Notify active sessions ──────────────────────────
        # Adapters are still connected here, so messages can be sent.
        await runner._notify_active_sessions_of_shutdown()
        logger.info(
            "Shutdown phase: notify_active_sessions done at +%.2fs",
            _phase_elapsed(),
        )

        # ── PHASE 2: Preemptive resume_pending ───────────────────────
        # Mark ALL active sessions as resume_pending BEFORE draining so
        # they can be auto-resumed even if the process gets SIGKILL'd
        # mid-drain.  Sessions that finish cleanly during the drain
        # window will have resume_pending cleared on their next turn.
        if not _budget_exhausted("preemptive_resume"):
            _pre_reason = (
                "restart_pre" if runner._restart_requested else "shutdown_pre"
            )
            for _sk, _agent in list(runner._running_agents.items()):
                if _agent is _AGENT_PENDING_SENTINEL:
                    continue
                try:
                    runner.session_store.mark_resume_pending(_sk, _pre_reason)
                except Exception as _e:
                    logger.debug(
                        "preemptive mark_resume_pending for %s: %s",
                        _sk, _e,
                    )

        # ── PHASE 3: Drain agents (capped) ──────────────────────────
        # The configured restart_drain_timeout (180s) is only for
        # voluntary /restart.  For SIGTERM from systemd, cap to 30s.
        _drain_started_at = time.monotonic()
        _drain_timeout_used = 0.0
        if _budget_exhausted("drain"):
            timed_out = bool(runner._running_agents)
            active_agents = {}
        else:
            _drain_cap = 30.0
            _drain_timeout_used = min(runner._restart_drain_timeout, _drain_cap)
            active_agents, timed_out = await runner._drain_active_agents(_drain_timeout_used)
        logger.info(
            "Shutdown phase: drain done at +%.2fs (drain took %.2fs, "
            "timed_out=%s, active_at_start=%d, active_now=%d)",
            _phase_elapsed(),
            time.monotonic() - _drain_started_at,
            timed_out,
            len(active_agents),
            runner._running_agent_count(),
        )

        if not timed_out:
            # Drain completed gracefully - clear any pre-drain
            # resume_pending markers so finished sessions don't carry
            # a stale flag.
            for _sk in list(runner._running_agents):
                try:
                    runner.session_store.clear_resume_pending(_sk)
                except Exception as _e:
                    logger.debug(
                        "clear_resume_pending after drain for %s: %s",
                        _sk, _e,
                    )

        if timed_out:
            logger.warning(
                "Gateway drain timed out after %.1fs with %d active agent(s); interrupting remaining work.",
                _drain_timeout_used,
                runner._running_agent_count(),
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
            # Iterate runner._running_agents (current) rather than the
            # drain-start ``active_agents`` snapshot - the snapshot
            # may include sessions that finished gracefully during
            # the drain window, and marking those falsely would give
            # them a stray restart-interruption system note on their
            # next turn even though their previous turn completed
            # cleanly.  Skip pending sentinels for the same reason
            # _interrupt_running_agents() does: their agent hasn't
            # started yet, there's nothing to interrupt, and the
            # session shouldn't carry a misleading resume flag.
            _resume_reason = (
                "restart_timeout" if runner._restart_requested else "shutdown_timeout"
            )
            for _sk, _agent in list(runner._running_agents.items()):
                if _agent is _AGENT_PENDING_SENTINEL:
                    continue
                try:
                    runner.session_store.mark_resume_pending(_sk, _resume_reason)
                except Exception as _e:
                    logger.debug(
                        "mark_resume_pending failed for %s: %s",
                        _sk, _e,
                    )
            runner._interrupt_running_agents(
                _INTERRUPT_REASON_GATEWAY_RESTART if runner._restart_requested else _INTERRUPT_REASON_GATEWAY_SHUTDOWN
            )
            interrupt_deadline = asyncio.get_running_loop().time() + 2.0
            while runner._running_agents and asyncio.get_running_loop().time() < interrupt_deadline:
                runner._update_runtime_status("draining")
                await asyncio.sleep(0.1)

            # Kill lingering tool subprocesses NOW, before we spend more
            # budget on adapter disconnect / session DB close.  Under
            # systemd (TimeoutStopSec bounded by drain_timeout+headroom),
            # deferring this to the end of stop() risks systemd escalating
            # to SIGKILL on the cgroup first - at which point bash/sleep
            # children left behind by an interrupted terminal tool get
            # killed by systemd instead of us (issue #8202).  The final
            # catch-all cleanup below still runs for the graceful path.
            _kill_tool_subprocesses("post-interrupt")
            logger.info(
                "Shutdown phase: post-interrupt tool kill done at +%.2fs",
                _phase_elapsed(),
            )

        if runner._restart_requested and runner._restart_detached:
            try:
                await runner._launch_detached_restart_command()
            except Exception as e:
                logger.error("Failed to launch detached gateway restart: %s", e)

        runner._finalize_shutdown_agents(active_agents)

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
                runner._cleanup_agent_resources(_agent)

        # ── PHASE 5: Disconnect adapters (with per-adapter timeout) ──
        if not _budget_exhausted("adapter_disconnect"):
            _adapter_timeout = runner._adapter_disconnect_timeout_secs()
            for platform, adapter in list(runner.adapters.items()):
                _adapter_started_at = time.monotonic()
                try:
                    await adapter.cancel_background_tasks()
                except Exception as e:
                    logger.debug("✗ %s background-task cancel error: %s", platform.value, e)
                try:
                    if _adapter_timeout > 0:
                        await asyncio.wait_for(adapter.disconnect(), timeout=_adapter_timeout)
                    else:
                        await adapter.disconnect()
                    logger.info(
                        "✓ %s disconnected (%.2fs)",
                        platform.value,
                        time.monotonic() - _adapter_started_at,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Timed out after %.1fs disconnecting %s; "
                        "continuing shutdown",
                        _adapter_timeout, platform.value,
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

        for _task in list(runner._background_tasks):
            if _task is runner._stop_task:
                continue
            _task.cancel()
        runner._background_tasks.clear()

        runner.adapters.clear()
        runner._running_agents.clear()
        runner._running_agents_ts.clear()
        runner._pending_messages.clear()
        runner._pending_approvals.clear()
        if hasattr(self, '_busy_ack_ts'):
            runner._busy_ack_ts.clear()
        runner._shutdown_event.set()

        # Global cleanup: kill any remaining tool subprocesses not tied
        # to a specific agent (catch-all for zombie prevention). On the
        # drain-timeout path we already did this earlier after agent
        # interrupt - this second call catches (a) the graceful path
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
        # actually exits - causing 'database is locked' errors when
        # the new gateway tries to open the same file.
        if not _budget_exhausted("sessiondb_close"):
            for _db_holder in (self, getattr(self, "session_store", None)):
                _db = getattr(_db_holder, "_db", None) if _db_holder else None
                if _db is None or not hasattr(_db, "close"):
                    continue
                try:
                    await asyncio.wait_for(
                        asyncio.get_running_loop().run_in_executor(None, _db.close),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "SessionDB close timed out after 5s; continuing",
                    )
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
        # suspends those sessions - giving users a clean slate instead
        # of resuming a half-finished tool loop.
        if not timed_out:
            try:
                (_hermes_home / ".clean_shutdown").touch()
            except Exception:
                pass
        else:
            logger.info(
                "Skipping .clean_shutdown marker - drain timed out with "
                "interrupted agents; next startup will suspend recently "
                "active sessions."
            )

        # Track sessions that were active at shutdown for stuck-loop
        # detection (#7536).  On each restart, the counter increments
        # for sessions that were running.  If a session hits the
        # threshold (3 consecutive restarts while active), the next
        # startup auto-suspends it - breaking the loop.
        if active_agents:
            runner._increment_restart_failure_counts(set(active_agents.keys()))

        if runner._restart_requested and runner._restart_command_source is None:
            try:
                atomic_json_write(
                    _planned_restart_notification_path(),
                    {
                        "requested_at": time.time(),
                        "via_service": bool(runner._restart_via_service),
                        "detached": bool(runner._restart_detached),
                    },
                    indent=None,
                )
            except Exception as e:
                logger.debug("Failed to write planned restart notification marker: %s", e)

        if runner._restart_requested and runner._restart_via_service:
            runner._launch_systemd_restart_shortcut()
            # systemd units use Restart=always, so a planned restart should
            # exit cleanly and still be relaunched.  Using TEMPFAIL here
            # makes systemd treat the operator-requested restart as a
            # failure and can trip stepped restart backoff.  launchd's
            # KeepAlive.SuccessfulExit=false needs a non-zero exit to
            # relaunch, so keep the old code on macOS.
            runner._exit_code = (
                GATEWAY_SERVICE_RESTART_EXIT_CODE
                if sys.platform == "darwin" or not os.environ.get("INVOCATION_ID")
                else 0
            )
            runner._exit_reason = runner._exit_reason or "Gateway restart requested"

        runner._draining = False
        runner._update_runtime_status("stopped", runner._exit_reason)
        logger.info("Gateway stopped (total teardown %.2fs)", _phase_elapsed())

    runner._stop_task = asyncio.create_task(_stop_impl())
    await runner._stop_task



async def process_handoff(
    runner,  # GatewayRunner instance
    row: dict,
) -> None:
    """Execute one handoff row. Raises on failure (caller marks failed)."""
    from gateway.config import Platform
    from gateway.session import SessionSource, build_session_key
    from gateway.platforms.base import MessageEvent

    cli_session_id = row["id"]
    platform_name = (row.get("handoff_platform") or "").strip().lower()
    if not platform_name:
        raise RuntimeError("handoff_platform is empty")

    # Resolve platform enum
    try:
        platform = Platform(platform_name)
    except (ValueError, KeyError):
        raise RuntimeError(f"unknown platform '{platform_name}'")

    # Adapter must be live
    adapter = runner.adapters.get(platform)
    if not adapter:
        raise RuntimeError(
            f"platform '{platform_name}' is not active in this gateway"
        )

    # Home channel must be configured
    home = runner.config.get_home_channel(platform)
    if not home or not home.chat_id:
        raise RuntimeError(
            f"no home channel configured for {platform_name}; "
            f"run /sethome on the desired chat first"
        )

    cli_title = row.get("title") or cli_session_id[:8]

    # Try to create a fresh thread on the destination so the handoff
    # has its own scrollback. Adapter returns None if threading isn't
    # supported (Matrix/WhatsApp/Signal/SMS) or if creation failed
    # (no permission, topics-mode off, parent is a DM, etc.). When
    # None we fall through to using the home channel directly — the
    # synthetic turn still lands; just without thread isolation.
    thread_name = f"Hermes — {cli_title}"
    try:
        new_thread_id = await adapter.create_handoff_thread(
            str(home.chat_id), thread_name,
        )
    except Exception as exc:
        logger.debug(
            "Handoff: create_handoff_thread raised on %s: %s",
            platform_name, exc, exc_info=True,
        )
        new_thread_id = None

    # Use the new thread if the adapter created one; otherwise fall
    # back to whatever thread (if any) the home channel was configured
    # with.
    effective_thread_id = new_thread_id or (
        str(home.thread_id) if home.thread_id else None
    )

    # Determine chat_type for the destination source. If we created a
    # thread, key the session_key as a thread (build_session_key sets
    # thread sessions to user-shared by default, which is what we
    # want — the synthetic turn and any later real-user message both
    # land on the same key without needing a user_id).
    if new_thread_id:
        dest_chat_type = "thread"
    else:
        # No thread — assume DM-style for the home channel. For
        # group/channel home channels without thread support
        # (Matrix/WhatsApp/Signal), the platform's own keying makes
        # the synthetic turn shared anyway (single-DM platforms).
        dest_chat_type = "dm"

    dest_source = SessionSource(
        platform=platform,
        chat_id=str(home.chat_id),
        chat_name=home.name,
        chat_type=dest_chat_type,
        user_id="system:handoff",
        user_name="Handoff",
        thread_id=effective_thread_id,
    )

    # Compute the gateway's session_key for that destination using the
    # same rules its adapters use, so switch_session targets the right
    # entry. For thread destinations build_session_key keys without
    # user_id (thread_sessions_per_user defaults to False) — so the
    # next real user message in the thread shares this same session.
    platform_cfg = runner.config.platforms.get(platform)
    extra = platform_cfg.extra if platform_cfg else {}
    session_key = build_session_key(
        dest_source,
        group_sessions_per_user=extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=extra.get("thread_sessions_per_user", False),
    )

    # Make sure there's an entry in the session_store for this key. If
    # the home channel has never been used, get_or_create_session
    # creates one; switch_session then re-points it.
    runner.session_store.get_or_create_session(dest_source)

    # Re-bind the destination key to the CLI session_id. switch_session
    # ends the prior session in SQLite and reopens the CLI session under
    # the new key. The CLI's transcript becomes the active one for the
    # gateway from this moment on.
    switched = runner.session_store.switch_session(session_key, cli_session_id)
    if switched is None:
        raise RuntimeError(
            f"could not switch session key {session_key} → {cli_session_id}"
        )

    # Evict any cached AIAgent for this session_key so the next dispatch
    # rebuilds it against the CLI session_id (mirrors /resume / /branch).
    runner._evict_cached_agent(session_key)

    # Cancel any in-flight running-agent state for the destination key
    # so the synthetic turn isn't queued behind a stale running flag.
    runner._release_running_agent_state(session_key)

    synthetic_text = (
        f"[Session was just handed off from CLI (\"{cli_title}\") to this "
        f"channel. The full prior conversation history is loaded above. "
        f"Briefly confirm you're working here and summarize what we were "
        f"working on, so the user can continue from this device.]"
    )

    synthetic_event = MessageEvent(
        text=synthetic_text,
        source=dest_source,
        internal=True,
    )

    logger.info(
        "Handoff: dispatching synthetic turn for CLI session %s → %s "
        "(home=%s, thread=%s, session_key=%s)",
        cli_session_id, platform_name, home.chat_id, effective_thread_id,
        session_key,
    )

    # Dispatch through the runner directly. Going through
    # adapter.handle_message would spawn a background task and we'd
    # lose synchronous error visibility; calling _handle_message inline
    # keeps the success/failure path observable for the watcher.
    response_text = await runner._handle_message(synthetic_event)
    if not response_text:
        # Streaming may have already delivered the response inline.
        # Either way, agent ran without raising — count as success.
        return

    # Send the agent's reply to the destination. Route to the new
    # thread if we created one; otherwise the configured home channel
    # (which may itself carry a thread_id).
    from typing import Dict, Any
    send_metadata: Dict[str, Any] = {}
    if effective_thread_id:
        send_metadata["thread_id"] = effective_thread_id
    try:
        result = await adapter.send(
            chat_id=str(home.chat_id),
            content=response_text,
            metadata=send_metadata or None,
        )
    except Exception as exc:
        raise RuntimeError(f"adapter.send failed: {exc}") from exc

    if not getattr(result, "success", True):
        err = getattr(result, "error", "send returned success=False")
        raise RuntimeError(f"adapter.send failed: {err}")
