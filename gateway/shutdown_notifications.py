"""
Shutdown notification utilities for GatewayRunner.

Extracted from gateway/run.py to reduce the God file size.
Provides functions for sending shutdown/restart notifications to active sessions.
"""

import logging
import os
import shlex
import sys
from typing import Dict, Any, Optional, Set, Tuple

from hermes_cli.enums import Platform
from gateway.session import parse_session_key as _parse_session_key

logger = logging.getLogger(__name__)


def _resolve_hermes_bin() -> Optional[list[str]]:
    """Resolve the Hermes update command as argv parts.

    Tries in order:
    1. ``shutil.which("hermes")`` — standard PATH lookup
    2. ``sys.executable -m hermes_cli.main`` — fallback when Hermes is running
       from a venv/module invocation and the ``hermes`` shim is not on PATH

    Returns argv parts ready for quoting/joining, or ``None`` if neither works.
    """
    import shutil

    hermes_bin = shutil.which("hermes")
    if hermes_bin:
        return [hermes_bin]

    try:
        import importlib.util

        if importlib.util.find_spec("hermes_cli") is not None:
            return [sys.executable, "-m", "hermes_cli.main"]
    except Exception:
        pass

    return None


async def notify_active_sessions_of_shutdown(
    runner: "GatewayRunner",
) -> None:
    """Send shutdown/restart notifications to active chats and home channels.

    Called at the very start of stop() — adapters are still connected so
    messages can be delivered. Best-effort: individual send failures are
    logged and swallowed so they never block the shutdown sequence.

    Args:
        runner: GatewayRunner instance
    """
    active = runner._snapshot_running_agents()
    restart_source = runner._restart_command_source if runner._restart_requested else None

    action = "restarting" if runner._restart_requested else "shutting down"
    hint = (
        "Your current task will be interrupted. "
        "Send any message after restart and I'll try to resume where you left off."
        if runner._restart_requested
        else "Your current task will be interrupted."
    )
    msg = f"⚠️ Gateway {action} — {hint}"

    notified: set[Tuple[str, str, Optional[str]]] = set()

    for session_key in active:
        source = None
        try:
            if getattr(runner, "session_store", None) is not None:
                runner.session_store._ensure_loaded()
                entry = runner.session_store._entries.get(session_key)
                source = getattr(entry, "origin", None) if entry else None
        except Exception as e:
            logger.debug(
                "Failed to load session origin for shutdown notification %s: %s",
                session_key,
                e,
            )

        if source is None:
            source = runner._get_cached_session_source(session_key)

        if source is not None:
            platform_str = source.platform.value
            chat_id = str(source.chat_id)
            thread_id = source.thread_id
        else:
            # Fall back to parsing the session key when no persisted
            # origin is available (legacy sessions/tests).
            _parsed = _parse_session_key(session_key)
            if not _parsed:
                continue
            platform_str = _parsed["platform"]
            chat_id = _parsed["chat_id"]
            thread_id = _parsed.get("thread_id")

        # Deduplicate only identical delivery targets. Thread/topic-aware
        # platforms can share a parent chat while still routing to distinct
        # destinations via metadata.
        dedup_key = (platform_str, chat_id, str(thread_id) if thread_id else None)
        if dedup_key in notified:
            continue

        try:
            platform = Platform(platform_str)
            adapter = runner.adapters.get(platform)
            if not adapter:
                continue

            platform_cfg = runner.config.platforms.get(platform)
            if platform_cfg is not None and not platform_cfg.gateway_restart_notification:
                logger.info(
                    "Shutdown notification suppressed for active session: %s has gateway_restart_notification=false",
                    platform_str,
                )
                continue

            reply_to_message_id = getattr(source, "message_id", None) if source is not None else None
            if reply_to_message_id is None and restart_source is not None:
                try:
                    restart_platform = restart_source.platform.value
                    restart_chat_id = str(restart_source.chat_id)
                    restart_thread_id = str(restart_source.thread_id) if restart_source.thread_id else None
                    if (restart_platform, restart_chat_id, restart_thread_id) == dedup_key:
                        reply_to_message_id = getattr(restart_source, "message_id", None)
                except Exception:
                    pass

            metadata = runner._thread_metadata_for_target(
                platform,
                chat_id,
                thread_id,
                chat_type=getattr(source, "chat_type", None) if source is not None else None,
                reply_to_message_id=reply_to_message_id,
                adapter=adapter,
            )

            result = await adapter.send(chat_id, msg, metadata=metadata)
            if result is not None and getattr(result, "success", True) is False:
                logger.debug(
                    "Failed to send shutdown notification to %s:%s: %s",
                    platform_str,
                    chat_id,
                    getattr(result, "error", "send returned success=False"),
                )
                continue

            notified.add(dedup_key)
            logger.info(
                "Sent shutdown notification to active chat %s:%s",
                platform_str, chat_id,
            )
        except Exception as e:
            logger.debug(
                "Failed to send shutdown notification to %s:%s: %s",
                platform_str, chat_id, e,
            )

    if runner._restart_requested and restart_source is not None:
        logger.debug("Skipping home-channel shutdown notifications for in-chat restart")
        return

    # Snapshot adapters up front: adapter.send() can hit a fatal error
    # path that pops the adapter from self.adapters (see _handle_fatal
    # elsewhere), which would otherwise trigger
    # ``RuntimeError: dictionary changed size during iteration`` —
    # observed in a user report during gateway shutdown.
    for platform, adapter in list(runner.adapters.items()):
        home = runner.config.get_home_channel(platform)
        if not home or not home.chat_id:
            continue

        platform_cfg = runner.config.platforms.get(platform)
        if platform_cfg is not None and not platform_cfg.gateway_restart_notification:
            logger.info(
                "Shutdown notification suppressed for home channel: %s has gateway_restart_notification=false",
                platform.value,
            )
            continue

        dedup_key = (platform.value, str(home.chat_id), str(home.thread_id) if home.thread_id else None)
        if dedup_key in notified:
            continue

        try:
            metadata = runner._thread_metadata_for_target(
                platform,
                home.chat_id,
                home.thread_id,
                adapter=adapter,
            )
            if metadata:
                result = await adapter.send(str(home.chat_id), msg, metadata=metadata)
            else:
                result = await adapter.send(str(home.chat_id), msg)
            if result is not None and getattr(result, "success", True) is False:
                logger.debug(
                    "Failed to send shutdown notification to home channel %s:%s: %s",
                    platform.value,
                    home.chat_id,
                    getattr(result, "error", "send returned success=False"),
                )
                continue

            notified.add(dedup_key)
            logger.info(
                "Sent shutdown notification to home channel %s:%s",
                platform.value,
                home.chat_id,
            )
        except Exception as e:
            logger.debug(
                "Failed to send shutdown notification to home channel %s:%s: %s",
                platform.value,
                home.chat_id,
                e,
            )


async def send_update_notification(
    runner,  # GatewayRunner instance
) -> bool:
    """If an update finished, notify the user.

    Returns False when the update is still running so a caller can retry
    later. Returns True after a definitive send/skip decision.

    This is the legacy notification path used when the streaming watcher
    cannot resolve the adapter (e.g. after a gateway restart where the
    platform hasn't reconnected yet).
    """
    import json
    import re
    from pathlib import Path

    _hermes_home = Path.home() / ".hermes"
    pending_path = _hermes_home / ".update_pending.json"
    claimed_path = _hermes_home / ".update_pending.claimed.json"
    output_path = _hermes_home / ".update_output.txt"
    exit_code_path = _hermes_home / ".update_exit_code"

    if not pending_path.exists() and not claimed_path.exists():
        return False

    cleanup = True
    active_pending_path = claimed_path
    try:
        if pending_path.exists():
            try:
                pending_path.replace(claimed_path)
            except FileNotFoundError:
                if not claimed_path.exists():
                    return True
        elif not claimed_path.exists():
            return True

        pending = json.loads(claimed_path.read_text())
        platform_str = pending.get("platform")
        chat_id = pending.get("chat_id")
        chat_type = pending.get("chat_type")
        thread_id = pending.get("thread_id")
        message_id = pending.get("message_id")

        if not exit_code_path.exists():
            logger.info("Update notification deferred: update still running")
            cleanup = False
            active_pending_path = pending_path
            claimed_path.replace(pending_path)
            return False

        exit_code_raw = exit_code_path.read_text().strip() or "1"
        exit_code = int(exit_code_raw)

        # Read the captured update output
        output = ""
        if output_path.exists():
            output = output_path.read_text()

        # Resolve adapter
        platform = Platform(platform_str)
        adapter = runner.adapters.get(platform)

        if adapter and chat_id:
            metadata = runner._thread_metadata_for_target(
                platform,
                chat_id,
                thread_id,
                chat_type=chat_type,
                reply_to_message_id=message_id,
                adapter=adapter,
            )
            # Strip ANSI escape codes for clean display
            output = re.sub(r'\x1b\[[0-9;]*m', '', output).strip()
            if output:
                if len(output) > 3500:
                    output = "…" + output[-3500:]
                if exit_code == 0:
                    msg = f"✅ Hermes update finished.\n\n```\n{output}\n```"
                else:
                    msg = f"❌ Hermes update failed.\n\n```\n{output}\n```"
            elif exit_code == 0:
                msg = "✅ Hermes update finished successfully."
            else:
                msg = "❌ Hermes update failed. Check the gateway logs or run `hermes update` manually for details."
            await adapter.send(chat_id, msg, metadata=metadata)
            logger.info(
                "Sent post-update notification to %s:%s (exit=%s)",
                platform_str,
                chat_id,
                exit_code,
            )
    except Exception as e:
        logger.warning("Post-update notification failed: %s", e)
    finally:
        if cleanup:
            active_pending_path.unlink(missing_ok=True)
            claimed_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            exit_code_path.unlink(missing_ok=True)

    return True


async def send_restart_notification(
    runner,  # GatewayRunner instance
) -> Optional[tuple[str, str, Optional[str]]]:
    """Notify the chat that initiated /restart that the gateway is back."""
    import json
    from pathlib import Path

    _hermes_home = Path.home() / ".hermes"
    notify_path = _hermes_home / ".restart_notify.json"
    if not notify_path.exists():
        return None

    try:
        data = json.loads(notify_path.read_text())
        platform_str = data.get("platform")
        chat_id = data.get("chat_id")
        chat_type = data.get("chat_type")
        thread_id = data.get("thread_id")
        message_id = data.get("message_id")

        if not platform_str or not chat_id:
            return None

        platform = Platform(platform_str)
        adapter = runner.adapters.get(platform)
        if not adapter:
            logger.debug(
                "Restart notification skipped: %s adapter not connected",
                platform_str,
            )
            return None

        platform_cfg = runner.config.platforms.get(platform)
        if platform_cfg is not None and not platform_cfg.gateway_restart_notification:
            logger.info(
                "Restart notification suppressed: %s has gateway_restart_notification=false",
                platform_str,
            )
            return None

        metadata = runner._thread_metadata_for_target(
            platform,
            chat_id,
            thread_id,
            chat_type=chat_type,
            reply_to_message_id=message_id,
            adapter=adapter,
        )
        result = await adapter.send(
            str(chat_id),
            "♻ Gateway restarted successfully. Your session continues.",
            metadata=metadata,
        )
        # adapter.send() catches provider errors (e.g. "Chat not found")
        # and returns SendResult(success=False) rather than raising, so
        # we must inspect the result before claiming success — otherwise
        # the log line is misleading and hides real delivery failures.
        if result is not None and getattr(result, "success", True) is False:
            logger.warning(
                "Restart notification to %s:%s was not delivered: %s",
                platform_str,
                chat_id,
                getattr(result, "error", "send returned success=False"),
            )
            return None

        logger.info(
            "Sent restart notification to %s:%s",
            platform_str,
            chat_id,
        )
        return str(platform_str), str(chat_id), str(thread_id) if thread_id else None
    except Exception as e:
        logger.warning("Restart notification failed: %s", e)
        return None
    finally:
        notify_path.unlink(missing_ok=True)


async def launch_detached_restart_command(
    runner,  # GatewayRunner instance
) -> None:
    """Launch a detached restart command that survives gateway exit.

    On Unix, uses setsid/bash to daemonize the restart. On Windows, spawns
    a Python watcher process that waits for current PID exit then restarts.
    """
    import shutil
    import subprocess

    hermes_cmd = _resolve_hermes_bin()
    if not hermes_cmd:
        logger.error("Could not locate hermes binary for detached /restart")
        return

    current_pid = os.getpid()

    # On Windows there's no bash/setsid chain — spawn a tiny Python
    # watcher directly via sys.executable instead.  The watcher polls
    # current_pid, waits for our exit, then runs `hermes gateway
    # restart` with detach flags so the respawn survives the CLI
    # that triggered the /restart command closing its console.
    if sys.platform == "win32":
        import textwrap
        from hermes_cli._subprocess_compat import windows_detach_popen_kwargs

        cmd_argv = [*hermes_cmd, "gateway", "restart"]
        watcher = textwrap.dedent(
            """
            import os, subprocess, sys, time
            pid = int(sys.argv[1])
            cmd = sys.argv[2:]
            deadline = time.monotonic() + 120

            def _alive(p):
                # On Windows, os.kill(pid, 0) is NOT a no-op — it maps to
                # GenerateConsoleCtrlEvent(0, pid) (bpo-14484). Use the
                # Win32 handle-based existence check instead.
                if os.name == 'nt':
                    import ctypes
                    k32 = ctypes.windll.kernel32
                    k32.OpenProcess.restype = ctypes.c_void_p
                    k32.WaitForSingleObject.restype = ctypes.c_uint
                    k32.GetLastError.restype = ctypes.c_uint
                    h = k32.OpenProcess(0x1000 | 0x100000, False, int(p))
                    if not h:
                        return k32.GetLastError() != 87
                    try:
                        return k32.WaitForSingleObject(h, 0) == 0x102
                    finally:
                        k32.CloseHandle(h)
                try:
                    os.kill(int(p), 0)
                    return True
                except ProcessLookupError:
                    return False
                except PermissionError:
                    return True
                except OSError:
                    return False

            while time.monotonic() < deadline:
                if not _alive(pid):
                    break
                time.sleep(0.2)
            _CREATE_NEW_PROCESS_GROUP = 0x00000200
            _DETACHED_PROCESS = 0x00000008
            _CREATE_NO_WINDOW = 0x08000000
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=_CREATE_NEW_PROCESS_GROUP | _DETACHED_PROCESS | _CREATE_NO_WINDOW,
            )
            """
        ).strip()
        subprocess.Popen(
            [sys.executable, "-c", watcher, str(current_pid), *cmd_argv],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **windows_detach_popen_kwargs(),
        )
        return

    cmd = " ".join(shlex.quote(part) for part in hermes_cmd)
    shell_cmd = (
        f"while kill -0 {current_pid} 2>/dev/null; do sleep 0.2; done; "
        f"{cmd} gateway restart"
    )
    setsid_bin = shutil.which("setsid")
    if setsid_bin:
        subprocess.Popen(
            [setsid_bin, "bash", "-lc", shell_cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    else:
        subprocess.Popen(
            ["bash", "-lc", shell_cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )


def launch_systemd_restart_shortcut(
    runner,  # GatewayRunner instance
) -> None:
    """Best-effort helper to bypass systemd's automatic restart delay.

    For planned in-chat restarts, the gateway exits cleanly so systemd does
    not record a failure.  However, units with RestartSteps still count
    automatic restarts and can delay repeated /restart tests.  A transient
    user service survives our cgroup teardown and explicitly starts the
    gateway as soon as this PID exits, while the unit keeps its normal
    backoff for real crash loops.
    """
    if sys.platform != "linux" or not os.environ.get("INVOCATION_ID"):
        return

    try:
        import shutil
        import subprocess

        systemd_run = shutil.which("systemd-run")
        systemctl = shutil.which("systemctl")
        if not systemd_run or not systemctl:
            return

        try:
            from hermes_cli.gateway import get_service_name

            service_name = get_service_name()
        except Exception:
            service_name = "hermes-gateway"

        current_pid = os.getpid()
        show = subprocess.run(
            [
                systemctl,
                "--user",
                "show",
                service_name,
                "--property=MainPID",
                "--value",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if (show.stdout or "").strip() != str(current_pid):
            return

        systemctl_user = "systemctl --user"
        service_arg = shlex.quote(service_name)
        shell_cmd = (
            f"while kill -0 {current_pid} 2>/dev/null; do sleep 0.2; done; "
            f"{systemctl_user} reset-failed {service_arg}; "
            f"{systemctl_user} restart {service_arg}"
        )
        unit_name = f"{service_name}-planned-restart-{current_pid}".replace(".", "-")
        subprocess.Popen(
            [
                systemd_run,
                "--user",
                "--collect",
                "--unit",
                unit_name,
                "/bin/sh",
                "-lc",
                shell_cmd,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        logger.info(
            "Launched systemd planned-restart helper for %s (pid=%s)",
            service_name,
            current_pid,
        )
    except Exception as e:
        logger.debug("Failed to launch systemd planned-restart helper: %s", e)


def is_stale_restart_redelivery(
    runner,  # GatewayRunner instance
    event,  # MessageEvent
) -> bool:
    """Return True if this /restart is a Telegram re-delivery we already handled.

    The previous gateway wrote ``.restart_last_processed.json`` with the
    triggering platform + update_id when it processed the /restart.  If
    we now see a /restart on the same platform with an update_id <= that
    recorded value AND the marker is recent (< 5 minutes), it's a
    redelivery and should be ignored.

    Only applies to Telegram today (the only platform that exposes a
    numeric cross-session update ordering); other platforms return False.
    """
    import json
    import time
    from pathlib import Path

    if event is None or event.source is None:
        return False
    if event.platform_update_id is None:
        return False
    if event.source.platform is None:
        return False
    # Only Telegram populates platform_update_id currently; be explicit
    # so future platforms aren't accidentally gated by this check.
    try:
        platform_value = event.source.platform.value
    except Exception:
        return False
    if platform_value != "telegram":
        return False

    try:
        _hermes_home = Path.home() / ".hermes"
        marker_path = _hermes_home / ".restart_last_processed.json"
        if not marker_path.exists():
            return False
        data = json.loads(marker_path.read_text())
    except Exception:
        return False

    if data.get("platform") != platform_value:
        return False
    recorded_uid = data.get("update_id")
    if not isinstance(recorded_uid, int):
        return False
    # Staleness guard: ignore markers older than 5 minutes.  A legitimately
    # old marker (e.g. crash recovery where notify never fired) should not
    # swallow a fresh /restart from the user.
    requested_at = data.get("requested_at")
    if isinstance(requested_at, (int, float)):
        if time.time() - requested_at > 300:
            return False
    return event.platform_update_id <= recorded_uid
