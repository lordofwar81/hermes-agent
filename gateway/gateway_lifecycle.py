"""Gateway lifecycle helpers: restart + agent resource cleanup.

Round 11 of gateway decomposition. Extracted from GatewayRunner. These
methods handle planned /restart (detached watcher processes, systemd shortcut)
and best-effort cleanup of temporary/cached agent resources. All stateless —
no instance state touched. _launch_detached_restart_command stays async.

Deps on run.py module globals (_resolve_hermes_bin, _hermes_home) are
imported lazily to avoid circular imports at module load.
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from gateway.platforms.base import MessageEvent

import logging
logger = logging.getLogger("gateway.run")


def _cleanup_agent_resources(agent: Any) -> None:
    """Best-effort cleanup for temporary or cached agent instances."""
    if agent is None:
        return
    try:
        if hasattr(agent, "shutdown_memory_provider"):
            # Pass the agent's own conversation transcript so memory
            # providers' ``on_session_end`` hooks see the real messages
            # instead of the empty default (#15165). ``_session_messages``
            # is set on ``AIAgent`` (run_agent.py:1518) and refreshed at
            # the end of every ``run_conversation`` turn via
            # ``_persist_session``; on an agent built through
            # ``object.__new__`` (test stubs) the attribute may be
            # absent, so ``getattr`` with a ``None`` default keeps the
            # call signature-compatible with the pre-fix behaviour
            # (``shutdown_memory_provider(messages=None)``).
            session_messages = getattr(agent, "_session_messages", None)
            if isinstance(session_messages, list):
                agent.shutdown_memory_provider(session_messages)
            else:
                agent.shutdown_memory_provider()
    except Exception:
        pass
    # Close tool resources (terminal sandboxes, browser daemons,
    # background processes, httpx clients) to prevent zombie
    # process accumulation.
    try:
        if hasattr(agent, "close"):
            agent.close()
    except Exception:
        pass
    # Auxiliary async clients (session_search/web/vision/etc.) live in a
    # process-global cache and are created inside worker threads. Clean up
    # any entries whose event loop is now dead so their httpx transports do
    # not accumulate across gateway turns.
    try:
        from agent.auxiliary_client import cleanup_stale_async_clients
        cleanup_stale_async_clients()
    except Exception:
        pass


async def _launch_detached_restart_command() -> None:
    import shutil
    import subprocess

    from gateway.run import _resolve_hermes_bin
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
        watcher_env = os.environ.copy()
        # This watcher is intentionally outside the running gateway. If it
        # inherits the gateway marker, `hermes gateway restart` refuses to
        # run as a self-restart loop guard and the gateway stays stopped.
        watcher_env.pop("_HERMES_GATEWAY", None)
        project_root = Path(__file__).resolve().parent.parent
        venv_dir = Path(watcher_env.get("VIRTUAL_ENV") or project_root / "venv")
        site_packages = venv_dir / "Lib" / "site-packages"
        if site_packages.exists():
            watcher_env["VIRTUAL_ENV"] = str(venv_dir)
            pythonpath = [str(project_root), str(site_packages)]
            if watcher_env.get("PYTHONPATH"):
                pythonpath.append(watcher_env["PYTHONPATH"])
            watcher_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath))
        subprocess.Popen(
            [sys.executable, "-c", watcher, str(current_pid), *cmd_argv],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=watcher_env,
            **windows_detach_popen_kwargs(),
        )
        return

    cmd = " ".join(shlex.quote(part) for part in hermes_cmd)
    shell_cmd = (
        f"while kill -0 {current_pid} 2>/dev/null; do sleep 0.2; done; "
        f"{cmd} gateway restart"
    )
    # Same marker scrub as the Windows watcher above: this watcher runs
    # `hermes gateway restart` from outside the gateway, but it inherits
    # _HERMES_GATEWAY=1 from us, and the CLI's self-restart loop guard
    # refuses to run when that marker is set — silently (DEVNULL), so the
    # gateway stops and never comes back.
    watcher_env = os.environ.copy()
    watcher_env.pop("_HERMES_GATEWAY", None)
    setsid_bin = shutil.which("setsid")
    if setsid_bin:
        subprocess.Popen(
            [setsid_bin, "bash", "-lc", shell_cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=watcher_env,
            start_new_session=True,
        )
    else:
        subprocess.Popen(
            ["bash", "-lc", shell_cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=watcher_env,
            start_new_session=True,
        )


def _launch_systemd_restart_shortcut() -> None:
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


def _is_stale_restart_redelivery(event: MessageEvent) -> bool:
    """Return True if this /restart is a Telegram re-delivery we already handled.

    The previous gateway wrote ``.restart_last_processed.json`` with the
    triggering platform + update_id when it processed the /restart.  If
    we now see a /restart on the same platform with an update_id <= that
    recorded value AND the marker is recent (< 5 minutes), it's a
    redelivery and should be ignored.

    Only applies to Telegram today (the only platform that exposes a
    numeric cross-session update ordering); other platforms return False.
    """
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
        from gateway.run import _hermes_home
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
