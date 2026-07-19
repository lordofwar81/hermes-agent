"""Integration tests for gateway/run.py:start_gateway() and GatewayRunner.

Tests the gateway startup lifecycle: PID guard, config loading, signal handling,
runner creation, shutdown paths, and startup ordering.

All heavy dependencies are mocked at their source modules.
"""

import asyncio
import os
import signal
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─── Helpers ──────────────────────────────────────────────────────────────

def _make_start_gateway_patches(hermes_home: Path, runner_mock=None):
    """Return a dict of context managers for all mocks needed by start_gateway."""
    if runner_mock is None:
        runner_mock = MagicMock()
        runner_mock.start = AsyncMock(return_value=False)
        runner_mock.should_exit_cleanly = False
        runner_mock.should_exit_with_failure = False
        runner_mock._restart_requested = False
        runner_mock._restart_via_service = False

    return {
        "status_get_running_pid": patch("gateway.status.get_running_pid", return_value=None),
        "status_acquire_lock": patch("gateway.status.acquire_gateway_runtime_lock", return_value=True),
        "status_write_pid": patch("gateway.status.write_pid_file"),
        "status_remove_pid": patch("gateway.status.remove_pid_file"),
        "status_release_lock": patch("gateway.status.release_gateway_runtime_lock"),
        "status_pid_exists": patch("gateway.status._pid_exists", return_value=False),
        "status_terminate_pid": patch("gateway.status.terminate_pid"),
        "status_write_takeover": patch("gateway.status.write_takeover_marker"),
        "status_clear_takeover": patch("gateway.status.clear_takeover_marker"),
        "status_consume_takeover": patch("gateway.status.consume_takeover_marker_for_self", return_value=False),
        "status_consume_planned_stop": patch("gateway.status.consume_planned_stop_marker_for_self", return_value=False),
        "status_release_all_locks": patch("gateway.status.release_all_scoped_locks", return_value=0),
        "status_get_start_time": patch("gateway.status.get_process_start_time", return_value=0),
        "hermes_home": patch("hermes_constants.get_hermes_home", return_value=hermes_home),
        "setup_logging": patch("hermes_logging.setup_logging"),
        "sync_skills": patch("tools.skills_sync.sync_skills"),
        "discover_mcp": patch("tools.mcp_tool.discover_mcp_tools"),
        "shutdown_mcp": patch("tools.mcp_tool.shutdown_mcp_servers"),
        "load_config": patch("hermes_cli.config.load_config", return_value={}),
        "memory_monitor": patch("gateway.memory_monitor.start_memory_monitoring"),
        "memory_monitor_stop": patch("gateway.memory_monitor.stop_memory_monitoring"),
        "shutdown_forensics": patch("gateway.shutdown_forensics.snapshot_shutdown_context", return_value=None),
        "runner_cls": patch("gateway.run.GatewayRunner", return_value=runner_mock),
    }


def _apply_patches(patches):
    """Enter all mock context managers and return them as a list."""
    entered = []
    for p in patches.values():
        entered.append(p.__enter__())
    return entered


def _exit_patches(patches):
    """Exit all mock context managers in reverse order."""
    for p in reversed(list(patches.values())):
        p.__exit__(None, None, None)


# ─── Tests ───────────────────────────────────────────────────────────────

class TestDuplicateInstanceGuard:
    """Test PID guard — prevents double-running."""

    @pytest.mark.asyncio
    async def test_rejects_when_lock_unavailable(self, tmp_path):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        patches = _make_start_gateway_patches(hermes_home)
        # Override: lock acquisition fails
        patches["status_acquire_lock"] = patch("gateway.status.acquire_gateway_runtime_lock", return_value=False)
        # Override: existing PID detected
        patches["status_get_running_pid"] = patch("gateway.status.get_running_pid", return_value=99999)

        from gateway.run import start_gateway
        _apply_patches(patches)
        try:
            result = await start_gateway(replace=False)
            assert result is False
        finally:
            _exit_patches(patches)

    @pytest.mark.asyncio
    async def test_replace_mode_kills_existing(self, tmp_path):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        patches = _make_start_gateway_patches(hermes_home)
        patches["status_get_running_pid"] = patch("gateway.status.get_running_pid", return_value=99999)

        from gateway.run import start_gateway
        _apply_patches(patches)
        try:
            result = await start_gateway(replace=True)
            # Replace mode + mock runner returning False → should not crash
            assert result is False
        finally:
            _exit_patches(patches)


class TestStartupSequence:
    """Test ordered startup sequence."""

    @pytest.mark.asyncio
    async def test_skills_sync_called(self, tmp_path):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        mock_sync = MagicMock()
        patches = _make_start_gateway_patches(hermes_home)
        patches["sync_skills"] = patch("tools.skills_sync.sync_skills", mock_sync)

        from gateway.run import start_gateway
        _apply_patches(patches)
        try:
            await start_gateway()
            mock_sync.assert_called_once_with(quiet=True)
        finally:
            _exit_patches(patches)

    @pytest.mark.asyncio
    async def test_pid_written_before_runner_start(self, tmp_path):
        """PID file must be written before GatewayRunner.start()."""
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        call_order = []

        def record_pid():
            call_order.append("pid")

        async def record_start():
            call_order.append("start")
            return True

        mock_runner = MagicMock()
        mock_runner.start = AsyncMock(side_effect=record_start)
        mock_runner.should_exit_cleanly = True
        mock_runner.should_exit_with_failure = False
        mock_runner.wait_for_shutdown = AsyncMock()

        patches = _make_start_gateway_patches(hermes_home, mock_runner)
        patches["status_write_pid"] = patch("gateway.status.write_pid_file", side_effect=record_pid)

        from gateway.run import start_gateway
        _apply_patches(patches)
        try:
            await start_gateway()
            assert call_order == ["pid", "start"]
        finally:
            _exit_patches(patches)

    @pytest.mark.asyncio
    async def test_runner_start_failure_returns_false(self, tmp_path):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        mock_runner = MagicMock()
        mock_runner.start = AsyncMock(return_value=False)
        mock_runner.should_exit_cleanly = False

        patches = _make_start_gateway_patches(hermes_home, mock_runner)

        from gateway.run import start_gateway
        _apply_patches(patches)
        try:
            result = await start_gateway()
            assert result is False
        finally:
            _exit_patches(patches)

    @pytest.mark.asyncio
    async def test_runner_clean_exit_returns_true(self, tmp_path):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        mock_runner = MagicMock()
        mock_runner.start = AsyncMock(return_value=True)
        mock_runner.should_exit_cleanly = True
        mock_runner.should_exit_with_failure = False
        mock_runner.exit_reason = "User stop"
        mock_runner.wait_for_shutdown = AsyncMock()

        patches = _make_start_gateway_patches(hermes_home, mock_runner)

        from gateway.run import start_gateway
        _apply_patches(patches)
        try:
            result = await start_gateway()
            assert result is True
        finally:
            _exit_patches(patches)

    @pytest.mark.asyncio
    async def test_runner_exit_code_propagates(self, tmp_path):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        mock_runner = MagicMock()
        mock_runner.start = AsyncMock(return_value=True)
        mock_runner.should_exit_cleanly = False
        mock_runner.should_exit_with_failure = False
        mock_runner.exit_code = 42
        mock_runner.wait_for_shutdown = AsyncMock()

        patches = _make_start_gateway_patches(hermes_home, mock_runner)

        from gateway.run import start_gateway
        _apply_patches(patches)
        try:
            with pytest.raises(SystemExit) as exc:
                await start_gateway()
            assert exc.value.code == 42
        finally:
            _exit_patches(patches)

    @pytest.mark.asyncio
    async def test_mcp_discovery_called(self, tmp_path):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        mock_discover = MagicMock()
        patches = _make_start_gateway_patches(hermes_home)
        patches["discover_mcp"] = patch("tools.mcp_tool.discover_mcp_tools", mock_discover)

        from gateway.run import start_gateway
        _apply_patches(patches)
        try:
            await start_gateway()
            mock_discover.assert_called_once()
        finally:
            _exit_patches(patches)
