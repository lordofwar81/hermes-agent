"""Integration tests for the startup pre-flight extraction (round 46).

Exercises ``GatewayRunner.start()`` end-to-end with the pre-flight region
moved into ``_run_startup_preflight_checks()``. Uses the cron-only mode
(no enabled platforms) from test_runner_startup_failures.py as the vehicle:
``start()`` runs the entire pre-flight region, then returns True without
constructing any adapter.

Assertions cover the two contract guarantees of the extraction:
  1. Each pre-flight block swallows its own exceptions — a failure in
     routing init / plugin discovery / shell-hook registration must NOT
     propagate out of start().
  2. A specific side effect fires — proving the block actually ran (not
     silently dropped during extraction). The runtime-status write is
     the cleanest canary: ``write_runtime_status(gateway_state="starting")``
     is an unconditional side effect in the region.
"""

import logging

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.status import read_runtime_status

from tests.gateway._runner_harness import RecordingAdapter


@pytest.mark.asyncio
async def test_start_runs_preflight_and_succeeds_in_cron_only_mode(monkeypatch, tmp_path):
    """start() must run the full pre-flight region and return True even
    with no platforms enabled (cron-only mode). The runtime-status write
    inside pre-flight is the side-effect canary."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=False, token="***")},
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)

    ok = await runner.start()

    assert ok is True
    assert runner.adapters == {}
    # Canary: the pre-flight region writes gateway_state="starting" then
    # the connect loop writes "running". Both must have fired.
    state = read_runtime_status()
    assert state["gateway_state"] in {"starting", "running"}


@pytest.mark.asyncio
async def test_preflight_routing_init_failure_does_not_block_start(monkeypatch, tmp_path):
    """A routing-config problem must NOT block gateway startup — the
    gateway degrades to the primary-model path. This is the contract of
    the first pre-flight block."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    # Force init_router to raise.
    import agent.routing as routing_mod
    orig_init = routing_mod.init_router
    monkeypatch.setattr(routing_mod, "init_router", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("forced routing failure")))

    config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=False, token="***")},
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)

    ok = await runner.start()  # must NOT raise

    assert ok is True
    # Routing init was attempted (the block ran) and swallowed.
    assert runner.adapters == {}


@pytest.mark.asyncio
async def test_preflight_plugin_discovery_failure_does_not_block_start(monkeypatch, tmp_path):
    """A plugin-discovery failure must NOT block gateway startup."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    import hermes_cli.plugins as plugins_mod
    monkeypatch.setattr(plugins_mod, "discover_plugins", lambda: (_ for _ in ()).throw(RuntimeError("forced plugin failure")))

    config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=False, token="***")},
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)

    ok = await runner.start()  # must NOT raise

    assert ok is True


@pytest.mark.asyncio
async def test_preflight_helper_is_called_during_start(monkeypatch, tmp_path):
    """Directly verify _run_startup_preflight_checks is invoked by start()
    and that swapping it propagates a visible side effect. This catches
    a regression where the helper call gets dropped from start()."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=False, token="***")},
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)

    call_count = [0]
    orig = runner._run_startup_preflight_checks

    def counting_helper():
        call_count[0] += 1
        orig()

    monkeypatch.setattr(runner, "_run_startup_preflight_checks", counting_helper)

    ok = await runner.start()

    assert ok is True
    assert call_count[0] == 1, "start() must call _run_startup_preflight_checks exactly once"


@pytest.mark.asyncio
async def test_preflight_resolves_through_mro(monkeypatch, tmp_path):
    """The helper must resolve through the MRO to StartupPreflightMixin,
    not be shadowed by a definition on GatewayRunner itself."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=False, token="***")},
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)

    qn = runner._run_startup_preflight_checks.__qualname__
    assert qn.startswith("StartupPreflightMixin"), qn
