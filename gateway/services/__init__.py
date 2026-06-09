"""
Gateway background services.

This package contains standalone service modules that run as background threads
within the gateway process. Each service is responsible for a specific background
task like cron scheduling or shutdown watching.

Services are started by GatewayRunner during gateway initialization and stopped
gracefully during shutdown.
"""

from gateway.services.cron_service import CronService, start_cron_ticker
from gateway.services.planned_stop_watcher import (
    PlannedStopWatcher,
    run_planned_stop_watcher,
)

__all__ = [
    "CronService",
    "start_cron_ticker",
    "PlannedStopWatcher",
    "run_planned_stop_watcher",
]
