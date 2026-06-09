"""
Gateway exit state management.

This module provides functions for accessing gateway exit state
and status information.
"""

from typing import Any, Optional


def should_exit_cleanly(runner: Any) -> bool:
    """Check if gateway should exit cleanly.

    Args:
        runner: GatewayRunner instance

    Returns:
        True if gateway should exit with success status
    """
    return getattr(runner, "_exit_cleanly", False)


def should_exit_with_failure(runner: Any) -> bool:
    """Check if gateway should exit with failure.

    Args:
        runner: GatewayRunner instance

    Returns:
        True if gateway should exit with failure status
    """
    return getattr(runner, "_exit_with_failure", False)


def exit_reason(runner: Any) -> Optional[str]:
    """Get the reason for gateway exit.

    Args:
        runner: GatewayRunner instance

    Returns:
        Exit reason string, or None if not set
    """
    return getattr(runner, "_exit_reason", None)


def exit_code(runner: Any) -> Optional[int]:
    """Get the gateway exit code.

    Args:
        runner: GatewayRunner instance

    Returns:
        Exit code integer, or None if not set
    """
    return getattr(runner, "_exit_code", None)
