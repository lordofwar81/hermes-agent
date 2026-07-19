"""Async + contextvar helpers extracted from GatewayRunner.

Round 14 of gateway decomposition. _run_in_executor_with_context runs blocking
work in the thread pool preserving contextvars; _safe_adapter_disconnect and
_connect_adapter_with_timeout wrap adapter connect/disconnect with timeouts.
All stateless; timeouts read from gateway_gateway_env.
"""

import asyncio
from contextvars import copy_context
from typing import Any

import logging
logger = logging.getLogger("gateway.run")


async def _run_in_executor_with_context(func, *args):
    """Run blocking work in the thread pool while preserving session contextvars."""
    loop = asyncio.get_running_loop()
    ctx = copy_context()
    return await loop.run_in_executor(None, ctx.run, func, *args)


async def _safe_adapter_disconnect(adapter, platform) -> None:
    """Call adapter.disconnect() defensively, swallowing any error.

    Used when adapter.connect() failed or raised — the adapter may
    have allocated partial resources (aiohttp.ClientSession, poll
    tasks, child subprocesses) that would otherwise leak and surface
    as "Unclosed client session" warnings at process exit.

    Must tolerate partial-init state and never raise, since callers
    use it inside error-handling blocks.
    """
    from gateway.gateway_gateway_env import _adapter_disconnect_timeout_secs
    timeout = _adapter_disconnect_timeout_secs()
    try:
        if timeout <= 0:
            await adapter.disconnect()
        else:
            await asyncio.wait_for(adapter.disconnect(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(
            "Timed out after %.1fs while disconnecting %s adapter; continuing shutdown",
            timeout,
            platform.value if platform is not None else "adapter",
        )
    except Exception as e:
        logger.debug(
            "Defensive %s disconnect after failed connect raised: %s",
            platform.value if platform is not None else "adapter",
            e,
        )


async def _connect_adapter_with_timeout(adapter, platform) -> bool:
    """Connect an adapter without allowing one platform to block others."""
    from gateway.gateway_gateway_env import _platform_connect_timeout_secs
    timeout = _platform_connect_timeout_secs()
    if timeout <= 0:
        return await adapter.connect()
    try:
        return await asyncio.wait_for(adapter.connect(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"{platform.value} connect timed out after {timeout:g}s"
        ) from exc
