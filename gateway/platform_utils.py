"""
Platform utility functions for GatewayRunner.

Extracted from gateway/run.py to reduce the God file size.
Provides functions for platform-specific operations.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gateway.run import GatewayRunner

logger = logging.getLogger(__name__)


async def deliver_platform_notice(
    runner,  # GatewayRunner instance
    source,
    content: str,
) -> None:
    """Deliver a setup/operational notice using platform-specific privacy rules."""
    adapter = runner.adapters.get(source.platform)
    if not adapter:
        return

    config = getattr(runner, "config", None)
    notice_delivery = "public"
    if config and hasattr(config, "get_notice_delivery"):
        notice_delivery = config.get_notice_delivery(source.platform)

    metadata = runner._thread_metadata_for_source(source)
    if notice_delivery == "private" and getattr(source, "user_id", None):
        try:
            result = await adapter.send_private_notice(
                source.chat_id,
                source.user_id,
                content,
                metadata=metadata,
            )
            if getattr(result, "success", False):
                return
        except Exception:
            logger.debug(
                "[%s] send_private_notice failed, falling back to public",
                getattr(source, "platform", "?"),
                exc_info=True,
            )

    await adapter.send(source.chat_id, content, metadata=metadata)


def adapter_enforces_own_access_policy(
    runner,  # GatewayRunner instance
    platform,  # Optional[Platform]
) -> bool:
    """Whether the adapter for *platform* gates access at intake itself.

    Mirrors ``BasePlatformAdapter.enforces_own_access_policy``. Adapters
    such as WeCom, Weixin, Yuanbao, QQBot, and WhatsApp evaluate their
    documented ``dm_policy`` / ``group_policy`` / ``allow_from`` config before a
    message is dispatched to the gateway, so a message that reaches
    ``_is_user_authorized`` has already been authorized by the adapter.
    Defaults to ``False`` when the adapter is unknown or doesn't expose
    the flag.
    """
    if not platform:
        return False
    # Some test helpers build a bare GatewayRunner via object.__new__ and
    # never set ``adapters``; treat a missing/empty map as "no adapter"
    # rather than raising (see pitfalls.md #17).
    adapters = getattr(runner, "adapters", None)
    if not adapters:
        return False
    adapter = adapters.get(platform)
    if adapter is None:
        return False
    return bool(getattr(adapter, "enforces_own_access_policy", False))
