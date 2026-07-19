"""Platform-adapter factory method for ``GatewayRunner``.

Round 36 of the god-file decomposition. Lifted verbatim from
gateway/run.py into gateway/create_adapter_mixin.py.

``_create_adapter`` is the platform-adapter factory: it checks the
plugin platform_registry first, then falls through to a built-in if/elif
chain over the core platforms, importing each adapter's module in-body
(and running its requirements check) only on the matching branch so a
missing optional dependency for one platform never blocks the others.

``self.*`` references resolve unchanged via the MRO. Behavior-neutral
lift matching the existing mixin pattern.

``gateway.run`` module-level runtime globals and free functions
(``logger``, ``_load_gateway_config``) are lazy-imported at the top of
the method body to avoid the circular import (``gateway.run`` imports
this mixin at module top). Stdlib (``os``), types (``Any``,
``Optional``), and non-circular module symbols (``BasePlatformAdapter``,
``Platform``, ``cfg_get``) are imported at module top. Every platform
adapter class, requirements checker, ``platform_registry``, and
``WEBSOCKETS_AVAILABLE`` are imported in-body within their matching
branch (already lazy in source) and kept verbatim.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter
from hermes_cli.config import cfg_get


class CreateAdapterMixin:
    def _create_adapter(
        self,
        platform: Platform,
        config: Any
    ) -> Optional[BasePlatformAdapter]:
        """Create the appropriate adapter for a platform.

        Checks the platform_registry first (plugin adapters), then falls
        through to the built-in if/elif chain for core platforms.
        """
        from gateway.run import _load_gateway_config, logger

        if hasattr(config, "extra") and isinstance(config.extra, dict):
            config.extra.setdefault(
                "group_sessions_per_user",
                self.config.group_sessions_per_user,
            )
            config.extra.setdefault(
                "thread_sessions_per_user",
                getattr(self.config, "thread_sessions_per_user", False),
            )

        # ── Plugin-registered platforms (checked first) ───────────────────
        try:
            from gateway.platform_registry import platform_registry
            if platform_registry.is_registered(platform.value):
                adapter = platform_registry.create_adapter(platform.value, config)
                if adapter is not None:
                    # Adapters that need a back-reference to the gateway runner
                    # (e.g. for cross-platform admin alerts) declare a
                    # ``gateway_runner`` attribute. Inject it after creation so
                    # plugin adapters don't need a custom factory signature.
                    if hasattr(adapter, "gateway_runner"):
                        adapter.gateway_runner = self
                    return adapter
                # Registered but failed to instantiate — don't silently fall
                # through to built-ins (there are none for plugin platforms).
                logger.error(
                    "Platform '%s' is registered but adapter creation failed "
                    "(check dependencies and config)",
                    platform.value,
                )
                return None
        except Exception as e:
            logger.debug("Platform registry lookup for '%s' failed: %s", platform.value, e)
        # Fall through to built-in adapters below

        if platform == Platform.TELEGRAM:
            from plugins.platforms.telegram.adapter import TelegramAdapter, check_telegram_requirements
            if not check_telegram_requirements():
                logger.warning("Telegram: python-telegram-bot not installed")
                return None
            adapter = TelegramAdapter(config)
            # Apply Telegram notification mode from config.  Controls whether
            # intermediate messages (tool progress, streaming, status) trigger
            # push notifications.  Supports ENV override for quick testing.
            _notify_mode = os.getenv("HERMES_TELEGRAM_NOTIFICATIONS", "")
            if not _notify_mode:
                try:
                    _gw_cfg = _load_gateway_config()
                    _raw = cfg_get(_gw_cfg, "display", "platforms", "telegram", "notifications")
                    if _raw not in {None, ""}:
                        _notify_mode = str(_raw).strip().lower()
                except Exception:
                    pass
            _notify_mode = _notify_mode or "important"
            if _notify_mode not in {"all", "important"}:
                logger.warning(
                    "Unknown telegram notifications mode '%s', "
                    "defaulting to 'important' (valid: all, important)",
                    _notify_mode,
                )
                _notify_mode = "important"
            adapter._notifications_mode = _notify_mode
            return adapter

        elif platform == Platform.WHATSAPP:
            from plugins.platforms.whatsapp.adapter import WhatsAppAdapter, check_whatsapp_requirements
            if not check_whatsapp_requirements():
                logger.warning("WhatsApp: Node.js not installed or bridge not configured")
                return None
            return WhatsAppAdapter(config)

        elif platform == Platform.WHATSAPP_CLOUD:
            from gateway.platforms.whatsapp_cloud import (
                WhatsAppCloudAdapter,
                check_whatsapp_cloud_requirements,
            )
            if not check_whatsapp_cloud_requirements():
                logger.warning(
                    "WhatsApp Cloud: aiohttp/httpx missing — reinstall hermes-agent"
                )
                return None
            return WhatsAppCloudAdapter(config)

        elif platform == Platform.SLACK:
            from plugins.platforms.slack.adapter import SlackAdapter, check_slack_requirements
            if not check_slack_requirements():
                logger.warning("Slack: slack-bolt not installed. Run: pip install 'hermes-agent[slack]'")
                return None
            return SlackAdapter(config)

        elif platform == Platform.SIGNAL:
            from gateway.platforms.signal import SignalAdapter, check_signal_requirements
            if not check_signal_requirements():
                logger.warning("Signal: SIGNAL_HTTP_URL or SIGNAL_ACCOUNT not configured")
                return None
            return SignalAdapter(config)

        elif platform == Platform.EMAIL:
            from plugins.platforms.email.adapter import EmailAdapter, check_email_requirements
            if not check_email_requirements():
                logger.warning("Email: EMAIL_ADDRESS, EMAIL_PASSWORD, EMAIL_IMAP_HOST, or EMAIL_SMTP_HOST not set")
                return None
            return EmailAdapter(config)

        elif platform == Platform.SMS:
            from plugins.platforms.sms.adapter import SmsAdapter, check_sms_requirements
            if not check_sms_requirements():
                logger.warning("SMS: aiohttp not installed or TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN not set")
                return None
            return SmsAdapter(config)

        elif platform == Platform.DINGTALK:
            from plugins.platforms.dingtalk.adapter import DingTalkAdapter, check_dingtalk_requirements
            if not check_dingtalk_requirements():
                logger.warning("DingTalk: dingtalk-stream not installed or DINGTALK_CLIENT_ID/SECRET not set")
                return None
            return DingTalkAdapter(config)

        elif platform == Platform.FEISHU:
            from plugins.platforms.feishu.adapter import FeishuAdapter, check_feishu_requirements
            if not check_feishu_requirements():
                logger.warning("Feishu: lark-oapi not installed or FEISHU_APP_ID/SECRET not set")
                return None
            return FeishuAdapter(config)

        elif platform == Platform.WECOM_CALLBACK:
            from plugins.platforms.wecom.callback_adapter import (
                WecomCallbackAdapter,
                check_wecom_callback_requirements,
            )
            if not check_wecom_callback_requirements():
                logger.warning("WeComCallback: aiohttp/httpx/defusedxml not installed")
                return None
            return WecomCallbackAdapter(config)

        elif platform == Platform.WECOM:
            from plugins.platforms.wecom.adapter import WeComAdapter, check_wecom_requirements
            if not check_wecom_requirements():
                logger.warning("WeCom: aiohttp not installed or WECOM_BOT_ID/SECRET not set")
                return None
            return WeComAdapter(config)

        elif platform == Platform.WEIXIN:
            from gateway.platforms.weixin import WeixinAdapter, check_weixin_requirements
            if not check_weixin_requirements():
                logger.warning("Weixin: aiohttp/cryptography not installed")
                return None
            return WeixinAdapter(config)

        elif platform == Platform.MATRIX:
            from plugins.platforms.matrix.adapter import MatrixAdapter, check_matrix_requirements
            if not check_matrix_requirements():
                logger.warning("Matrix: mautrix not installed or credentials not set. Run: pip install 'mautrix[encryption]'")
                return None
            return MatrixAdapter(config)

        elif platform == Platform.API_SERVER:
            from gateway.platforms.api_server import APIServerAdapter, check_api_server_requirements
            if not check_api_server_requirements():
                logger.warning("API Server: aiohttp not installed")
                return None
            return APIServerAdapter(config)

        elif platform == Platform.WEBHOOK:
            from gateway.platforms.webhook import WebhookAdapter, check_webhook_requirements
            if not check_webhook_requirements():
                logger.warning("Webhook: aiohttp not installed")
                return None
            adapter = WebhookAdapter(config)
            adapter.gateway_runner = self  # For cross-platform delivery
            return adapter

        elif platform == Platform.MSGRAPH_WEBHOOK:
            from gateway.platforms.msgraph_webhook import (
                MSGraphWebhookAdapter,
                check_msgraph_webhook_requirements,
            )
            if not check_msgraph_webhook_requirements():
                logger.warning("MSGraph webhook: aiohttp not installed")
                return None
            return MSGraphWebhookAdapter(config)

        elif platform == Platform.BLUEBUBBLES:
            from gateway.platforms.bluebubbles import BlueBubblesAdapter, check_bluebubbles_requirements
            if not check_bluebubbles_requirements():
                logger.warning("BlueBubbles: aiohttp/httpx missing or BLUEBUBBLES_SERVER_URL/BLUEBUBBLES_PASSWORD not configured")
                return None
            return BlueBubblesAdapter(config)

        elif platform == Platform.QQBOT:
            from gateway.platforms.qqbot import QQAdapter, check_qq_requirements
            if not check_qq_requirements():
                logger.warning("QQBot: aiohttp/httpx missing or QQ_APP_ID/QQ_CLIENT_SECRET not configured")
                return None
            return QQAdapter(config)

        elif platform == Platform.YUANBAO:
            from gateway.platforms.yuanbao import YuanbaoAdapter, WEBSOCKETS_AVAILABLE
            if not WEBSOCKETS_AVAILABLE:
                logger.warning("Yuanbao: websockets not installed. Run: pip install websockets")
                return None
            return YuanbaoAdapter(config)

        return None
