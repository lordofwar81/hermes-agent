"""
Adapter factory for creating platform adapters.

Extracted from GatewayRunner._create_adapter to provide a focused module
for platform adapter instantiation.
"""

import logging
import os
from typing import Any, Optional

from gateway.platforms.base import BasePlatformAdapter
from hermes_shared import Platform
from hermes_cli.config import cfg_get


logger = logging.getLogger(__name__)


def _load_gateway_config():
    """Lazy load gateway config to avoid circular imports."""
    from gateway.config import load_gateway_config
    return load_gateway_config()


def create_adapter(
    runner,
    platform: Platform,
    config: Any,
) -> Optional[BasePlatformAdapter]:
    """Create the appropriate adapter for a platform.

    Checks the platform_registry first (plugin adapters), then falls
    through to the built-in if/elif chain for core platforms.
    """
    if hasattr(config, "extra") and isinstance(config.extra, dict):
        config.extra.setdefault(
            "group_sessions_per_user",
            runner.config.group_sessions_per_user,
        )
        config.extra.setdefault(
            "thread_sessions_per_user",
            getattr(runner.config, "thread_sessions_per_user", False),
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
                    adapter.gateway_runner = runner
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
        from gateway.platforms.telegram import TelegramAdapter, check_telegram_requirements
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
        from gateway.platforms.whatsapp import WhatsAppAdapter, check_whatsapp_requirements
        if not check_whatsapp_requirements():
            logger.warning("WhatsApp: Node.js not installed or bridge not configured")
            return None
        return WhatsAppAdapter(config)

    elif platform == Platform.SLACK:
        from gateway.platforms.slack import SlackAdapter, check_slack_requirements
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

    elif platform == Platform.HOMEASSISTANT:
        from gateway.platforms.homeassistant import HomeAssistantAdapter, check_ha_requirements
        if not check_ha_requirements():
            logger.warning("HomeAssistant: aiohttp not installed or HASS_TOKEN not set")
            return None
        return HomeAssistantAdapter(config)

    elif platform == Platform.EMAIL:
        from gateway.platforms.email import EmailAdapter, check_email_requirements
        if not check_email_requirements():
            logger.warning("Email: EMAIL_ADDRESS, EMAIL_PASSWORD, EMAIL_IMAP_HOST, or EMAIL_SMTP_HOST not set")
            return None
        return EmailAdapter(config)

    elif platform == Platform.SMS:
        from gateway.platforms.sms import SmsAdapter, check_sms_requirements
        if not check_sms_requirements():
            logger.warning("SMS: aiohttp not installed or TWILIO_ACCOUNT_SID/TWILION_AUTH_TOKEN not set")
            return None
        return SmsAdapter(config)

    elif platform == Platform.DINGTALK:
        from gateway.platforms.dingtalk import DingTalkAdapter, check_dingtalk_requirements
        if not check_dingtalk_requirements():
            logger.warning("DingTalk: dingtalk-stream not installed or DINGTALK_CLIENT_ID/SECRET not set")
            return None
        return DingTalkAdapter(config)

    elif platform == Platform.FEISHU:
        from gateway.platforms.feishu import FeishuAdapter, check_feishu_requirements
        if not check_feishu_requirements():
            logger.warning("Feishu: lark-oapi not installed or FEISHU_APP_ID/SECRET not set")
            return None
        return FeishuAdapter(config)

    elif platform == Platform.WECOM_CALLBACK:
        from gateway.platforms.wecom_callback import (
            WecomCallbackAdapter,
            check_wecom_callback_requirements,
        )
        if not check_wecom_callback_requirements():
            logger.warning("WeComCallback: aiohttp/httpx/defusedxml not installed")
            return None
        return WecomCallbackAdapter(config)

    elif platform == Platform.WECOM:
        from gateway.platforms.wecom import WeComAdapter, check_wecom_requirements
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
        from gateway.platforms.matrix import MatrixAdapter, check_matrix_requirements
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
        adapter.gateway_runner = runner  # For cross-platform delivery
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
