"""
Gateway utility modules.

This package contains utility functions extracted from gateway/run.py
to improve code organization and maintainability.

Modules:
- gateway_helpers: Platform validation, error handling, secret redaction
- config_resolvers: Configuration loading and environment variable resolution
- message_builders: Message construction and formatting utilities
- status_helpers: Status message handling and notification utilities
"""

from gateway.utils.gateway_helpers import (
    # Platform and validation
    _gateway_platform_value,
    _is_transient_network_error,
    _gateway_loop_exception_handler,

    # Secret and error handling
    _redact_gateway_user_facing_secrets,
    _gateway_provider_error_reply,
    _looks_like_gateway_provider_error,
    _sanitize_gateway_final_response,

    # Telegram-specific
    _telegramize_command_mentions,

    # Timestamp and duration
    _coerce_gateway_timestamp,
    _format_duration,
    _probe_audio_duration,

    # Control messages
    _is_control_interrupt_message,

    # Process notifications
    _format_gateway_process_notification,

    # Status helpers
    _prepare_gateway_status_message,
    _send_or_update_status_coro,

    # Constants
    _GATEWAY_PROVIDER_ERROR_SHAPE_RE,
    _GATEWAY_SECRET_PATTERNS,
    _CONTROL_INTERRUPT_MESSAGES,
    _INTERRUPT_REASON_STOP,
    _INTERRUPT_REASON_RESET,
    _INTERRUPT_REASON_TIMEOUT,
    _INTERRUPT_REASON_SSE_DISCONNECT,
    _INTERRUPT_REASON_GATEWAY_SHUTDOWN,
    _INTERRUPT_REASON_GATEWAY_RESTART,
)

from gateway.utils.config_resolvers import (
    _float_env,
    _resolve_hermes_bin,
    _auto_continue_freshness_window,
    _gateway_agent_timeout,
    _gateway_model_provider,
    _gateway_fallback_model,
    _get_gateway_platform_value,
)

from gateway.utils.message_builders import (
    _build_replay_entry,
    _build_gateway_agent_history,
    _wrap_current_message_with_observed_context,
    _build_media_placeholder,
    _build_media_collection_placeholders,
    _normalize_empty_agent_response,
    _format_gateway_process_notification,
    _skill_slug_from_frontmatter,
)

from gateway.utils.status_helpers import (
    _prepare_gateway_status_message as _prepare_status_message,
    _send_or_update_status_coro as _send_status,
    _last_transcript_timestamp,
    _uses_telegram_observed_group_context,
    _dequeue_pending_event,
    _build_status_notification,
)

__all__ = [
    # gateway_helpers
    "_gateway_platform_value",
    "_is_transient_network_error",
    "_gateway_loop_exception_handler",
    "_redact_gateway_user_facing_secrets",
    "_gateway_provider_error_reply",
    "_looks_like_gateway_provider_error",
    "_sanitize_gateway_final_response",
    "_telegramize_command_mentions",
    "_coerce_gateway_timestamp",
    "_format_duration",
    "_probe_audio_duration",
    "_is_control_interrupt_message",
    "_format_gateway_process_notification",
    "_prepare_gateway_status_message",
    "_send_or_update_status_coro",
    "_GATEWAY_PROVIDER_ERROR_SHAPE_RE",
    "_GATEWAY_SECRET_PATTERNS",
    "_CONTROL_INTERRUPT_MESSAGES",
    "_INTERRUPT_REASON_STOP",
    "_INTERRUPT_REASON_RESET",
    "_INTERRUPT_REASON_TIMEOUT",
    "_INTERRUPT_REASON_SSE_DISCONNECT",
    "_INTERRUPT_REASON_GATEWAY_SHUTDOWN",
    "_INTERRUPT_REASON_GATEWAY_RESTART",
    # config_resolvers
    "_float_env",
    "_resolve_hermes_bin",
    "_auto_continue_freshness_window",
    "_gateway_agent_timeout",
    "_gateway_model_provider",
    "_gateway_fallback_model",
    "_get_gateway_platform_value",
    # message_builders
    "_build_replay_entry",
    "_build_gateway_agent_history",
    "_wrap_current_message_with_observed_context",
    "_build_media_placeholder",
    "_build_media_collection_placeholders",
    "_normalize_empty_agent_response",
    "_format_gateway_process_notification",
    "_skill_slug_from_frontmatter",
    # status_helpers
    "_prepare_status_message",
    "_send_status",
    "_last_transcript_timestamp",
    "_uses_telegram_observed_group_context",
    "_dequeue_pending_event",
    "_build_status_notification",
]
