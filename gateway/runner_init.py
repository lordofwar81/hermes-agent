"""
GatewayRunner initialization helpers.

Extracted from gateway/run.py __init__ method to improve modularity.
This module contains helper functions that initialize the various
components of a GatewayRunner instance.

Functions are grouped by concern:
- Config loading
- Session and delivery setup
- State tracking dictionaries
- Voice mode initialization
- Session database
- Security and approval checks
- Maintenance tasks
"""

import logging
import itertools
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional

from hermes_cli.config import cfg_get

from gateway import config_loaders, voice_mode
from gateway.config import GatewayConfig

logger = logging.getLogger(__name__)


def load_ephemeral_config(config: GatewayConfig) -> Dict[str, Any]:
    """Load ephemeral configuration from config.yaml / env vars.

    These values are injected at API-call time only and never persisted.

    Returns:
        Dict with keys: prefill_messages, ephemeral_system_prompt,
        reasoning_config, service_tier, show_reasoning, busy_input_mode,
        busy_text_mode, restart_drain_timeout, provider_routing,
        fallback_model
    """
    return {
        "prefill_messages": config_loaders.load_prefill_messages(config),
        "ephemeral_system_prompt": config_loaders.load_ephemeral_system_prompt(
            config
        ),
        "reasoning_config": config_loaders.load_reasoning_config(config),
        "service_tier": config_loaders.load_service_tier(config),
        "show_reasoning": config_loaders.load_show_reasoning(config),
        "busy_input_mode": config_loaders.load_busy_input_mode(config),
        "busy_text_mode": config_loaders.load_busy_text_mode(config),
        "restart_drain_timeout": config_loaders.load_restart_drain_timeout(config),
        "provider_routing": config_loaders.load_provider_routing(config),
        "fallback_model": config_loaders.load_fallback_model(config),
    }


def initialize_session_store_and_router(config: GatewayConfig):
    """Initialize session store and delivery router.

    The session store is wired with process registry for reset protection.

    Returns:
        Tuple of (session_store, delivery_router)
    """
    from gateway.session_store import SessionStore
    from gateway.delivery_router import DeliveryRouter
    from tools.process_registry import process_registry

    session_store = SessionStore(
        config.sessions_dir,
        config,
        has_active_processes_fn=lambda key: process_registry.has_active_for_session(
            key
        ),
    )
    delivery_router = DeliveryRouter(config)
    return session_store, delivery_router


def initialize_agent_cache() -> tuple:
    """Initialize the AIAgent cache structure.

    Returns:
        Tuple of (agent_cache OrderedDict, agent_cache_lock Lock)
    """
    return OrderedDict(), threading.Lock()


def initialize_session_state_tracking() -> Dict[str, Any]:
    """Initialize all session-tracking dictionaries.

    Returns:
        Dict with all tracking dictionaries initialized
    """
    return {
        "running_agents": {},  # session_key -> AIAgent
        "running_agents_ts": {},  # session_key -> start timestamp
        "pending_messages": {},  # session_key -> queued message during interrupt
        "last_resolved_model": {},  # session_key -> last successful model
        "queued_events": {},  # session_key -> List[MessageEvent] for /queue
        "pending_native_image_paths_by_session": {},  # session_key -> List[path]
        "busy_ack_ts": {},  # session_key -> last busy-ack timestamp
        "session_run_generation": {},  # session_key -> generation counter
        "session_sources": OrderedDict(),  # LRU cache of SessionSources
        "session_sources_max": 512,  # cap for LRU cache
        "session_model_overrides": {},  # session_key -> model override dict
        "session_reasoning_overrides": {},  # session_key -> reasoning config dict
        "pending_approvals": {},  # session_key -> approval dict
        "failed_platforms": {},  # Platform -> retry state dict
        "update_prompt_pending": {},  # session_key -> bool
        "recent_voice_transcripts": {},  # (guild_id, user_id) -> List[(ts, text)]
    }


def initialize_voice_mode(config: GatewayConfig) -> Dict[str, str]:
    """Initialize per-chat voice reply mode.

    Returns:
        Dict mapping session_key to voice mode ("off" | "voice_only" | "all")
    """
    return voice_mode.load_voice_modes(config)


def initialize_session_db(config: GatewayConfig) -> Optional[Any]:
    """Initialize session database for session_search tool support.

    Also runs opportunistic maintenance (auto-prune, VACUUM) if configured.

    Returns:
        SessionDB instance or None if unavailable
    """
    session_db = None
    try:
        from hermes_state import SessionDB

        session_db = SessionDB()
    except Exception as e:
        logger.warning("SQLite session store not available: %s", e)

    # Opportunistic state.db maintenance
    if session_db is not None:
        try:
            from hermes_cli.config import load_config

            sess_cfg = load_config().get("sessions") or {}
            if sess_cfg.get("auto_prune", False):
                session_db.maybe_auto_prune_and_vacuum(
                    retention_days=int(sess_cfg.get("retention_days", 90)),
                    min_interval_hours=int(sess_cfg.get("min_interval_hours", 24)),
                    vacuum=bool(sess_cfg.get("vacuum_after_prune", True)),
                    sessions_dir=config.sessions_dir,
                )
        except Exception as exc:
            logger.debug("state.db auto-maintenance skipped: %s", exc)

    return session_db


def initialize_security_checks() -> None:
    """Run startup security and approval mode checks.

    Logs a warning if gateway is in manual approval mode with no automated
    risk assessor (tirith disabled AND no auxiliary.approval model).
    """
    try:
        from hermes_cli.config import load_config
        from tools.tirith_security import ensure_installed

        # Ensure tirith security scanner is available
        ensure_installed(log_failures=False)

        # Check approval mode configuration
        appr_cfg = load_config()
        appr_mode = str(
            cfg_get(appr_cfg, "approvals", "mode", default="manual") or "manual"
        ).strip().lower()
        tirith_on = bool(
            cfg_get(appr_cfg, "security", "tirith_enabled", default=True)
        )
        aux_approval = cfg_get(appr_cfg, "auxiliary", "approval", default=None)

        if appr_mode == "manual" and not tirith_on and not aux_approval:
            logger.warning(
                "Gateway approvals.mode=manual with no automated risk "
                "assessor (security.tirith_enabled is false and "
                "auxiliary.approval is unset): dangerous commands and "
                "execute_code scripts will BLOCK until a human approves "
                "them in chat. Enable security.tirith_enabled or configure "
                "auxiliary.approval for unattended operation."
            )
    except Exception:
        logger.debug("approvals.mode startup check skipped", exc_info=True)


def initialize_checkpoint_maintenance() -> None:
    """Run opportunistic shadow-repo cleanup for checkpoint repos.

    Deletes orphan/stale checkpoint repos under ~/.hermes/checkpoints/.
    Opt-in via checkpoints.auto_prune.
    """
    try:
        from hermes_cli.config import load_config
        from tools.checkpoint_manager import maybe_auto_prune_checkpoints

        ckpt_cfg = load_config().get("checkpoints") or {}
        if ckpt_cfg.get("auto_prune", False):
            maybe_auto_prune_checkpoints(
                retention_days=int(ckpt_cfg.get("retention_days", 7)),
                min_interval_hours=int(ckpt_cfg.get("min_interval_hours", 24)),
                delete_orphans=bool(ckpt_cfg.get("delete_orphans", True)),
                max_total_size_mb=int(ckpt_cfg.get("max_total_size_mb", 500)),
            )
    except Exception as exc:
        logger.debug("checkpoint auto-maintenance skipped: %s", exc)


def get_active_profile_name() -> str:
    """Get the active kanban notifier profile name.

    Returns:
        Profile name or "default" if unavailable
    """
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name() or "default"
    except Exception:
        return "default"


def initialize_slash_confirm_counter() -> itertools.count:
    """Initialize counter for slash-confirm ID generation.

    Returns:
        itertools.count starting at 1
    """
    return itertools.count(1)


def initialize_background_tasks() -> set:
    """Initialize set for tracking background tasks.

    Returns:
        Empty set to hold background task references
    """
    return set()


def initialize_teams_pipeline_runtime() -> tuple:
    """Initialize Teams meeting pipeline runtime placeholders.

    Returns:
        Tuple of (pipeline_runtime, pipeline_runtime_error)
    """
    return None, None


def initialize_pairing_store():
    """Initialize DM pairing store for code-based user authorization.

    Returns:
        PairingStore instance
    """
    from gateway.pairing import PairingStore

    return PairingStore()


def initialize_hooks():
    """Initialize event hook system.

    Returns:
        HookRegistry instance
    """
    from gateway.hooks import HookRegistry

    return HookRegistry()
