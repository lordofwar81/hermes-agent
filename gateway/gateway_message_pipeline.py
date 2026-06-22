"""Message enrichment + routing helpers extracted from GatewayRunner.

Round 14 of gateway decomposition. _enrich_message_with_vision auto-analyzes
attached images; _decide_image_input_mode resolves native vs text image
routing; _set_session_env binds contextvars; _enrich_async_delegation_routing
back-fills routing fields on async events; _read_user_config reads config.yaml;
_thread_metadata_for_target builds synthetic-send metadata. All stateless.
"""

import json
from typing import Any, Dict, List, Optional

from gateway.platforms.base import Platform
from gateway.session import SessionContext

import logging
logger = logging.getLogger("gateway.run")


def _decide_image_input_mode() -> str:
    """Resolve the image-input routing for the currently active model.

    Returns ``"native"`` (attach pixels on the user turn) or ``"text"``
    (pre-analyze with vision_analyze and prepend the description). See
    agent/image_routing.py for the full decision table.

    The active provider/model are read from config.yaml so the decision
    tracks ``/model`` switches automatically on the next message.
    """
    try:
        from agent.image_routing import decide_image_input_mode
        from agent.auxiliary_client import _read_main_model, _read_main_provider
        from hermes_cli.config import load_config

        cfg = load_config()
        provider = _read_main_provider()
        model = _read_main_model()
        return decide_image_input_mode(provider, model, cfg)
    except Exception as exc:
        logger.debug("image_routing: decision failed, falling back to text — %s", exc)
        return "text"


async def _enrich_message_with_vision(
    user_text: str,
    image_paths: List[str],
) -> str:
    """
    Auto-analyze user-attached images with the vision tool and prepend
    the descriptions to the message text.

    Each image is analyzed with a general-purpose prompt.  The resulting
    description *and* the local cache path are injected so the model can:
      1. Immediately understand what the user sent (no extra tool call).
      2. Re-examine the image with vision_analyze if it needs more detail.

    Args:
        user_text:   The user's original caption / message text.
        image_paths: List of local file paths to cached images.

    Returns:
        The enriched message string with vision descriptions prepended.
    """
    from tools.vision_tools import vision_analyze_tool
    from agent.memory_manager import sanitize_context

    analysis_prompt = (
        "Describe everything visible in this image in thorough detail. "
        "Include any text, code, data, objects, people, layout, colors, "
        "and any other notable visual information."
    )

    enriched_parts = []
    for path in image_paths:
        try:
            logger.debug("Auto-analyzing user image: %s", path)
            result_json = await vision_analyze_tool(
                image_url=path,
                user_prompt=analysis_prompt,
            )
            result = json.loads(result_json)
            if result.get("success"):
                description = result.get("analysis", "")
                description = sanitize_context(description)
                enriched_parts.append(
                    f"[The user sent an image~ Here's what I can see:\n{description}]\n"
                    f"[If you need a closer look, use vision_analyze with "
                    f"image_url: {path} ~]"
                )
            else:
                enriched_parts.append(
                    "[The user sent an image but I couldn't quite see it "
                    "this time (>_<) You can try looking at it yourself "
                    f"with vision_analyze using image_url: {path}]"
                )
        except Exception as e:
            logger.error("Vision auto-analysis error: %s", e)
            enriched_parts.append(
                f"[The user sent an image but something went wrong when I "
                f"tried to look at it~ You can try examining it yourself "
                f"with vision_analyze using image_url: {path}]"
            )

    # Combine: vision descriptions first, then the user's original text
    if enriched_parts:
        prefix = "\n\n".join(enriched_parts)
        if user_text:
            return f"{prefix}\n\n{user_text}"
        return prefix
    return user_text


def _set_session_env(context: SessionContext) -> list:
    """Set session context variables for the current async task.

    Uses ``contextvars`` instead of ``os.environ`` so that concurrent
    gateway messages cannot overwrite each other's session state.

    Returns a list of reset tokens; pass them to ``_clear_session_env``
    in a ``finally`` block.
    """
    from gateway.session_context import set_session_vars
    return set_session_vars(
        platform=context.source.platform.value,
        chat_id=context.source.chat_id,
        chat_name=context.source.chat_name or "",
        thread_id=str(context.source.thread_id) if context.source.thread_id else "",
        user_id=str(context.source.user_id) if context.source.user_id else "",
        user_name=str(context.source.user_name) if context.source.user_name else "",
        session_key=context.session_key,
        message_id=str(context.source.message_id) if context.source.message_id else "",
    )


def _enrich_async_delegation_routing(evt: dict) -> None:
    """Fill platform/chat_id/thread_id/chat_type on an async-delegation event.

    Async-delegation completion events only carry ``session_key`` (the
    daemon worker has no access to the per-message routing metadata the
    terminal background watcher captures at spawn time). Parse the
    session_key into the routing fields ``_build_process_event_source``
    expects. Best-effort: a CLI-origin event (empty session_key) is left
    as-is and simply won't route on the gateway.
    """
    if evt.get("platform"):
        return  # already enriched
    from gateway.run import _parse_session_key
    parsed = _parse_session_key(evt.get("session_key", "") or "")
    if not parsed:
        return
    evt["platform"] = parsed.get("platform", "")
    evt["chat_type"] = parsed.get("chat_type", "")
    evt["chat_id"] = parsed.get("chat_id", "")
    if parsed.get("thread_id"):
        evt["thread_id"] = parsed["thread_id"]


def _thread_metadata_for_source(
    source,
    reply_to_message_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build the metadata dict platforms need for thread-aware replies."""
    return _thread_metadata_for_target(
        getattr(source, "platform", None),
        getattr(source, "chat_id", None),
        getattr(source, "thread_id", None),
        chat_type=getattr(source, "chat_type", None),
        reply_to_message_id=reply_to_message_id or getattr(source, "message_id", None),
    )


def _read_user_config() -> Dict[str, Any]:
    """Read the user's raw config.yaml (cached) for gate lookups.

    Used by slash-confirm gates that must reflect on-disk state changes
    (e.g. a prior "Always Approve" click) without a gateway restart.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _thread_metadata_for_target(
    platform: Optional[Platform],
    chat_id: Optional[str],
    thread_id: Optional[str],
    *,
    chat_type: Optional[str] = None,
    reply_to_message_id: Optional[str] = None,
    adapter: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Build thread metadata for synthetic sends that only have routing state."""
    from gateway.gateway_telegram_topics import _is_telegram_dm_topic_target
    if thread_id is None:
        return None
    metadata: Dict[str, Any] = {"thread_id": thread_id}
    if _is_telegram_dm_topic_target(
        platform,
        chat_id,
        thread_id,
        chat_type=chat_type,
        adapter=adapter,
    ):
        metadata["telegram_dm_topic_reply_fallback"] = True
        # Telegram DM topic lanes need direct_messages_topic_id in metadata
        # so synthetic/queued messages (goal continuations, status notices)
        # route to the correct topic even when reply anchor is unavailable.
        tid = str(thread_id)
        if tid and tid not in {"", "1"}:
            metadata["direct_messages_topic_id"] = tid
        if reply_to_message_id is not None:
            metadata["telegram_reply_to_message_id"] = str(reply_to_message_id)
    return metadata
