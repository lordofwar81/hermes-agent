"""
Kanban artifact delivery utilities for GatewayRunner.

Extracted from gateway/run.py to reduce the God file size.
Provides functions for uploading artifact files referenced by kanban tasks.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import quote as _quote

from gateway.platforms.base import BasePlatformAdapter

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}


async def deliver_kanban_artifacts(
    *,
    adapter,
    chat_id: str,
    metadata: dict,
    event_payload: Optional[dict],
    task: Any,
) -> None:
    """Upload artifact files referenced by a completed kanban task.

    Workers passing ``kanban_complete(artifacts=[...])`` ship absolute
    file paths through the completion event so downstream humans get
    the deliverable as a native upload instead of a path printed in
    chat.

    Sources scanned, in priority order:
      1. ``event_payload['artifacts']`` (explicit list — preferred)
      2. ``event_payload['summary']`` (truncated first line)
      3. ``task.result`` (legacy fallback)

    Files are deduplicated, missing files are silently skipped (the
    path may have been mentioned for reference only), and delivery
    errors are logged but do not break the notifier loop.

    Args:
        adapter: Platform adapter for sending files
        chat_id: Target chat ID
        metadata: Message metadata dict
        event_payload: Kanban completion event payload
        task: Completed task object (may have result attribute)
    """
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(path: str) -> None:
        if not path:
            return
        expanded = os.path.expanduser(path)
        if expanded in seen:
            return
        if not os.path.isfile(expanded):
            return
        seen.add(expanded)
        candidates.append(expanded)

    # 1. Explicit artifacts list in payload.
    if isinstance(event_payload, dict):
        raw = event_payload.get("artifacts")
        if isinstance(raw, (list, tuple)):
            for item in raw:
                if isinstance(item, str):
                    _add(item)

        # 2. Paths embedded in the payload summary.
        summary = event_payload.get("summary")
        if isinstance(summary, str) and summary:
            paths, _ = adapter.extract_local_files(summary)
            for p in paths:
                _add(p)

    # 3. Legacy: paths embedded in task.result.
    if task is not None and getattr(task, "result", None):
        result_text = str(task.result)
        paths, _ = adapter.extract_local_files(result_text)
        for p in paths:
            _add(p)

    if not candidates:
        return

    candidates = BasePlatformAdapter.filter_local_delivery_paths(candidates)
    if not candidates:
        return

    # Partition images so they ride a single send_multiple_images call
    # on platforms that support batch image uploads (Signal/Slack RPCs).
    image_paths = [p for p in candidates if Path(p).suffix.lower() in _IMAGE_EXTS]
    other_paths = [p for p in candidates if Path(p).suffix.lower() not in _IMAGE_EXTS]

    if image_paths:
        try:
            batch = [(f"file://{_quote(p)}", "") for p in image_paths]
            await adapter.send_multiple_images(
                chat_id=chat_id, images=batch, metadata=metadata,
            )
        except Exception as exc:
            logger.warning(
                "kanban notifier: image batch upload failed: %s", exc,
            )

    for path in other_paths:
        ext = Path(path).suffix.lower()
        try:
            if ext in _VIDEO_EXTS:
                await adapter.send_video(
                    chat_id=chat_id, video_path=path, metadata=metadata,
                )
            else:
                await adapter.send_document(
                    chat_id=chat_id, file_path=path, metadata=metadata,
                )
        except Exception as exc:
            logger.warning(
                "kanban notifier: artifact upload (%s) failed: %s",
                path, exc,
            )
