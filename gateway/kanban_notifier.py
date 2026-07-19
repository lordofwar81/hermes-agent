"""Kanban notification helpers for the gateway.

Extracted from gateway/run.py to reduce the God file size.
Provides sync helpers for kanban subscription cursor management
and artifact delivery.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote as _quote

logger = logging.getLogger(__name__)


def kanban_advance(
    sub: dict,
    cursor: int,
    board: Optional[str] = None,
) -> None:
    """Sync helper: advance a subscription's cursor."""
    from hermes_cli import kanban_db as _kb

    conn = _kb.connect(board=board)
    try:
        _kb.advance_notify_cursor(
            conn,
            task_id=sub["task_id"],
            platform=sub["platform"],
            chat_id=sub["chat_id"],
            thread_id=sub.get("thread_id") or "",
            new_cursor=cursor,
        )
    finally:
        conn.close()


def kanban_unsub(
    sub: dict,
    board: Optional[str] = None,
) -> None:
    from hermes_cli import kanban_db as _kb

    conn = _kb.connect(board=board)
    try:
        _kb.remove_notify_sub(
            conn,
            task_id=sub["task_id"],
            platform=sub["platform"],
            chat_id=sub["chat_id"],
            thread_id=sub.get("thread_id") or "",
        )
    finally:
        conn.close()


def kanban_rewind(
    sub: dict,
    claimed_cursor: int,
    old_cursor: int,
    board: Optional[str] = None,
) -> None:
    """Sync helper: undo a claimed notification cursor after send failure."""
    from hermes_cli import kanban_db as _kb

    conn = _kb.connect(board=board)
    try:
        _kb.rewind_notify_cursor(
            conn,
            task_id=sub["task_id"],
            platform=sub["platform"],
            chat_id=sub["chat_id"],
            thread_id=sub.get("thread_id") or "",
            claimed_cursor=claimed_cursor,
            old_cursor=old_cursor,
        )
    finally:
        conn.close()


async def deliver_kanban_artifacts(
    *,
    adapter: Any,
    chat_id: str,
    metadata: dict,
    event_payload: Optional[dict],
    task: Any,
) -> None:
    """Upload artifact files referenced by a completed kanban task."""
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

    if isinstance(event_payload, dict):
        raw = event_payload.get("artifacts")
        if isinstance(raw, (list, tuple)):
            for item in raw:
                if isinstance(item, str):
                    _add(item)

        summary = event_payload.get("summary")
        if isinstance(summary, str) and summary:
            paths, _ = adapter.extract_local_files(summary)
            for p in paths:
                _add(p)

    if task is not None and getattr(task, "result", None):
        result_text = str(task.result)
        paths, _ = adapter.extract_local_files(result_text)
        for p in paths:
            _add(p)

    if not candidates:
        return

    from gateway.platforms.base import BasePlatformAdapter

    candidates = BasePlatformAdapter.filter_local_delivery_paths(candidates)
    if not candidates:
        return

    _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    _VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}

    image_paths = [p for p in candidates if Path(p).suffix.lower() in _IMAGE_EXTS]
    other_paths = [p for p in candidates if Path(p).suffix.lower() not in _IMAGE_EXTS]

    if image_paths:
        try:
            batch = [(f"file://{_quote(p)}", "") for p in image_paths]
            await adapter.send_multiple_images(
                chat_id=chat_id, images=batch, metadata=metadata,
            )
        except Exception as exc:
            logger.warning("kanban notifier: image batch upload failed: %s", exc)

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
            logger.warning("kanban notifier: artifact upload (%s) failed: %s", path, exc)
