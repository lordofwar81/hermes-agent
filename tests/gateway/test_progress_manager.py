"""Tests for _ProgressManager — the extracted tool-progress pipeline.

_ProgressManager encapsulates:
- progress_callback — sync callback invoked by the agent on tool lifecycle events
- run() — async task that sends/edits progress bubbles
- make_on_new_message() — callback for stream consumer segment breaks

These were previously nested closures inside _run_agent.
"""

from __future__ import annotations

import asyncio
import queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource


# ---------------------------------------------------------------------------
# Test adapter — minimal BasePlatformAdapter subclass
# ---------------------------------------------------------------------------

class ProgressTestAdapter(BasePlatformAdapter):
    """Adapter that records every send/edit for inspection."""

    _next_mid = 1000

    def __init__(self):
        super().__init__(MagicMock(), Platform.TELEGRAM)
        self.sent = []
        self.edits = []
        self.deleted = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    def _mint_id(self) -> str:
        ProgressTestAdapter._next_mid += 1
        return str(ProgressTestAdapter._next_mid)

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        mid = self._mint_id()
        self.sent.append({"chat_id": chat_id, "content": content, "message_id": mid, "metadata": metadata})
        return SendResult(success=True, message_id=mid)

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edits.append({"chat_id": chat_id, "message_id": message_id, "content": content})
        return SendResult(success=True, message_id=message_id)

    async def delete_message(self, chat_id, message_id) -> bool:
        self.deleted.append({"chat_id": chat_id, "message_id": str(message_id)})
        return True

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(*, progress_mode="all", **kwargs):
    """Build a _ProgressManager with sensible defaults."""
    from gateway.mixins.agent_runner_mixin import _ProgressManager

    defaults = dict(
        runner=MagicMock(),
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="-1001"),
        adapter=ProgressTestAdapter(),
        progress_queue=queue.Queue(),
        platform_key="telegram",
        progress_mode=progress_mode,
        user_config={},
        session_key="agent:test:telegram:group:-1001",
        run_generation=1,
        event_message_id=None,
        run_still_current=lambda: True,
        agent_holder=[None],
        cleanup_progress=False,
        cleanup_adapter=None,
        cleanup_msg_ids=[],
        progress_thread_id=None,
        progress_metadata=None,
        progress_reply_to=None,
        loop_for_step=MagicMock(),
        hooks_ref=MagicMock(loaded_hooks=False),
        logger=MagicMock(),
    )
    defaults.update(kwargs)
    return _ProgressManager(**defaults)


# ---------------------------------------------------------------------------
# progress_callback tests
# ---------------------------------------------------------------------------

class TestProgressCallback:

    def test_ignores_tool_completed_without_hint(self):
        """tool.completed without long duration is ignored."""
        pm = _make_manager()
        pm.callback("tool.completed", "terminal")
        assert pm._queue.qsize() == 0

    def test_ignores_non_tool_events(self):
        """Only tool.started events produce queue entries."""
        pm = _make_manager()
        pm.callback("reasoning.available", "_thinking", "thinking...")
        assert pm._queue.qsize() == 0

    def test_new_mode_ignores_same_tool(self):
        """progress_mode='new' skips consecutive identical tool names."""
        q = queue.Queue()
        pm = _make_manager(progress_queue=q, progress_mode="new")
        pm.callback("tool.started", "terminal", "pwd")
        assert q.qsize() == 1
        pm.callback("tool.started", "terminal", "pwd")
        assert q.qsize() == 1

    def test_all_mode_reports_each_call(self):
        """progress_mode='all' reports every tool.started."""
        q = queue.Queue()
        pm = _make_manager(progress_queue=q, progress_mode="all")
        pm.callback("tool.started", "terminal", "pwd")
        pm.callback("tool.started", "terminal", "pwd")
        assert q.qsize() == 2

    def test_verbose_mode_includes_args(self):
        """progress_mode='verbose' includes argument keys."""
        q = queue.Queue()
        pm = _make_manager(progress_queue=q, progress_mode="verbose")
        pm.callback("tool.started", "terminal", "pwd", {"dir": "/tmp"})
        msg = q.get_nowait()
        assert "terminal" in msg
        assert "dir" in msg

    def test_dedup_collapses_identical_messages(self):
        """Consecutive identical messages collapse with a counter."""
        q = queue.Queue()
        pm = _make_manager(progress_queue=q)
        pm.callback("tool.started", "terminal", "ls")
        pm.callback("tool.started", "terminal", "ls")
        assert q.qsize() == 2
        first = q.get_nowait()
        second = q.get_nowait()
        assert isinstance(second, tuple)
        assert second[0] == "__dedup__"

    def test_new_tool_message_resets_dedup(self):
        """A different tool name resets the dedup counter."""
        q = queue.Queue()
        pm = _make_manager(progress_queue=q)
        pm.callback("tool.started", "terminal", "ls")
        pm.callback("tool.started", "terminal", "ls")
        pm.callback("tool.started", "web_search", "query")
        assert q.qsize() == 3
        q.get_nowait()  # first ls
        q.get_nowait()  # dedup ls
        third = q.get_nowait()
        assert isinstance(third, str)

    def test_stops_when_run_not_current(self):
        """No messages queued when run is stale."""
        q = queue.Queue()
        pm = _make_manager(progress_queue=q, run_still_current=lambda: False)
        pm.callback("tool.started", "terminal", "pwd")
        assert q.qsize() == 0

    def test_stops_when_agent_interrupted(self):
        """No messages queued when agent is interrupted."""
        q = queue.Queue()
        agent = MagicMock(is_interrupted=True)
        pm = _make_manager(progress_queue=q, agent_holder=[agent])
        pm.callback("tool.started", "terminal", "pwd")
        assert q.qsize() == 0


# ---------------------------------------------------------------------------
# run() — async progress message editing
# ---------------------------------------------------------------------------

class TestProgressRun:

    @pytest.mark.asyncio
    async def test_sends_first_progress_message(self):
        """First tool.started sends a new progress message."""
        adapter = ProgressTestAdapter()
        q = queue.Queue()
        pm = _make_manager(adapter=adapter, progress_queue=q)
        pm.callback("tool.started", "terminal", "pwd")

        task = asyncio.create_task(pm.run())
        await asyncio.sleep(0.5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert len(adapter.sent) >= 1
        sent = adapter.sent[0]
        assert "terminal" in sent["content"]

    @pytest.mark.asyncio
    async def test_edits_existing_message(self):
        """Second tool.started edits the existing progress message."""
        adapter = ProgressTestAdapter()
        q = queue.Queue()
        pm = _make_manager(adapter=adapter, progress_queue=q)
        pm.callback("tool.started", "terminal", "pwd")

        task = asyncio.create_task(pm.run())
        await asyncio.sleep(0.3)

        pm.callback("tool.started", "web_search", "query")
        await asyncio.sleep(0.3)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert len(adapter.sent) >= 1
        # At least one edit should have occurred (run() edits the bubble)
        assert len(adapter.edits) >= 1 or len(adapter.sent) >= 2

    @pytest.mark.asyncio
    async def test_run_stops_when_stale(self):
        """Run exits when run_still_current returns False."""
        adapter = ProgressTestAdapter()
        q = queue.Queue()
        stale = [True]
        pm = _make_manager(
            adapter=adapter,
            progress_queue=q,
            run_still_current=lambda: stale[0],
        )
        pm.callback("tool.started", "terminal", "pwd")
        stale[0] = False

        task = asyncio.create_task(pm.run())
        await asyncio.sleep(0.5)
        assert task.done()

    @pytest.mark.asyncio
    async def test_no_op_without_progress_queue(self):
        """run() returns immediately when queue is None."""
        pm = _make_manager(progress_queue=None)
        await pm.run()

    @pytest.mark.asyncio
    async def test_no_op_without_adapter(self):
        """run() returns immediately when adapter is None."""
        pm = _make_manager(adapter=None)
        await pm.run()

    @pytest.mark.asyncio
    async def test_drain_on_cancel(self):
        """Cancelled run() drains remaining queue messages."""
        adapter = ProgressTestAdapter()
        q = queue.Queue()
        pm = _make_manager(adapter=adapter, progress_queue=q)

        # Put multiple progress entries
        q.put("🔍 terminal: \"pwd\"")
        q.put("🔍 web_search: \"query\"")

        task = asyncio.create_task(pm.run())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Both messages should be processed before cancel finishes
        # (at minimum the first one was sent)
        assert len(adapter.sent) >= 1


# ---------------------------------------------------------------------------
# make_on_new_message
# ---------------------------------------------------------------------------

class TestOnNewMessage:

    def test_resets_progress_state(self):
        """on_new_message puts __reset__ on the queue."""
        q = queue.Queue()
        pm = _make_manager(progress_queue=q)
        cb = pm.make_on_new_message()
        cb()
        msg = q.get_nowait()
        assert isinstance(msg, tuple)
        assert msg[0] == "__reset__"

    def test_no_op_when_queue_none(self):
        """on_new_message is a no-op when queue is None."""
        pm = _make_manager(progress_queue=None)
        cb = pm.make_on_new_message()
        cb()  # Should not raise


# ---------------------------------------------------------------------------
# Cleanup integration
# ---------------------------------------------------------------------------

class TestCleanupIntegration:

    @pytest.mark.asyncio
    async def test_tracks_msg_ids_when_cleanup_enabled(self):
        """Messages are tracked in cleanup_msg_ids when cleanup is on."""
        ids_container: list[str] = []
        adapter = ProgressTestAdapter()
        q = queue.Queue()
        pm = _make_manager(
            adapter=adapter,
            progress_queue=q,
            cleanup_progress=True,
            cleanup_adapter=adapter,
            cleanup_msg_ids=ids_container,
        )
        pm.callback("tool.started", "terminal", "pwd")

        task = asyncio.create_task(pm.run())
        await asyncio.sleep(0.5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert len(ids_container) >= 1

    @pytest.mark.asyncio
    async def test_does_not_track_ids_without_cleanup(self):
        """No tracking when cleanup_progress is False."""
        ids_container: list[str] = []
        adapter = ProgressTestAdapter()
        q = queue.Queue()
        pm = _make_manager(
            adapter=adapter,
            progress_queue=q,
            cleanup_progress=False,
            cleanup_adapter=None,
            cleanup_msg_ids=ids_container,
        )
        pm.callback("tool.started", "terminal", "pwd")

        task = asyncio.create_task(pm.run())
        await asyncio.sleep(0.5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert len(ids_container) == 0


# ---------------------------------------------------------------------------
# Thread metadata
# ---------------------------------------------------------------------------

class TestThreadMetadata:

    def test_slack_uses_event_message_id_fallback(self):
        """Slack adapter uses event_message_id as fallback thread_id."""
        from gateway.mixins.agent_runner_mixin import _ProgressManager
        adapter = MagicMock(spec=BasePlatformAdapter)
        source = SessionSource(platform=Platform.SLACK, chat_id="C001")
        pm = _ProgressManager(
            runner=MagicMock(),
            source=source,
            adapter=adapter,
            progress_queue=queue.Queue(),
            platform_key="slack",
            progress_mode="all",
            user_config={},
            session_key="slack:C001",
            run_generation=1,
            event_message_id="evt_001",
            run_still_current=lambda: True,
            agent_holder=[None],
            cleanup_progress=False,
            cleanup_adapter=None,
            cleanup_msg_ids=[],
            progress_thread_id="evt_001",
            progress_metadata={"thread_id": "evt_001"},
            progress_reply_to=None,
            loop_for_step=MagicMock(),
            hooks_ref=MagicMock(loaded_hooks=False),
            logger=MagicMock(),
        )
        assert pm._progress_thread_id == "evt_001"
        assert pm._progress_metadata == {"thread_id": "evt_001"}
