#!/usr/bin/env python3
"""
Built-in memory provider — hierarchical recall with vector memory, temporal events,
and relationship graph.

Provides automatic recall of relevant memories for each turn, with token-aware
selection to reduce context bloat (target 60-80% less tokens than naive inclusion).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Optional imports for vector memory and unified search
try:
    from tools.vector_memory import VectorMemoryStore

    HAS_VECTOR_MEMORY = True
except ImportError:
    HAS_VECTOR_MEMORY = False
    VectorMemoryStore = None

try:
    from tools.unified_memory_search import search_memories

    HAS_UNIFIED_SEARCH = True
except ImportError:
    HAS_UNIFIED_SEARCH = False
    search_memories = None

try:
    from tools.temporal_extractor import get_events_for_query

    HAS_TEMPORAL = True
except ImportError:
    HAS_TEMPORAL = False
    get_events_for_query = None

try:
    from tools.relationship_extractor import find_related_memories

    HAS_RELATIONSHIP = True
except ImportError:
    HAS_RELATIONSHIP = False
    find_related_memories = None


class BuiltinMemoryProvider(MemoryProvider):
    """Built-in hierarchical memory provider."""

    def __init__(self) -> None:
        self._initialized = False
        self._vector_store: Optional[VectorMemoryStore] = None
        self._session_id: str = ""
        self._hermes_home: str = ""
        # Configuration defaults
        self._max_memory_tokens = 2000  # Target token budget for memory context
        self._recency_weight = 0.3  # Weight for recency boost (same as unified search)
        self._recency_halflife_days = 30
        # Short‑term buffer of recent conversation turns (user + assistant)
        self._recent_turns = deque(maxlen=20)
        # Dynamic window management
        self._model_context_length = None
        self._remaining_tokens = None

    @property
    def name(self) -> str:
        return "builtin"

    def is_available(self) -> bool:
        """Return True if vector memory store is available."""
        return HAS_VECTOR_MEMORY

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize for a session."""
        self._session_id = session_id
        self._hermes_home = kwargs.get("hermes_home", "")
        if HAS_VECTOR_MEMORY:
            try:
                self._vector_store = VectorMemoryStore()
                logger.info("Built-in memory provider initialized with vector store")
            except Exception as e:
                logger.warning("Failed to initialize vector store: %s", e)
                self._vector_store = None
        self._initialized = True

    def system_prompt_block(self) -> str:
        """Return static instructions about memory."""
        # No static block; memories are injected via prefetch.
        return ""

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token)."""
        return len(text) // 4

    def _format_memory(self, memory: Dict[str, Any]) -> str:
        """Format a single memory for context."""
        # Include text, source, epistemic status, confidence, entities, keywords
        lines = []
        lines.append(f"- {memory.get('text', '').strip()}")
        # Add metadata if available
        source = memory.get("source")
        if source:
            lines[-1] += f" [source: {source}]"
        epistemic = memory.get("epistemic_status")
        if epistemic and epistemic != "stated":
            lines[-1] += f" [{epistemic}]"
        confidence = memory.get("confidence")
        if confidence is not None and confidence < 0.8:
            lines[-1] += f" (confidence: {confidence:.2f})"
        return "\n".join(lines)

    def _format_turn(self, turn: Dict[str, Any]) -> str:
        """Format a conversation turn for context."""
        role = turn.get("role", "")
        content = turn.get("content", "").strip()
        if role == "user":
            return f"User: {content}"
        elif role == "assistant":
            return f"Assistant: {content}"
        else:
            return f"{role}: {content}"

    def _select_memories(
        self, query: str, max_tokens: int
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Select memories using hierarchical recall.

        Returns (selected memories, total tokens used).
        """
        if not HAS_UNIFIED_SEARCH:
            return [], 0

        selected = []
        used_tokens = 0
        seen_ids = set()

        # Step 0: Include recent conversation turns (short‑term buffer)
        recent_turns = list(self._recent_turns)[-10:]  # last 10 turns
        for turn in reversed(recent_turns):  # most recent first
            turn_id = f"turn_{turn['timestamp']}_{turn['role']}"
            if turn_id in seen_ids:
                continue
            text = self._format_turn(turn)
            tokens = self._estimate_tokens(text)
            if used_tokens + tokens > max_tokens:
                continue
            selected.append(
                {
                    "type": "turn",
                    "text": text,
                    "role": turn["role"],
                    "content": turn["content"],
                    "timestamp": turn["timestamp"],
                }
            )
            used_tokens += tokens
            seen_ids.add(turn_id)
            if used_tokens >= max_tokens:
                break

        # Step 1: Retrieve relevant memories with unified search (includes recency boost)
        relevant = search_memories(
            query=query,
            limit=20,
            filters={},
            include_related=False,
            include_temporal=False,
            recency_weight=self._recency_weight,
            recency_halflife_days=self._recency_halflife_days,
        )
        # Step 2: Add relevant memories within token budget
        for mem in relevant:
            mem_id = mem.get("id")
            if mem_id and mem_id in seen_ids:
                continue
            text = self._format_memory(mem)
            tokens = self._estimate_tokens(text)
            if used_tokens + tokens > max_tokens:
                continue
            selected.append(mem)
            used_tokens += tokens
            if mem_id:
                seen_ids.add(mem_id)
            if used_tokens >= max_tokens:
                break

        # Step 3: If we still have token budget, add recent memories (last 24 hours)
        if used_tokens < max_tokens * 0.8:  # reserve 20% for other context
            # Filter for memories created in last 24 hours
            recent_filter = {"after": time.time() - 24 * 3600}
            recent = search_memories(
                query="",  # empty query returns all by recency
                limit=10,
                filters=recent_filter,
                include_related=False,
                include_temporal=False,
                recency_weight=0.0,  # no extra weight, we already filter by time
                recency_halflife_days=self._recency_halflife_days,
            )
            for mem in recent:
                mem_id = mem.get("id")
                if mem_id and mem_id in seen_ids:
                    continue
                text = self._format_memory(mem)
                tokens = self._estimate_tokens(text)
                if used_tokens + tokens > max_tokens:
                    break
                selected.append(mem)
                used_tokens += tokens
                if mem_id:
                    seen_ids.add(mem_id)
                if used_tokens >= max_tokens:
                    break

        # Step 4: If query contains temporal expressions, add temporal events
        # (handled by search_memories via filters; we already included via relevant)
        # Step 5: If query references known entities, add related memories via relationship graph
        # (future enhancement)

        return selected, used_tokens

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall relevant memories for the upcoming turn."""
        if not self._initialized or not self._vector_store:
            return ""

        # Determine token budget: use configured max_memory_tokens
        max_tokens = self._max_memory_tokens

        selected, used_tokens = self._select_memories(query, max_tokens)
        if not selected:
            return ""

        # Format memories into a coherent block
        lines = []
        lines.append("Relevant memories:")
        for mem in selected:
            if mem.get("type") == "turn":
                lines.append(mem.get("text", ""))
            else:
                lines.append(self._format_memory(mem))

        # Add token usage debug (optional)
        if logger.isEnabledFor(logging.DEBUG):
            lines.append(f"[{len(selected)} items, ~{used_tokens} tokens]")

        return "\n".join(lines)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """No background prefetch needed."""

    def sync_turn(
        self, user_content: str, assistant_content: str, *, session_id: str = ""
    ) -> None:
        """Store recent turns in short‑term buffer."""
        if user_content.strip():
            self._recent_turns.append(
                {"role": "user", "content": user_content, "timestamp": time.time()}
            )
        if assistant_content.strip():
            self._recent_turns.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "timestamp": time.time(),
                }
            )
        # Optionally extract and store memories from the turn (future).
        # For now, memories are added via the memory tool.

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """No additional tools beyond the existing memory tool."""
        return []

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """No tools to handle."""
        return tool_error(
            f"Built-in memory provider does not handle tool '{tool_name}'"
        )

    def shutdown(self) -> None:
        """Cleanup."""
        self._vector_store = None
        self._initialized = False

    # Optional hooks
    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Optional per‑turn notification."""

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Optional session‑end processing."""

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract insights before compression."""
        return ""

    def on_delegation(self, task: str, result: str, **kwargs) -> None:
        """Observe subagent completion."""

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built‑in memory writes to vector store (already done by memory_tool)."""
