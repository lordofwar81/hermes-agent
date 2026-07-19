#!/usr/bin/env python3
"""Deprecated — kept only for backward-compatible constant exports.

The ``BuiltinMemoryProvider`` class that lived here was never instantiated in
production: the builtin memory path uses ``tools.memory_tool.MemoryStore``
directly (see ``agent_init.py``), bypassing the MemoryProvider abstraction.
The class was removed as dead code; the module is preserved because
``tests/test_search_memories.py`` imports ``HAS_UNIFIED_SEARCH`` from it.

If you are looking for the active memory providers, see
``plugins/memory/`` (holographic is the current ``memory.provider``).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Preserved for backward compatibility — tests import HAS_UNIFIED_SEARCH.
try:
    from tools.unified_memory_search import search_memories  # noqa: F401

    HAS_UNIFIED_SEARCH = True
except ImportError:
    HAS_UNIFIED_SEARCH = False
    search_memories = None

try:
    from tools.vector_memory import VectorMemoryStore  # noqa: F401

    HAS_VECTOR_MEMORY = True
except ImportError:
    HAS_VECTOR_MEMORY = False
    VectorMemoryStore = None
