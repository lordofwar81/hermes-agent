#!/usr/bin/env python3
"""Test contradiction detection integration."""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from tools.memory_tool import memory_tool, MemoryStore
from tools.vector_memory import VectorMemoryStore
from tools.contradiction_detector import handle_memory_with_contradiction_check

store = MemoryStore()
store.load_from_disk()

vector_store = VectorMemoryStore()
print("Vector store count:", vector_store.count())

# Add first memory
print("Adding first memory...")
result1 = memory_tool(
    action="add",
    target="vector",
    content="The user prefers Python over Java.",
    store=store,
)
print("Result1:", result1)

# Add similar memory (could be contradictory)
print("Adding second similar memory...")
result2 = memory_tool(
    action="add",
    target="vector",
    content="The user prefers Java over Python.",
    store=store,
)
print("Result2:", result2)

# Direct contradiction detection test
print("Direct contradiction detection test...")
detection = handle_memory_with_contradiction_check(
    "The user prefers Java over Python.", vector_store.table
)
print("Detection:", detection)

# List memories
memories = vector_store.list_memories(limit=5)
for mem in memories:
    print(f"  - {mem.get('text')} [{mem.get('epistemic_status')}]")
