#!/usr/bin/env python3
import sys

sys.path.insert(0, ".")
import logging

logging.basicConfig(level=logging.DEBUG)
from tools.vector_memory import VectorMemoryStore

store = VectorMemoryStore()
print("Store initialized")
print("Table exists?", store.table is not None)
if store.table:
    print("Count:", store.count())
    # Try to list memories
    memories = store.list_memories(limit=5)
    print(f"Found {len(memories)} memories")
    for mem in memories:
        print(f"  ID: {mem.get('id')}, text: {mem.get('text')[:50]}...")
    # Try to add a memory
    mem_id = store.add_memory(
        text="Test memory from vector store",
        source="test",
        memory_type="test",
        session_id="test_session",
    )
    print(f"Added memory ID: {mem_id}")
    if mem_id:
        # Search
        results = store.search("test memory", top_k=5)
        print(f"Search results: {len(results)}")
        for r in results:
            print(f"  Sim: {r.similarity:.3f} - {r.text[:60]}...")
else:
    print("No table, skipping operations")
