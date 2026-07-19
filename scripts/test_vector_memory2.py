#!/usr/bin/env python3
import sys

sys.path.insert(0, ".")
import logging

logging.basicConfig(level=logging.DEBUG)
from tools.vector_memory import VectorMemoryStore

store = VectorMemoryStore()
print("Count:", store.count())
memories = store.list_memories(limit=10)
print(f"List memories returned {len(memories)} rows")
for i, mem in enumerate(memories):
    print(f"{i}: id={mem.get('id')}, text={mem.get('text')[:60]}")
    break  # just first
# Test search
results = store.search("test", top_k=5)
print(f"Search results: {len(results)}")
for r in results[:3]:
    print(f"  {r.similarity:.3f} - {r.text[:80]}")
