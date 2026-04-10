#!/usr/bin/env python3
import sys

sys.path.insert(0, ".")
import logging

logging.basicConfig(level=logging.INFO)
from tools.vector_memory import VectorMemoryStore

store = VectorMemoryStore()
memories = store.list_memories(limit=1)
if memories:
    mem_id = memories[0]["id"]
    print(f"Testing update_epistemic_status on {mem_id}")
    success = store.update_epistemic_status(mem_id, "verified", confidence=0.9)
    print(f"Success: {success}")
    # Retrieve to verify
    mem = store.get_memory(mem_id)
    if mem:
        print(
            f"Updated status: {mem.get('epistemic_status')}, confidence: {mem.get('confidence')}"
        )
else:
    print("No memories found")
