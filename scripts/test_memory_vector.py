#!/usr/bin/env python3
"""Test vector memory integration in memory_tool."""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from tools.memory_tool import memory_tool, MemoryStore

# Create a dummy store (not used for vector target)
store = MemoryStore()
store.load_from_disk()

print("Testing vector memory add...")
result = memory_tool(
    action="add", target="vector", content="Test vector memory entry", store=store
)
print("Result:", result)

# Parse JSON
import json

parsed = json.loads(result)
if parsed.get("success"):
    memory_id = parsed.get("memory_id")
    print(f"Added memory ID: {memory_id}")

    # Verify
    print("Testing verify...")
    result2 = memory_tool(
        action="verify", target="vector", old_text=memory_id, store=store
    )
    print("Verify result:", result2)

    # Contradict
    print("Testing contradict...")
    result3 = memory_tool(
        action="contradict", target="vector", old_text=memory_id, store=store
    )
    print("Contradict result:", result3)

    # Retract
    print("Testing retract...")
    result4 = memory_tool(
        action="retract", target="vector", old_text=memory_id, store=store
    )
    print("Retract result:", result4)
else:
    print("Failed to add vector memory:", parsed.get("error"))
