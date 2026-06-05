#!/usr/bin/env python3
import sys

sys.path.insert(0, ".")
import logging

logging.basicConfig(level=logging.WARNING)
from tools.memory_tool import MemoryStore

store = MemoryStore()
store.load_from_disk()
print("Built-in memory store loaded")
print("Memory entries:", len(store.memory_entries))
print("User entries:", len(store.user_entries))
# Test add
result = store.add("memory", "Test entry from validation")
print("Add result:", result.get("success"), result.get("message", ""))
# Clean up: remove the added entry
if result.get("success"):
    store.remove("memory", "Test entry from validation")
    print("Cleaned up")
print("Built-in memory functional")
