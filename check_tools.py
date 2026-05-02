#!/usr/bin/env python3
import sys

sys.path.insert(0, ".")
from tools.registry import registry

print("Registered tools:")
for name, entry in registry._tools.items():
    print(f"{name}: toolset={entry.toolset}, check={entry.check_fn()}")

print("\nTools in 'memory' toolset:")
for name, entry in registry._tools.items():
    if entry.toolset == "memory":
        print(f"  {name}")
