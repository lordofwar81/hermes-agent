#!/usr/bin/env python3
import sys

sys.path.insert(0, ".")
import json
import logging

logging.basicConfig(level=logging.DEBUG)
from tools.unified_memory_search import unified_memory_search_tool

args = {"query": "test", "mode": "vector", "top_k": 5}
result = unified_memory_search_tool(args, task_id="test")
print("Result:")
print(result)
parsed = json.loads(result)
print(f"Error?: {parsed.get('error')}")
print(f"Count: {parsed.get('count', 0)}")
if "results" in parsed:
    for r in parsed["results"][:3]:
        print(f"  {r.get('memory_id')}: {r.get('text')[:80]}")
