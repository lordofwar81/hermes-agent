#!/usr/bin/env python3
import sys

sys.path.insert(0, ".")
import json
import logging

logging.basicConfig(level=logging.DEBUG)
from tools.memory_export import memory_export_tool
import tempfile
import os

output_path = tempfile.mktemp(suffix=".json")
args = {
    "format": "json",
    "output_path": output_path,
    "include_epistemic": True,
    "include_vectors": False,
    "backup": False,
}
result = memory_export_tool(args, task_id="test")
print("Result:")
parsed = json.loads(result)
print(json.dumps(parsed, indent=2))
if parsed.get("success"):
    print(f"Exported to {parsed.get('output_path')}")
    # read file to verify
    with open(output_path, "r") as f:
        data = json.load(f)
        print(f"Exported {data.get('memory_count')} memories")
    os.unlink(output_path)
else:
    print(f"Error: {parsed.get('error')}")
