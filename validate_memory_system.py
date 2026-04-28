#!/usr/bin/env python3
"""
Validate the enhanced memory system.
"""

import sys

sys.path.insert(0, ".")
import json
import logging
import tempfile
import os
from pathlib import Path

logging.basicConfig(level=logging.WARNING)  # suppress debug logs


def test_embedding():
    """Test embedding endpoint."""
    try:
        import requests

        resp = requests.post(
            "http://localhost:11434/v1/embeddings",
            json={"model": "mxbai-embed-large-v1", "input": "test"},
            timeout=10,
        )
        if resp.status_code == 200:
            emb = resp.json()["data"][0]["embedding"]
            return True, f"Embedding works (dim={len(emb)})"
        else:
            return False, f"Embedding endpoint error: {resp.status_code}"
    except Exception as e:
        return False, f"Embedding endpoint failed: {e}"


def test_vector_store():
    """Test VectorMemoryStore basic operations."""
    from tools.vector_memory import VectorMemoryStore

    try:
        store = VectorMemoryStore()
        if store.table is None:
            return False, "Vector store table not found"
        count_before = store.count()
        # Add a test memory
        mem_id = store.add_memory(
            text="Validation test memory",
            source="validation",
            memory_type="test",
            session_id="validation_session",
        )
        if not mem_id:
            return False, "Failed to add memory"
        # Verify count increased
        count_after = store.count()
        if count_after <= count_before:
            return False, "Count did not increase"
        # Search
        results = store.search("validation test", top_k=2)
        if len(results) == 0:
            return False, "Search returned no results"
        # Update epistemic status
        if not store.update_epistemic_status(mem_id, "verified", confidence=0.8):
            return False, "Failed to update epistemic status"
        # Get memory
        mem = store.get_memory(mem_id)
        if not mem:
            return False, "Failed to retrieve memory"
        # List memories
        list_mem = store.list_memories(limit=5)
        if not list_mem:
            return False, "List memories returned empty"
        # Delete test memory (cleanup)
        if not store.delete_memory(mem_id):
            return False, "Failed to delete test memory"
        return (
            True,
            f"Vector store operations OK (count: {count_before} -> {store.count()})",
        )
    except Exception as e:
        return False, f"Vector store test failed: {e}"


def test_memory_search():
    """Test memory_search tool."""
    from tools.unified_memory_search import unified_memory_search_tool

    try:
        args = {"query": "validation", "mode": "vector", "top_k": 5}
        result = unified_memory_search_tool(args, task_id="validation")
        data = json.loads(result)
        if "error" in data:
            return False, f"Memory search error: {data['error']}"
        if data.get("count", 0) >= 0:
            return True, f"Memory search OK (found {data.get('count')} results)"
        else:
            return False, "Memory search returned invalid count"
    except Exception as e:
        return False, f"Memory search test failed: {e}"


def test_memory_export():
    """Test memory_export tool."""
    from tools.memory_export import memory_export_tool

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "memories.json"
            args = {"format": "json", "output_path": str(out_path)}
            result = memory_export_tool(args, task_id="validation")
            data = json.loads(result)
            if "error" in data:
                return False, f"Memory export error: {data['error']}"
            if data.get("success") and data.get("memory_count", 0) >= 0:
                if out_path.exists():
                    return True, f"Memory export OK ({data['memory_count']} memories)"
                else:
                    return False, "Export file not created"
            else:
                return False, "Memory export reported failure"
    except Exception as e:
        return False, f"Memory export test failed: {e}"


def main():
    print("=== Memory System Validation ===\n")
    tests = [
        ("Embedding Endpoint", test_embedding),
        ("Vector Memory Store", test_vector_store),
        ("Memory Search Tool", test_memory_search),
        ("Memory Export Tool", test_memory_export),
    ]
    passed = 0
    total = len(tests)
    for name, test_fn in tests:
        print(f"{name}: ", end="")
        success, msg = test_fn()
        if success:
            print(f"✅ PASS - {msg}")
            passed += 1
        else:
            print(f"❌ FAIL - {msg}")
    print(f"\nTotal: {passed}/{total} passed")
    score = passed / total * 100
    print(f"Score: {score:.0f}%")
    if score >= 90:
        print("🎉 Memory system operational at 9/10 state!")
    else:
        print("⚠️ Memory system needs improvements.")
    sys.exit(0 if score >= 90 else 1)


if __name__ == "__main__":
    main()
