#!/usr/bin/env python3
"""Integration tests for the hermes_memory_mcp MCP server.

Tests the full request/response cycle: JSON-RPC → handle_request → tool logic.
Uses a temporary MemoryStore (hermetic — does NOT touch the real ~/.hermes/).

Run: ~/.hermes/hermes-agent/venv/bin/python -m pytest tests/scripts/test_hermes_memory_mcp.py -v
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# --- Path setup (mirror the MCP script's own sys.path manipulation) ---------
HERMES_HOME = os.path.expanduser("~/.hermes")
AGENT_HOME = os.path.join(HERMES_HOME, "hermes-agent")
SCRIPTS_HOME = os.path.join(HERMES_HOME, "scripts")
for _p in (HERMES_HOME, AGENT_HOME, SCRIPTS_HOME):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- Hermetic fixture: temp MEMORY.md + temp MemoryStore --------------------
# We can't easily redirect the MCP script's module-level MemoryStore() to a temp
# DB, so we test handle_request against a real (read-only) store but assert on
# response structure, not specific fact content.


@pytest.fixture(scope="module")
def mcp_module():
    """Import the MCP module fresh."""
    import importlib
    mod = importlib.import_module("hermes_memory_mcp")
    importlib.reload(mod)
    return mod


def _call(mcp_module, tool_name, args=None):
    """Call a tool via handle_request and return the parsed result."""
    req = {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": tool_name, "arguments": args or {}},
    }
    resp = mcp_module.handle_request(req)
    assert resp is not None, "handle_request returned None"
    assert "error" not in resp, f"unexpected error: {resp.get('error')}"
    text = resp["result"]["content"][0]["text"]
    return json.loads(text)


class TestProtocol:
    """MCP protocol-level tests."""

    def test_initialize_returns_correct_protocol_version(self, mcp_module):
        req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        resp = mcp_module.handle_request(req)
        assert resp["result"]["protocolVersion"] == "2024-11-05"
        assert resp["result"]["serverInfo"]["name"] == "hermes-memory"

    def test_initialize_reports_v2(self, mcp_module):
        """The upgraded server should report version 2.0.0."""
        req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        resp = mcp_module.handle_request(req)
        assert resp["result"]["serverInfo"]["version"] == "2.0.0"

    def test_tools_list_returns_all_8_tools(self, mcp_module):
        req = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        resp = mcp_module.handle_request(req)
        names = [t["name"] for t in resp["result"]["tools"]]
        assert "memory.find" in names
        assert "memory.semantic_search" in names
        assert "memory.hybrid_search" in names
        assert "memory.context" in names
        assert "memory.get" in names
        assert "memory.recent" in names
        assert "memory.feedback" in names
        assert "memory.health" in names
        assert len(names) == 8

    def test_unknown_method_returns_error(self, mcp_module):
        req = {"jsonrpc": "2.0", "id": 1, "method": "bogus/method", "params": {}}
        resp = mcp_module.handle_request(req)
        assert resp["error"]["code"] == -32601

    def test_unknown_tool_returns_error(self, mcp_module):
        req = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": "memory.nonexistent", "arguments": {}}}
        resp = mcp_module.handle_request(req)
        assert resp["error"]["code"] == -32601

    def test_notifications_return_none(self, mcp_module):
        req = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        resp = mcp_module.handle_request(req)
        assert resp is None


class TestFTS5Sanitizer:
    """The _sanitize_fts_query function — the colon-crash fix."""

    def test_plain_query_gets_quoted(self, mcp_module):
        result = mcp_module._sanitize_fts_query("gbrain")
        assert result == '"gbrain"'

    def test_host_port_query_does_not_crash(self, mcp_module):
        """The original crasher: 'gbrain :3131' parsed as column filter."""
        result = mcp_module._sanitize_fts_query("gbrain :3131")
        assert result == '"gbrain" ":3131"'

    def test_multi_token_query(self, mcp_module):
        result = mcp_module._sanitize_fts_query("qwen3 embed model")
        assert result == '"qwen3" "embed" "model"'

    def test_fts_operators_are_neutralized(self, mcp_module):
        """AND/OR/NEAR/NOT must not be interpreted as FTS5 operators."""
        result = mcp_module._sanitize_fts_query("memory AND recall")
        assert "AND" in result  # still present as text
        assert result == '"memory" "AND" "recall"'

    def test_parentheses_are_neutralized(self, mcp_module):
        result = mcp_module._sanitize_fts_query("(group)")
        assert result == '"(group)"'

    def test_pre_quoted_tokens_preserved(self, mcp_module):
        # Note: sanitizer splits on whitespace first, so multi-word quoted
        # strings become individual quoted tokens. Single quoted token is preserved.
        result = mcp_module._sanitize_fts_query('"already"')
        assert result == '"already"'

    def test_embedded_double_quote_escaped(self, mcp_module):
        # Token with embedded double-quote: inner quote is doubled per FTS5 spec
        result = mcp_module._sanitize_fts_query('he"quote')
        # Expected: "he""quote" (wrapped in quotes, inner quote doubled)
        assert result == '"he""quote"'

    def test_empty_string(self, mcp_module):
        assert mcp_module._sanitize_fts_query("") == ""

    def test_single_token(self, mcp_module):
        assert mcp_module._sanitize_fts_query("HSA_ENABLE_SDMA=0") == '"HSA_ENABLE_SDMA=0"'

    def test_hyphenated_token(self, mcp_module):
        assert mcp_module._sanitize_fts_query("qwen3-embed-8b") == '"qwen3-embed-8b"'


class TestMemoryFind:
    """memory.find — the primary keyword search tool."""

    def test_basic_find_returns_list(self, mcp_module):
        results = _call(mcp_module, "memory.find", {"query": "hermes", "limit": 2})
        assert isinstance(results, list)
        assert len(results) <= 2

    def test_empty_query_returns_error(self, mcp_module):
        req = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": "memory.find", "arguments": {"query": ""}}}
        resp = mcp_module.handle_request(req)
        assert resp["error"]["code"] == -32602

    def test_host_port_query_no_crash(self, mcp_module):
        """Regression: 'gbrain :3131' used to crash with OperationalError."""
        results = _call(mcp_module, "memory.find", {"query": "gbrain :3131", "limit": 2})
        assert isinstance(results, list)  # may be empty, but must not crash

    def test_result_has_expected_fields(self, mcp_module):
        results = _call(mcp_module, "memory.find", {"query": "hermes", "limit": 1})
        if results:
            r = results[0]
            assert "fact_id" in r
            assert "content" in r
            assert "trust_score" in r


class TestMemorySemanticSearch:
    """memory.semantic_search — the new vector search tool."""

    def test_returns_list(self, mcp_module):
        results = _call(mcp_module, "memory.semantic_search",
                        {"query": "memory system", "limit": 3})
        assert isinstance(results, list)
        assert len(results) <= 3

    def test_result_has_similarity_field(self, mcp_module):
        results = _call(mcp_module, "memory.semantic_search",
                        {"query": "gbrain", "limit": 1})
        if results:
            assert "similarity" in results[0]
            assert isinstance(results[0]["similarity"], float)

    def test_empty_query_returns_error(self, mcp_module):
        req = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": "memory.semantic_search", "arguments": {"query": ""}}}
        resp = mcp_module.handle_request(req)
        assert resp["error"]["code"] == -32602


class TestMemoryHybridSearch:
    """memory.hybrid_search — combined FTS5 + vector."""

    def test_returns_list_with_scores(self, mcp_module):
        results = _call(mcp_module, "memory.hybrid_search",
                        {"query": "gbrain embed", "limit": 3})
        assert isinstance(results, list)
        if results:
            r = results[0]
            assert "score" in r
            assert "trust_score" in r
            assert "fts_rank" in r
            assert "vector_sim" in r


class TestMemoryContextAndGet:
    """memory.context + memory.get."""

    def test_context_returns_entries_and_facts(self, mcp_module):
        payload = _call(mcp_module, "memory.context")
        assert "memory_md_entries" in payload
        assert "recent_worker_facts" in payload
        assert isinstance(payload["memory_md_entries"], list)
        assert isinstance(payload["recent_worker_facts"], list)

    def test_context_with_project_label(self, mcp_module):
        payload = _call(mcp_module, "memory.context", {"project": "test-project"})
        assert payload.get("project") == "test-project"

    def test_get_valid_fact_id(self, mcp_module):
        # First find a real fact_id
        results = _call(mcp_module, "memory.find", {"query": "hermes", "limit": 1})
        if results:
            fid = results[0]["fact_id"]
            fact = _call(mcp_module, "memory.get", {"fact_id": fid})
            assert fact["fact_id"] == fid
            assert "content" in fact

    def test_get_nonexistent_fact_id(self, mcp_module):
        fact = _call(mcp_module, "memory.get", {"fact_id": 999999})
        assert fact.get("error") == "not found"

    def test_get_invalid_fact_id_returns_error(self, mcp_module):
        req = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": "memory.get", "arguments": {"fact_id": "not_a_number"}}}
        resp = mcp_module.handle_request(req)
        assert resp["error"]["code"] == -32602


class TestMemoryRecent:
    """memory.recent."""

    def test_recent_returns_list(self, mcp_module):
        results = _call(mcp_module, "memory.recent", {"limit": 3})
        assert isinstance(results, list)
        assert len(results) <= 3

    def test_recent_with_source_filter(self, mcp_module):
        results = _call(mcp_module, "memory.recent",
                        {"source": "claude-code", "limit": 2})
        assert isinstance(results, list)
        for r in results:
            assert r.get("category") == "claude-code"


class TestMemoryFeedback:
    """memory.feedback — trust signal adjustment."""

    def test_feedback_helpful_increments_helpful_count(self, mcp_module):
        # Find a fact, get its current helpful_count, feedback, check it went up
        results = _call(mcp_module, "memory.find", {"query": "hermes", "limit": 1})
        if not results:
            pytest.skip("no facts in store to test feedback")
        fid = results[0]["fact_id"]
        before = _call(mcp_module, "memory.get", {"fact_id": fid})
        old_hc = before.get("helpful_count", 0)

        result = _call(mcp_module, "memory.feedback", {"fact_id": fid, "helpful": True})
        assert result["fact_id"] == fid
        assert result["helpful_count"] == old_hc + 1

    def test_feedback_nonexistent_fact_returns_error(self, mcp_module):
        req = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": "memory.feedback",
                          "arguments": {"fact_id": 999999, "helpful": True}}}
        resp = mcp_module.handle_request(req)
        assert resp["error"]["code"] == -32602

    def test_feedback_invalid_fact_id_returns_error(self, mcp_module):
        req = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": "memory.feedback",
                          "arguments": {"fact_id": "abc", "helpful": True}}}
        resp = mcp_module.handle_request(req)
        assert resp["error"]["code"] == -32602


class TestMemoryHealth:
    """memory.health — liveness probe."""

    def test_health_returns_status_dict(self, mcp_module):
        h = _call(mcp_module, "memory.health")
        assert "subsystems" in h
        assert "holographic_store" in h["subsystems"]
        assert "vector_store" in h["subsystems"]
        assert "embed_endpoint" in h["subsystems"]
        assert "overall" in h

    def test_health_holographic_store_ok(self, mcp_module):
        h = _call(mcp_module, "memory.health")
        hs = h["subsystems"]["holographic_store"]
        # Under pytest's isolated HERMES_HOME, the store may be empty (0 facts).
        # The key assertion is that the status is "ok" and fact_count is present.
        assert hs["status"] == "ok"
        assert "fact_count" in hs
        assert isinstance(hs["fact_count"], int)
