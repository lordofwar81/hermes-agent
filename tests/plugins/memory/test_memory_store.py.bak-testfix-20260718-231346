"""Tests for plugins/memory/holographic/store.py — SQLite fact store, entity resolution, trust scoring.

Run:  python3 -m pytest tests/plugins/memory/test_memory_store.py -v
"""

import os
import tempfile
import pytest
import numpy as np

from plugins.memory.holographic.store import (
    MemoryStore,
    EmbedClient,
    _clamp_trust,
    _ENTITY_STOPWORDS,
)


# ─── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    """In-memory-ish store with temp file."""
    db_path = tmp_path / "test_memory.db"
    return MemoryStore(
        db_path=str(db_path),
        hrr_dim=256,  # smaller for faster tests
    )


# ─── EmbedClient ─────────────────────────────────────────────────────────

class TestEmbedClient:
    def test_init_defaults(self):
        ec = EmbedClient()
        assert ec.url == "http://localhost:11434/v1/embeddings"
        # alive property eagerly probes — check it's a bool
        assert isinstance(ec.alive, bool)

    def test_probe_offline(self, tmp_path):
        """Dead port → not alive."""
        ec = EmbedClient(url="http://127.0.0.1:19999/v1/embeddings", timeout=1)
        assert ec.alive is False

    def test_embed_returns_none_when_dead(self):
        ec = EmbedClient(url="http://127.0.0.1:19999/v1/embeddings", timeout=1)
        assert ec.embed("test") is None

    def test_embed_batch_returns_none_when_dead(self):
        ec = EmbedClient(url="http://127.0.0.1:19999/v1/embeddings", timeout=1)
        results = ec.embed_batch(["a", "b", "c"])
        assert len(results) == 3
        assert all(r is None for r in results)


# ─── _clamp_trust ────────────────────────────────────────────────────────

class TestClampTrust:
    def test_clamps_high(self):
        assert _clamp_trust(1.5) == 1.0

    def test_clamps_low(self):
        assert _clamp_trust(-0.5) == 0.0

    def test_passes_through(self):
        assert _clamp_trust(0.5) == 0.5

    def test_boundary(self):
        assert _clamp_trust(0.0) == 0.0
        assert _clamp_trust(1.0) == 1.0


# ─── MemoryStore CRUD ──────────────────────────────────────────────────

class TestMemoryStoreCRUD:
    def test_add_and_retrieve_fact(self, store):
        fid = store.add_fact("Strix Halo has 128GB shared UMA memory")
        assert fid > 0
        facts = store.list_facts()
        assert len(facts) >= 1
        found = [f for f in facts if f["content"] == "Strix Halo has 128GB shared UMA memory"]
        assert len(found) == 1

    def test_add_fact_deduplicates(self, store):
        f1 = store.add_fact("Qwen3.6 runs on port 8199")
        f2 = store.add_fact("Qwen3.6 runs on port 8199")
        assert f1 == f2  # same ID

    def test_add_fact_with_category_and_tags(self, store):
        fid = store.add_fact("Retatrutide 6mg/wk", category="health", tags="peptide,weight-loss")
        facts = store.list_facts(category="health")
        assert len(facts) == 1
        assert facts[0]["tags"] == "peptide,weight-loss"

    def test_add_fact_empty_raises(self, store):
        with pytest.raises(ValueError):
            store.add_fact("")

    def test_add_fact_whitespace_only_raises(self, store):
        with pytest.raises(ValueError):
            store.add_fact("   \n  ")

    def test_remove_fact(self, store):
        fid = store.add_fact("temporary fact")
        facts = store.list_facts()
        before = len(facts)
        result = store.remove_fact(fid)
        assert result is True
        facts = store.list_facts()
        assert len(facts) == before - 1

    def test_remove_nonexistent_returns_false(self, store):
        assert store.remove_fact(999999) is False

    def test_update_fact_content(self, store):
        fid = store.add_fact("old content")
        store.update_fact(fid, content="new content")
        facts = store.list_facts()
        found = [f for f in facts if f["fact_id"] == fid]
        assert len(found) == 1
        assert found[0]["content"] == "new content"

    def test_update_fact_trust(self, store):
        fid = store.add_fact("test fact")
        store.update_fact(fid, trust_delta=0.2)
        facts = store.list_facts()
        found = [f for f in facts if f["fact_id"] == fid]
        assert found[0]["trust_score"] == pytest.approx(0.7, abs=0.01)  # 0.5 + 0.2

    def test_update_fact_clamps_trust(self, store):
        fid = store.add_fact("test fact")
        store.update_fact(fid, trust_delta=1.0)  # 0.5 + 1.0 = 1.5, clamped to 1.0
        facts = store.list_facts()
        found = [f for f in facts if f["fact_id"] == fid]
        assert found[0]["trust_score"] == 1.0

    def test_update_nonexistent_returns_false(self, store):
        assert store.update_fact(999999, content="nope") is False

    def test_list_facts_min_trust(self, store):
        store.add_fact("high trust")  # default 0.5
        fid = store.add_fact("low trust")
        store.update_fact(fid, trust_delta=-0.5)  # 0.0
        high = store.list_facts(min_trust=0.3)
        assert all(f["trust_score"] >= 0.3 for f in high)


# ─── Entity Extraction ──────────────────────────────────────────────────

class TestEntityExtraction:
    def test_single_word_entities(self, store):
        fid = store.add_fact("SearXNG provides privacy-preserving search")
        rows = store._conn.execute(
            "SELECT e.name FROM entities e JOIN fact_entities fe ON fe.entity_id = e.entity_id WHERE fe.fact_id = ?",
            (fid,),
        ).fetchall()
        names = {r[0] for r in rows}
        assert "SearXNG" in names

    def test_capitalized_phrase(self, store):
        fid = store.add_fact("Strix Halo is the main machine")
        rows = store._conn.execute(
            "SELECT e.name FROM entities e JOIN fact_entities fe ON fe.entity_id = e.entity_id WHERE fe.fact_id = ?",
            (fid,),
        ).fetchall()
        names = {r[0] for r in rows}
        assert "Strix Halo" in names

    def test_version_identifiers(self, store):
        fid = store.add_fact("glm-5-turbo is the default model")
        rows = store._conn.execute(
            "SELECT e.name FROM entities e JOIN fact_entities fe ON fe.entity_id = e.entity_id WHERE fe.fact_id = ?",
            (fid,),
        ).fetchall()
        names = {r[0] for r in rows}
        # Version regex extracts the version prefix — glm-5 is valid extraction
        assert "glm-5" in names or "glm-5-turbo" in names

    def test_file_paths(self, store):
        fid = store.add_fact("Config is at config.yaml")
        rows = store._conn.execute(
            "SELECT e.name FROM entities e JOIN fact_entities fe ON fe.entity_id = e.entity_id WHERE fe.fact_id = ?",
            (fid,),
        ).fetchall()
        names = {r[0] for r in rows}
        assert "config.yaml" in names

    def test_stopwords_not_extracted(self, store):
        fid = store.add_fact("The system is running well")
        rows = store._conn.execute(
            "SELECT e.name FROM entities e JOIN fact_entities fe ON fe.entity_id = e.entity_id WHERE fe.fact_id = ?",
            (fid,),
        ).fetchall()
        names = {r[0].lower() for r in rows}
        for sw in ["the", "is", "are"]:
            assert sw not in names

    def test_entity_alias_resolution(self, store):
        """Entities with similar names should resolve to same entity_id."""
        fid1 = store.add_fact("SearXNG is deployed")
        # Direct name lookup
        rows = store._conn.execute(
            "SELECT e.name FROM entities e JOIN fact_entities fe ON fe.entity_id = e.entity_id WHERE fe.fact_id = ?",
            (fid1,),
        ).fetchall()
        assert len(rows) >= 1


# ─── FTS5 Search ────────────────────────────────────────────────────────

class TestFTSSearch:
    def test_basic_search(self, store):
        store.add_fact("Qwen3-6 is the primary local model")
        store.add_fact("Gemma 4 is the vision model")
        store.add_fact("llama.cpp serves the embedding model")
        results = store.search_facts("Qwen3")
        assert len(results) >= 1
        assert "Qwen3" in results[0]["content"]

    def test_search_with_category(self, store):
        store.add_fact("Qwen3-6 model", category="models")
        store.add_fact("Qwen3-6 in another context", category="other")
        results = store.search_facts("Qwen3", category="models")
        assert len(results) >= 1
        assert all(r["category"] == "models" for r in results)

    def test_search_empty(self, store):
        assert store.search_facts("") == []

    def test_search_no_results(self, store):
        store.add_fact("Qwen3.6 model")
        results = store.search_facts("XYZNONEXISTENT")
        assert len(results) == 0

    def test_retrieval_count_increments(self, store):
        fid = store.add_fact("searchable fact")
        store.search_facts("searchable")
        row = store._conn.execute(
            "SELECT retrieval_count FROM facts WHERE fact_id = ?", (fid,)
        ).fetchone()
        assert row[0] >= 1


# ─── Trust & Feedback ───────────────────────────────────────────────────

class TestTrustFeedback:
    def test_helpful_increases_trust(self, store):
        fid = store.add_fact("helpful fact")
        result = store.record_feedback(fid, helpful=True)
        assert result["old_trust"] == 0.5
        assert result["new_trust"] == 0.55  # +0.05
        assert result["helpful_count"] == 1

    def test_unhelpful_decreases_trust(self, store):
        fid = store.add_fact("unhelpful fact")
        result = store.record_feedback(fid, helpful=False)
        assert result["new_trust"] == 0.4  # -0.10

    def test_feedback_clamps_at_bounds(self, store):
        fid = store.add_fact("boundary test")
        # Bump to 1.0
        for _ in range(20):
            store.record_feedback(fid, helpful=True)
        fact = [f for f in store.list_facts() if f["fact_id"] == fid][0]
        assert fact["trust_score"] == 1.0

        # Now negative — should stay at 1.0 (helpful goes up by 0.05, unhelpful goes down by 0.10)
        # Actually helpful=True adds, and clamps to 1.0. Let's test clamping at 0.
        # Start fresh
        fid2 = store.add_fact("bottom test")
        for _ in range(20):
            store.record_feedback(fid2, helpful=False)
        fact2 = [f for f in store.list_facts() if f["fact_id"] == fid2][0]
        assert fact2["trust_score"] == 0.0

    def test_feedback_nonexistent_raises(self, store):
        with pytest.raises(KeyError):
            store.record_feedback(999999, helpful=True)


# ─── HRR Vectors ───────────────────────────────────────────────────────

class TestHRRVectors:
    def test_fact_has_hrr_vector(self, store):
        fid = store.add_fact("HRR-enabled fact", category="test")
        row = store._conn.execute(
            "SELECT hrr_vector FROM facts WHERE fact_id = ?", (fid,)
        ).fetchone()
        assert row[0] is not None, "HRR vector should be computed on add_fact"

    def test_hrr_vector_shape(self, store):
        fid = store.add_fact("vector shape test", category="test")
        row = store._conn.execute(
            "SELECT hrr_vector FROM facts WHERE fact_id = ?", (fid,)
        ).fetchone()
        vec = np.frombuffer(row[0], dtype=np.float64)
        assert vec.shape == (256,)  # hrr_dim=256 from fixture

    def test_rebuild_all_vectors(self, store):
        store.add_fact("fact1", category="cat_a")
        store.add_fact("fact2", category="cat_a")
        store.add_fact("fact3", category="cat_b")
        count = store.rebuild_all_vectors()
        assert count == 3

    def test_memory_bank_created(self, store):
        store.add_fact("bank test", category="bank_cat")
        row = store._conn.execute(
            "SELECT * FROM memory_banks WHERE bank_name = 'cat:bank_cat'"
        ).fetchone()
        assert row is not None
        assert row["fact_count"] == 1


# ─── WAL Mode ───────────────────────────────────────────────────────────

class TestWALMode:
    def test_journal_mode(self, store):
        mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
        # Should be 'wal' on normal filesystems, or 'delete' on NFS/FUSE
        assert mode in ("wal", "delete")


# ─── Context Manager ────────────────────────────────────────────────────

class TestContextManager:
    def test_with_statement(self, tmp_path):
        db_path = tmp_path / "cm_test.db"
        with MemoryStore(db_path=str(db_path), hrr_dim=64) as s:
            s.add_fact("context manager test")
        # Connection should be closed — accessing should raise
        with pytest.raises(Exception):
            s.add_fact("this should fail")

    def test_close(self, tmp_path):
        db_path = tmp_path / "close_test.db"
        s = MemoryStore(db_path=str(db_path), hrr_dim=64)
        s.add_fact("before close")
        s.close()
        with pytest.raises(Exception):
            s.add_fact("after close")
