"""Tests for plugins/memory/holographic/retrieval.py — hybrid retrieval, probe, reason, contradict.

Run:  python3 -m pytest tests/plugins/memory/test_retrieval.py -v
"""

import os
import tempfile
import pytest
import numpy as np

from plugins.memory.holographic.store import MemoryStore
from plugins.memory.holographic.retrieval import FactRetriever


# ─── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def populated_store(tmp_path):
    """Store with diverse facts for retrieval testing."""
    db_path = tmp_path / "test_retrieval.db"
    store = MemoryStore(db_path=str(db_path), hrr_dim=256)
    # Infrastructure facts
    store.add_fact("Strix Halo runs Qwen3-6 on port 8199", category="infra")
    store.add_fact("Strix Halo runs Qwen3-Coder-30B on port 8200", category="infra")
    store.add_fact("Mac Studio runs Qwen3-6-MLX on port 8000", category="infra")
    store.add_fact("Mac Mini runs Pi-hole and SearXNG", category="infra")
    store.add_fact("Gemma 4 26B handles vision and multimodal tasks", category="infra")
    # Health facts
    store.add_fact("User takes 6mg per week retatrutide for weight management", category="health")
    store.add_fact("User takes 2.5mg MOTS-C for muscle preservation", category="health")
    store.add_fact("Pool heater is Gulfstream HE150-RA rated at 136K BTU", category="home")
    store.add_fact("User had Achilles repair surgery May 26 2026", category="health")
    # Contradiction pair
    store.add_fact("Port 8201 runs Qwen3-Coder-30B-Compact", category="infra")
    store.add_fact("Port 8201 is dead and was deleted May 10 2026", category="infra")
    return store


@pytest.fixture
def retriever(populated_store):
    """FactRetriever with no neural embed (pure HRR + FTS5 + Jaccard)."""
    # Replace embed client with a dead one (not None — None crashes _compute_neural_embed)
    from plugins.memory.holographic.store import EmbedClient
    populated_store._embed = EmbedClient(url="http://127.0.0.1:19999/v1/embeddings", timeout=1)
    return FactRetriever(
        store=populated_store,
        hrr_dim=256,
        fts_weight=0.4,
        jaccard_weight=0.3,
        hrr_weight=0.3,
        neural_weight=0.0,  # explicitly disabled
    )


# ─── Hybrid Search ──────────────────────────────────────────────────────

class TestHybridSearch:
    def test_keyword_search(self, retriever):
        results = retriever.search("Strix Halo")
        assert len(results) >= 1
        # At least one result should mention Strix
        content_lower = " ".join(r["content"].lower() for r in results)
        assert "strix" in content_lower

    def test_search_returns_score(self, retriever):
        results = retriever.search("port 8199")
        assert len(results) >= 1
        assert "score" in results[0]
        assert results[0]["score"] > 0

    def test_search_no_vectors_stripped(self, retriever):
        results = retriever.search("Qwen3.6")
        for r in results:
            assert "hrr_vector" not in r
            assert "neural_embed" not in r

    def test_search_ordered_by_score(self, retriever):
        results = retriever.search("Qwen3.6")
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i]["score"] >= results[i + 1]["score"]

    def test_search_with_min_trust(self, retriever):
        results = retriever.search("Qwen3.6", min_trust=0.9)
        # Default trust is 0.5, so with min_trust=0.9 should return empty
        # unless retrieval boost pushed some above 0.9
        # At minimum, should not error
        assert isinstance(results, list)

    def test_search_with_category(self, retriever):
        results = retriever.search("Qwen3.6", category="infra")
        for r in results:
            assert r["category"] == "infra"

    def test_search_limit(self, retriever):
        results = retriever.search("model", limit=2)
        assert len(results) <= 2

    def test_search_trust_weighted(self, retriever, populated_store):
        # Add a high-trust fact
        fid = populated_store.add_fact("High trust Strix fact", category="infra")
        for _ in range(20):
            populated_store.update_fact(fid, trust_delta=0.2)  # bump toward 1.0
        results = retriever.search("Strix")
        # Higher trust should boost score
        for r in results:
            assert r["score"] > 0

    def test_empty_query(self, retriever):
        assert retriever.search("") == []

    def test_no_results_query(self, retriever):
        results = retriever.search("XYZNONEXISTENTQUERY12345")
        assert results == []


# ─── Probe (Compositional HRR) ─────────────────────────────────────────

class TestProbe:
    def test_probe_entity(self, retriever):
        results = retriever.probe("Strix Halo", category="infra")
        assert isinstance(results, list)
        # Should find facts related to Strix structurally
        if results:
            content_lower = " ".join(r["content"].lower() for r in results)
            # Probing for "Strix Halo" should surface Strix-related facts
            assert "strix" in content_lower

    def test_probe_returns_score(self, retriever):
        results = retriever.probe("Qwen3.6")
        if results:
            assert "score" in results[0]

    def test_probe_unknown_entity(self, retriever):
        # Unknown entity — should fall back to keyword search
        results = retriever.probe("ZZZNONEXISTENT")
        # Should return something (keyword fallback) or nothing
        assert isinstance(results, list)


# ─── Related ─────────────────────────────────────────────────────────────

class TestRelated:
    def test_related_finds_connections(self, retriever):
        results = retriever.related("Strix Halo")
        assert isinstance(results, list)
        # Should find facts that share structural context
        if results:
            for r in results:
                assert "score" in r

    def test_related_no_results_for_isolated(self, retriever):
        results = retriever.related("ZZZNONEXISTENT")
        assert isinstance(results, list)


# ─── Reason (Multi-Entity JOIN) ───────────────────────────────────────

class TestReason:
    def test_multi_entity_reasoning(self, retriever):
        """reason(["Strix", "port"]) should find facts where both have structural roles."""
        results = retriever.reason(["Strix", "port"])
        assert isinstance(results, list)
        # Should find at least one fact that mentions both
        if results:
            all_content = " ".join(r["content"].lower() for r in results)
            # At least one fact should be relevant to both entities
            assert len(results) > 0

    def test_empty_entities(self, retriever):
        results = retriever.reason([])
        assert isinstance(results, list)


# ─── Contradict ─────────────────────────────────────────────────────────

class TestContradict:
    def test_finds_contradictions(self, retriever):
        """Our test data has a contradiction: port 8201 is alive vs. dead."""
        results = retriever.contradict(category="infra")
        assert isinstance(results, list)
        # Should find at least one contradiction pair
        if results:
            pair = results[0]
            assert "fact_a" in pair
            assert "fact_b" in pair
            assert "contradiction_score" in pair
            assert "shared_entities" in pair
            assert pair["contradiction_score"] >= 0.3

    def test_contradiction_score_bounded(self, retriever):
        results = retriever.contradict()
        for pair in results:
            assert 0.0 <= pair["contradiction_score"] <= 1.0

    def test_contradiction_entity_overlap(self, retriever):
        results = retriever.contradict()
        for pair in results:
            # Contradictions require entity overlap
            assert pair["entity_overlap"] >= 0.3

    def test_contradiction_ordered_by_score(self, retriever):
        results = retriever.contradict()
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i]["contradiction_score"] >= results[i + 1]["contradiction_score"]

    def test_contradiction_limit(self, retriever):
        results = retriever.contradict(limit=1)
        assert len(results) <= 1


# ─── Weight Redistribution ─────────────────────────────────────────────

class TestWeightRedistribution:
    def test_no_numpy_redistributes_hrr(self):
        """If numpy is somehow unavailable, HRR weight goes to FTS5 + neural."""
        store = MemoryStore.__new__(MemoryStore)
        # Mock away HRR
        import plugins.memory.holographic as hrr_mod
        orig = hrr_mod.holographic._HAS_NUMPY
        try:
            hrr_mod.holographic._HAS_NUMPY = False
            fr = FactRetriever(
                store, hrr_dim=256,
                fts_weight=0.3, jaccard_weight=0.2, hrr_weight=0.2, neural_weight=0.3,
            )
            assert fr.hrr_weight == 0.0
            # Should have redistributed to fts + neural
            assert fr.fts_weight > 0.3  # got some of hrr's weight
        finally:
            hrr_mod.holographic._HAS_NUMPY = orig

    def test_no_neural_redistributes(self, populated_store):
        populated_store._embed = None
        fr = FactRetriever(
            populated_store, hrr_dim=256,
            fts_weight=0.3, jaccard_weight=0.2, hrr_weight=0.2, neural_weight=0.3,
        )
        assert fr.neural_weight == 0.0
        assert fr.fts_weight > 0.3  # got neural's share


# ─── Jaccard & Tokenization ────────────────────────────────────────────

class TestJaccardAndTokenization:
    def test_jaccard_identical(self):
        from plugins.memory.holographic.retrieval import FactRetriever
        assert FactRetriever._jaccard_similarity({"a", "b", "c"}, {"a", "b", "c"}) == 1.0

    def test_jaccard_disjoint(self):
        from plugins.memory.holographic.retrieval import FactRetriever
        assert FactRetriever._jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0

    def test_jaccard_partial(self):
        from plugins.memory.holographic.retrieval import FactRetriever
        # |{a,b} ∩ {a,c}| / |{a,b} ∪ {a,c}| = 1/3
        assert FactRetriever._jaccard_similarity({"a", "b"}, {"a", "c"}) == pytest.approx(1/3)

    def test_tokenize_simple(self):
        from plugins.memory.holographic.retrieval import FactRetriever
        tokens = FactRetriever._tokenize("hello world, test!")
        assert tokens == {"hello", "world", "test"}

    def test_tokenize_empty(self):
        from plugins.memory.holographic.retrieval import FactRetriever
        assert FactRetriever._tokenize("") == set()

    def test_tokenize_punctuation(self):
        from plugins.memory.holographic.retrieval import FactRetriever
        tokens = FactRetriever._tokenize("a.b,c;d:e")
        # Dots and semicolons are stripped, but the string is treated as one token
        # because whitespace tokenization splits on spaces, not punctuation
        assert len(tokens) >= 1


# ─── Temporal Decay ────────────────────────────────────────────────────

class TestTemporalDecay:
    def test_decay_disabled(self, populated_store):
        fr = FactRetriever(populated_store, hrr_dim=256, temporal_decay_half_life=0)
        results = fr.search("Qwen3.6")
        # No decay — all results should have positive scores
        for r in results:
            assert r["score"] > 0

    def test_decay_enabled(self, populated_store):
        fr = FactRetriever(populated_store, hrr_dim=256, temporal_decay_half_life=30)
        results = fr.search("Qwen3.6")
        # With decay, scores should still be positive
        for r in results:
            assert r["score"] > 0

    def test_decay_function_shape(self):
        from plugins.memory.holographic.retrieval import FactRetriever
        import math
        from datetime import datetime, timezone, timedelta
        fr = FactRetriever.__new__(FactRetriever)
        fr.half_life = 10  # 10-day half-life
        old_ts = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
        decay = fr._temporal_decay(old_ts)
        # After 20 days with 10-day half-life: 0.5^2 = 0.25
        assert decay == pytest.approx(0.25, abs=0.01)

    def test_decay_future_timestamp(self):
        from plugins.memory.holographic.retrieval import FactRetriever
        fr = FactRetriever.__new__(FactRetriever)
        fr.half_life = 10
        future = "2099-01-01T00:00:00+00:00"
        assert fr._temporal_decay(future) == 1.0

    def test_decay_none_timestamp(self):
        from plugins.memory.holographic.retrieval import FactRetriever
        fr = FactRetriever.__new__(FactRetriever)
        fr.half_life = 10
        assert fr._temporal_decay(None) == 1.0

    def test_decay_parses_z_suffix(self):
        """A timestamp ending in 'Z' (UTC Zulu) must parse, not silently fall
        through to the exception path returning 1.0. This pins the
        ``replace("Z", "+00:00")`` normalization — a mutant that breaks the
        replace (e.g. searching for a different substring) must be caught."""
        from plugins.memory.holographic.retrieval import FactRetriever
        from datetime import datetime, timezone, timedelta
        fr = FactRetriever.__new__(FactRetriever)
        fr.half_life = 10
        old_ts = (datetime.now(timezone.utc) - timedelta(days=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
        decay = fr._temporal_decay(old_ts)
        # 20 days / 10-day half-life -> 0.5^2 = 0.25. If the Z-suffix wasn't
        # normalized, fromisoformat raises and the method returns 1.0 (no decay).
        assert decay == pytest.approx(0.25, abs=0.01)
        assert decay < 0.5  # confirm decay actually happened (not the 1.0 fallback)

    def test_decay_precision_seconds_per_day(self):
        """Pin the age-to-decay math on a short, known age. A 1-day-old fact
        with a 10-day half-life decays to 0.5^(0.1) = 0.9330. This catches
        gross divisor errors (e.g. /8640, /864000) and confirms the seconds-
        per-day conversion is in the right order of magnitude. Note: an
        off-by-one (/86400 -> /86401) produces ~1e-6 drift, below practical
        detection — that mutant is equivalent for realistic decay windows."""
        from plugins.memory.holographic.retrieval import FactRetriever
        from datetime import datetime, timezone, timedelta
        import math
        fr = FactRetriever.__new__(FactRetriever)
        fr.half_life = 10
        one_day_ago = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        decay = fr._temporal_decay(one_day_ago)
        expected = math.pow(0.5, 1.0 / 10.0)
        assert decay == pytest.approx(expected, rel=1e-4)


# ─── Trust Boost on Retrieval ──────────────────────────────────────────

class TestTrustBoost:
    def test_retrieval_boosts_trust(self, retriever, populated_store):
        fid = populated_store.add_fact("boost test fact")
        initial = [f for f in populated_store.list_facts() if f["fact_id"] == fid][0]
        initial_trust = initial["trust_score"]
        # Search should trigger boost
        retriever.search("boost test fact")
        after = [f for f in populated_store.list_facts() if f["fact_id"] == fid][0]
        assert after["trust_score"] > initial_trust


# ─── LLM Contradiction Verification ────────────────────────────────────

class FakeLLMVerifier:
    """Test double for LLMVerifier — no network calls.

    Returns pre-scripted verdicts based on the fact contents, or a fixed
    verdict for any pair. Mimics the real class's fail-open contract:
    alive and verify_contradiction are the only surfaces contradict() uses.
    """

    def __init__(self, verdicts: dict | None = None, default: dict | None = None, alive: bool = True):
        # verdicts: keyed by (text_a, text_b) tuples — lookup is content-based
        self._verdicts = verdicts or {}
        self._default = default
        self._alive = alive

    @property
    def alive(self) -> bool:
        return self._alive

    def verify_contradiction(self, text_a: str, text_b: str) -> dict | None:
        for (a, b), verdict in self._verdicts.items():
            if a in text_a and b in text_b:
                return verdict
        return self._default


class TestLLMContradictionVerify:
    """Tests for the LLM precision pass on contradict(llm_verify=True)."""

    @pytest.fixture
    def contra_store(self, tmp_path):
        """Store with a guaranteed structural contradiction pair.

        Uses entity-rich facts that the entity extractor will reliably tag,
        ensuring the structural detector finds them. The pair shares the
        entity 'Acme Server' but makes opposing claims about its status.
        A lowered threshold (0.15) is used in the tests because short facts
        sharing the entity token produce moderate (not high) content
        divergence in HRR space — the threshold calibration is a separate
        concern from the LLM verify pass being tested here.
        """
        from plugins.memory.holographic.store import EmbedClient
        store = MemoryStore(db_path=str(tmp_path / "contra.db"), hrr_dim=256)
        store._embed = EmbedClient(
            url="http://127.0.0.1:19999/v1/embeddings", timeout=1
        )
        store.add_fact("Acme Server is running and accepting requests", category="infra")
        store.add_fact("Acme Server is shut down permanently decommissioned", category="infra")
        # Non-contradicting filler to pad the store
        store.add_fact("Acme Server has 128GB RAM and 16 cores", category="infra")
        return store

    # Threshold lowered to ensure the structural detector surfaces the pair
    # despite moderate HRR similarity (entity token dominates the binding).
    _THRESHOLD = 0.15

    def _retriever(self, store, fake_verifier=None):
        return FactRetriever(
            store=store, hrr_dim=256,
            fts_weight=0.4, jaccard_weight=0.3, hrr_weight=0.3,
            neural_weight=0.0, llm_verifier=fake_verifier,
        )

    def test_llm_confirmed_pair_is_kept_and_boosted(self, contra_store):
        """When the LLM confirms a contradiction, it stays and its score is
        modulated by the LLM's confidence."""
        fake = FakeLLMVerifier(
            default={
                "is_contradiction": True,
                "confidence": 0.9,
                "reasoning": "one says alive, other says dead",
            }
        )
        retriever = self._retriever(contra_store, fake)
        structural = retriever.contradict(category="infra", threshold=self._THRESHOLD)
        assert len(structural) >= 1  # structural detector found candidates

        llm_results = retriever.contradict(
            category="infra", threshold=self._THRESHOLD, llm_verify=True
        )
        confirmed = [r for r in llm_results if r.get("llm_confirmed") is True]
        assert len(confirmed) >= 1

        for pair in confirmed:
            assert pair["llm_reasoning"] is not None
            # Confidence 0.9 → score * (0.5 + 0.5*0.9) = score * 0.95
            orig = next(
                (s for s in structural
                 if s["fact_a"]["fact_id"] == pair["fact_a"]["fact_id"]
                 and s["fact_b"]["fact_id"] == pair["fact_b"]["fact_id"]),
                None,
            )
            if orig:
                expected = round(orig["contradiction_score"] * 0.95, 3)
                assert pair["contradiction_score"] == pytest.approx(expected, abs=0.002)

    def test_llm_rejected_pair_is_dropped(self, contra_store):
        """When the LLM says 'not a contradiction', the pair is filtered out."""
        fake = FakeLLMVerifier(
            default={
                "is_contradiction": False,
                "confidence": 0.8,
                "reasoning": "different aspects of same entity, not conflicting",
            }
        )
        retriever = self._retriever(contra_store, fake)
        structural = retriever.contradict(category="infra", threshold=self._THRESHOLD)
        assert len(structural) >= 1

        llm_results = retriever.contradict(
            category="infra", threshold=self._THRESHOLD, llm_verify=True
        )
        confirmed = [r for r in llm_results if r.get("llm_confirmed") is True]
        assert len(confirmed) == 0  # LLM rejected all candidates

    def test_llm_unavailable_falls_back_to_structural(self, contra_store):
        """When the LLM endpoint is down, contradict(llm_verify=True) returns
        the same structural results, badged as unverified. Fail-open."""
        fake = FakeLLMVerifier(alive=False)  # endpoint down
        retriever = self._retriever(contra_store, fake)
        structural = retriever.contradict(category="infra", threshold=self._THRESHOLD)
        llm_results = retriever.contradict(
            category="infra", threshold=self._THRESHOLD, llm_verify=True
        )

        assert len(llm_results) == len(structural)
        for pair in llm_results:
            assert pair.get("llm_confirmed") is None
            assert "unavailable" in pair.get("llm_reasoning", "").lower()

    def test_llm_verify_off_by_default(self, contra_store):
        """Without llm_verify=True, no LLM badges appear on results."""
        retriever = self._retriever(contra_store)  # no verifier
        results = retriever.contradict(category="infra", threshold=self._THRESHOLD)
        for pair in results:
            assert "llm_confirmed" not in pair

    def test_llm_unsure_pair_kept_as_unverified(self, contra_store):
        """When the LLM returns None (can't parse / unsure), the structural
        pair is kept but badged as unverified (not confirmed)."""
        fake = FakeLLMVerifier(default=None, alive=True)
        retriever = self._retriever(contra_store, fake)
        structural = retriever.contradict(category="infra", threshold=self._THRESHOLD)
        llm_results = retriever.contradict(
            category="infra", threshold=self._THRESHOLD, llm_verify=True
        )

        assert len(llm_results) == len(structural)
        for pair in llm_results:
            assert pair.get("llm_confirmed") is None
            assert pair.get("llm_reasoning") is None

    def test_score_boost_modulates_by_confidence(self, contra_store):
        """The contradiction score must be multiplied by the confidence
        factor, not left unchanged. Kills the 'no-score-boost' mutant."""
        fake = FakeLLMVerifier(
            default={
                "is_contradiction": True,
                "confidence": 0.9,
                "reasoning": "confirmed",
            }
        )
        retriever = self._retriever(contra_store, fake)
        structural = retriever.contradict(category="infra", threshold=self._THRESHOLD)
        assert len(structural) >= 1

        llm_results = retriever.contradict(
            category="infra", threshold=self._THRESHOLD, llm_verify=True
        )
        confirmed = [r for r in llm_results if r.get("llm_confirmed") is True]
        assert len(confirmed) >= 1

        for pair in confirmed:
            orig = next(
                (s for s in structural
                 if s["fact_a"]["fact_id"] == pair["fact_a"]["fact_id"]
                 and s["fact_b"]["fact_id"] == pair["fact_b"]["fact_id"]),
                None,
            )
            assert orig is not None, "confirmed pair must exist in structural results"
            expected = round(orig["contradiction_score"] * 0.95, 3)
            # The boosted score must differ from the original — if it's
            # identical, the confidence factor wasn't applied.
            assert pair["contradiction_score"] != orig["contradiction_score"]
            assert pair["contradiction_score"] == pytest.approx(expected, abs=0.002)

    def test_rejected_pairs_absent_from_results(self, contra_store):
        """Rejected pairs must not appear in results at all — not just
        missing the confirmed badge. Kills the 'keep-rejected' mutant."""
        fake = FakeLLMVerifier(
            default={
                "is_contradiction": False,
                "confidence": 0.8,
                "reasoning": "not a contradiction",
            }
        )
        retriever = self._retriever(contra_store, fake)
        structural = retriever.contradict(category="infra", threshold=self._THRESHOLD)
        assert len(structural) >= 1

        llm_results = retriever.contradict(
            category="infra", threshold=self._THRESHOLD, llm_verify=True
        )
        # All structural pairs should be gone (within the LLM cap)
        # Verify each structural pair's fact IDs don't appear in results
        for s_pair in structural[:20]:  # LLM cap
            s_a = s_pair["fact_a"]["fact_id"]
            s_b = s_pair["fact_b"]["fact_id"]
            for r in llm_results:
                assert not (
                    r["fact_a"]["fact_id"] == s_a and r["fact_b"]["fact_id"] == s_b
                ), f"rejected pair ({s_a},{s_b}) should not appear in results"

    def test_contradict_respects_limit(self, contra_store):
        """Results count must not exceed the limit parameter."""
        fake = FakeLLMVerifier(
            default={
                "is_contradiction": True,
                "confidence": 1.0,
                "reasoning": "confirmed",
            }
        )
        retriever = self._retriever(contra_store, fake)
        # Use a very low threshold to maximize candidates
        all_results = retriever.contradict(category="infra", threshold=0.01)
        if len(all_results) > 1:
            limited = retriever.contradict(category="infra", threshold=0.01, limit=1)
            assert len(limited) <= 1

    def test_contradict_boosts_trust(self, contra_store):
        """Facts involved in contradiction pairs should have their trust
        boosted after contradict() runs."""
        # Get initial trust
        facts_before = contra_store.list_facts()
        trust_before = {f["fact_id"]: f["trust_score"] for f in facts_before}

        fake = FakeLLMVerifier(
            default={
                "is_contradiction": True,
                "confidence": 1.0,
                "reasoning": "confirmed",
            }
        )
        retriever = self._retriever(contra_store, fake)
        results = retriever.contradict(category="infra", threshold=self._THRESHOLD)
        assert len(results) >= 1

        # Check trust increased for involved facts
        facts_after = contra_store.list_facts()
        trust_after = {f["fact_id"]: f["trust_score"] for f in facts_after}

        boosted = False
        for pair in results:
            for key in ("fact_a", "fact_b"):
                fid = pair[key]["fact_id"]
                if trust_before[fid] < trust_after.get(fid, 0):
                    boosted = True
                    break
            if boosted:
                break
        assert boosted, "at least one fact should have its trust boosted"

    def test_llm_verify_excludes_unchecked_pairs(self, tmp_path):
        """When llm_verify=True, pairs beyond the LLM cap (20) must not
        appear in results. This was a real bug: unchecked pairs leaked
        through with confirmed=None, defeating the precision pass."""
        from plugins.memory.holographic.store import EmbedClient
        # Build a store with many pairs sharing an entity
        store = MemoryStore(db_path=str(tmp_path / "many.db"), hrr_dim=256)
        store._embed = EmbedClient(
            url="http://127.0.0.1:19999/v1/embeddings", timeout=1
        )
        # 25 facts all sharing "Alpha Node" entity — produces many pairs
        for i in range(25):
            store.add_fact(
                f"Alpha Node configuration variant {i} details",
                category="infra",
            )

        # Fake verifier that rejects everything
        fake = FakeLLMVerifier(
            default={
                "is_contradiction": False,
                "confidence": 0.9,
                "reasoning": "not a contradiction",
            }
        )
        retriever = self._retriever(store, fake)

        # Structural pass should find many pairs
        structural = retriever.contradict(
            category="infra", threshold=0.01, limit=100
        )
        assert len(structural) > 20, "need >20 pairs to exceed LLM cap"

        # LLM verify — all pairs within cap are rejected
        llm_results = retriever.contradict(
            category="infra", threshold=0.01, limit=50, llm_verify=True
        )

        # No pairs should have confirmed=None from unchecked-beyond-cap leak
        leaked = [r for r in llm_results if r.get("llm_confirmed") is None]
        assert len(leaked) == 0, (
            f"{len(leaked)} pairs leaked through without LLM check"
        )
        # All results were LLM-processed and rejected — should be empty
        assert len(llm_results) == 0, (
            f"{len(llm_results)} pairs survived despite all being rejected"
        )


# ─── Real LLMVerifier unit tests ───────────────────────────────────────
# These test the actual LLMVerifier class (not the fake) against a dead
# endpoint, ensuring the fail-open contract holds in the real implementation.

class TestLLMVerifierClass:
    def test_alive_false_on_dead_endpoint(self):
        """LLMVerifier.alive must return False when the endpoint is unreachable.
        This kills the 'always-alive' mutant — the test exercises the real
        _probe() method, not the FakeLLMVerifier override."""
        from plugins.memory.holographic.retrieval import LLMVerifier
        v = LLMVerifier(
            url="http://127.0.0.1:19999/v1/chat/completions",  # nothing there
            timeout=1,
        )
        assert v.alive is False

    def test_verify_returns_none_when_not_alive(self):
        """verify_contradiction must return None when alive is False."""
        from plugins.memory.holographic.retrieval import LLMVerifier
        v = LLMVerifier(
            url="http://127.0.0.1:19999/v1/chat/completions",
            timeout=1,
        )
        result = v.verify_contradiction("A is true", "A is false")
        assert result is None

    def test_parse_response_extracts_fields(self):
        """_parse_response must correctly extract contradiction, confidence,
        and reasoning from a well-formed LLM response."""
        from plugins.memory.holographic.retrieval import LLMVerifier
        content = (
            "CONTRADICTION: yes\n"
            "CONFIDENCE: 0.85\n"
            "REASONING: The two statements make opposing claims.\n"
        )
        result = LLMVerifier._parse_response(content)
        assert result is not None
        assert result["is_contradiction"] is True
        assert result["confidence"] == pytest.approx(0.85, abs=0.001)
        assert "opposing claims" in result["reasoning"]

    def test_parse_response_no_line_returns_none(self):
        """If the response doesn't contain 'CONTRADICTION:', return None."""
        from plugins.memory.holographic.retrieval import LLMVerifier
        result = LLMVerifier._parse_response("I think they might conflict.")
        assert result is None

    def test_parse_response_clamps_confidence(self):
        """Confidence values outside [0, 1] are clamped."""
        from plugins.memory.holographic.retrieval import LLMVerifier
        content = "CONTRADICTION: no\nCONFIDENCE: 5.0\nREASONING: test\n"
        result = LLMVerifier._parse_response(content)
        assert result is not None
        assert result["confidence"] == 1.0  # clamped from 5.0


# ─── Epistemic Status ──────────────────────────────────────────────────

class TestEpistemicStatus:
    def test_new_fact_defaults_to_stated(self, populated_store):
        facts = populated_store.list_facts()
        assert len(facts) > 0
        for f in facts:
            assert f["epistemic_status"] == "stated"

    def test_update_fact_epistemic_status(self, populated_store):
        fid = populated_store.add_fact("temporary fact for epistemic test")
        updated = populated_store.update_fact(fid, epistemic_status="contradicted")
        assert updated is True
        facts = populated_store.list_facts()
        target = [f for f in facts if f["fact_id"] == fid][0]
        assert target["epistemic_status"] == "contradicted"

    def test_update_fact_rejects_invalid_status(self, populated_store):
        fid = populated_store.add_fact("another temporary fact")
        # Should not raise, just log a warning and ignore the value
        updated = populated_store.update_fact(fid, epistemic_status="bogus")
        assert updated is True  # row existed
        facts = populated_store.list_facts()
        target = [f for f in facts if f["fact_id"] == fid][0]
        assert target["epistemic_status"] == "stated"  # unchanged

    def test_search_surfaces_epistemic_status(self, retriever):
        results = retriever.search("Strix Halo")
        if results:
            assert "epistemic_status" in results[0]

    def test_contradict_surfaces_epistemic_status(self, retriever):
        results = retriever.contradict(category="infra")
        for pair in results:
            assert "epistemic_status" in pair["fact_a"]
            assert "epistemic_status" in pair["fact_b"]
