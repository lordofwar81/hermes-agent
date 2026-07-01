"""Hybrid keyword/BM25 retrieval for the memory store.

Ported from KIK memory_agent.py — combines FTS5 full-text search with
Jaccard similarity reranking and trust-weighted scoring.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .store import MemoryStore

try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]


# ── LLM contradiction verifier ────────────────────────────────────────
# Precision pass on top of the structural contradict() detector. The
# structural detector flags any fact pair with high entity overlap + low
# content similarity; this class asks a local LLM whether the pair is a
# *real* semantic contradiction, killing false positives (e.g. two facts
# about the same entity that merely describe different aspects).
#
# Fail-open by design: any failure (no endpoint, timeout, parse error)
# returns None, and the caller falls back to the structural score alone.

_DEFAULT_LLM_URL = os.environ.get(
    "HERMES_CONTRADICT_LLM_URL",
    "http://192.168.1.149:8000/v1/chat/completions",  # Mac Studio, Qwen3.6-35B
)
_DEFAULT_LLM_MODEL = os.environ.get(
    "HERMES_CONTRADICT_LLM_MODEL",
    "mlx-community/Qwen3.6-35B-A3B-4bit",
)
_DEFAULT_LLM_KEY = os.environ.get("LLM_SERVER_KEY", "notempty")
_LLM_TIMEOUT = 15  # seconds — fail fast, don't block the detector


class LLMVerifier:
    """Minimal OpenAI-compatible chat client for contradiction verification.

    One instance is cheap to construct (no connection held). alive is
    probed once on first use and cached. All public methods degrade to
    None on any failure — never raises to the caller.
    """

    def __init__(
        self,
        url: str = _DEFAULT_LLM_URL,
        model: str = _DEFAULT_LLM_MODEL,
        api_key: str = _DEFAULT_LLM_KEY,
        timeout: int = _LLM_TIMEOUT,
    ):
        self.url = url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._alive: bool | None = None

    @property
    def alive(self) -> bool:
        if self._alive is not None:
            return self._alive
        self._alive = self._probe()
        return self._alive

    def _probe(self) -> bool:
        try:
            models_url = self.url.replace("/chat/completions", "/models")
            req = urllib.request.Request(
                models_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status == 200
        except Exception:
            logger.debug("LLM verifier probe failed", exc_info=True)
            return False

    def verify_contradiction(
        self, text_a: str, text_b: str
    ) -> dict | None:
        """Ask the LLM whether two statements genuinely contradict.

        Returns dict {is_contradiction: bool, confidence: float, reasoning: str}
        or None if the LLM is unavailable or the response couldn't be parsed.
        """
        if not self.alive:
            return None

        prompt = (
            "You are a contradiction detection system. Compare these two "
            "statements and determine if they make conflicting claims about "
            "the same subject.\n\n"
            f'Statement A: "{text_a}"\n\n'
            f'Statement B: "{text_b}"\n\n'
            "Two facts about the same entity that describe different aspects "
            "are NOT contradictions (e.g. 'Alice lives in Paris' and 'Alice "
            "likes jazz' are compatible). Only answer YES if the statements "
            "cannot both be true at the same time.\n\n"
            "Respond EXACTLY in this format, nothing else:\n"
            "CONTRADICTION: yes or no\n"
            "CONFIDENCE: 0.0-1.0\n"
            "REASONING: <one sentence>\n"
        )

        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150,
            "temperature": 0.1,
            "stream": False,
            # Qwen3.x emits a chain-of-thought preamble by default that eats
            # the token budget before the structured answer appears. Suppress
            # it so the model emits the parseable CONTRADICTION/CONFIDENCE/
            # REASONING block directly. Harmless on non-Qwen servers (ignored).
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode("utf-8")

        try:
            req = urllib.request.Request(
                self.url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
        except Exception:
            logger.debug("LLM contradiction verify call failed", exc_info=True)
            return None

        return self._parse_response(content)

    @staticmethod
    def _parse_response(content: str) -> dict | None:
        """Parse the LLM's structured response. Returns None on malformed input."""
        is_contra = False
        confidence = 0.5
        reasoning = ""
        for line in content.strip().splitlines():
            low = line.lower()
            if low.startswith("contradiction:"):
                is_contra = "yes" in low
            elif low.startswith("confidence:"):
                m = re.search(r"[\d.]+", low)
                if m:
                    try:
                        confidence = max(0.0, min(1.0, float(m.group())))
                    except ValueError:
                        pass
            elif low.startswith("reasoning:"):
                reasoning = line.split(":", 1)[1].strip()
        # If we got at least the CONTRADICTION line, it's a valid parse
        if "contradiction:" in content.lower():
            return {
                "is_contradiction": is_contra,
                "confidence": confidence,
                "reasoning": reasoning,
            }
        return None


class FactRetriever:
    """Multi-strategy fact retrieval with trust-weighted scoring."""

    def __init__(
        self,
        store: MemoryStore,
        temporal_decay_half_life: int = 0,  # days, 0 = disabled
        fts_weight: float = 0.3,
        jaccard_weight: float = 0.2,
        hrr_weight: float = 0.2,
        neural_weight: float = 0.3,
        hrr_dim: int = 1024,
        llm_verifier: LLMVerifier | None = None,
    ):
        self.store = store
        self.half_life = temporal_decay_half_life
        self.hrr_dim = hrr_dim

        # Auto-redistribute weights if components unavailable
        total_assigned = fts_weight + jaccard_weight + hrr_weight + neural_weight
        if total_assigned <= 0:
            fts_weight = 1.0

        if hrr_weight > 0 and not hrr._HAS_NUMPY:
            # Redistribute HRR weight to FTS5 + neural
            fts_weight += hrr_weight * 0.5
            neural_weight += hrr_weight * 0.5
            hrr_weight = 0.0

        # Check neural embed availability
        neural_available = (
            hasattr(store, "_embed")
            and store._embed is not None
            and store._embed.alive
        )
        if not neural_available and neural_weight > 0:
            fts_weight += neural_weight * 0.6
            jaccard_weight += neural_weight * 0.4
            neural_weight = 0.0

        self.fts_weight = fts_weight
        self.jaccard_weight = jaccard_weight
        self.hrr_weight = hrr_weight
        self.neural_weight = neural_weight
        # LLM verifier for the contradict() precision pass. None by default;
        # contradict() lazily constructs one if llm_verify=True and no override
        # is provided. This keeps non-contradict callers free of LLM latency.
        self.llm_verifier: LLMVerifier | None = llm_verifier

    def search(
        self,
        query: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Hybrid search: FTS5 candidates → Jaccard rerank → trust weighting.

        Pipeline:
        1. FTS5 search: Get limit*3 candidates from SQLite full-text search
        2. Jaccard boost: Token overlap between query and fact content
        3. Trust weighting: final_score = relevance * trust_score
        4. Temporal decay (optional): decay = 0.5^(age_days / half_life)

        Returns list of dicts with fact data + 'score' field, sorted by score desc.
        """
        # Stage 1: Get FTS5 candidates (more than limit for reranking headroom)
        candidates = self._fts_candidates(query, category, min_trust, limit * 3)

        # Fallback: if FTS5 returns nothing but neural embeds are available,
        # scan all facts with neural_embed and score by cosine similarity.
        if not candidates and self.neural_weight > 0:
            return self._neural_search(query, category, min_trust, limit)

        if not candidates:
            return []

        # Stage 2: Compute query neural embedding once (if available)
        query_neural = None
        query_neural_norm = 0.0  # precomputed norm; invariant across candidates
        if self.neural_weight > 0:
            try:
                query_neural = self.store._embed.embed(query)
                if query_neural is not None:
                    import numpy as np
                    query_neural = np.asarray(query_neural, dtype=np.float32)
                    query_neural_norm = float(np.linalg.norm(query_neural))
            except Exception:
                query_neural = None

        # Stage 3: Rerank with Jaccard + HRR + Neural + trust + optional decay
        query_tokens = self._tokenize(query)
        scored = []

        # Hoist query-invariant computations out of the per-candidate loop.
        # encode_text(query) is independent of the fact being scored; without
        # hoisting it was recomputed once per candidate (the dominant cost in
        # the HRR branch — see profile: encode_text ≫ bytes_to_phases).
        query_vec = None
        if self.hrr_weight > 0:
            query_vec = hrr.encode_text(query, self.hrr_dim)
        # Pre-tokenize candidates once; tags are reused for both jaccard union.
        tokenized = [(f, self._tokenize(f["content"]), self._tokenize(f.get("tags", ""))) for f in candidates]

        for fact, content_tokens, tag_tokens in tokenized:
            all_tokens = content_tokens | tag_tokens

            jaccard = self._jaccard_similarity(query_tokens, all_tokens)
            fts_score = fact.get("fts_rank", 0.0)

            # HRR similarity
            if query_vec is not None and fact.get("hrr_vector"):
                fact_vec = hrr.bytes_to_phases(fact["hrr_vector"])
                hrr_sim = (hrr.similarity(query_vec, fact_vec) + 1.0) / 2.0  # shift to [0,1]
            else:
                hrr_sim = 0.5  # neutral

            # Neural embedding similarity (cosine)
            neural_sim = 0.5  # neutral default
            if query_neural is not None and fact.get("neural_embed"):
                try:
                    import numpy as np
                    fact_embed = np.frombuffer(fact["neural_embed"], dtype=np.float32)
                    # Cosine similarity — query norm precomputed above
                    dot = float(np.dot(query_neural, fact_embed))
                    norm_f = float(np.linalg.norm(fact_embed))
                    if query_neural_norm > 0 and norm_f > 0:
                        neural_sim = max(0.0, dot / (query_neural_norm * norm_f))  # clamp to [0,1]
                except Exception:
                    neural_sim = 0.5

            # Combine FTS5 + Jaccard + HRR + Neural
            relevance = (self.fts_weight * fts_score
                        + self.jaccard_weight * jaccard
                        + self.hrr_weight * hrr_sim
                        + self.neural_weight * neural_sim)

            # Trust weighting
            score = relevance * fact["trust_score"]

            # Optional temporal decay
            if self.half_life > 0:
                score *= self._temporal_decay(fact.get("updated_at") or fact.get("created_at"))

            fact["score"] = score
            scored.append(fact)

        # Sort by score descending, return top limit
        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:limit]
        # Strip raw vectors — callers expect JSON-serializable dicts
        for fact in results:
            fact.pop("hrr_vector", None)
            fact.pop("neural_embed", None)

        # Increment retrieval_count for all returned facts
        if results:
            fact_ids = [f["fact_id"] for f in results]
            placeholders = ",".join("?" * len(fact_ids))
            self.store._conn.execute(
                f"UPDATE facts SET retrieval_count = retrieval_count + 1 WHERE fact_id IN ({placeholders})",
                fact_ids,
            )
            self.store._conn.commit()

        self._boost_retrieved_facts(results)
        return results

    def probe(
        self,
        entity: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Compositional entity query using HRR algebra.

        Unbinds entity from memory bank to extract associated content.
        This is NOT keyword search — it uses algebraic structure to find facts
        where the entity plays a structural role.

        Falls back to FTS5 search if numpy unavailable.
        """
        if not hrr._HAS_NUMPY:
            # Fallback to keyword search on entity name
            return self.search(entity, category=category, limit=limit)

        conn = self.store._conn

        # Encode entity as role-bound vector
        role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)
        entity_vec = hrr.encode_atom(entity.lower(), self.hrr_dim)
        probe_key = hrr.bind(entity_vec, role_entity)

        # Try category-specific bank first, then all facts
        if category:
            bank_name = f"cat:{category}"
            bank_row = conn.execute(
                "SELECT vector FROM memory_banks WHERE bank_name = ?",
                (bank_name,),
            ).fetchone()
            if bank_row:
                bank_vec = hrr.bytes_to_phases(bank_row["vector"])
                extracted = hrr.unbind(bank_vec, probe_key)
                # Use extracted signal to score individual facts
                return self._score_facts_by_vector(
                    extracted, category=category, limit=limit
                )

        # Score against individual fact vectors directly
        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   hrr_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        if not rows:
            # Final fallback: keyword search
            return self.search(entity, category=category, limit=limit)

        # role_content is a fixed role vector; encode once, not per-fact.
        role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)
        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"))
            # Unbind probe key from fact to see if entity is structurally present
            residual = hrr.unbind(fact_vec, probe_key)
            # Compare residual against content signal
            content_vec = hrr.bind(hrr.encode_text(fact["content"], self.hrr_dim), role_content)
            sim = hrr.similarity(residual, content_vec)
            fact["score"] = (sim + 1.0) / 2.0 * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:limit]

        # Increment retrieval_count for all returned facts
        if results:
            fact_ids = [f["fact_id"] for f in results]
            placeholders = ",".join("?" * len(fact_ids))
            conn.execute(
                f"UPDATE facts SET retrieval_count = retrieval_count + 1 WHERE fact_id IN ({placeholders})",
                fact_ids,
            )
            conn.commit()

        self._boost_retrieved_facts(results)
        return results

    def related(
        self,
        entity: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Discover facts that share structural connections with an entity.

        Unlike probe (which finds facts *about* an entity), related finds
        facts that are connected through shared context — e.g., other entities
        mentioned alongside this one, or content that overlaps structurally.

        Falls back to FTS5 search if numpy unavailable.
        """
        if not hrr._HAS_NUMPY:
            return self.search(entity, category=category, limit=limit)

        conn = self.store._conn

        # Encode entity as a bare atom (not role-bound — we want ANY structural match)
        entity_vec = hrr.encode_atom(entity.lower(), self.hrr_dim)

        # Get all facts with vectors
        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   hrr_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        if not rows:
            return self.search(entity, category=category, limit=limit)

        # Role vectors are fixed atoms; encode once, not per-fact (2x per iter).
        role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)
        role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)

        # Score each fact by how much the entity's atom appears in its vector
        # This catches both role-bound entity matches AND content word matches
        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"))

            # Check structural similarity: unbind entity from fact
            residual = hrr.unbind(fact_vec, entity_vec)
            # A high-similarity residual to ANY known role vector means this entity
            # plays a structural role in the fact

            entity_role_sim = hrr.similarity(residual, role_entity)
            content_role_sim = hrr.similarity(residual, role_content)
            # Take the max — entity could appear in either role
            best_sim = max(entity_role_sim, content_role_sim)

            fact["score"] = (best_sim + 1.0) / 2.0 * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:limit]

        # Increment retrieval_count for all returned facts
        if results:
            fact_ids = [f["fact_id"] for f in results]
            placeholders = ",".join("?" * len(fact_ids))
            conn.execute(
                f"UPDATE facts SET retrieval_count = retrieval_count + 1 WHERE fact_id IN ({placeholders})",
                fact_ids,
            )
            conn.commit()

        self._boost_retrieved_facts(results)
        return results

    def reason(
        self,
        entities: list[str],
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Multi-entity compositional query — vector-space JOIN.

        Given multiple entities, algebraically intersects their structural
        connections to find facts related to ALL of them simultaneously.
        This is compositional reasoning that no embedding DB can do.

        Example: reason(["peppi", "backend"]) finds facts where peppi AND
        backend both play structural roles — without keyword matching.

        Falls back to FTS5 search if numpy unavailable.
        """
        if not hrr._HAS_NUMPY or not entities:
            # Fallback: search with all entities as keywords
            query = " ".join(entities)
            return self.search(query, category=category, limit=limit)

        conn = self.store._conn
        role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)

        # For each entity, compute what the bank "remembers" about it
        # by unbinding entity+role from each fact vector
        entity_residuals = []
        for entity in entities:
            entity_vec = hrr.encode_atom(entity.lower(), self.hrr_dim)
            probe_key = hrr.bind(entity_vec, role_entity)
            entity_residuals.append(probe_key)

        # Get all facts with vectors
        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   hrr_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        if not rows:
            query = " ".join(entities)
            return self.search(query, category=category, limit=limit)

        # Score each fact by how much EACH entity is structurally present.
        # A fact scores high only if ALL entities have structural presence
        # (AND semantics via min, vs OR which would use mean/max).
        role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)

        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"))

            entity_scores = []
            for probe_key in entity_residuals:
                residual = hrr.unbind(fact_vec, probe_key)
                sim = hrr.similarity(residual, role_content)
                entity_scores.append(sim)

            min_sim = min(entity_scores)
            fact["score"] = (min_sim + 1.0) / 2.0 * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:limit]

        # Increment retrieval_count for all returned facts
        if results:
            fact_ids = [f["fact_id"] for f in results]
            placeholders = ",".join("?" * len(fact_ids))
            conn.execute(
                f"UPDATE facts SET retrieval_count = retrieval_count + 1 WHERE fact_id IN ({placeholders})",
                fact_ids,
            )
            conn.commit()

        self._boost_retrieved_facts(results)
        return results

    def contradict(
        self,
        category: str | None = None,
        threshold: float = 0.2,
        limit: int = 10,
        llm_verify: bool = False,
    ) -> list[dict]:
        """Find potentially contradictory facts via entity overlap + content divergence.

        Two facts contradict when they share entities (same subject) but have
        low content-vector similarity (different claims). This is automated
        memory hygiene — no other memory system does this.

        Default threshold 0.2 (was 0.3): lowered because real fact corpora
        produce moderate HRR similarity even for genuine contradictions — the
        entity token dominates the binding. At 0.3 the detector returned zero
        pairs on the 152-fact production DB. The LLM verify pass handles the
        false positives a lower threshold admits.

        If llm_verify=True, each structurally-detected candidate pair is
        confirmed by a local LLM precision pass. Confirmed pairs get a
        boosted score and `llm_confirmed: true`; rejected pairs are dropped.
        Fail-open: if the LLM endpoint is unavailable or returns nothing,
        structural results are returned unfiltered (the LLM is a precision
        refinement, not a dependency).

        Returns pairs of facts with a contradiction score.
        Falls back to empty list if numpy unavailable.
        """
        if not hrr._HAS_NUMPY:
            return []

        conn = self.store._conn

        # Get all facts with vectors and their linked entities
        where = "WHERE f.hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND f.category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT f.fact_id, f.content, f.category, f.tags, f.trust_score,
                   f.epistemic_status, f.created_at, f.updated_at, f.hrr_vector
            FROM facts f
            {where}
            """,
            params,
        ).fetchall()

        if len(rows) < 2:
            return []

        # Guard against O(n²) explosion on large fact stores.
        # At 500 facts, that's ~125K comparisons — acceptable.
        # Above that, only check the most recently updated facts.
        _MAX_CONTRADICT_FACTS = 500
        if len(rows) > _MAX_CONTRADICT_FACTS:
            rows = sorted(rows, key=lambda r: r["updated_at"] or r["created_at"], reverse=True)
            rows = rows[:_MAX_CONTRADICT_FACTS]

        # Build entity sets per fact — single batched query instead of one per
        # fact (was N+1: 500 separate SELECTs on a full store). Scope the query
        # to exactly the fact_ids we are about to compare.
        fact_ids = [row["fact_id"] for row in rows]
        id_ph = ",".join("?" * len(fact_ids))
        entity_rows = conn.execute(
            f"""
            SELECT fe.fact_id, e.name FROM entities e
            JOIN fact_entities fe ON fe.entity_id = e.entity_id
            WHERE fe.fact_id IN ({id_ph})
            """,
            fact_ids,
        ).fetchall()
        fact_entities: dict[int, set[str]] = {fid: set() for fid in fact_ids}
        for er in entity_rows:
            fact_entities[er["fact_id"]].add(er["name"].lower())

        # Decode each fact's HRR vector exactly once. The pair loop used to
        # re-decode every vector for every pair it appeared in (≈2N decodes per
        # fact → 199k decodes on a 200-fact store; now 200).
        fact_vectors: dict[int, "np.ndarray"] = {}
        for row in rows:
            fact_vectors[row["fact_id"]] = hrr.bytes_to_phases(row["hrr_vector"])

        # Compare all pairs: high entity overlap + low content similarity = contradiction
        facts = [dict(r) for r in rows]
        contradictions = []

        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                f1, f2 = facts[i], facts[j]
                ents1 = fact_entities.get(f1["fact_id"], set())
                ents2 = fact_entities.get(f2["fact_id"], set())

                if not ents1 or not ents2:
                    continue

                # Entity overlap (Jaccard)
                entity_overlap = len(ents1 & ents2) / len(ents1 | ents2) if (ents1 | ents2) else 0.0

                if entity_overlap < 0.3:
                    continue  # Not enough entity overlap to be contradictory

                # Content similarity via HRR vectors (decoded once above)
                content_sim = hrr.similarity(fact_vectors[f1["fact_id"]],
                                             fact_vectors[f2["fact_id"]])

                # High entity overlap + low content similarity = potential contradiction
                # contradiction_score: higher = more contradictory
                contradiction_score = entity_overlap * (1.0 - (content_sim + 1.0) / 2.0)

                if contradiction_score >= threshold:
                    # Strip hrr_vector from output (not JSON serializable)
                    f1_clean = {k: v for k, v in f1.items() if k != "hrr_vector"}
                    f2_clean = {k: v for k, v in f2.items() if k != "hrr_vector"}
                    contradictions.append({
                        "fact_a": f1_clean,
                        "fact_b": f2_clean,
                        "entity_overlap": round(entity_overlap, 3),
                        "content_similarity": round(content_sim, 3),
                        "contradiction_score": round(contradiction_score, 3),
                        "shared_entities": sorted(ents1 & ents2),
                    })

        contradictions.sort(key=lambda x: x["contradiction_score"], reverse=True)

        # ── LLM precision pass (fail-open) ────────────────────────────
        # The structural detector's known weakness: two facts sharing entities
        # but describing different aspects (not a real conflict). The LLM pass
        # confirms genuine semantic contradictions and drops false positives.
        # Cap at 20 pairs to bound LLM latency (~15s/pair worst case).
        if llm_verify and contradictions:
            verifier = self.llm_verifier or LLMVerifier()
            if verifier.alive:
                _MAX_LLM_PAIRS = 20
                verified: list[dict] = []
                for pair in contradictions[:_MAX_LLM_PAIRS]:
                    verdict = verifier.verify_contradiction(
                        pair["fact_a"]["content"], pair["fact_b"]["content"]
                    )
                    if verdict is None:
                        # LLM couldn't answer for this pair — keep the
                        # structural result, badge as unverified.
                        pair["llm_confirmed"] = None
                        pair["llm_reasoning"] = None
                        verified.append(pair)
                    elif verdict["is_contradiction"]:
                        # Confirmed: boost score by LLM confidence factor.
                        pair["llm_confirmed"] = True
                        pair["llm_reasoning"] = verdict["reasoning"]
                        pair["contradiction_score"] = round(
                            pair["contradiction_score"]
                            * (0.5 + 0.5 * verdict["confidence"]),
                            3,
                        )
                        verified.append(pair)
                    else:
                        # LLM says not a real contradiction — drop it.
                        logger.debug(
                            "LLM rejected contradiction pair (facts %s/%s): %s",
                            pair["fact_a"].get("fact_id"),
                            pair["fact_b"].get("fact_id"),
                            verdict.get("reasoning"),
                        )
                # Re-sort verified + any pairs beyond the LLM cap (kept as-is)
                results = verified + contradictions[_MAX_LLM_PAIRS:]
                results.sort(
                    key=lambda x: x["contradiction_score"], reverse=True
                )
            else:
                # LLM unreachable — return structural results, badged.
                for pair in contradictions:
                    pair["llm_confirmed"] = None
                    pair["llm_reasoning"] = "LLM endpoint unavailable"
                results = contradictions
        else:
            results = contradictions

        results = results[:limit]
        # Boost trust for all facts involved in contradiction pairs
        try:
            unique_facts = []
            seen_ids: set[int] = set()
            for c in results:
                for key in ("fact_a", "fact_b"):
                    fid = c.get(key, {}).get("fact_id")
                    if fid is not None and fid not in seen_ids:
                        unique_facts.append({"fact_id": fid})
                        seen_ids.add(fid)
            self._boost_retrieved_facts(unique_facts)
        except Exception:
            logger.debug("Failed to boost contradiction facts", exc_info=True)
        return results

    def _score_facts_by_vector(
        self,
        target_vec: "np.ndarray",
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Score facts by similarity to a target vector."""
        conn = self.store._conn

        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   hrr_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"))
            sim = hrr.similarity(target_vec, fact_vec)
            fact["score"] = (sim + 1.0) / 2.0 * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:limit]
        self._boost_retrieved_facts(results)
        return results

    def _neural_search(
        self,
        query: str,
        category: str | None,
        min_trust: float,
        limit: int,
    ) -> list[dict]:
        """Pure neural cosine similarity search — fallback when FTS5 has no candidates.

        Used when the query doesn't contain exact tokens from any fact but
        semantic similarity should still find relevant results.
        """
        try:
            import numpy as np
            query_vec = self.store._embed.embed(query)
            if query_vec is None:
                return []
            query_vec = np.asarray(query_vec, dtype=np.float32)
            norm_q = float(np.linalg.norm(query_vec))
            if norm_q == 0:
                return []
        except Exception:
            return []

        conn = self.store._conn
        where = "WHERE neural_embed IS NOT NULL AND trust_score >= ?"
        params: list = [min_trust]
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   neural_embed
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        if not rows:
            return []

        scored = []
        for row in rows:
            fact = dict(row)
            fact.pop("neural_embed", None)
            fact_embed = np.frombuffer(row["neural_embed"], dtype=np.float32)
            norm_f = float(np.linalg.norm(fact_embed))
            if norm_f == 0:
                continue
            cosine = float(np.dot(query_vec, fact_embed)) / (norm_q * norm_f)
            cosine = max(0.0, cosine)  # clamp
            fact["score"] = cosine * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:limit]

        # Increment retrieval_count for all returned facts
        if results:
            fact_ids = [f["fact_id"] for f in results]
            placeholders = ",".join("?" * len(fact_ids))
            conn.execute(
                f"UPDATE facts SET retrieval_count = retrieval_count + 1 WHERE fact_id IN ({placeholders})",
                fact_ids,
            )
            conn.commit()

        self._boost_retrieved_facts(results)
        return results

    def _fts_candidates(
        self,
        query: str,
        category: str | None,
        min_trust: float,
        limit: int,
    ) -> list[dict]:
        """Get raw FTS5 candidates from the store.

        Uses the store's database connection directly for FTS5 MATCH
        with rank scoring. Normalizes FTS5 rank to [0, 1] range.
        """
        conn = self.store._conn

        # Build query - FTS5 rank is negative (lower = better match)
        # We need to join facts_fts with facts to get all columns
        params: list = []
        where_clauses = ["facts_fts MATCH ?"]
        params.append(query)

        if category:
            where_clauses.append("f.category = ?")
            params.append(category)

        where_clauses.append("f.trust_score >= ?")
        params.append(min_trust)

        where_sql = " AND ".join(where_clauses)

        sql = f"""
            SELECT f.*, facts_fts.rank as fts_rank_raw
            FROM facts_fts
            JOIN facts f ON f.fact_id = facts_fts.rowid
            WHERE {where_sql}
            ORDER BY facts_fts.rank
            LIMIT ?
        """
        params.append(limit)

        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception:
            # FTS5 MATCH can fail on malformed queries — fall back to empty
            return []

        if not rows:
            return []

        # Normalize FTS5 rank: rank is negative, lower = better
        # Convert to positive score in [0, 1] range
        raw_ranks = [abs(row["fts_rank_raw"]) for row in rows]
        max_rank = max(raw_ranks) if raw_ranks else 1.0
        max_rank = max(max_rank, 1e-6)  # avoid div by zero

        results = []
        for row, raw_rank in zip(rows, raw_ranks):
            fact = dict(row)
            fact.pop("fts_rank_raw", None)
            fact["fts_rank"] = raw_rank / max_rank  # normalize to [0, 1]
            results.append(fact)

        return results

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Simple whitespace tokenization with lowercasing.

        Strips common punctuation. No stemming/lemmatization (Phase 1).
        """
        if not text:
            return set()
        # Split on whitespace, lowercase, strip punctuation
        tokens = set()
        for word in text.lower().split():
            cleaned = word.strip(".,;:!?\"'()[]{}#@<>")
            if cleaned:
                tokens.add(cleaned)
        return tokens

    @staticmethod
    def _jaccard_similarity(set_a: set, set_b: set) -> float:
        """Jaccard similarity coefficient: |A ∩ B| / |A ∪ B|."""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _temporal_decay(self, timestamp_str: str | None) -> float:
        """Exponential decay: 0.5^(age_days / half_life_days).

        Returns 1.0 if decay is disabled or timestamp is missing.
        """
        if not self.half_life or not timestamp_str:
            return 1.0

        try:
            if isinstance(timestamp_str, str):
                # Parse ISO format timestamp from SQLite
                ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            else:
                ts = timestamp_str

            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
            if age_days < 0:
                return 1.0

            return math.pow(0.5, age_days / self.half_life)
        except (ValueError, TypeError):
            return 1.0

    # ------------------------------------------------------------------
    # Trust boost on retrieval
    # ------------------------------------------------------------------

    def _boost_trust(self, fact_id: int) -> None:
        """Boost a single fact's trust_score by +0.02, capped at 1.0."""
        try:
            conn = self.store._conn
            conn.execute(
                "UPDATE facts SET trust_score = MIN(trust_score + 0.02, 1.0) "
                "WHERE fact_id = ?",
                (fact_id,),
            )
            conn.commit()
        except Exception:
            logger.debug(
                "Failed to boost trust for fact_id=%s", fact_id, exc_info=True
            )

    def _boost_retrieved_facts(self, facts: list[dict]) -> None:
        """Batch boost trust scores for retrieved facts (+0.02 each, capped at 1.0).

        Uses ``executemany`` with a single commit for efficiency.
        Silently ignores any error so retrieval never fails.
        """
        try:
            if not facts:
                return
            conn = self.store._conn
            fact_ids = [
                (f["fact_id"],) for f in facts if "fact_id" in f
            ]
            if not fact_ids:
                return
            conn.executemany(
                "UPDATE facts SET trust_score = MIN(trust_score + 0.02, 1.0) "
                "WHERE fact_id = ?",
                fact_ids,
            )
            conn.commit()
            logger.debug("Boosted trust for %d retrieved facts", len(fact_ids))
        except Exception:
            logger.debug(
                "Failed to batch-boost trust for retrieved facts", exc_info=True
            )
