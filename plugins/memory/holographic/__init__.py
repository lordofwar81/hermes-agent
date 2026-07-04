"""hermes-memory-store — holographic memory plugin using MemoryProvider interface.

Registers as a MemoryProvider plugin, giving the agent structured fact storage
with entity resolution, trust scoring, and HRR-based compositional retrieval.

Original plugin by dusterbloom (PR #2351), adapted to the MemoryProvider ABC.

Config in $HERMES_HOME/config.yaml (profile-scoped):
  plugins:
    hermes-memory-store:
      db_path: $HERMES_HOME/memory_store.db   # omit to use the default
      auto_extract: false
      default_trust: 0.5
      min_trust_threshold: 0.3
      temporal_decay_half_life: 0
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error
from .store import MemoryStore
from .retrieval import FactRetriever
from hermes_cli.config import cfg_get

logger = logging.getLogger(__name__)


def _str_to_bool(value) -> bool:
    """Coerce schema-passed values (str/bool) to bool. MCP args arrive as strings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


# ---------------------------------------------------------------------------
# Tool schemas (unchanged from original PR)
# ---------------------------------------------------------------------------

FACT_STORE_SCHEMA = {
    "name": "fact_store",
    "description": (
        "Deep structured memory with algebraic reasoning. "
        "Use alongside the memory tool — memory for always-on context, "
        "fact_store for deep recall and compositional queries.\n\n"
        "ACTIONS (simple → powerful):\n"
        "• add — Store a fact the user would expect you to remember.\n"
        "• search — Keyword lookup ('editor config', 'deploy process').\n"
        "• probe — Entity recall: ALL facts about a person/thing.\n"
        "• related — What connects to an entity? Structural adjacency.\n"
        "• reason — Compositional: facts connected to MULTIPLE entities simultaneously.\n"
        "• contradict — Memory hygiene: find facts making conflicting claims. Pass llm_verify=true for an LLM precision pass that filters false positives.\n"
        "• update/remove/list — CRUD operations.\n\n"
        "IMPORTANT: Before answering questions about the user, ALWAYS probe or reason first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "search", "probe", "related", "reason", "contradict", "update", "remove", "list"],
            },
            "content": {"type": "string", "description": "Fact content (required for 'add')."},
            "query": {"type": "string", "description": "Search query (required for 'search')."},
            "entity": {"type": "string", "description": "Entity name for 'probe'/'related'."},
            "entities": {"type": "array", "items": {"type": "string"}, "description": "Entity names for 'reason'."},
            "fact_id": {"type": "integer", "description": "Fact ID for 'update'/'remove'."},
            "category": {"type": "string", "enum": ["user_pref", "project", "tool", "general"]},
            "tags": {"type": "string", "description": "Comma-separated tags."},
            "trust_delta": {"type": "number", "description": "Trust adjustment for 'update'."},
            "epistemic_status": {"type": "string", "enum": ["stated", "inferred", "verified", "contradicted", "retracted"], "description": "Set fact's epistemic status (for 'update'). Use 'contradicted' to retire a superseded fact."},
            "min_trust": {"type": "number", "description": "Minimum trust filter (default: 0.3)."},
            "limit": {"type": "integer", "description": "Max results (default: 10)."},
            "llm_verify": {"type": "boolean", "description": "If true, confirm each candidate pair with a local LLM precision pass (default: false). Drops pairs the LLM says aren't real contradictions."},
        },
        "required": ["action"],
    },
}

FACT_FEEDBACK_SCHEMA = {
    "name": "fact_feedback",
    "description": (
        "Rate a fact after using it. Mark 'helpful' if accurate, 'unhelpful' if outdated. "
        "This trains the memory — good facts rise, bad facts sink."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["helpful", "unhelpful"]},
            "fact_id": {"type": "integer", "description": "The fact ID to rate."},
        },
        "required": ["action", "fact_id"],
    },
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_plugin_config() -> dict:
    from hermes_constants import get_hermes_home
    config_path = get_hermes_home() / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path, encoding="utf-8-sig") as f:
            all_config = yaml.safe_load(f) or {}
        return cfg_get(all_config, "plugins", "hermes-memory-store", default={}) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class HolographicMemoryProvider(MemoryProvider):
    """Holographic memory with structured facts, entity resolution, and HRR retrieval."""

    def __init__(self, config: dict | None = None):
        self._config = config or _load_plugin_config()
        self._store = None
        self._retriever = None
        self._vector_store = None  # dual-mode: LanceDB store for semantic recall
        self._min_trust = float(self._config.get("min_trust_threshold", 0.3))

    @property
    def name(self) -> str:
        return "holographic"

    def is_available(self) -> bool:
        return True  # SQLite is always available, numpy is optional

    def save_config(self, values, hermes_home):
        """Write config to config.yaml under plugins.hermes-memory-store."""
        from pathlib import Path
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            existing = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8-sig") as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("plugins", {})
            existing["plugins"]["hermes-memory-store"] = values
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception:
            pass

    def get_config_schema(self):
        from hermes_constants import display_hermes_home
        _default_db = f"{display_hermes_home()}/memory_store.db"
        return [
            {"key": "db_path", "description": "SQLite database path", "default": _default_db},
            {"key": "auto_extract", "description": "Auto-extract facts at session end", "default": "false", "choices": ["true", "false"]},
            {"key": "default_trust", "description": "Default trust score for new facts", "default": "0.5"},
            {"key": "hrr_dim", "description": "HRR vector dimensions", "default": "1024"},
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home
        _hermes_home = str(get_hermes_home())
        _default_db = _hermes_home + "/memory_store.db"
        db_path = self._config.get("db_path", _default_db)
        # Expand $HERMES_HOME in user-supplied paths so config values like
        # "$HERMES_HOME/memory_store.db" or "~/.hermes/memory_store.db" both
        # resolve to the active profile's directory.
        if isinstance(db_path, str):
            db_path = db_path.replace("$HERMES_HOME", _hermes_home)
            db_path = db_path.replace("${HERMES_HOME}", _hermes_home)
        default_trust = float(self._config.get("default_trust", 0.5))
        hrr_dim = int(self._config.get("hrr_dim", 1024))
        hrr_weight = float(self._config.get("hrr_weight", 0.2))
        neural_weight = float(self._config.get("neural_weight", 0.3))
        temporal_decay = int(self._config.get("temporal_decay_half_life", 0))

        self._store = MemoryStore(db_path=db_path, default_trust=default_trust, hrr_dim=hrr_dim)
        self._retriever = FactRetriever(
            store=self._store,
            temporal_decay_half_life=temporal_decay,
            hrr_weight=hrr_weight,
            neural_weight=neural_weight,
            hrr_dim=hrr_dim,
        )
        self._session_id = session_id

        # Dual-mode: connect the LanceDB vector store for semantic recall alongside
        # HRR. Best-effort — if LanceDB/embeddings are unavailable the provider
        # degrades gracefully to HRR-only (prefetch handles a None vector_store).
        try:
            from tools.vector_memory import VectorMemoryStore
            self._vector_store = VectorMemoryStore()
            logger.debug("Holographic dual-mode: vector store connected")
        except Exception as exc:
            self._vector_store = None
            logger.debug("Holographic dual-mode: vector store unavailable (%s)", exc)

    def system_prompt_block(self) -> str:
        if not self._store:
            return ""
        try:
            total = self._store._conn.execute(
                "SELECT COUNT(*) FROM facts"
            ).fetchone()[0]
        except Exception:
            total = 0
        if total == 0:
            return (
                "# Holographic Memory\n"
                "Active. Empty fact store — proactively add facts the user would expect you to remember.\n"
                "Use fact_store(action='add') to store durable structured facts about people, projects, preferences, decisions.\n"
                "Use fact_feedback to rate facts after using them (trains trust scores)."
            )
        return (
            f"# Holographic Memory\n"
            f"Active. {total} facts stored with entity resolution and trust scoring.\n"
            f"Use fact_store to search, probe entities, reason across entities, or add facts.\n"
            f"Use fact_feedback to rate facts after using them (trains trust scores)."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._retriever or not query:
            return ""
        try:
            # --- Unified retrieval: HRR + vector (semantic) + Memory Tree (episodic) ---
            # RRF-fuse all three ranked lists so the agent sees the best facts from
            # any modality. gbrain (knowledge graph) is intentionally NOT folded in
            # here — it's an HTTP+OAuth MCP call (~150-500ms) too expensive for a
            # per-turn prefetch. The agent can query gbrain on demand via MCP tools
            # when it needs structured knowledge-graph recall.
            limit = 5
            # Stage 1 fetches a wider candidate pool (20) so Stage 2 (cross-encoder)
            # has enough to rerank down to the final limit. Each arm over-fetches;
            # RRF fuses the union; reranker selects the top by relevance.
            pool = 20
            hrr_results = self._retriever.search(
                query, min_trust=self._min_trust, limit=pool
            ) or []

            vec_results = []
            if self._vector_store is not None:
                try:
                    vec_results = self._vector_store.search(query, top_k=pool) or []
                except Exception as ve:
                    logger.debug("Vector prefetch failed: %s", ve)

            tree_results = self._memory_tree_search(query, limit=pool)
            session_results = self._session_search(query, limit=pool)

            fused = self._rrf_fuse(hrr_results, vec_results, tree_results,
                                   session_results, limit=pool)
            if not fused:
                return ""
            # Stage 2: cross-encoder rerank for precision. RRF gave us broad
            # recall (up to 20 candidates); the cross-encoder re-scores by genuine
            # query-doc relevance so a long multi-term fact that merely overlaps
            # on terms no longer dominates a short fact that actually answers.
            # Graceful fallback: if the model is unavailable, keep RRF order.
            try:
                from .reranker import rerank as _rerank, is_available as _rerank_ok
                if _rerank_ok() and len(fused) > limit:
                    docs = [content for _, content, _ in fused]
                    reranked = _rerank(query, docs, top_k=limit)
                    fused = [fused[i] for i, _ in reranked]
            except Exception as re_exc:
                logger.debug("Rerank stage failed, using RRF order: %s", re_exc)
            lines = []
            for score, content, badge in fused[:limit]:
                lines.append(f"- [{score:.2f}{badge}] {content}")
            return "## Holographic Memory\n" + "\n".join(lines)
        except Exception as e:
            logger.debug("Holographic prefetch failed: %s", e)
            return ""

    def _memory_tree_search(self, query: str, *, limit: int = 5) -> list[dict]:
        """Lightweight Memory Tree content search (episodic chunks).

        Direct SQLite LIKE query — same shape as MemoryTree.search_chunks but
        without importing the full memory_tree package (keeps the provider
        decoupled). Returns dicts with 'content' and 'score' for RRF fusion.
        Sub-millisecond; safe for per-turn prefetch.
        """
        import sqlite3
        from hermes_constants import get_hermes_home
        db_path = get_hermes_home() / "memory_tree.db"
        if not db_path.exists():
            return []
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            # Search admitted/sealed chunks by keyword; rank by importance score.
            rows = conn.execute(
                """SELECT c.content_preview, sc.total as score
                   FROM mem_tree_chunks c
                   LEFT JOIN mem_tree_scores sc ON c.id = sc.chunk_id
                   WHERE c.lifecycle_status IN ('admitted', 'buffered', 'sealed')
                     AND (sc.dropped = 0 OR sc.dropped IS NULL)
                     AND c.content_preview LIKE ?
                   ORDER BY sc.total DESC LIMIT ?""",
                (f"%{query}%", limit),
            ).fetchall()
            conn.close()
            return [{"content": r["content_preview"], "score": float(r["score"] or 0)}
                    for r in rows if r["content_preview"]]
        except Exception as exc:
            logger.debug("Memory Tree prefetch failed: %s", exc)
            return []

    def _session_search(self, query: str, *, limit: int = 5) -> list[dict]:
        """Lightweight state.db FTS5 search over session messages (episodic recall).

        state.db holds every message the agent has ever processed (67k+ rows) but
        was previously an island — not fused into prefetch. This gives the "did we
        literally discuss X?" modality that HRR/VEC/TREE (which index facts/chunks,
        not raw dialogue) cannot provide. Direct FTS5 MATCH query; sub-20ms on the
        full corpus. Returns dicts with 'content' and 'score' for RRF fusion.
        """
        import sqlite3
        from hermes_constants import get_hermes_home
        db_path = get_hermes_home() / "state.db"
        if not db_path.exists():
            return []
        # Build an FTS5 MATCH query: each term is double-quoted per the FTS5
        # query syntax spec so that special characters (hyphens, asterisks,
        # colons, parentheses) in user input do not cause syntax errors or
        # unintended boolean operators. Strip existing double-quotes from
        # the raw input first to prevent nested-quote injection.
        terms = [t for t in query.replace('"', "").split() if len(t) >= 2]
        if not terms:
            return []
        match_expr = " ".join(f'"{t}"' for t in terms[:6])  # cap at 6 terms
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            # FTS5 bm25() returns negative scores; more negative = more relevant.
            # Normalize to a 0-1 score (closer to 0 bm25 = higher relevance).
            rows = conn.execute(
                """SELECT substr(m.content, 1, 500) as content,
                          -rank as score
                   FROM messages_fts f
                   JOIN messages m ON m.rowid = f.rowid
                   WHERE messages_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (match_expr, limit),
            ).fetchall()
            conn.close()
            if not rows:
                return []
            # Normalize bm25 scores to 0-1 for RRF (rank-based, so exact value
            # matters less than relative ordering; RRF uses rank position anyway).
            max_s = max(float(r["score"] or 0) for r in rows) or 1.0
            return [{"content": r["content"], "score": float(r["score"] or 0) / max_s}
                    for r in rows if r["content"]]
        except Exception as exc:
            logger.debug("Session FTS prefetch failed: %s", exc)
            return []

    @staticmethod
    def _rrf_fuse(hrr_results: list, vec_results: list, tree_results: list | None = None,
                  session_results: list | None = None,
                  *, limit: int, k: int = 60):
        """Reciprocal Rank Fusion of HRR + vector + Memory Tree results.

        Returns a list of (fused_score, content, badge) tuples, de-duplicated by
        normalized content. badge marks the contributing modality(ies). HRR
        trust_score and Tree importance score are folded into the ranking as
        tiebreaks so high-trust facts surface even when ranks are close.
        """
        scored: dict[str, dict] = {}

        for rank, r in enumerate(hrr_results):
            content = (r.get("content") or "").strip()
            if not content:
                continue
            key = content.lower()[:200]
            trust = float(r.get("trust_score", r.get("trust", 0)) or 0)
            entry = scored.setdefault(key, {"content": content, "rrf": 0.0, "sources": set(), "trust": 0.0})
            entry["rrf"] += 1.0 / (k + rank + 1)
            entry["sources"].add("HRR")
            entry["trust"] = max(entry["trust"], trust)

        for rank, sr in enumerate(vec_results):
            content = (getattr(sr, "text", "") or "").strip()
            if not content:
                continue
            key = content.lower()[:200]
            entry = scored.setdefault(key, {"content": content, "rrf": 0.0, "sources": set(), "trust": 0.0})
            entry["rrf"] += 1.0 / (k + rank + 1)
            entry["sources"].add("VEC")

        for rank, r in enumerate(tree_results or []):
            content = (r.get("content") or "").strip()
            if not content:
                continue
            key = content.lower()[:200]
            tree_score = float(r.get("score", 0) or 0)
            entry = scored.setdefault(key, {"content": content, "rrf": 0.0, "sources": set(), "trust": 0.0})
            entry["rrf"] += 1.0 / (k + rank + 1)
            entry["sources"].add("TREE")
            entry["trust"] = max(entry["trust"], tree_score)

        for rank, r in enumerate(session_results or []):
            content = (r.get("content") or "").strip()
            if not content:
                continue
            key = content.lower()[:200]
            entry = scored.setdefault(key, {"content": content, "rrf": 0.0, "sources": set(), "trust": 0.0})
            entry["rrf"] += 1.0 / (k + rank + 1)
            entry["sources"].add("SESS")

        ranked = sorted(
            scored.values(),
            key=lambda e: (e["rrf"] + e["trust"] * 0.01),
            reverse=True,
        )[:limit]
        out = []
        for e in ranked:
            badge = " (" + "+".join(sorted(e["sources"])) + ")"
            out.append((e["rrf"], e["content"], badge))
        return out

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        # Holographic memory stores explicit facts via tools, not auto-sync.
        # The on_session_end hook handles auto-extraction if configured.
        pass

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [FACT_STORE_SCHEMA, FACT_FEEDBACK_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "fact_store":
            return self._handle_fact_store(args)
        elif tool_name == "fact_feedback":
            return self._handle_fact_feedback(args)
        return tool_error(f"Unknown tool: {tool_name}")

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._config.get("auto_extract", False):
            return
        if not self._store or not messages:
            return
        self._auto_extract_facts(messages)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes as facts."""
        if action == "add" and self._store and content:
            try:
                category = "user_pref" if target == "user" else "general"
                self._store.add_fact(content, category=category)
            except Exception as e:
                logger.debug("Holographic memory_write mirror failed: %s", e)

    def shutdown(self) -> None:
        self._store = None
        self._retriever = None

    # -- Tool handlers -------------------------------------------------------

    def _handle_fact_store(self, args: dict) -> str:
        try:
            action = args["action"]
            store = self._store
            retriever = self._retriever

            if action == "add":
                fact_id = store.add_fact(
                    args["content"],
                    category=args.get("category", "general"),
                    tags=args.get("tags", ""),
                )
                return json.dumps({"fact_id": fact_id, "status": "added"})

            elif action == "search":
                results = retriever.search(
                    args["query"],
                    category=args.get("category"),
                    min_trust=float(args.get("min_trust", self._min_trust)),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "probe":
                results = retriever.probe(
                    args["entity"],
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "related":
                results = retriever.related(
                    args["entity"],
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "reason":
                entities = args.get("entities", [])
                if not entities:
                    return tool_error("reason requires 'entities' list")
                results = retriever.reason(
                    entities,
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "contradict":
                results = retriever.contradict(
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                    llm_verify=_str_to_bool(args.get("llm_verify", False)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "update":
                updated = store.update_fact(
                    int(args["fact_id"]),
                    content=args.get("content"),
                    trust_delta=float(args["trust_delta"]) if "trust_delta" in args else None,
                    tags=args.get("tags"),
                    category=args.get("category"),
                    epistemic_status=args.get("epistemic_status"),
                )
                return json.dumps({"updated": updated})

            elif action == "remove":
                removed = store.remove_fact(int(args["fact_id"]))
                return json.dumps({"removed": removed})

            elif action == "list":
                facts = store.list_facts(
                    category=args.get("category"),
                    min_trust=float(args.get("min_trust", 0.0)),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"facts": facts, "count": len(facts)})

            else:
                return tool_error(f"Unknown action: {action}")

        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    def _handle_fact_feedback(self, args: dict) -> str:
        try:
            fact_id = int(args["fact_id"])
            helpful = args["action"] == "helpful"
            result = self._store.record_feedback(fact_id, helpful=helpful)
            return json.dumps(result)
        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    # -- Auto-extraction (on_session_end) ------------------------------------

    def _auto_extract_facts(self, messages: list) -> None:
        _PREF_PATTERNS = [
            re.compile(r'\bI\s+(?:prefer|like|love|use|want|need)\s+(.+)', re.IGNORECASE),
            re.compile(r'\bmy\s+(?:favorite|preferred|default)\s+\w+\s+is\s+(.+)', re.IGNORECASE),
            re.compile(r'\bI\s+(?:always|never|usually)\s+(.+)', re.IGNORECASE),
        ]
        _DECISION_PATTERNS = [
            re.compile(r'\bwe\s+(?:decided|agreed|chose)\s+(?:to\s+)?(.+)', re.IGNORECASE),
            re.compile(r'\bthe\s+project\s+(?:uses|needs|requires)\s+(.+)', re.IGNORECASE),
        ]

        extracted = 0
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str) or len(content) < 10:
                continue

            for pattern in _PREF_PATTERNS:
                if pattern.search(content):
                    try:
                        self._store.add_fact(content[:400], category="user_pref")
                        extracted += 1
                    except Exception:
                        pass
                    break

            for pattern in _DECISION_PATTERNS:
                if pattern.search(content):
                    try:
                        self._store.add_fact(content[:400], category="project")
                        extracted += 1
                    except Exception:
                        pass
                    break

        if extracted:
            logger.info("Auto-extracted %d facts from conversation", extracted)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the holographic memory provider with the plugin system."""
    config = _load_plugin_config()
    provider = HolographicMemoryProvider(config=config)
    ctx.register_memory_provider(provider)
