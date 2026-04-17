"""
SQLite-backed fact store with entity resolution and trust scoring.
Single-user Hermes memory store plugin.
"""

import fcntl
import logging
import re
import sqlite3
import threading
import urllib.request
import urllib.error
from contextlib import contextmanager
from pathlib import Path

try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL UNIQUE,
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hrr_vector      BLOB,
    neural_embed    BLOB
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    entity_type TEXT DEFAULT 'unknown',
    aliases     TEXT DEFAULT '',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_entities (
    fact_id   INTEGER REFERENCES facts(fact_id),
    entity_id INTEGER REFERENCES entities(entity_id),
    PRIMARY KEY (fact_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_facts_trust    ON facts(trust_score DESC);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
CREATE INDEX IF NOT EXISTS idx_entities_name  ON entities(name);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(content, tags, content=facts, content_rowid=fact_id);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TABLE IF NOT EXISTS memory_banks (
    bank_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_name  TEXT NOT NULL UNIQUE,
    vector     BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    fact_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# Trust adjustment constants
_HELPFUL_DELTA = 0.05
_UNHELPFUL_DELTA = -0.10
_TRUST_MIN = 0.0
_TRUST_MAX = 1.0

# ── Entity extraction patterns ──────────────────────────────────────
# Each pattern targets a distinct structural signal in fact text.
# Order matters: higher-specificity patterns first.

# 1. Multi-word capitalized phrases  e.g. "TurboQuant", "Bosgame M5"
_RE_CAPITALIZED = re.compile(
    r"\b((?:[A-Z][a-z]*\.\w+|[A-Z][a-z]+|[A-Z]+)"
    r"(?:\s+(?:[A-Z][a-z]*\.\w+|[A-Z][a-z]+|[A-Z]+))+)\b"
)

# 2. Single-word PascalCase/camelCase identifiers  e.g. "SearXNG", "TurboQuant"
#    Catches mixed-case words with internal capitals — NOT simple all-caps like API
_RE_TECH_TERM = re.compile(r"\b([A-Z][a-z]+[A-Z][a-zA-Z]*)\b")

# 3. Quoted terms (double then single)
_RE_DOUBLE_QUOTE = re.compile(r'"([^"]+)"')
_RE_SINGLE_QUOTE = re.compile(r"'([^']+)'")

# 4. AKA patterns  e.g. "Guido aka BDFL"
_RE_AKA = re.compile(
    r"(\w+(?:\s+\w+)*)\s+(?:aka|also known as)\s+(\w+(?:\s+\w+)*)",
    re.IGNORECASE,
)

# 6. Parenthetical labels  e.g. "(Vulkan-only)", "(JSON API)"
#    Skips pure dates like "(Apr 5)" and compound lists
_RE_PAREN_LABEL = re.compile(r"\(([^,)]{2,35})\)")

# 7. File paths and config files  e.g. "/home/user/llama.cpp", "config.yaml"
#    Requires a leading / or specific extension. Skips bare URLs.
_RE_FILEPATH = re.compile(
    r"((?:/[\w.-]+)+/\S+?\.\w{1,5}"
    r"|[\w][\w.-]*\.(?:yaml|py|db|json|toml|cfg|conf|md|txt|sh|gguf|bin))\b"
)

# 7. Version-like strings  e.g. "glm-5-turbo", "qwen2.5-0.5b", "v2.0"
_RE_VERSION_ID = re.compile(r"\b([a-zA-Z][\w.-]*(?:-\d[\d.]*(?:[a-z]\d*)?))\b")

# 8. Key-value labels before colons  e.g. "System:", "Search stack:"
_RE_KEY_LABEL = re.compile(r"^([\w\s]{2,25}?):", re.MULTILINE)

# Stopwords — never extract these as entities
_ENTITY_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "used",
        "using",
        "with",
        "for",
        "and",
        "but",
        "or",
        "not",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "too",
        "very",
        "just",
        "also",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "when",
        "where",
        "why",
        "if",
        "else",
        "from",
        "into",
        "to",
        "in",
        "on",
        "at",
        "by",
        "up",
        "about",
        "after",
        "before",
        "over",
        "under",
        "between",
        "through",
        "during",
        "without",
        "within",
        "per",
        "its",
        "it",
        "he",
        "she",
        "they",
        "them",
        "his",
        "her",
        "their",
        "my",
        "your",
        "our",
        "me",
        "him",
        "us",
        "we",
        "you",
        "any",
        "own",
        "now",
        "new",
        "old",
        "first",
        "last",
        "next",
        "same",
        "only",
        # SQL/technical keywords that aren't useful as entities
        "text",
        "default",
        "integer",
        "null",
        "blob",
        "primary",
        "autoincrement",
        "bytes",
        "values",
        "table",
        "column",
        "schema",
        "text default",
        # Epistemic states (used as values, not entities)
        "stated",
        "inferred",
        "verified",
        "contradicted",
        "retracted",
    }
)

# Pattern for entities to skip — fragments that look like garbage
_RE_SKIP_FRAGMENT = re.compile(
    r"^(?:bytes?\s+\d|row|line|field|file|page|test fact|not\s+/)",
    re.IGNORECASE,
)


def _clamp_trust(value: float) -> float:
    return max(_TRUST_MIN, min(_TRUST_MAX, value))


_log = logging.getLogger(__name__)

# ── Neural embed client ────────────────────────────────────────────

# Default embed server — local llama.cpp with mxbai-embed-large-v1
_DEFAULT_EMBED_URL = "http://localhost:11434/v1/embeddings"
_DEFAULT_EMBED_MODEL = "mxbai-embed-large-v1-f16.gguf"
_DEFAULT_EMBED_KEY = "notempty"
_EMBED_TIMEOUT = 5  # seconds — fail fast, don't block memory writes


class EmbedClient:
    """Minimal OpenAI-compatible embed client. Graceful on failure."""

    def __init__(
        self,
        url: str = _DEFAULT_EMBED_URL,
        model: str = _DEFAULT_EMBED_MODEL,
        api_key: str = _DEFAULT_EMBED_KEY,
        timeout: int = _EMBED_TIMEOUT,
    ):
        self.url = url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._alive: bool | None = None  # None = not yet probed

    @property
    def alive(self) -> bool:
        """Check if embed server is reachable (cached after first call)."""
        if self._alive is not None:
            return self._alive
        self._alive = self._probe()
        return self._alive

    def _probe(self) -> bool:
        try:
            req = urllib.request.Request(
                self.url.replace("/embeddings", "/models"),
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout):
                return True
        except Exception:
            return False

    def embed(self, text: str) -> "np.ndarray | None":
        """Get embedding for a single text. Returns None on any failure."""
        if not self.alive:
            return None

        try:
            import json
            import numpy as np

            payload = json.dumps(
                {
                    "input": text,
                    "model": self.model,
                }
            ).encode()

            req = urllib.request.Request(
                self.url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read())
                vec = np.array(result["data"][0]["embedding"], dtype=np.float32)
                return vec
        except Exception:
            _log.debug("Embed request failed, marking server as down")
            self._alive = False
            return None

    def embed_batch(self, texts: list[str]) -> list["np.ndarray | None"]:
        """Embed multiple texts. Returns None for any that fail."""
        return [self.embed(t) for t in texts]


class MemoryStore:
    """SQLite-backed fact store with entity resolution and trust scoring."""

    def __init__(
        self,
        db_path: "str | Path | None" = None,
        default_trust: float = 0.5,
        hrr_dim: int = 1024,
        embed_client: EmbedClient | None = None,
    ) -> None:
        if db_path is None:
            from hermes_constants import get_hermes_home

            db_path = str(get_hermes_home() / "memory_store.db")
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_trust = _clamp_trust(default_trust)
        self.hrr_dim = hrr_dim
        self._hrr_available = hrr._HAS_NUMPY
        self._embed = embed_client or EmbedClient()
        self._conn: sqlite3.Connection = None  # type: ignore[assignment]
        self._lock = threading.RLock()
        self._lock_fd = None
        self._init_db()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables, enable WAL mode, check integrity."""
        # Acquire file-level lock for multi-process safety
        lock_path = self.db_path.with_suffix(self.db_path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_fd = open(lock_path, "w")
        fcntl.flock(self._lock_fd, fcntl.LOCK_EX)

        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=30.0,
        )
        self._conn.row_factory = sqlite3.Row

        # Check integrity before proceeding
        try:
            result = self._conn.execute("PRAGMA integrity_check").fetchone()
            if result[0] != "ok":
                logger.error(f"Database integrity check failed: {result[0]}")
                self._conn.close()
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                self._lock_fd.close()
                self._lock_fd = None
                raise RuntimeError(f"Database corruption detected: {result[0]}")
        except sqlite3.DatabaseError as e:
            logger.error(f"Database corrupted: {e}")
            self._conn.close()
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            self._lock_fd.close()
            self._lock_fd = None
            raise

        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        # Migrate: add hrr_vector column if missing (safe for existing databases)
        columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(facts)").fetchall()
        }
        if "hrr_vector" not in columns:
            self._conn.execute("ALTER TABLE facts ADD COLUMN hrr_vector BLOB")
            self._conn.commit()
            # Backfill vectors for any pre-existing facts that were inserted
            # via raw SQL and bypassed add_fact()'s HRR computation.
            if self._hrr_available:
                rows = self._conn.execute(
                    "SELECT fact_id, content FROM facts WHERE hrr_vector IS NULL"
                ).fetchall()
                if rows:
                    categories: set[str] = set()
                    for row in rows:
                        self._compute_hrr_vector(row["fact_id"], row["content"])
                        categories.add(
                            self._conn.execute(
                                "SELECT category FROM facts WHERE fact_id = ?",
                                (row["fact_id"],),
                            ).fetchone()["category"]
                        )
                    for cat in categories:
                        self._rebuild_bank(cat)
                    logger.info(
                        "HRR migration: backfilled vectors for %d existing facts",
                        len(rows),
                    )
        # Migrate: add neural_embed column if missing
        if "neural_embed" not in columns:
            self._conn.execute("ALTER TABLE facts ADD COLUMN neural_embed BLOB")
            self._conn.commit()
            _log.info("Added neural_embed column to facts table")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_fact(
        self,
        content: str,
        category: str = "general",
        tags: str = "",
    ) -> int:
        """Insert a fact and return its fact_id.

        Deduplicates by content (UNIQUE constraint). On duplicate, returns
        the existing fact_id without modifying the row. Extracts entities from
        the content and links them to the fact.
        """
        with self._lock:
            content = content.strip()
            if not content:
                raise ValueError("content must not be empty")

            try:
                cur = self._conn.execute(
                    """
                    INSERT INTO facts (content, category, tags, trust_score)
                    VALUES (?, ?, ?, ?)
                    """,
                    (content, category, tags, self.default_trust),
                )
                self._conn.commit()
                fact_id: int = cur.lastrowid  # type: ignore[assignment]
            except sqlite3.IntegrityError:
                # Duplicate content — return existing id
                row = self._conn.execute(
                    "SELECT fact_id FROM facts WHERE content = ?", (content,)
                ).fetchone()
                return int(row["fact_id"])

            # Entity extraction and linking
            for name in self._extract_entities(content):
                entity_id = self._resolve_entity(name)
                self._link_fact_entity(fact_id, entity_id)

            # Compute HRR vector after entity linking
            self._compute_hrr_vector(fact_id, content)

            # Compute neural embedding (async-safe, graceful on failure)
            self._compute_neural_embed(fact_id, content)

            self._rebuild_bank(category)

            return fact_id

    def search_facts(
        self,
        query: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Full-text search over facts using FTS5.

        Returns a list of fact dicts ordered by FTS5 rank, then trust_score
        descending. Also increments retrieval_count for matched facts.
        """
        with self._lock:
            query = query.strip()
            if not query:
                return []

            params: list = [query, min_trust]
            category_clause = ""
            if category is not None:
                category_clause = "AND f.category = ?"
                params.append(category)
            params.append(limit)

            sql = f"""
                SELECT f.fact_id, f.content, f.category, f.tags,
                       f.trust_score, f.retrieval_count, f.helpful_count,
                       f.created_at, f.updated_at
                FROM facts f
                JOIN facts_fts fts ON fts.rowid = f.fact_id
                WHERE facts_fts MATCH ?
                  AND f.trust_score >= ?
                  {category_clause}
                ORDER BY fts.rank, f.trust_score DESC
                LIMIT ?
            """

            rows = self._conn.execute(sql, params).fetchall()
            results = [self._row_to_dict(r) for r in rows]

            if results:
                ids = [r["fact_id"] for r in results]
                placeholders = ",".join("?" * len(ids))
                self._conn.execute(
                    f"UPDATE facts SET retrieval_count = retrieval_count + 1 WHERE fact_id IN ({placeholders})",
                    ids,
                )
                self._conn.commit()

            return results

    def update_fact(
        self,
        fact_id: int,
        content: str | None = None,
        trust_delta: float | None = None,
        tags: str | None = None,
        category: str | None = None,
    ) -> bool:
        """Partially update a fact. Trust is clamped to [0, 1].

        Returns True if the row existed, False otherwise.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, trust_score FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if row is None:
                return False

            assignments: list[str] = ["updated_at = CURRENT_TIMESTAMP"]
            params: list = []

            if content is not None:
                assignments.append("content = ?")
                params.append(content.strip())
            if tags is not None:
                assignments.append("tags = ?")
                params.append(tags)
            if category is not None:
                assignments.append("category = ?")
                params.append(category)
            if trust_delta is not None:
                new_trust = _clamp_trust(row["trust_score"] + trust_delta)
                assignments.append("trust_score = ?")
                params.append(new_trust)

            params.append(fact_id)
            self._conn.execute(
                f"UPDATE facts SET {', '.join(assignments)} WHERE fact_id = ?",
                params,
            )
            self._conn.commit()

            # If content changed, re-extract entities
            if content is not None:
                self._conn.execute(
                    "DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,)
                )
                for name in self._extract_entities(content):
                    entity_id = self._resolve_entity(name)
                    self._link_fact_entity(fact_id, entity_id)
                self._conn.commit()

            # Recompute HRR vector if content changed
            if content is not None:
                self._compute_hrr_vector(fact_id, content)
            # Rebuild bank for relevant category
            cat = (
                category
                or self._conn.execute(
                    "SELECT category FROM facts WHERE fact_id = ?", (fact_id,)
                ).fetchone()["category"]
            )
            self._rebuild_bank(cat)

            return True

    def remove_fact(self, fact_id: int) -> bool:
        """Delete a fact and its entity links. Returns True if the row existed."""
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, category FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if row is None:
                return False

            self._conn.execute(
                "DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,)
            )
            self._conn.execute("DELETE FROM facts WHERE fact_id = ?", (fact_id,))
            self._conn.commit()
            self._rebuild_bank(row["category"])
            return True

    def list_facts(
        self,
        category: str | None = None,
        min_trust: float = 0.0,
        limit: int = 50,
    ) -> list[dict]:
        """Browse facts ordered by trust_score descending.

        Optionally filter by category and minimum trust score.
        """
        with self._lock:
            params: list = [min_trust]
            category_clause = ""
            if category is not None:
                category_clause = "AND category = ?"
                params.append(category)
            params.append(limit)

            sql = f"""
                SELECT fact_id, content, category, tags, trust_score,
                       retrieval_count, helpful_count, created_at, updated_at
                FROM facts
                WHERE trust_score >= ?
                  {category_clause}
                ORDER BY trust_score DESC
                LIMIT ?
            """
            rows = self._conn.execute(sql, params).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def record_feedback(self, fact_id: int, helpful: bool) -> dict:
        """Record user feedback and adjust trust asymmetrically.

        helpful=True  -> trust += 0.05, helpful_count += 1
        helpful=False -> trust -= 0.10

        Returns a dict with fact_id, old_trust, new_trust, helpful_count.
        Raises KeyError if fact_id does not exist.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, trust_score, helpful_count FROM facts WHERE fact_id = ?",
                (fact_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"fact_id {fact_id} not found")

            old_trust: float = row["trust_score"]
            delta = _HELPFUL_DELTA if helpful else _UNHELPFUL_DELTA
            new_trust = _clamp_trust(old_trust + delta)

            helpful_increment = 1 if helpful else 0
            self._conn.execute(
                """
                UPDATE facts
                SET trust_score    = ?,
                    helpful_count  = helpful_count + ?,
                    updated_at     = CURRENT_TIMESTAMP
                WHERE fact_id = ?
                """,
                (new_trust, helpful_increment, fact_id),
            )
            self._conn.commit()

            return {
                "fact_id": fact_id,
                "old_trust": old_trust,
                "new_trust": new_trust,
                "helpful_count": row["helpful_count"] + helpful_increment,
            }

    # ------------------------------------------------------------------
    # Entity helpers
    # ------------------------------------------------------------------

    def _extract_entities(self, text: str) -> list[str]:
        """Extract entity candidates from fact text using layered regex rules.

        Patterns applied (in priority order):
        1. Multi-word capitalized phrases    e.g. "Bosgame M5", "Local Vulkan"
        2. Single-word technical identifiers  e.g. "SearXNG", "ROCm", "SQLite"
        3. Double-quoted terms               e.g. "Python"
        4. Single-quoted terms               e.g. 'pytest'
        5. AKA patterns                      e.g. "Guido aka BDFL" -> two entities
        6. Parenthetical labels              e.g. "(Apr 5)", "(JSON API)"
        7. File paths and config files       e.g. "/home/user/llama.cpp"
        8. Version-like identifiers          e.g. "glm-5-turbo", "v2.0"
        9. Key-value labels at line starts   e.g. "System:", "Search stack:"

        Returns a deduplicated list preserving first-seen order.
        Single-character and stopword matches are filtered.
        """
        seen: set[str] = set()
        candidates: list[str] = []

        def _add(name: str) -> None:
            stripped = name.strip()
            if (
                stripped
                and len(stripped) >= 2
                and stripped.lower() not in _ENTITY_STOPWORDS
                and not _RE_SKIP_FRAGMENT.match(stripped)
                and stripped.lower() not in seen
            ):
                seen.add(stripped.lower())
                candidates.append(stripped)

        # 1. Multi-word capitalized phrases
        for m in _RE_CAPITALIZED.finditer(text):
            _add(m.group(1))

        # 2. Single-word technical identifiers (PascalCase, camelCase, ALL_CAPS)
        for m in _RE_TECH_TERM.finditer(text):
            _add(m.group(1))

        # 3-4. Quoted terms
        for m in _RE_DOUBLE_QUOTE.finditer(text):
            _add(m.group(1))
        for m in _RE_SINGLE_QUOTE.finditer(text):
            _add(m.group(1))

        # 5. AKA patterns
        for m in _RE_AKA.finditer(text):
            _add(m.group(1))
            _add(m.group(2))

        # 6. Parenthetical labels
        for m in _RE_PAREN_LABEL.finditer(text):
            label = m.group(1).strip()
            # Skip date-only parentheticals like "Apr 5" or "10/04/2026"
            if not re.match(
                r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                r"\s+\d{1,2}|\d{1,2}[/\s]\d{1,2}[/\s]\d{2,4}$",
                label,
                re.IGNORECASE,
            ):
                _add(label)

        # 7. File paths
        for m in _RE_FILEPATH.finditer(text):
            _add(m.group(1))

        # 8. Version identifiers
        for m in _RE_VERSION_ID.finditer(text):
            _add(m.group(1))

        # 9. Key-value labels
        for m in _RE_KEY_LABEL.finditer(text):
            _add(m.group(1).strip())

        return candidates

    def _resolve_entity(self, name: str) -> int:
        """Find an existing entity by name or alias (case-insensitive) or create one.

        Returns the entity_id.
        """
        # Exact name match
        row = self._conn.execute(
            "SELECT entity_id FROM entities WHERE name LIKE ?", (name,)
        ).fetchone()
        if row is not None:
            return int(row["entity_id"])

        # Search aliases — aliases stored as comma-separated; use LIKE with % boundaries
        alias_row = self._conn.execute(
            """
            SELECT entity_id FROM entities
            WHERE ',' || aliases || ',' LIKE '%,' || ? || ',%'
            """,
            (name,),
        ).fetchone()
        if alias_row is not None:
            return int(alias_row["entity_id"])

        # Create new entity
        cur = self._conn.execute("INSERT INTO entities (name) VALUES (?)", (name,))
        self._conn.commit()
        return int(cur.lastrowid)  # type: ignore[return-value]

    def _link_fact_entity(self, fact_id: int, entity_id: int) -> None:
        """Insert into fact_entities, silently ignore if the link already exists."""
        self._conn.execute(
            """
            INSERT OR IGNORE INTO fact_entities (fact_id, entity_id)
            VALUES (?, ?)
            """,
            (fact_id, entity_id),
        )
        self._conn.commit()

    def _compute_hrr_vector(self, fact_id: int, content: str) -> None:
        """Compute and store HRR vector for a fact. No-op if numpy unavailable."""
        with self._lock:
            if not self._hrr_available:
                return

            # Get entities linked to this fact
            rows = self._conn.execute(
                """
                SELECT e.name FROM entities e
                JOIN fact_entities fe ON fe.entity_id = e.entity_id
                WHERE fe.fact_id = ?
                """,
                (fact_id,),
            ).fetchall()
            entities = [row["name"] for row in rows]

            vector = hrr.encode_fact(content, entities, self.hrr_dim)
            self._conn.execute(
                "UPDATE facts SET hrr_vector = ? WHERE fact_id = ?",
                (hrr.phases_to_bytes(vector), fact_id),
            )
            self._conn.commit()

    def _compute_neural_embed(self, fact_id: int, content: str) -> None:
        """Compute and cache neural embedding for a fact. Graceful on failure."""
        if not self._embed.alive:
            return

        vec = self._embed.embed(content)
        if vec is not None:
            try:
                import numpy as np

                self._conn.execute(
                    "UPDATE facts SET neural_embed = ? WHERE fact_id = ?",
                    (vec.tobytes(), fact_id),
                )
                self._conn.commit()
            except Exception:
                pass  # Non-critical — HRR + FTS5 still work

    def _rebuild_bank(self, category: str) -> None:
        """Full rebuild of a category's memory bank from all its fact vectors."""
        with self._lock:
            if not self._hrr_available:
                return

            bank_name = f"cat:{category}"
            rows = self._conn.execute(
                "SELECT hrr_vector FROM facts WHERE category = ? AND hrr_vector IS NOT NULL",
                (category,),
            ).fetchall()

            if not rows:
                self._conn.execute(
                    "DELETE FROM memory_banks WHERE bank_name = ?", (bank_name,)
                )
                self._conn.commit()
                return

            vectors = [hrr.bytes_to_phases(row["hrr_vector"]) for row in rows]
            bank_vector = hrr.bundle(*vectors)
            fact_count = len(vectors)

            # Check SNR
            hrr.snr_estimate(self.hrr_dim, fact_count)

            self._conn.execute(
                """
                INSERT INTO memory_banks (bank_name, vector, dim, fact_count, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(bank_name) DO UPDATE SET
                    vector = excluded.vector,
                    dim = excluded.dim,
                    fact_count = excluded.fact_count,
                    updated_at = excluded.updated_at
                """,
                (bank_name, hrr.phases_to_bytes(bank_vector), self.hrr_dim, fact_count),
            )
            self._conn.commit()

    def rebuild_all_vectors(self, dim: int | None = None) -> int:
        """Recompute all HRR vectors + banks from text. For recovery/migration.

        Returns the number of facts processed.
        """
        with self._lock:
            if not self._hrr_available:
                return 0

            if dim is not None:
                self.hrr_dim = dim

            rows = self._conn.execute(
                "SELECT fact_id, content, category FROM facts"
            ).fetchall()

            categories: set[str] = set()
            for row in rows:
                self._compute_hrr_vector(row["fact_id"], row["content"])
                categories.add(row["category"])

            for category in categories:
                self._rebuild_bank(category)

            return len(rows)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """Convert a sqlite3.Row to a plain dict."""
        return dict(row)

    def close(self) -> None:
        """Close the database connection and release locks."""
        if self._conn:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass
            self._conn.close()
            self._conn = None  # type: ignore[assignment]
        if self._lock_fd:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            self._lock_fd.close()
            self._lock_fd = None

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
