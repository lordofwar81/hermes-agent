#!/usr/bin/env python3
"""
BM25 Memory Store - Keyword-based search complementing vector similarity.

Features:
- SQLite FTS5 table: `memories_fts`
- Keyword extraction with stopword removal
- BM25 ranking (normalized 0.0-1.0)
- Filters: source, memory_type, epistemic_status
- RRF fusion: `rrf_fusion(vector, bm25, alpha=0.7)`

Usage:
    from tools.bm25_memory import BM25MemoryStore, rrf_fusion

    store = BM25MemoryStore()

    # Add to BM25 index
    store.add_memory("vm_123", "User prefers Python")

    # Search
    results = store.search("Python", top_k=10)

    # RRF fusion
    fused = rrf_fusion(vector_results, bm25_results, alpha=0.7)
"""

import logging
import re
import sqlite3
import string
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Default SQLite database path
DEFAULT_DB_PATH = Path.home() / ".hermes" / "vector_memory" / "bm25_store.db"

# Stopwords (common English)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "i",
    "you",
    "we",
    "they",
    "this",
    "that",
    "these",
    "those",
    "my",
    "your",
    "our",
    "their",
    "me",
    "him",
    "her",
    "us",
    "them",
}

# Minimum keyword length
MIN_KEYWORD_LENGTH = 3

# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclass
class BM25Result:
    """BM25 search result."""

    memory_id: str
    text: str
    score: float  # BM25 score (higher is better)
    source: str = ""
    memory_type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "text": self.text,
            "score": self.score,
            "source": self.source,
            "memory_type": self.memory_type,
        }


# -----------------------------------------------------------------------------
# Keyword extraction
# -----------------------------------------------------------------------------


def extract_keywords(text: str) -> List[str]:
    """
    Extract keywords from text using simple tokenization and stopword removal.

    Args:
        text: Input text

    Returns:
        List of keywords (lowercased, no punctuation)
    """
    if not text:
        return []

    # Remove punctuation and convert to lowercase
    text_lower = text.lower()
    # Replace punctuation with spaces
    for punct in string.punctuation:
        text_lower = text_lower.replace(punct, " ")

    # Split on whitespace
    tokens = text_lower.split()

    # Filter stopwords and short tokens
    keywords = []
    for token in tokens:
        if (
            len(token) >= MIN_KEYWORD_LENGTH
            and token not in STOPWORDS
            and not token.isdigit()
        ):
            keywords.append(token)

    return keywords


# -----------------------------------------------------------------------------
# BM25 Memory Store
# -----------------------------------------------------------------------------


class BM25MemoryStore:
    """SQLite FTS5-based keyword search store."""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize BM25 store.

        Args:
            db_path: Path to SQLite database file (default: ~/.hermes/vector_memory/bm25_store.db)
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = None
        self._connect()

    def _connect(self):
        """Establish database connection and create tables if needed."""
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row

        # Enable FTS5 extension
        self._conn.execute("PRAGMA journal_mode=WAL")

        # Create memories table (metadata)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT DEFAULT '',
                memory_type TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create FTS5 virtual table for full-text search
        # Note: FTS5 must be compiled into SQLite
        try:
            self._conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(
                    memory_id UNINDEXED,
                    text,
                    source UNINDEXED,
                    memory_type UNINDEXED,
                    content='memories',
                    content_rowid='rowid'
                )
            """)
        except sqlite3.OperationalError as e:
            if "fts5" in str(e):
                logger.error(
                    "FTS5 not available in SQLite. Please recompile SQLite with FTS5 support."
                )
                raise
            else:
                raise

        # Create triggers to keep FTS5 table in sync
        self._conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, memory_id, text, source, memory_type)
                VALUES (new.rowid, new.memory_id, new.text, new.source, new.memory_type);
            END
        """)

        self._conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, memory_id, text, source, memory_type)
                VALUES ('delete', old.rowid, old.memory_id, old.text, old.source, old.memory_type);
            END
        """)

        self._conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, memory_id, text, source, memory_type)
                VALUES ('delete', old.rowid, old.memory_id, old.text, old.source, old.memory_type);
                INSERT INTO memories_fts(rowid, memory_id, text, source, memory_type)
                VALUES (new.rowid, new.memory_id, new.text, new.source, new.memory_type);
            END
        """)

        self._conn.commit()

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def add_memory(
        self,
        memory_id: str,
        text: str,
        source: str = "",
        memory_type: str = "",
    ) -> bool:
        """
        Add a memory to BM25 index.

        Args:
            memory_id: Unique identifier (should match vector store ID)
            text: Memory text content
            source: Source of memory (e.g., "user", "agent", "system")
            memory_type: Type of memory (e.g., "fact", "preference", "note")

        Returns:
            True if successful, False otherwise
        """
        try:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO memories (memory_id, text, source, memory_type)
                VALUES (?, ?, ?, ?)
                """,
                (memory_id, text, source, memory_type),
            )
            self._conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to add memory {memory_id}: {e}")
            return False

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """
        Sanitize a search query for safe use with FTS5 MATCH.

        FTS5 interprets special characters/operators in MATCH expressions:
          - ':'  → column filter (e.g. "M3:term" → column M3 → error)
          - '*'  → prefix search
          - '"'  → phrase delimiter
          - 'NEAR', 'AND', 'OR', 'NOT' → boolean operators

        We strip all FTS5 special syntax and rebuild a safe token-only query
        joined with OR so that any token can match independently.

        Args:
            query: Raw user query text

        Returns:
            Sanitized query safe for FTS5 MATCH
        """
        if not query:
            return ""

        # Remove double-quote characters (phrase delimiters)
        cleaned = query.replace('"', "")

        # Replace colons and other FTS5 operator chars with spaces
        # to prevent column-filter interpretation (e.g. "M3:" → "M3 ")
        for ch in (":", "*", "+", "-", "^", "#", "@"):
            cleaned = cleaned.replace(ch, " ")

        # Remove FTS5 keyword operators (case-insensitive, word-boundary)
        cleaned = re.sub(
            r"\b(NEAR|AND|OR|NOT)\b",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )

        # Collapse whitespace and strip
        tokens = cleaned.split()

        if not tokens:
            return ""

        # Wrap each token in double-quotes for literal matching and join with OR
        # so FTS5 treats every token as a plain term
        safe_tokens = [f'"{t}"' for t in tokens]
        return " OR ".join(safe_tokens)

    def search(
        self,
        query: str,
        top_k: int = 10,
        source: Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> List[BM25Result]:
        """
        Search memories using BM25 ranking.

        Args:
            query: Search query (auto-sanitized for FTS5 safety)
            top_k: Maximum number of results
            source: Filter by source (optional)
            memory_type: Filter by memory type (optional)

        Returns:
            List of BM25Result objects sorted by relevance
        """
        if not query:
            return []

        # Sanitize query to prevent FTS5 syntax errors (e.g. "no such column: M3")
        safe_query = self._sanitize_fts_query(query)
        if not safe_query:
            return []

        # Build WHERE clause for filters
        where_clauses = []
        params = []

        # Basic FTS5 match (using sanitized query)
        where_clauses.append("memories_fts MATCH ?")
        params.append(safe_query)

        if source:
            where_clauses.append("source = ?")
            params.append(source)

        if memory_type:
            where_clauses.append("memory_type = ?")
            params.append(memory_type)

        where_sql = " AND ".join(where_clauses)

        # Use FTS5 bm25() function for ranking
        # bm25() returns lower values for better matches, so we use ORDER BY bm25(memories_fts)
        sql = f"""
            SELECT memory_id, text, source, memory_type,
                   bm25(memories_fts) as bm25_score
            FROM memories_fts
            WHERE {where_sql}
            ORDER BY bm25_score
            LIMIT ?
        """
        params.append(top_k)

        try:
            cursor = self._conn.execute(sql, params)
            rows = cursor.fetchall()

            # Convert BM25 scores to normalized 0.0-1.0 range (lower bm25() is better)
            results = []
            for row in rows:
                # bm25() typically returns negative values; normalize
                bm25_raw = row["bm25_score"]
                # Simple normalization: convert to positive score where 0 = perfect match
                # For simplicity, we'll use 1 / (1 + abs(bm25_raw))
                score = 1.0 / (1.0 + abs(bm25_raw)) if bm25_raw != 0 else 1.0

                results.append(
                    BM25Result(
                        memory_id=row["memory_id"],
                        text=row["text"],
                        score=score,
                        source=row["source"],
                        memory_type=row["memory_type"],
                    )
                )

            return results
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory from BM25 index."""
        try:
            self._conn.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
            self._conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve memory metadata by ID."""
        cursor = self._conn.execute(
            "SELECT * FROM memories WHERE memory_id = ?",
            (memory_id,),
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

    def count(self) -> int:
        """Return total number of memories in index."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM memories")
        return cursor.fetchone()[0]


# -----------------------------------------------------------------------------
# RRF Fusion
# -----------------------------------------------------------------------------


def rrf_fusion(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[BM25Result],
    alpha: float = 0.7,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF) of vector and BM25 results.

    Args:
        vector_results: List of vector search results (each must have 'memory_id' and 'similarity' score)
        bm25_results: List of BM25Result objects
        alpha: Weight for vector results (0.0-1.0), BM25 weight = 1 - alpha
        top_k: Maximum number of fused results

    Returns:
        List of fused results sorted by RRF score
    """
    # Build rank dictionaries
    vector_ranks = {}
    for i, result in enumerate(vector_results):
        memory_id = result.get("memory_id")
        if memory_id:
            vector_ranks[memory_id] = i + 1  # rank starts at 1

    bm25_ranks = {}
    for i, result in enumerate(bm25_results):
        bm25_ranks[result.memory_id] = i + 1

    # Combine memory IDs
    all_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())

    # Compute RRF scores
    rrf_scores = []
    for memory_id in all_ids:
        vector_rank = vector_ranks.get(memory_id, top_k * 2)  # penalty for missing
        bm25_rank = bm25_ranks.get(memory_id, top_k * 2)

        # RRF formula: score = alpha * (1 / (k + vector_rank)) + (1 - alpha) * (1 / (k + bm25_rank))
        # where k is a constant (typically 60)
        k = 60
        vector_score = 1.0 / (k + vector_rank)
        bm25_score = 1.0 / (k + bm25_rank)

        rrf_score = alpha * vector_score + (1 - alpha) * bm25_score

        rrf_scores.append(
            {
                "memory_id": memory_id,
                "rrf_score": rrf_score,
                "vector_rank": vector_rank if memory_id in vector_ranks else None,
                "bm25_rank": bm25_rank if memory_id in bm25_ranks else None,
            }
        )

    # Sort by RRF score descending
    rrf_scores.sort(key=lambda x: x["rrf_score"], reverse=True)

    return rrf_scores[:top_k]


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BM25 memory store test")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add a memory")
    add_parser.add_argument("id", help="Memory ID")
    add_parser.add_argument("text", help="Memory text")
    add_parser.add_argument("--source", default="", help="Source")
    add_parser.add_argument(
        "--type", dest="memory_type", default="", help="Memory type"
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Search memories")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--top-k", type=int, default=10, help="Number of results"
    )

    # Stats command
    subparsers.add_parser("stats", help="Show store statistics")

    args = parser.parse_args()

    store = BM25MemoryStore()

    try:
        if args.command == "add":
            success = store.add_memory(
                args.id, args.text, args.source, args.memory_type
            )
            print(f"Added: {success}")
        elif args.command == "search":
            results = store.search(args.query, top_k=args.top_k)
            print(f"Found {len(results)} results:")
            for result in results:
                print(f"  ID: {result.memory_id}")
                print(f"  Text: {result.text[:80]}...")
                print(f"  Score: {result.score:.4f}")
                print()
        elif args.command == "stats":
            count = store.count()
            print(f"Total memories: {count}")
        else:
            parser.print_help()
    finally:
        store.close()
