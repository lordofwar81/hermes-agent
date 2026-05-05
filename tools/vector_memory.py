#!/usr/bin/env python3
"""
Vector Memory Store - LanceDB-based vector memory with embeddings.

Features:
- LanceDB table with 1024-dim embeddings
- Semantic search via vector similarity
- Epistemic status tracking (stated, inferred, verified, contradicted, retracted)
- Entity and keyword indexing
- Integration with BM25 for hybrid search

Usage:
    from tools.vector_memory import VectorMemoryStore

    store = VectorMemoryStore()

    # Add a memory
    memory_id = store.add_memory(
        text="User prefers Python over JavaScript",
        source="user",
        memory_type="preference",
        session_id="session_123"
    )

    # Search
    results = store.search("Python", top_k=10)

    # Update epistemic status
    store.update_epistemic_status(memory_id, "verified", confidence=0.9)
"""

import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import requests

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Default vector memory path
DEFAULT_VECTOR_MEMORY_PATH = Path.home() / ".hermes" / "vector_memory"

# Embedding endpoint (same as adaptive_context_manager.py)
EMBED_ENDPOINT = "http://localhost:11434/v1"
EMBED_MODEL = "mxbai-embed-large-v1"

# Default epistemic status values
EPISTEMIC_STATUSES = ["stated", "inferred", "verified", "contradicted", "retracted"]

# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclass
class MemoryRecord:
    """A memory record in the vector store."""

    id: str
    text: str
    source: str
    memory_type: str
    session_id: str
    created_at: float
    access_count: int
    epistemic_status: str
    confidence: float
    entities: List[str]
    keywords: List[str]
    related_ids: List[str]
    version: int
    # Note: vector field is stored separately in LanceDB


@dataclass
class SearchResult:
    """Search result with similarity score."""

    id: str
    text: str
    source: str
    memory_type: str
    similarity: float
    epistemic_status: str
    confidence: float
    entities: List[str]
    keywords: List[str]
    created_at: float


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding vector for text using local Ollama endpoint."""
    # Truncate to avoid exceeding embedding model's token limit
    if len(text) > 800:
        text = text[:800]
        logger.debug("Truncated text for embedding (max 800 chars)")

    try:
        response = requests.post(
            f"{EMBED_ENDPOINT}/embeddings",
            headers={"Content-Type": "application/json"},
            json={"model": EMBED_MODEL, "input": text},
            timeout=30,
        )
        if response.status_code == 200:
            result = response.json()
            return result["data"][0]["embedding"]
        else:
            logger.error(
                f"Embedding API error: {response.status_code} - {response.text}"
            )
            return None
    except Exception as e:
        logger.error(f"Failed to get embedding: {e}")
        return None


# -----------------------------------------------------------------------------
# Vector Memory Store
# -----------------------------------------------------------------------------


class VectorMemoryStore:
    """LanceDB-based vector memory store."""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize vector memory store.

        Args:
            db_path: Path to LanceDB directory (default: ~/.hermes/vector_memory)
        """
        self.db_path = db_path or DEFAULT_VECTOR_MEMORY_PATH
        self.db = None
        self.table = None
        self._connect()

    def _connect(self):
        """Connect to LanceDB and open the memory_vectors table."""
        try:
            import lancedb

            self.db = lancedb.connect(str(self.db_path))

            # Check if table exists
            tables = self.db.list_tables()
            # Handle both ListTablesResponse object and plain list
            if hasattr(tables, "tables"):
                actual_tables = tables.tables
            else:
                actual_tables = tables
            if "memory_vectors" in actual_tables:
                self.table = self.db.open_table("memory_vectors")
                logger.debug(f"Opened existing table 'memory_vectors'")
            else:
                # Table doesn't exist - we'll skip vector operations
                logger.warning(f"Table 'memory_vectors' not found in {self.db_path}")
                self.table = None

        except ImportError:
            logger.error("lancedb module not available")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to LanceDB: {e}")
            raise

    def add_memory(
        self,
        text: str,
        source: str = "user",
        memory_type: str = "general",
        session_id: str = "",
        epistemic_status: str = "stated",
        confidence: float = 0.5,
        entities: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        related_ids: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Add a memory to the vector store.

        Args:
            text: Memory text content
            source: Source of memory (user, system, inferred)
            memory_type: Type of memory (preference, fact, instruction, etc.)
            session_id: Session ID where memory was created
            epistemic_status: Epistemic status (stated, inferred, verified, etc.)
            confidence: Confidence score (0.0-1.0)
            entities: List of entities mentioned
            keywords: List of keywords
            related_ids: List of related memory IDs

        Returns:
            Memory ID if successful, None otherwise
        """
        # Generate embedding
        vector = get_embedding(text)
        if vector is None:
            logger.error("Failed to generate embedding, memory not added")
            return None

        # Generate ID
        memory_id = str(uuid.uuid4())

        # Prepare data
        data = {
            "id": memory_id,
            "vector": vector,
            "text": text,
            "source": source,
            "memory_type": memory_type,
            "session_id": session_id,
            "created_at": time.time(),
            "access_count": 0,
            "epistemic_status": epistemic_status,
            "confidence": float(confidence),
            "entities": entities or [],
            "keywords": keywords or [],
            "related_ids": related_ids or [],
            "version": 2,
        }

        try:
            self.table.add([data])
            logger.debug(f"Added memory {memory_id}")
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            return None

        # --- BM25 auto-sync: mirror to SQLite FTS5 store ---
        try:
            from tools.bm25_memory import BM25MemoryStore

            bm25 = BM25MemoryStore()
            bm25.add_memory(
                memory_id=memory_id,
                text=text,
                source=source,
                memory_type=memory_type,
            )
            bm25.close()
            logger.debug(f"BM25 auto-sync: mirrored {memory_id}")
        except Exception as bm25_err:
            # Non-fatal: vector write succeeded; BM25 is supplementary
            logger.warning(f"BM25 auto-sync failed for {memory_id}: {bm25_err}")

        return memory_id

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search memories by semantic similarity.

        Args:
            query: Search query text
            top_k: Maximum number of results
            min_similarity: Minimum similarity threshold
            filters: Optional filters (e.g., {"source": "user", "epistemic_status": "verified"})

        Returns:
            List of search results
        """
        # Generate embedding for query
        query_vector = get_embedding(query)
        if query_vector is None:
            logger.error("Failed to generate query embedding")
            return []

        # Build query
        try:
            query_builder = self.table.search(query_vector)
            query_builder = query_builder.limit(top_k)

            # Apply filters if provided
            if filters:
                for key, value in filters.items():
                    if key in ["source", "memory_type", "epistemic_status"]:
                        safe_value = self._escape_filter_value(value)
                        query_builder = query_builder.where(f"{key} = '{safe_value}'")
                    elif key == "min_confidence":
                        query_builder = query_builder.where(
                            f"confidence >= {float(value)}"
                        )
                    elif key == "before":
                        query_builder = query_builder.where(
                            f"created_at <= {float(value)}"
                        )
                    elif key == "after":
                        query_builder = query_builder.where(
                            f"created_at >= {float(value)}"
                        )

            # Execute query
            results = query_builder.to_list()

            # Convert to SearchResult objects
            search_results = []
            for row in results:
                similarity = 1.0 - row["_distance"] if "_distance" in row else 0.0

                if similarity >= min_similarity:
                    result = SearchResult(
                        id=row.get("id", ""),
                        text=row.get("text", ""),
                        source=row.get("source", ""),
                        memory_type=row.get("memory_type", ""),
                        similarity=similarity,
                        epistemic_status=row.get("epistemic_status", "stated"),
                        confidence=row.get("confidence", 0.5),
                        entities=row.get("entities", []),
                        keywords=row.get("keywords", []),
                        created_at=row.get("created_at", 0.0),
                    )
                    search_results.append(result)

            # Sort by similarity (descending)
            search_results.sort(key=lambda x: x.similarity, reverse=True)
            return search_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _escape_filter_value(self, value: Any) -> str:
        """Escape a value for safe use in LanceDB .where() filter strings."""
        if isinstance(value, (int, float)):
            return str(value)
        # Escape single quotes by doubling them (SQL standard)
        return str(value).replace("'", "''")

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a memory by ID."""
        try:
            safe_id = self._escape_filter_value(memory_id)
            results = (
                self.table.search().where(f"id = '{safe_id}'").limit(1).to_list()
            )
            if results:
                return results[0]
            return None
        except Exception as e:
            logger.error(f"Failed to get memory {memory_id}: {e}")
            return None

    def update_epistemic_status(
        self,
        memory_id: str,
        status: str,
        confidence: Optional[float] = None,
    ) -> bool:
        """
        Update epistemic status of a memory.

        Args:
            memory_id: Memory ID
            status: New epistemic status
            confidence: New confidence score (optional)

        Returns:
            True if successful
        """
        if status not in EPISTEMIC_STATUSES:
            logger.error(f"Invalid epistemic status: {status}")
            return False

        try:
            # Get current memory
            memory = self.get_memory(memory_id)
            if not memory:
                logger.error(f"Memory {memory_id} not found")
                return False

            # Update
            update_data = {"id": memory_id, "epistemic_status": status}
            if confidence is not None:
                update_data["confidence"] = float(confidence)

            # LanceDB update
            self.table.update(where=f"id = '{memory_id}'", values=update_data)
            logger.debug(f"Updated epistemic status of {memory_id} to {status}")
            return True

        except Exception as e:
            logger.error(f"Failed to update epistemic status: {e}")
            return False

    def increment_access_count(self, memory_id: str) -> bool:
        """Increment access count for a memory."""
        try:
            memory = self.get_memory(memory_id)
            if not memory:
                return False

            current_count = memory.get("access_count", 0)
            self.table.update(
                where=f"id = '{memory_id}'", values={"access_count": current_count + 1}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to increment access count: {e}")
            return False

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        try:
            self.table.delete(where=f"id = '{memory_id}'")
            logger.debug(f"Deleted memory {memory_id}")
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False

        # --- BM25 auto-sync: remove from SQLite FTS5 store ---
        try:
            from tools.bm25_memory import BM25MemoryStore

            bm25 = BM25MemoryStore()
            bm25.delete_memory(memory_id)
            bm25.close()
            logger.debug(f"BM25 auto-sync: deleted {memory_id}")
        except Exception as bm25_err:
            logger.warning(f"BM25 auto-sync delete failed for {memory_id}: {bm25_err}")

        return True

    def list_memories(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        List memories with optional filtering.

        Args:
            limit: Maximum number of memories
            offset: Offset for pagination
            filters: Optional filters

        Returns:
            List of memory dictionaries
        """
        try:
            df = self.table.to_pandas()
            if df.empty:
                return []

            # Apply filters
            if filters:
                for key, value in filters.items():
                    if key in ["source", "memory_type", "epistemic_status"]:
                        df = df[df[key] == value]
                    elif key == "min_confidence":
                        df = df[df["confidence"] >= float(value)]

            # Apply offset and limit
            total = len(df)
            if offset >= total:
                return []
            end = min(offset + limit, total)
            df_subset = df.iloc[offset:end]

            # Convert to list of dicts
            memories = []
            for _, row in df_subset.iterrows():
                mem = {}
                for col in df.columns:
                    if col == "vector":
                        continue  # skip vector column
                    val = row[col]
                    # Convert numpy/pandas types to Python types
                    if hasattr(val, "tolist"):
                        val = val.tolist()
                    elif hasattr(val, "item"):
                        val = val.item()
                    mem[col] = val
                memories.append(mem)
            return memories
        except Exception as e:
            logger.error(f"Failed to list memories: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return []

    def count(self) -> int:
        """Get total number of memories."""
        try:
            return self.table.count_rows()
        except Exception as e:
            logger.error(f"Failed to count memories: {e}")
            return 0


# -----------------------------------------------------------------------------
# Integration with contradiction detector
# -----------------------------------------------------------------------------

try:
    from tools.contradiction_detector import add_epistemic_methods_to_class

    # Monkey-patch the class to add epistemic methods
    add_epistemic_methods_to_class(VectorMemoryStore)
except ImportError:
    pass


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vector Memory Store test")
    parser.add_argument("--add", help="Text to add as memory")
    parser.add_argument("--search", help="Search query")
    parser.add_argument("--list", action="store_true", help="List memories")
    parser.add_argument("--count", action="store_true", help="Count memories")
    parser.add_argument("--top-k", type=int, default=10, help="Top K results")

    args = parser.parse_args()

    store = VectorMemoryStore()

    if args.add:
        memory_id = store.add_memory(args.add)
        if memory_id:
            print(f"Added memory: {memory_id}")
        else:
            print("Failed to add memory")

    if args.search:
        results = store.search(args.search, top_k=args.top_k)
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"{i + 1}. [{result.similarity:.3f}] {result.text[:80]}...")
            print(
                f"   ID: {result.id}, Source: {result.source}, Status: {result.epistemic_status}"
            )

    if args.list:
        memories = store.list_memories(limit=10)
        print(f"Recent memories ({len(memories)}):")
        for i, mem in enumerate(memories):
            print(f"{i + 1}. {mem.get('text', '')[:80]}...")
            print(f"   ID: {mem.get('id')}, Created: {mem.get('created_at')}")

    if args.count:
        count = store.count()
        print(f"Total memories: {count}")
