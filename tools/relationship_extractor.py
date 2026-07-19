#!/usr/bin/env python3
"""
Relationship Extractor - Extract typed relationships between memories.

Uses pattern-first + LLM fallback (via llm_extractor) to detect relationships:
is_a, part_of, related_to, contradicts, precedes, same_as, author_of, located_in.

Relationships are stored in SQLite graph for traversal and relational retrieval.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .llm_extractor import extract_relationship

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# SQLite storage for relationships
# -----------------------------------------------------------------------------


def get_relationship_db_path() -> Path:
    """Return path to relationship graph SQLite database."""
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "relationship_graph.db"


def init_relationship_db():
    """Initialize relationship graph database."""
    db_path = get_relationship_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_memory_id TEXT NOT NULL,
                target_memory_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                direction TEXT NOT NULL,  -- forward, backward, bidirectional
                confidence REAL,
                extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_source ON relationships (source_memory_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_target ON relationships (target_memory_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_type ON relationships (relationship_type)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_both ON relationships (source_memory_id, target_memory_id)"
        )
        conn.commit()
    finally:
        conn.close()


def store_relationship(
    source_memory_id: str,
    target_memory_id: str,
    relationship_type: str,
    direction: str,
    confidence: float,
):
    """
    Store a relationship between two memories.

    Args:
        source_memory_id: Source memory ID (subject)
        target_memory_id: Target memory ID (object)
        relationship_type: Type of relationship
        direction: forward (source -> target), backward, or bidirectional
        confidence: Confidence score (0.0-1.0)
    """
    init_relationship_db()
    conn = sqlite3.connect(str(get_relationship_db_path()))
    try:
        cursor = conn.cursor()
        # Avoid duplicate relationships (same source, target, type)
        cursor.execute(
            """
            SELECT id FROM relationships 
            WHERE source_memory_id = ? AND target_memory_id = ? AND relationship_type = ?
            """,
            (source_memory_id, target_memory_id, relationship_type),
        )
        if cursor.fetchone():
            logger.debug(
                f"Relationship already exists: {source_memory_id} -> {target_memory_id} ({relationship_type})"
            )
            return

        cursor.execute(
            """
            INSERT INTO relationships (source_memory_id, target_memory_id, relationship_type, direction, confidence)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                source_memory_id,
                target_memory_id,
                relationship_type,
                direction,
                confidence,
            ),
        )
        conn.commit()
        logger.debug(
            f"Stored relationship: {source_memory_id} -> {target_memory_id} ({relationship_type})"
        )
    except Exception as e:
        logger.error(f"Failed to store relationship: {e}")
        conn.rollback()
    finally:
        conn.close()


def get_relationships_for_memory(memory_id: str) -> List[Dict[str, Any]]:
    """Retrieve all relationships where memory_id is source or target."""
    init_relationship_db()
    conn = sqlite3.connect(str(get_relationship_db_path()))
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT source_memory_id, target_memory_id, relationship_type, direction, confidence
            FROM relationships
            WHERE source_memory_id = ? OR target_memory_id = ?
            ORDER BY confidence DESC
            """,
            (memory_id, memory_id),
        )
        rows = cursor.fetchall()
        relationships = []
        for row in rows:
            relationships.append(
                {
                    "source_memory_id": row[0],
                    "target_memory_id": row[1],
                    "relationship_type": row[2],
                    "direction": row[3],
                    "confidence": row[4],
                }
            )
        return relationships
    except Exception as e:
        logger.error(f"Failed to retrieve relationships: {e}")
        return []
    finally:
        conn.close()


def find_related_memories(
    memory_id: str,
    relationship_type: Optional[str] = None,
    direction: Optional[str] = None,
    min_confidence: float = 0.0,
) -> List[Tuple[str, str, float]]:
    """
    Find memories related to the given memory.

    Returns list of (related_memory_id, relationship_type, confidence).
    """
    init_relationship_db()
    conn = sqlite3.connect(str(get_relationship_db_path()))
    try:
        cursor = conn.cursor()
        query = """
            SELECT 
                CASE WHEN source_memory_id = ? THEN target_memory_id ELSE source_memory_id END,
                relationship_type,
                confidence
            FROM relationships
            WHERE (source_memory_id = ? OR target_memory_id = ?)
                AND confidence >= ?
        """
        params = [memory_id, memory_id, memory_id, min_confidence]

        if relationship_type:
            query += " AND relationship_type = ?"
            params.append(relationship_type)
        if direction:
            query += " AND direction = ?"
            params.append(direction)

        query += " ORDER BY confidence DESC"

        cursor.execute(query, params)
        return cursor.fetchall()
    except Exception as e:
        logger.error(f"Failed to find related memories: {e}")
        return []
    finally:
        conn.close()


def traverse_graph(
    start_memory_id: str,
    max_depth: int = 3,
    relationship_filter: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Breadth-first traversal of relationship graph.

    Returns list of nodes with depth and path info.
    """
    from collections import deque

    init_relationship_db()
    conn = sqlite3.connect(str(get_relationship_db_path()))
    try:
        visited = {start_memory_id}
        queue = deque([(start_memory_id, 0, [])])  # (memory_id, depth, path)
        results = []

        while queue:
            current_id, depth, path = queue.popleft()
            if depth > max_depth:
                continue

            # Get relationships from current node
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT source_memory_id, target_memory_id, relationship_type, direction, confidence
                FROM relationships
                WHERE source_memory_id = ? OR target_memory_id = ?
                """,
                (current_id, current_id),
            )
            rows = cursor.fetchall()

            for source, target, rel_type, direction, confidence in rows:
                if relationship_filter and rel_type not in relationship_filter:
                    continue

                # Determine neighbor
                neighbor = target if source == current_id else source
                if neighbor in visited:
                    continue

                visited.add(neighbor)
                new_path = path + [(current_id, rel_type, neighbor)]
                results.append(
                    {
                        "memory_id": neighbor,
                        "depth": depth + 1,
                        "path": new_path,
                        "relationship_type": rel_type,
                        "confidence": confidence,
                    }
                )
                queue.append((neighbor, depth + 1, new_path))

        return results
    except Exception as e:
        logger.error(f"Graph traversal failed: {e}")
        return []
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# Relationship extraction between memories
# -----------------------------------------------------------------------------


def extract_relationships_for_memory(
    memory_id: str,
    text: str,
    vector_store,
    top_k: int = 5,
    use_llm_fallback: bool = True,
):
    """
    Extract relationships between new memory and existing similar memories.

    Args:
        memory_id: ID of the new memory
        text: Text content of the new memory
        vector_store: VectorMemoryStore instance for similarity search
        top_k: Number of similar memories to compare with
        use_llm_fallback: Whether to use LLM fallback for extraction
    """
    try:
        # Search for similar memories
        similar_results = vector_store.search(text, top_k=top_k)
        if not similar_results:
            logger.debug(f"No similar memories found for {memory_id}")
            return

        for result in similar_results:
            if result.id == memory_id:
                continue  # skip self

            # Get the similar memory's text (we need to fetch from vector store)
            similar_memory = vector_store.get_memory(result.id)
            if not similar_memory:
                continue
            similar_text = similar_memory.get("text", "")
            if not similar_text:
                continue

            # Extract relationship
            rel = extract_relationship(
                text,
                similar_text,
                use_llm_fallback=use_llm_fallback,
            )
            if not rel:
                continue

            # Determine direction: if direction is 'backward', swap source/target
            direction = rel.get("direction", "bidirectional")
            source_id = memory_id
            target_id = result.id
            if direction == "backward":
                source_id, target_id = target_id, source_id
                direction = "forward"  # store as forward after swapping

            # Store relationship
            store_relationship(
                source_memory_id=source_id,
                target_memory_id=target_id,
                relationship_type=rel["relationship"],
                direction=direction,
                confidence=rel.get("confidence", 0.5),
            )
            logger.debug(
                f"Extracted relationship {rel['relationship']} between {memory_id} and {result.id}"
            )

    except Exception as e:
        logger.error(f"Relationship extraction failed for memory {memory_id}: {e}")


# -----------------------------------------------------------------------------
# Integration with memory addition
# -----------------------------------------------------------------------------


def extract_and_store_relationships(
    memory_id: str, text: str, vector_store, use_llm_fallback: bool = True
):
    """
    Extract relationships for a new memory and store them.

    This should be called after adding a vector memory.
    """
    try:
        extract_relationships_for_memory(
            memory_id,
            text,
            vector_store,
            top_k=5,
            use_llm_fallback=use_llm_fallback,
        )
        logger.info(f"Relationship extraction completed for memory {memory_id}")
    except Exception as e:
        logger.error(f"Relationship extraction failed for memory {memory_id}: {e}")


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Relationship extractor CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # extract
    extract_parser = subparsers.add_parser(
        "extract", help="Extract relationship between two texts"
    )
    extract_parser.add_argument("text1", help="First text")
    extract_parser.add_argument("text2", help="Second text")
    extract_parser.add_argument("--llm", action="store_true", help="Use LLM fallback")

    # store
    store_parser = subparsers.add_parser(
        "store", help="Extract relationships for a memory (requires vector store)"
    )
    store_parser.add_argument("memory_id", help="Memory ID")
    store_parser.add_argument(
        "--text", help="Memory text (if not provided, read from stdin)"
    )
    store_parser.add_argument(
        "--top-k", type=int, default=5, help="Number of similar memories to compare"
    )

    # query
    query_parser = subparsers.add_parser("query", help="Query relationships")
    query_parser.add_argument("memory_id", help="Memory ID to query")
    query_parser.add_argument("--type", help="Filter by relationship type")
    query_parser.add_argument("--direction", help="Filter by direction")
    query_parser.add_argument(
        "--min-confidence", type=float, default=0.0, help="Minimum confidence"
    )

    # traverse
    traverse_parser = subparsers.add_parser(
        "traverse", help="Traverse relationship graph"
    )
    traverse_parser.add_argument("memory_id", help="Starting memory ID")
    traverse_parser.add_argument(
        "--max-depth", type=int, default=3, help="Maximum traversal depth"
    )
    traverse_parser.add_argument(
        "--types", help="Comma-separated relationship types to include"
    )

    args = parser.parse_args()

    if args.command == "extract":
        result = extract_relationship(args.text1, args.text2, use_llm_fallback=args.llm)
        print(json.dumps(result, indent=2))

    elif args.command == "store":
        # This requires vector store; we'll skip for CLI simplicity
        print("Store command requires vector store integration; use from code.")

    elif args.command == "query":
        results = find_related_memories(
            args.memory_id,
            relationship_type=args.type,
            direction=args.direction,
            min_confidence=args.min_confidence,
        )
        for mem_id, rel_type, conf in results:
            print(f"{mem_id} ({rel_type}) confidence={conf:.3f}")

    elif args.command == "traverse":
        types = args.types.split(",") if args.types else None
        results = traverse_graph(
            args.memory_id,
            max_depth=args.max_depth,
            relationship_filter=types,
        )
        for node in results:
            path_str = (
                " -> ".join([f"{src}[{rel}]" for src, rel, _ in node["path"]])
                + f" -> {node['memory_id']}"
            )
            print(f"Depth {node['depth']}: {path_str}")

    else:
        parser.print_help()
