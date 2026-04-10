#!/usr/bin/env python3
"""
Temporal Extractor - Extract event timestamps from text and link to memories.

Uses pattern-first + LLM fallback (via llm_extractor) to detect explicit or implied
timestamps. Can be called during memory addition or during hippocampus consolidation.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date

from .llm_extractor import extract_temporal

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# SQLite storage for temporal events
# -----------------------------------------------------------------------------


def get_temporal_db_path() -> Path:
    """Return path to temporal events SQLite database."""
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "temporal_memory.db"


def init_temporal_db():
    """Initialize temporal events database."""
    db_path = get_temporal_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS temporal_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                event_text TEXT NOT NULL,
                start_date TEXT,  -- YYYY-MM-DD or NULL
                end_date TEXT,    -- YYYY-MM-DD or NULL
                confidence REAL,
                extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (memory_id) REFERENCES memory_vectors(id) ON DELETE CASCADE
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_id ON temporal_events (memory_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_start_date ON temporal_events (start_date)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_end_date ON temporal_events (end_date)"
        )
        conn.commit()
    finally:
        conn.close()


def store_temporal_events(memory_id: str, events: List[Dict[str, Any]]):
    """
    Store temporal events for a memory.

    Args:
        memory_id: Vector memory ID
        events: List of event dicts from extract_temporal
    """
    init_temporal_db()
    conn = sqlite3.connect(str(get_temporal_db_path()))
    try:
        cursor = conn.cursor()
        # Delete existing events for this memory (replace on re-extraction)
        cursor.execute("DELETE FROM temporal_events WHERE memory_id = ?", (memory_id,))

        for event in events:
            cursor.execute(
                """
                INSERT INTO temporal_events (memory_id, event_text, start_date, end_date, confidence)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    event.get("text", ""),
                    event.get("start"),
                    event.get("end"),
                    event.get("confidence", 0.5),
                ),
            )
        conn.commit()
        logger.debug(f"Stored {len(events)} temporal events for memory {memory_id}")
    except Exception as e:
        logger.error(f"Failed to store temporal events: {e}")
        conn.rollback()
    finally:
        conn.close()


def get_temporal_events(memory_id: str) -> List[Dict[str, Any]]:
    """Retrieve temporal events for a memory."""
    init_temporal_db()
    conn = sqlite3.connect(str(get_temporal_db_path()))
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT event_text, start_date, end_date, confidence FROM temporal_events WHERE memory_id = ? ORDER BY id",
            (memory_id,),
        )
        rows = cursor.fetchall()
        events = []
        for row in rows:
            events.append(
                {
                    "text": row[0],
                    "start": row[1],
                    "end": row[2],
                    "confidence": row[3],
                }
            )
        return events
    except Exception as e:
        logger.error(f"Failed to retrieve temporal events: {e}")
        return []
    finally:
        conn.close()


def search_memories_by_temporal(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    date: Optional[str] = None,
    limit: int = 100,
) -> List[Tuple[str, str]]:
    """
    Search memories by temporal constraints.

    Args:
        start_date: Memories with events starting after this date (YYYY-MM-DD)
        end_date: Memories with events ending before this date
        date: Exact date (search for events containing this date)
        limit: Maximum results

    Returns:
        List of (memory_id, event_text) pairs
    """
    init_temporal_db()
    conn = sqlite3.connect(str(get_temporal_db_path()))
    try:
        cursor = conn.cursor()
        query = "SELECT DISTINCT memory_id, event_text FROM temporal_events WHERE 1=1"
        params = []

        if date:
            query += " AND (start_date = ? OR end_date = ? OR (start_date <= ? AND end_date >= ?))"
            params.extend([date, date, date, date])
        else:
            if start_date:
                query += " AND (end_date IS NULL OR end_date >= ?)"
                params.append(start_date)
            if end_date:
                query += " AND (start_date IS NULL OR start_date <= ?)"
                params.append(end_date)

        query += " LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        return cursor.fetchall()
    except Exception as e:
        logger.error(f"Temporal search failed: {e}")
        return []
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# Integration with memory addition
# -----------------------------------------------------------------------------


def extract_and_store_temporal(
    memory_id: str, text: str, use_llm_fallback: bool = True
):
    """
    Extract temporal events from text and store them linked to memory_id.

    This should be called after adding a vector memory.
    """
    try:
        result = extract_temporal(text, use_llm_fallback=use_llm_fallback)
        events = result.get("events", [])
        if events:
            store_temporal_events(memory_id, events)
            logger.info(
                f"Extracted {len(events)} temporal events for memory {memory_id}"
            )
        else:
            logger.debug(f"No temporal events found in memory {memory_id}")
    except Exception as e:
        logger.error(f"Temporal extraction failed for memory {memory_id}: {e}")


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Temporal extractor CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # extract
    extract_parser = subparsers.add_parser(
        "extract", help="Extract temporal events from text"
    )
    extract_parser.add_argument("text", help="Text to analyze")
    extract_parser.add_argument("--llm", action="store_true", help="Use LLM fallback")

    # store
    store_parser = subparsers.add_parser(
        "store", help="Store temporal events for a memory"
    )
    store_parser.add_argument("memory_id", help="Memory ID")
    store_parser.add_argument(
        "--text", help="Text to extract from (if not provided, read from stdin)"
    )

    # search
    search_parser = subparsers.add_parser(
        "search", help="Search memories by temporal constraints"
    )
    search_parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    search_parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    search_parser.add_argument("--date", help="Exact date (YYYY-MM-DD)")
    search_parser.add_argument("--limit", type=int, default=100, help="Result limit")

    args = parser.parse_args()

    if args.command == "extract":
        result = extract_temporal(args.text, use_llm_fallback=args.llm)
        print(json.dumps(result, indent=2))

    elif args.command == "store":
        text = args.text
        if not text:
            text = sys.stdin.read().strip()
        if not text:
            print("Error: No text provided")
            sys.exit(1)
        extract_and_store_temporal(args.memory_id, text, use_llm_fallback=True)
        print(f"Stored temporal events for memory {args.memory_id}")

    elif args.command == "search":
        results = search_memories_by_temporal(
            start_date=args.start,
            end_date=args.end,
            date=args.date,
            limit=args.limit,
        )
        for memory_id, event_text in results:
            print(f"{memory_id}: {event_text}")

    else:
        parser.print_help()
