#!/usr/bin/env python3
"""
Unified Memory Search - Single entry point for all memory search operations.

Features:
- Modes: vector, bm25, hybrid
- Temporal queries: `before:2026-03-20`, `after:2026-03-15`, `time_range:7d`
- Filters: source, memory_type, epistemic_status
- RRF fusion with alpha=0.7

Tool Schema: memory_search

Usage:
    from tools.unified_memory_search import UnifiedMemorySearch

    searcher = UnifiedMemorySearch(vector_memory, bm25_store)

    # Hybrid search
    results_json = searcher.search(
        query="Python time_range:30d",
        mode="hybrid",
        top_k=10,
        alpha=0.7,
    )

    # Vector-only
    results_json = searcher.search(
        query="Python",
        mode="vector",
        top_k=10,
    )
"""

import json
import logging
import math
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from tools.registry import registry

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Query Parser
# -----------------------------------------------------------------------------


def _parse_relative_date_expr(expr: str) -> Optional[Dict[str, Any]]:
    """
    Parse relative date expression to filter dict.

    Returns dict with keys: before, after, or time_range_days.
    Returns None if not a relative date expression.
    """
    expr_lower = expr.lower()
    now = datetime.now()

    # yesterday, today, tomorrow
    if expr_lower == "yesterday":
        yesterday = now - timedelta(days=1)
        return {
            "before": yesterday.strftime("%Y-%m-%d"),
            "after": yesterday.strftime("%Y-%m-%d"),
        }
    elif expr_lower == "today":
        return {"before": now.strftime("%Y-%m-%d"), "after": now.strftime("%Y-%m-%d")}
    elif expr_lower == "tomorrow":
        tomorrow = now + timedelta(days=1)
        return {
            "before": tomorrow.strftime("%Y-%m-%d"),
            "after": tomorrow.strftime("%Y-%m-%d"),
        }

    # last week, this week, next week
    elif expr_lower == "last week":
        start = now - timedelta(days=now.weekday() + 7)
        end = start + timedelta(days=6)
        return {"before": end.strftime("%Y-%m-%d"), "after": start.strftime("%Y-%m-%d")}
    elif expr_lower == "this week":
        start = now - timedelta(days=now.weekday())
        end = start + timedelta(days=6)
        return {"before": end.strftime("%Y-%m-%d"), "after": start.strftime("%Y-%m-%d")}
    elif expr_lower == "next week":
        start = now - timedelta(days=now.weekday()) + timedelta(days=7)
        end = start + timedelta(days=6)
        return {"before": end.strftime("%Y-%m-%d"), "after": start.strftime("%Y-%m-%d")}

    # last month, this month, next month
    elif expr_lower == "last month":
        first_day_last_month = (now.replace(day=1) - timedelta(days=1)).replace(day=1)
        last_day_last_month = now.replace(day=1) - timedelta(days=1)
        return {
            "before": last_day_last_month.strftime("%Y-%m-%d"),
            "after": first_day_last_month.strftime("%Y-%m-%d"),
        }
    elif expr_lower == "this month":
        first_day_this_month = now.replace(day=1)
        next_month = now.replace(day=28) + timedelta(
            days=4
        )  # ensure we get to next month
        last_day_this_month = next_month.replace(day=1) - timedelta(days=1)
        return {
            "before": last_day_this_month.strftime("%Y-%m-%d"),
            "after": first_day_this_month.strftime("%Y-%m-%d"),
        }
    elif expr_lower == "next month":
        first_day_next_month = (now.replace(day=28) + timedelta(days=4)).replace(day=1)
        next_next_month = (
            first_day_next_month.replace(day=28) + timedelta(days=4)
        ).replace(day=1)
        last_day_next_month = next_next_month - timedelta(days=1)
        return {
            "before": last_day_next_month.strftime("%Y-%m-%d"),
            "after": first_day_next_month.strftime("%Y-%m-%d"),
        }

    # past N days
    match = re.match(r"past\s+(\d+)\s+days?", expr_lower)
    if match:
        days = int(match.group(1))
        return {"time_range_days": days}

    # since YYYY-MM
    match = re.match(r"since\s+(\d{4}-\d{2})", expr_lower)
    if match:
        year_month = match.group(1)
        # Assume first day of month
        after_date = f"{year_month}-01"
        return {"after": after_date}

    # last year, this year, next year
    elif expr_lower == "last year":
        last_year = now.year - 1
        return {"after": f"{last_year}-01-01", "before": f"{last_year}-12-31"}
    elif expr_lower == "this year":
        this_year = now.year
        return {"after": f"{this_year}-01-01", "before": f"{this_year}-12-31"}
    elif expr_lower == "next year":
        next_year = now.year + 1
        return {"after": f"{next_year}-01-01", "before": f"{next_year}-12-31"}

    return None


@dataclass
class ParsedQuery:
    """Parsed search query with filters."""

    text: str  # The actual search text
    filters: Dict[str, Any]

    def __str__(self):
        return f"text='{self.text}', filters={self.filters}"


def parse_query(query: str) -> ParsedQuery:
    """
    Parse a search query containing filters.

    Supported filters:
        before:YYYY-MM-DD    - memories created before date
        after:YYYY-MM-DD     - memories created after date
        time_range:Nd        - memories within last N days
        source:value         - filter by source
        memory_type:value    - filter by memory type
        epistemic:value      - filter by epistemic status

    Natural language temporal expressions:
        "last week", "past 30 days", "since March", "yesterday", "this month", etc.
        (converted to before/after/time_range_days filters)

    Example:
        "Python before:2026-03-20 source:user" → text="Python", filters={before:..., source:...}
        "Python last week" → text="Python", filters={before:..., after:...}
    """
    filters = {}
    text_parts = []

    # First pass: extract relative date expressions (multi-word)
    # We'll process the query as a whole, removing matched patterns
    remaining_text = query

    # Pattern for relative date expressions (multi-word)
    # We'll use _parse_relative_date_expr on each word and also try consecutive pairs
    words = query.split()
    i = 0
    while i < len(words):
        word = words[i]
        matched = False

        # Check colon filters
        filter_patterns = [
            (r"before:(\d{4}-\d{2}-\d{2})", "before"),
            (r"after:(\d{4}-\d{2}-\d{2})", "after"),
            (r"time_range:(\d+)d", "time_range_days"),
            (r"source:([^\s]+)", "source"),
            (r"memory_type:([^\s]+)", "memory_type"),
            (r"epistemic:([^\s]+)", "epistemic_status"),
            (r"related_to:([^\s]+)", "related_to"),
        ]
        for pattern, filter_key in filter_patterns:
            match = re.match(pattern, word, re.IGNORECASE)
            if match:
                value = match.group(1)
                if filter_key == "time_range_days":
                    filters["time_range_days"] = int(value)
                else:
                    filters[filter_key] = value
                matched = True
                break

        # If not a colon filter, check for relative date expression
        if not matched:
            # Try single word
            rel_filter = _parse_relative_date_expr(word)
            if rel_filter:
                # Merge into filters (overwrite if conflict)
                filters.update(rel_filter)
                matched = True
            else:
                # Try two-word phrase with next word
                if i + 1 < len(words):
                    two_words = f"{word} {words[i + 1]}"
                    rel_filter = _parse_relative_date_expr(two_words)
                    if rel_filter:
                        filters.update(rel_filter)
                        matched = True
                        i += 1  # skip next word
                # Try three-word phrase (e.g., "past 30 days")
                if not matched and i + 2 < len(words):
                    three_words = f"{word} {words[i + 1]} {words[i + 2]}"
                    rel_filter = _parse_relative_date_expr(three_words)
                    if rel_filter:
                        filters.update(rel_filter)
                        matched = True
                        i += 2  # skip two words

        if not matched:
            text_parts.append(word)

        i += 1

    text = " ".join(text_parts)
    return ParsedQuery(text=text, filters=filters)


# -----------------------------------------------------------------------------
# Unified Memory Search
# -----------------------------------------------------------------------------


class UnifiedMemorySearch:
    """Unified interface for vector and BM25 memory search."""

    # Recency weighting parameters
    RECENCY_HALFLIFE_DAYS = 30  # memories older than this get half weight
    RECENCY_WEIGHT = (
        0.3  # weight of recency boost (0.0 = ignore recency, 1.0 = only recency)
    )

    def __init__(self, vector_store, bm25_store):
        """
        Initialize unified searcher.

        Args:
            vector_store: LanceDB table object
            bm25_store: BM25MemoryStore instance
        """
        self.vector_store = vector_store
        self.bm25_store = bm25_store

    def _apply_recency_boost(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply recency boost to search results."""
        if not results:
            return results

        now = time.time()
        boosted = []
        for res in results:
            created_at = res.get("created_at", 0)
            if not created_at:
                boosted.append(res)
                continue

            # Age in days
            age_days = (now - created_at) / (24 * 3600)
            # Exponential decay boost
            boost = math.exp(-age_days / self.RECENCY_HALFLIFE_DAYS)
            # Weighted adjustment
            adjusted_boost = 1.0 - self.RECENCY_WEIGHT + self.RECENCY_WEIGHT * boost

            # Apply to similarity (vector) or score (BM25)
            if "similarity" in res:
                res = res.copy()
                res["similarity"] = res["similarity"] * adjusted_boost
                res["recency_boost"] = adjusted_boost
            elif "score" in res:
                res = res.copy()
                res["score"] = res["score"] * adjusted_boost
                res["recency_boost"] = adjusted_boost

            boosted.append(res)

        return boosted

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 10,
        alpha: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Perform memory search.

        Args:
            query: Search query (may include filters)
            mode: "vector", "bm25", or "hybrid"
            top_k: Maximum number of results
            alpha: Weight for vector results in hybrid search (0.0-1.0)
            **kwargs: Additional search parameters

        Returns:
            JSON string with search results
        """
        # Parse query
        parsed = parse_query(query)
        logger.debug(f"Parsed query: {parsed}")

        # Apply filters to results (post-filtering for simplicity)
        # In production, would push filters to database queries

        # Perform search based on mode
        if mode == "vector":
            results = self._vector_search(parsed, top_k)
        elif mode == "bm25":
            results = self._bm25_search(parsed, top_k)
        elif mode == "hybrid":
            results = self._hybrid_search(parsed, top_k, alpha)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Apply temporal filters
        if (
            "before" in parsed.filters
            or "after" in parsed.filters
            or "time_range_days" in parsed.filters
        ):
            results = self._apply_temporal_filters(results, parsed.filters)

        # Apply other filters (source, memory_type, epistemic_status)
        results = self._apply_other_filters(results, parsed.filters)

        # Format results
        formatted = self._format_results(results, mode)
        return json.dumps(formatted, indent=2)

    def _vector_search(self, parsed: ParsedQuery, top_k: int) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""
        if not self.vector_store:
            return []

        try:
            # Get embedding for query text
            from .contradiction_detector import get_embedding

            embedding = get_embedding(parsed.text)
            if not embedding:
                return []

            # Search vector store
            results = self.vector_store.search(embedding).limit(top_k).to_pandas()

            # Convert to list of dicts
            vector_results = []
            for _, row in results.iterrows():
                # Compute similarity from distance
                similarity = 1.0
                if "_distance" in row:
                    similarity = 1.0 - float(row["_distance"])
                # Handle NaN values
                import math

                epistemic = row.get("epistemic_status", "stated")
                if epistemic is None or (
                    isinstance(epistemic, float) and math.isnan(epistemic)
                ):
                    epistemic = "stated"
                confidence = row.get("confidence", 0.5)
                if confidence is None or (
                    isinstance(confidence, float) and math.isnan(confidence)
                ):
                    confidence = 0.5
                vector_results.append(
                    {
                        "memory_id": row.get("id", ""),
                        "text": row.get("text", ""),
                        "similarity": similarity,
                        "source": row.get("source", ""),
                        "memory_type": row.get("memory_type", ""),
                        "epistemic_status": epistemic,
                        "confidence": float(confidence),
                        "created_at": row.get("created_at", 0),
                    }
                )

            vector_results = self._apply_recency_boost(vector_results)
            return vector_results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _bm25_search(self, parsed: ParsedQuery, top_k: int) -> List[Dict[str, Any]]:
        """Perform BM25 keyword search."""
        if not self.bm25_store:
            return []

        try:
            bm25_results = self.bm25_store.search(parsed.text, top_k=top_k)

            # Convert BM25Result to dict format
            results = []
            for result in bm25_results:
                results.append(
                    {
                        "memory_id": result.memory_id,
                        "text": result.text,
                        "score": result.score,
                        "source": result.source,
                        "memory_type": result.memory_type,
                        "epistemic_status": "stated",  # BM25 store doesn't track epistemic status
                        "confidence": 0.5,
                        "created_at": 0,
                    }
                )

            results = self._apply_recency_boost(results)
            return results
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def _hybrid_search(
        self,
        parsed: ParsedQuery,
        top_k: int,
        alpha: float,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search using RRF fusion."""
        vector_results = self._vector_search(parsed, top_k * 2)  # Get more for fusion
        bm25_results = self._bm25_search(parsed, top_k * 2)

        # Convert to format expected by RRF fusion
        vector_for_fusion = [
            {"memory_id": r["memory_id"], "similarity": r.get("similarity", 0.5)}
            for r in vector_results
        ]

        bm25_for_fusion = []
        for result in bm25_results:
            # Convert BM25 dict to BM25Result-like object
            from .bm25_memory import BM25Result

            bm25_obj = BM25Result(
                memory_id=result["memory_id"],
                text=result["text"],
                score=result["score"],
                source=result["source"],
                memory_type=result["memory_type"],
            )
            bm25_for_fusion.append(bm25_obj)

        # Apply RRF fusion
        from .bm25_memory import rrf_fusion

        fused = rrf_fusion(vector_for_fusion, bm25_for_fusion, alpha=alpha, top_k=top_k)

        # Merge back full result data
        merged_results = []
        for fused_item in fused:
            memory_id = fused_item["memory_id"]

            # Find full data from either vector or BM25 results
            full_data = None
            for vec in vector_results:
                if vec["memory_id"] == memory_id:
                    full_data = vec
                    break

            if not full_data:
                for bm25 in bm25_results:
                    if bm25["memory_id"] == memory_id:
                        full_data = bm25
                        break

            if full_data:
                full_data["rrf_score"] = fused_item["rrf_score"]
                full_data["vector_rank"] = fused_item["vector_rank"]
                full_data["bm25_rank"] = fused_item["bm25_rank"]
                merged_results.append(full_data)

        return merged_results

    def _apply_temporal_filters(
        self,
        results: List[Dict[str, Any]],
        filters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Filter results based on temporal constraints."""
        filtered = []

        for result in results:
            created_at = result.get("created_at", 0)
            if not created_at:
                continue

            # Convert to datetime if needed
            if isinstance(created_at, (int, float)):
                dt = datetime.fromtimestamp(created_at)
            else:
                try:
                    dt = datetime.fromisoformat(str(created_at))
                except:
                    continue

            include = True

            # before filter
            if "before" in filters:
                try:
                    before_date = datetime.fromisoformat(filters["before"])
                    if dt >= before_date:
                        include = False
                except:
                    pass

            # after filter
            if "after" in filters:
                try:
                    after_date = datetime.fromisoformat(filters["after"])
                    if dt <= after_date:
                        include = False
                except:
                    pass

            # time_range_days filter
            if "time_range_days" in filters:
                days = filters["time_range_days"]
                cutoff = datetime.now() - timedelta(days=days)
                if dt < cutoff:
                    include = False

            if include:
                filtered.append(result)

        return filtered

    def _apply_other_filters(
        self,
        results: List[Dict[str, Any]],
        filters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Filter results based on source, memory_type, epistemic_status, related_to."""
        filtered = []

        # Precompute related memory IDs if related_to filter present
        related_ids = None
        if "related_to" in filters:
            try:
                from .relationship_extractor import find_related_memories

                related_ids = set(
                    r[0] for r in find_related_memories(filters["related_to"])
                )
            except ImportError:
                # Relationship extractor not available, ignore filter
                related_ids = None

        for result in results:
            include = True

            # source filter
            if "source" in filters and result.get("source") != filters["source"]:
                include = False

            # memory_type filter
            if (
                "memory_type" in filters
                and result.get("memory_type") != filters["memory_type"]
            ):
                include = False

            # epistemic_status filter
            if (
                "epistemic_status" in filters
                and result.get("epistemic_status") != filters["epistemic_status"]
            ):
                include = False

            # related_to filter
            if related_ids is not None and result.get("memory_id") not in related_ids:
                include = False

            if include:
                filtered.append(result)

        return filtered

    def _format_results(
        self,
        results: List[Dict[str, Any]],
        mode: str,
    ) -> Dict[str, Any]:
        """Format results for JSON output."""
        return {
            "mode": mode,
            "count": len(results),
            "results": results,
        }


# -----------------------------------------------------------------------------
# Tool integration
# -----------------------------------------------------------------------------


def unified_memory_search_tool(args: Dict[str, Any], **kwargs) -> str:
    """
    Tool handler for unified memory search.

    Expected args:
        query: str - Search query
        mode: str (optional) - "vector", "bm25", or "hybrid" (default: "hybrid")
        top_k: int (optional) - Maximum results (default: 10)
        alpha: float (optional) - Vector weight for hybrid (default: 0.7)
    """
    try:
        import lancedb
        from .bm25_memory import BM25MemoryStore

        # Get parameters
        query = args.get("query", "")
        if not query:
            return json.dumps({"error": "Query is required", "results": []})

        mode = args.get("mode", "hybrid")
        top_k = args.get("top_k", 10)
        alpha = args.get("alpha", 0.7)

        # Initialize stores
        db_path = Path.home() / ".hermes" / "vector_memory"
        db = lancedb.connect(str(db_path))

        # Check if table exists
        tables = db.list_tables()
        # Handle both ListTablesResponse object and plain list
        if hasattr(tables, "tables"):
            actual_tables = tables.tables
        else:
            actual_tables = tables
        if "memory_vectors" not in actual_tables:
            return json.dumps(
                {
                    "error": "Vector memory table not found",
                    "results": [],
                    "suggestion": "Add memories first using memory tool",
                }
            )

        vector_store = db.open_table("memory_vectors")
        bm25_store = BM25MemoryStore()

        # Perform search
        searcher = UnifiedMemorySearch(vector_store, bm25_store)
        results_json = searcher.search(
            query=query,
            mode=mode,
            top_k=top_k,
            alpha=alpha,
        )

        return results_json

    except ImportError as e:
        return json.dumps({"error": f"Required module missing: {e}", "results": []})
    except Exception as e:
        logger.error(f"Memory search failed: {e}")
        return json.dumps({"error": f"Search failed: {str(e)}", "results": []})


def check_unified_search_requirements() -> bool:
    """Check if requirements for unified search are met."""
    try:
        import lancedb
        from .bm25_memory import BM25MemoryStore

        return True
    except ImportError:
        return False


# Schema for tool registration
UNIFIED_SEARCH_SCHEMA = {
    "name": "memory_search",
    "description": (
        "Search memories using vector similarity, keyword search, or hybrid approach. "
        "Supports filters: before:YYYY-MM-DD, after:YYYY-MM-DD, time_range:Nd, "
        "source:value, memory_type:value, epistemic:value"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (text + optional filters)",
            },
            "mode": {
                "type": "string",
                "enum": ["vector", "bm25", "hybrid"],
                "description": "Search mode (default: hybrid)",
                "default": "hybrid",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of results (default: 10)",
                "default": 10,
            },
            "alpha": {
                "type": "number",
                "description": "Weight for vector results in hybrid search (0.0-1.0, default: 0.7)",
                "default": 0.7,
            },
        },
        "required": ["query"],
    },
}

# Register the tool
registry.register(
    name="memory_search",
    toolset="memory",
    schema=UNIFIED_SEARCH_SCHEMA,
    handler=unified_memory_search_tool,
    check_fn=check_unified_search_requirements,
    emoji="🔍",
)

# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unified memory search test")
    parser.add_argument("query", help="Search query")
    parser.add_argument(
        "--mode", choices=["vector", "bm25", "hybrid"], default="hybrid"
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.7)

    args = parser.parse_args()

    # Note: This test requires vector and BM25 stores to be initialized
    print("Unified search test - requires initialized stores")
    print(f"Query: {args.query}")
    print(f"Mode: {args.mode}")
    print(f"Top K: {args.top_k}")
    print(f"Alpha: {args.alpha}")

    # Try to initialize stores
    try:
        import lancedb
        from .bm25_memory import BM25MemoryStore

        db_path = Path.home() / ".hermes" / "vector_memory"
        db = lancedb.connect(db_path)
        vector_store = db.open_table("memory_vectors")

        bm25_store = BM25MemoryStore()

        searcher = UnifiedMemorySearch(vector_store, bm25_store)
        results_json = searcher.search(
            query=args.query,
            mode=args.mode,
            top_k=args.top_k,
            alpha=args.alpha,
        )

        print("\nResults:")
        print(results_json)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
