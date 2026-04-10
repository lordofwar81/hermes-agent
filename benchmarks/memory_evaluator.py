#!/usr/bin/env python3
"""
Memory System Evaluator - LOCOMO‑style benchmark for temporal reasoning,
relationship extraction, and hybrid search.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.vector_memory import VectorMemoryStore
from tools.bm25_memory import BM25MemoryStore
from tools.unified_memory_search import UnifiedMemorySearch

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Test runner
# -----------------------------------------------------------------------------


class MemoryEvaluator:
    """Run benchmark test cases against memory system."""

    def __init__(self, use_temp_store: bool = True):
        """
        Initialize evaluator.

        Args:
            use_temp_store: If True, use temporary LanceDB directory (isolated).
        """
        self.use_temp_store = use_temp_store
        self.temp_dir = None
        self.vector_store = None
        self.bm25_store = None
        self.searcher = None

    def setup(self):
        """Initialize stores."""
        if self.use_temp_store:
            self.temp_dir = tempfile.mkdtemp(prefix="hermes_bench_")
            db_path = Path(self.temp_dir) / "vector_memory"
            logger.info(f"Using temporary store at {db_path}")
            self.vector_store = VectorMemoryStore(db_path=db_path)
        else:
            logger.warning(
                "Using production vector memory store (may affect real data)"
            )
            self.vector_store = VectorMemoryStore()

        self.bm25_store = BM25MemoryStore()
        self.searcher = UnifiedMemorySearch(self.vector_store.table, self.bm25_store)

    def teardown(self):
        """Clean up temporary resources."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            logger.debug(f"Removed temporary directory {self.temp_dir}")

    def add_memory(self, memory_spec: Dict[str, Any]) -> Optional[str]:
        """Add a test memory with optional created_at offset."""
        # Adjust created_at if offset specified
        created_at = time.time()
        if "created_at_offset_days" in memory_spec:
            offset = memory_spec["created_at_offset_days"]
            created_at += offset * 24 * 3600

        # Temporarily monkey-patch get_embedding to avoid API calls?
        # For now rely on real embedding (requires Ollama endpoint).
        # We'll skip if embedding fails.
        try:
            memory_id = self.vector_store.add_memory(
                text=memory_spec["text"],
                source=memory_spec.get("source", "user"),
                memory_type=memory_spec.get("memory_type", "fact"),
                session_id="benchmark",
                epistemic_status=memory_spec.get("epistemic_status", "stated"),
                confidence=memory_spec.get("confidence", 0.5),
            )
            if memory_id and "created_at_offset_days" in memory_spec:
                # Override created_at in LanceDB (hack: not directly supported)
                # We'll skip for now; temporal filters rely on real created_at.
                pass
            return memory_id
        except Exception as e:
            logger.warning(f"Failed to add memory {memory_spec.get('id')}: {e}")
            return None

    def run_query(self, query_spec: Dict[str, Any]) -> List[str]:
        """Run a query and return list of returned memory IDs."""
        try:
            results_json = self.searcher.search(
                query=query_spec["query"],
                mode=query_spec.get("mode", "hybrid"),
                top_k=query_spec.get("top_k", 10),
                alpha=query_spec.get("alpha", 0.7),
            )
            results = json.loads(results_json)
            return [r["memory_id"] for r in results["results"]]
        except Exception as e:
            logger.warning(f"Query failed: {e}")
            return []

    def evaluate_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case and compute metrics."""
        case_id = test_case["id"]
        logger.info(f"Running test case: {case_id}")

        # Add memories
        memory_id_map = {}  # test_id -> real_id
        for mem_spec in test_case["memories"]:
            test_id = mem_spec["id"]
            real_id = self.add_memory(mem_spec)
            if real_id:
                memory_id_map[test_id] = real_id
            else:
                logger.warning(f"Memory {test_id} failed to add")

        # Wait a bit for indexing (if any)
        time.sleep(0.5)

        # Run queries
        query_results = []
        for query_spec in test_case.get("queries", []):
            # Replace test IDs with real IDs in expected list
            expected_test_ids = query_spec.get("expected_memory_ids", [])
            expected_real_ids = [
                memory_id_map[tid] for tid in expected_test_ids if tid in memory_id_map
            ]

            retrieved_ids = self.run_query(query_spec)

            # Compute precision, recall, F1
            relevant_retrieved = set(retrieved_ids) & set(expected_real_ids)
            precision = (
                len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0.0
            )
            recall = (
                len(relevant_retrieved) / len(expected_real_ids)
                if expected_real_ids
                else 0.0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            query_results.append(
                {
                    "query": query_spec["query"],
                    "expected": expected_real_ids,
                    "retrieved": retrieved_ids,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )

        # Clean up memories? Not needed for temp store.

        return {
            "case_id": case_id,
            "description": test_case["description"],
            "queries": query_results,
            "summary": {
                "avg_precision": sum(q["precision"] for q in query_results)
                / len(query_results)
                if query_results
                else 0.0,
                "avg_recall": sum(q["recall"] for q in query_results)
                / len(query_results)
                if query_results
                else 0.0,
                "avg_f1": sum(q["f1"] for q in query_results) / len(query_results)
                if query_results
                else 0.0,
            },
        }

    def run_benchmark(self, test_set_path: Path) -> Dict[str, Any]:
        """Run entire benchmark suite."""
        with open(test_set_path, "r") as f:
            test_set = json.load(f)

        self.setup()
        results = []
        try:
            for test_case in test_set["test_cases"]:
                result = self.evaluate_test_case(test_case)
                results.append(result)
        finally:
            self.teardown()

        # Aggregate scores
        total_queries = sum(len(r["queries"]) for r in results)
        avg_precision = (
            sum(q["precision"] for r in results for q in r["queries"]) / total_queries
            if total_queries
            else 0.0
        )
        avg_recall = (
            sum(q["recall"] for r in results for q in r["queries"]) / total_queries
            if total_queries
            else 0.0
        )
        avg_f1 = (
            sum(q["f1"] for r in results for q in r["queries"]) / total_queries
            if total_queries
            else 0.0
        )

        return {
            "name": test_set["name"],
            "description": test_set["description"],
            "results": results,
            "overall": {
                "avg_precision": avg_precision,
                "avg_recall": avg_recall,
                "avg_f1": avg_f1,
                "total_test_cases": len(results),
                "total_queries": total_queries,
            },
        }


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run memory system benchmarks")
    parser.add_argument(
        "test_set",
        nargs="?",
        default="memory_test_set.json",
        help="Path to test set JSON",
    )
    parser.add_argument("--output", "-o", help="Output results JSON file")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Use production memory store (dangerous)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    test_set_path = Path(__file__).parent / args.test_set
    if not test_set_path.exists():
        logger.error(f"Test set not found: {test_set_path}")
        sys.exit(1)

    evaluator = MemoryEvaluator(use_temp_store=not args.production)
    results = evaluator.run_benchmark(test_set_path)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Benchmark: {results['name']}")
    print(f"Description: {results['description']}")
    print("=" * 60)
    for case in results["results"]:
        print(f"\nTest Case: {case['case_id']}")
        for q in case["queries"]:
            print(f"  Query: '{q['query']}'")
            print(
                f"    Precision: {q['precision']:.3f}, Recall: {q['recall']:.3f}, F1: {q['f1']:.3f}"
            )
    print("\n" + "=" * 60)
    overall = results["overall"]
    print(
        f"Overall: Precision={overall['avg_precision']:.3f}, Recall={overall['avg_recall']:.3f}, F1={overall['avg_f1']:.3f}"
    )
    print(
        f"Total test cases: {overall['total_test_cases']}, Total queries: {overall['total_queries']}"
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results saved to {args.output}")
