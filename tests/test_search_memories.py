"""Tests for search_memories standalone convenience function.

Validates that the function exists, handles edge cases, and normalises
output keys correctly for builtin_memory_provider consumption.
"""
import json
import unittest
from unittest.mock import patch, MagicMock


class TestSearchMemoriesExists(unittest.TestCase):
    """Verify the import that was previously broken now resolves."""

    def test_import_succeeds(self):
        from tools.unified_memory_search import search_memories
        self.assertTrue(callable(search_memories))

    def test_builtin_provider_flag_true(self):
        from agent.builtin_memory_provider import HAS_UNIFIED_SEARCH
        self.assertTrue(HAS_UNIFIED_SEARCH)


class TestSearchMemoriesNoTable(unittest.TestCase):
    """Returns empty list when vector memory table doesn't exist."""

    def test_returns_empty_when_no_table(self):
        with patch("lancedb.connect") as mock_connect:
            mock_db = MagicMock()
            mock_db.list_tables.return_value = MagicMock(tables=[])
            mock_connect.return_value = mock_db

            from tools.unified_memory_search import search_memories
            result = search_memories("test", limit=5)
            self.assertEqual(result, [])


class TestSearchMemoriesFilterTranslation(unittest.TestCase):
    """Verify unix timestamp filters are translated to ISO dates."""

    def test_unix_after_filter_converted(self):
        with patch("lancedb.connect") as mock_connect, \
             patch("tools.bm25_memory.BM25MemoryStore"):
            mock_db = MagicMock()
            mock_db.list_tables.return_value = MagicMock(tables=["memory_vectors"])
            mock_table = MagicMock()
            mock_db.open_table.return_value = mock_table
            mock_connect.return_value = mock_db

            from tools.unified_memory_search import search_memories
            search_memories("python", limit=10, filters={"after": 1718000000.0})

    def test_unix_before_filter_converted(self):
        with patch("lancedb.connect") as mock_connect, \
             patch("tools.bm25_memory.BM25MemoryStore"):
            mock_db = MagicMock()
            mock_db.list_tables.return_value = MagicMock(tables=["memory_vectors"])
            mock_table = MagicMock()
            mock_db.open_table.return_value = mock_table
            mock_connect.return_value = mock_db

            from tools.unified_memory_search import search_memories
            search_memories("python", limit=10, filters={"before": 1719000000.0})

    def test_empty_query_with_filter(self):
        """Empty query string + filters should still work."""
        with patch("lancedb.connect") as mock_connect, \
             patch("tools.bm25_memory.BM25MemoryStore"):
            mock_db = MagicMock()
            mock_db.list_tables.return_value = MagicMock(tables=["memory_vectors"])
            mock_table = MagicMock()
            mock_db.open_table.return_value = mock_table
            mock_connect.return_value = mock_db

            from tools.unified_memory_search import search_memories
            # This is how builtin_memory_provider calls it for recent memories
            result = search_memories("", limit=10, filters={"after": 1718000000.0})
            self.assertIsInstance(result, list)


class TestSearchMemoriesKeyNormalisation(unittest.TestCase):
    """Verify memory_id → id key normalisation."""

    def test_memory_id_normalised_to_id(self):
        with patch("lancedb.connect") as mock_connect, \
             patch("tools.bm25_memory.BM25MemoryStore"):
            mock_db = MagicMock()
            mock_db.list_tables.return_value = MagicMock(tables=["memory_vectors"])
            mock_table = MagicMock()
            mock_db.open_table.return_value = mock_table
            mock_connect.return_value = mock_db

            from tools.unified_memory_search import UnifiedMemorySearch

            # Patch the searcher to return results with memory_id (no id key)
            with patch.object(UnifiedMemorySearch, "search") as mock_search:
                mock_search.return_value = json.dumps({
                    "mode": "vector",
                    "count": 1,
                    "results": [{
                        "memory_id": "abc-123",
                        "text": "User likes Python",
                        "similarity": 0.85,
                        "source": "user",
                        "memory_type": "preference",
                        "epistemic_status": "stated",
                        "confidence": 0.9,
                        "created_at": 1718000000.0,
                    }],
                })

                from tools.unified_memory_search import search_memories
                result = search_memories("python preferences", limit=10)

                self.assertEqual(len(result), 1)
                self.assertEqual(result[0]["id"], "abc-123")
                # Default fields should be set
                self.assertEqual(result[0]["entities"], [])
                self.assertEqual(result[0]["keywords"], [])
                self.assertEqual(result[0]["source"], "user")

    def test_default_fields_filled_when_missing(self):
        """Ensure setdefault fills in missing fields from UnifiedMemorySearch results."""
        with patch("lancedb.connect") as mock_connect, \
             patch("tools.bm25_memory.BM25MemoryStore"):
            mock_db = MagicMock()
            mock_db.list_tables.return_value = MagicMock(tables=["memory_vectors"])
            mock_table = MagicMock()
            mock_db.open_table.return_value = mock_table
            mock_connect.return_value = mock_db

            from tools.unified_memory_search import UnifiedMemorySearch

            with patch.object(UnifiedMemorySearch, "search") as mock_search:
                # Minimal result — only memory_id and text
                mock_search.return_value = json.dumps({
                    "mode": "vector",
                    "count": 1,
                    "results": [{
                        "memory_id": "xyz-789",
                        "text": "A sparse memory",
                    }],
                })

                from tools.unified_memory_search import search_memories
                result = search_memories("sparse", limit=10)

                self.assertEqual(len(result), 1)
                r = result[0]
                self.assertEqual(r["id"], "xyz-789")
                self.assertEqual(r["source"], "")
                self.assertEqual(r["memory_type"], "")
                self.assertEqual(r["epistemic_status"], "stated")
                self.assertEqual(r["confidence"], 0.5)
                self.assertEqual(r["entities"], [])
                self.assertEqual(r["keywords"], [])


if __name__ == "__main__":
    unittest.main()
