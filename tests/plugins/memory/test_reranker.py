"""Tests for the holographic cross-encoder reranker (Stage 2 precision).

The reranker is an enhancement layer with graceful fallback: when the model is
unavailable it must return documents in original order so prefetch degrades to
RRF-only. These tests pin that contract without requiring the (80MB) model to be
downloaded — they force the disabled path and assert the fallback shape.
"""
from __future__ import annotations

import importlib

import pytest

reranker = importlib.import_module("plugins.memory.holographic.reranker")


@pytest.fixture(autouse=True)
def _reset_state():
    """Force the reranker into the disabled state before each test, then reset."""
    saved = (reranker._model, reranker._disabled, reranker._load_attempted)
    reranker._model = None
    reranker._disabled = True
    reranker._load_attempted = True
    yield
    reranker._model, reranker._disabled, reranker._load_attempted = saved


class TestRerankFallback:
    def test_empty_documents_returns_empty(self):
        assert reranker.rerank("query", [], top_k=5) == []

    def test_fewer_docs_than_top_k_returns_all(self):
        docs = ["alpha", "beta"]
        result = reranker.rerank("q", docs, top_k=5)
        assert len(result) == 2
        # Fallback preserves original order: (index, 0.0)
        assert result == [(0, 0.0), (1, 0.0)]

    def test_top_k_truncates(self):
        docs = [f"doc {i}" for i in range(10)]
        result = reranker.rerank("q", docs, top_k=3)
        assert len(result) == 3
        assert result == [(0, 0.0), (1, 0.0), (2, 0.0)]

    def test_fallback_scores_are_zero(self):
        result = reranker.rerank("q", ["a", "b"], top_k=2)
        assert all(score == 0.0 for _, score in result)

    def test_indices_are_original_positions(self):
        """Fallback must return ORIGINAL indices, not re-ranked ones, so the
        caller can map back to the fused-list positions correctly."""
        docs = ["first", "second", "third"]
        result = reranker.rerank("q", docs, top_k=3)
        assert [idx for idx, _ in result] == [0, 1, 2]


class TestIsAvailable:
    def test_returns_false_when_disabled(self):
        assert reranker.is_available() is False

    def test_returns_true_when_model_loaded(self):
        reranker._model = object()  # non-None sentinel
        reranker._disabled = False
        assert reranker.is_available() is True
