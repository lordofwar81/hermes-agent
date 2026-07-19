"""Cross-encoder reranker for holographic prefetch (Stage 2 precision).

SOTA pipeline: Stage 1 RRF fusion (recall, broad) -> Stage 2 cross-encoder
(precision, deep query-doc scoring). The cross-encoder scores genuine relevance,
not term overlap, so a long multi-term fact that merely mentions the query no
longer dominates top-k over a short fact that actually answers it.

Lazy-loaded + cached at module level (one model instance across all prefetch
calls). CPU-only torch (no CUDA dep on this box). Model: ms-marco-MiniLM-L-6-v2
(~80MB, already in HF cache). Inference: ~30ms for 20 candidates.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None
_lock = threading.Lock()
_load_attempted = False
_disabled = False  # set True if deps missing, to avoid re-trying every call


def _load():
    """Lazily load the cross-encoder. Returns (model, tokenizer) or (None, None)."""
    global _model, _tokenizer, _load_attempted, _disabled
    if _disabled:
        return None, None
    if _model is not None:
        return _model, _tokenizer
    with _lock:
        if _model is not None:
            return _model, _tokenizer
        if _load_attempted:
            return None, None
        _load_attempted = True
        try:
            import torch  # noqa: F401 — transformers needs it importable
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            _tokenizer = AutoTokenizer.from_pretrained(model_id)
            _model = AutoModelForSequenceClassification.from_pretrained(model_id)
            _model.eval()
            logger.info("cross-encoder reranker loaded: %s", model_id)
        except Exception as e:
            logger.warning("cross-encoder unavailable, prefetch will use RRF only: %s", e)
            _disabled = True
            return None, None
    return _model, _tokenizer


def is_available() -> bool:
    """Cheap check: has the model been loaded (or is it loadable)?"""
    if _model is not None:
        return True
    if _disabled:
        return False
    _load()
    return _model is not None


def rerank(query: str, documents: list[str], top_k: int = 5) -> list[tuple[int, float]]:
    """Rerank documents by relevance to query. Returns [(orig_index, score), ...].

    Uses the cross-encoder single-logit relevance score (higher = more relevant).
    If the model is unavailable, returns documents in original order (graceful
    fallback — callers should treat reranking as an enhancement, not a hard dep).
    """
    if not documents:
        return []
    model, tokenizer = _load()
    if model is None:
        return [(i, 0.0) for i in range(min(top_k, len(documents)))]

    try:
        import torch
        pairs = [(query, doc) for doc in documents]
        feats = tokenizer(pairs, padding=True, truncation=True,
                          return_tensors="pt", max_length=256)
        with torch.no_grad():
            logits = model(**feats).logits.squeeze(-1)
        scores = logits.tolist()
        # Sort by score descending, return (orig_index, score) for top_k
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:top_k]
        return ranked
    except Exception as e:
        logger.debug("rerank failed, falling back to RRF order: %s", e)
        return [(i, 0.0) for i in range(min(top_k, len(documents)))]
