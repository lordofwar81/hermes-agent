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
_last_fail_ts: float = 0.0  # timestamp of last load failure (0 = never failed)
_RETRY_COOLDOWN_SECONDS: float = 600.0  # 10 min — retry load after this long (audit H-1)

# _disabled removed: a permanent disable flag meant a transient failure (missing
# transformers, model download timeout, OOM) stuck the process on RRF-only ranking
# forever. Now _last_fail_ts + cooldown gates retries so recovery is automatic.


def _load():
    """Lazily load the cross-encoder. Returns (model, tokenizer) or (None, None).

    Uses a cooldown instead of permanent disable: after a load failure, retries
    are suppressed for _RETRY_COOLDOWN_SECONDS, then allowed again. This lets a
    long-lived process recover when transformers is installed, the model finishes
    downloading, or a transient network/OOM condition clears (audit H-1).
    """
    global _model, _tokenizer, _load_attempted, _last_fail_ts
    import time as _time
    if _model is not None:
        return _model, _tokenizer
    # Cooldown gate: if we failed recently, don't retry yet.
    if _last_fail_ts and (_time.time() - _last_fail_ts) < _RETRY_COOLDOWN_SECONDS:
        return None, None
    with _lock:
        if _model is not None:
            return _model, _tokenizer
        _load_attempted = True
        try:
            import torch  # noqa: F401 — transformers needs it importable
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            _tokenizer = AutoTokenizer.from_pretrained(model_id)
            _model = AutoModelForSequenceClassification.from_pretrained(model_id)
            _model.eval()
            _last_fail_ts = 0.0  # success — clear the cooldown
            logger.info("cross-encoder reranker loaded: %s", model_id)
        except Exception as e:
            _last_fail_ts = _time.time()
            logger.warning(
                "cross-encoder unavailable, prefetch will use RRF only "
                "(retry in %ds): %s", int(_RETRY_COOLDOWN_SECONDS), e
            )
            return None, None
    return _model, _tokenizer


def is_available() -> bool:
    """Cheap check: has the model been loaded (or is it loadable)?

    Returns False if the last load attempt failed and the cooldown hasn't
    expired, but will re-attempt after _RETRY_COOLDOWN_SECONDS. This means
    is_available() can flip from False to True without a process restart
    once the blocker (missing deps, network) resolves.
    """
    if _model is not None:
        return True
    _load()
    return _model is not None


def reset_cache() -> None:
    """Force the reranker to re-probe on the next call.

    Called by HolographicMemoryProvider.initialize() so each new session re-attempts
    the load instead of inheriting a stale failure state from the process start.
    """
    global _model, _tokenizer, _load_attempted, _last_fail_ts
    with _lock:
        # Keep a successfully-loaded model (no need to reload), only clear failure state.
        if _model is None:
            _load_attempted = False
            _last_fail_ts = 0.0


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
