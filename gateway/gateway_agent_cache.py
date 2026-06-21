"""Agent cache-key helpers extracted from gateway/run.py GatewayRunner.

Round 8 of gateway decomposition. _agent_config_signature is a pure function
of its arguments (no self/cls state) that computes a stable hash for the
cached-agent invalidation key. Moved verbatim as a module function.

The related _extract_cache_busting_config / _extract_honcho_cache_busting_config
methods are @classmethod with cls-level memoization state (_HONCHO_CACHE_BUSTING_MEMO)
and were NOT moved in this round — they need memoization-state migration,
a different extraction pattern. Kept in GatewayRunner for now.
"""

import hashlib
import json as _j


def _agent_config_signature(
    model: str,
    runtime: dict,
    enabled_toolsets: list,
    ephemeral_prompt: str,
    cache_keys: dict | None = None,
    user_id: str | None = None,
    user_id_alt: str | None = None,
) -> str:
    """Compute a stable string key from agent config values.

    When this signature changes between messages, the cached AIAgent is
    discarded and rebuilt.  When it stays the same, the cached agent is
    reused — preserving the frozen system prompt and tool schemas for
    prompt cache hits.

    ``cache_keys`` is an optional flat dict of additional config values
    that should invalidate the cache when they change.  Callers pass
    the output of ``_extract_cache_busting_config(user_config)`` so
    edits to model.context_length / compression.* in config.yaml are
    picked up on the next gateway message without a manual restart.

    ``user_id`` and ``user_id_alt`` are the runtime user identities
    carried by the current message's gateway source.  They participate
    in the cache key because the Honcho memory provider freezes them
    into ``HonchoSessionManager`` at first-message init (see
    ``plugins/memory/honcho/__init__.py::_do_session_init``).  Without
    them in the signature, a shared-thread session_key (one in which
    ``build_session_key`` intentionally omits the participant ID,
    e.g. ``thread_sessions_per_user=False``) would reuse the cached
    AIAgent across distinct users, causing the second user's messages
    to be attributed to the first user's resolved Honcho peer.  This
    broke #27371's per-user-peer contract in multi-user gateways.
    Per-user agent rebuilds in shared threads trade prompt-cache
    warmth for correct memory attribution.
    """
    import hashlib, json as _j

    # Fingerprint the FULL credential string instead of using a short
    # prefix. OAuth/JWT-style tokens frequently share a common prefix
    # (e.g. "eyJhbGci"), which can cause false cache hits across auth
    # switches if only the first few characters are considered.
    _api_key = str(runtime.get("api_key", "") or "")
    _api_key_fingerprint = hashlib.sha256(_api_key.encode()).hexdigest() if _api_key else ""

    _cache_keys_sorted = sorted((cache_keys or {}).items())

    blob = _j.dumps(
        [
            model,
            _api_key_fingerprint,
            runtime.get("base_url", ""),
            runtime.get("provider", ""),
            runtime.get("api_mode", ""),
            sorted(enabled_toolsets) if enabled_toolsets else [],
            # reasoning_config excluded — it's set per-message on the
            # cached agent and doesn't affect system prompt or tools.
            ephemeral_prompt or "",
            _cache_keys_sorted,
            str(user_id or ""),
            str(user_id_alt or ""),
        ],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
