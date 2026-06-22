"""Session-source cache + run-generation token methods for ``GatewayRunner``.

Round 22 of the god-file decomposition. Extracted verbatim into a mixin.
Two cohesive, self-contained bookkeeping subsystems lifted together:

1. **Session-source LRU cache** — ``_cache_session_source`` /
   ``_get_cached_session_source`` (``self._session_sources``,
   ``self._session_sources_max``).
2. **Run-generation token** — ``_begin_session_run_generation`` /
   ``_invalidate_session_run_generation`` / ``_is_session_run_current``
   (``self._session_run_generation``, accessed via ``self.__dict__`` so
   ``getattr`` auto-vivification on test doubles does not mask a missing
   store).

``self.*`` references resolve unchanged via the MRO. Behavior-neutral lift
matching the existing mixin pattern.

``logger`` is lazy-imported inside the methods that use it to avoid a
circular import (gateway.run imports this mixin at module top).
``OrderedDict`` and ``dataclasses`` import at module top.
"""

from __future__ import annotations

import dataclasses
from collections import OrderedDict


class GatewaySessionCacheMixin:
    """Session-source cache + run-generation token methods for ``GatewayRunner``."""

    def _cache_session_source(self, session_key: str, source) -> None:
        if not session_key or source is None:
            return
        cached_sources = getattr(self, "_session_sources", None)
        if cached_sources is None:
            cached_sources = OrderedDict()
            self._session_sources = cached_sources
        try:
            cached_sources[session_key] = dataclasses.replace(source)
        except Exception:
            from gateway.run import logger
            logger.debug("Failed to cache live session source for %s", session_key, exc_info=True)
            return
        # LRU: mark as most-recently-used and trim to max size.
        try:
            cached_sources.move_to_end(session_key)
            max_size = getattr(self, "_session_sources_max", 512)
            while len(cached_sources) > max_size:
                cached_sources.popitem(last=False)
        except Exception:
            pass

    def _get_cached_session_source(self, session_key: str):
        if not session_key:
            return None
        cached_sources = getattr(self, "_session_sources", None)
        if not cached_sources:
            return None
        source = cached_sources.get(session_key)
        if source is not None:
            try:
                cached_sources.move_to_end(session_key)
            except Exception:
                pass
        return source

    def _begin_session_run_generation(self, session_key: str) -> int:
        """Claim a fresh run generation token for ``session_key``.

        Every top-level gateway turn gets a monotonically increasing token.
        If a later command like /stop or /new invalidates that token while the
        old worker is still unwinding, the late result can be recognized and
        dropped instead of bleeding into the fresh session.
        """
        if not session_key:
            return 0
        generations = self.__dict__.get("_session_run_generation")
        if generations is None:
            generations = {}
            self._session_run_generation = generations
        next_generation = int(generations.get(session_key, 0)) + 1
        generations[session_key] = next_generation
        return next_generation

    def _invalidate_session_run_generation(self, session_key: str, *, reason: str = "") -> int:
        """Invalidate any in-flight run token for ``session_key``."""
        generation = self._begin_session_run_generation(session_key)
        if reason:
            from gateway.run import logger
            logger.info(
                "Invalidated run generation for %s → %d (%s)",
                session_key,
                generation,
                reason,
            )
        return generation

    def _is_session_run_current(self, session_key: str, generation: int) -> bool:
        """Return True when ``generation`` is still current for ``session_key``."""
        if not session_key:
            return True
        generations = self.__dict__.get("_session_run_generation") or {}
        return int(generations.get(session_key, 0)) == int(generation)
