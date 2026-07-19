"""Crawl4AI extract plugin — bundled, auto-loaded.

Backed by a user-hosted Crawl4AI instance (URL configured via ``CRAWL4AI_URL``).
Extract-only — pair with a search provider (parallel/tavily/exa) for
``web_search`` calls.
"""

from __future__ import annotations

from plugins.web.crawl4ai.provider import Crawl4AIWebSearchProvider


def register(ctx) -> None:
    """Register the Crawl4AI provider with the plugin context."""
    ctx.register_web_search_provider(Crawl4AIWebSearchProvider())
