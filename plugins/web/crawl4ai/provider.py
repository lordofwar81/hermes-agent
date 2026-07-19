"""Crawl4AI web extract — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`.

Crawl4AI is an open-source, self-hosted web scraper that returns
LLM-ready Markdown. It does NOT provide web search — only URL extraction.
``supports_search()`` returns False.

Config keys this provider responds to::

    web:
      extract_backend: "crawl4ai"     # explicit per-capability
      backend: "crawl4ai"            # shared fallback

Env var::

    CRAWL4AI_URL=http://192.168.1.232:11235

API contract::

    POST /crawl
    Body: {"urls": ["https://example.com"]}
    Response: {"success": true, "results": [{url, markdown, html, metadata, links, media, ...}]}
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List

import httpx

from agent.web_search_provider import WebSearchProvider
from tools.website_policy import check_website_access

logger = logging.getLogger(__name__)


def _crawl4ai_url() -> str:
    """Return CRAWL4AI_URL from Hermes config-aware env, falling back to process env."""
    try:
        from hermes_cli.config import get_env_value

        val = get_env_value("CRAWL4AI_URL")
    except Exception:
        val = None
    if val is None:
        val = os.getenv("CRAWL4AI_URL", "")
    return (val or "").strip().rstrip("/")


class Crawl4AIWebSearchProvider(WebSearchProvider):
    """Crawl4AI extract provider — search-only backends need an extract partner."""

    @property
    def name(self) -> str:
        return "crawl4ai"

    @property
    def display_name(self) -> str:
        return "Crawl4AI"

    def is_available(self) -> bool:
        """Return True when CRAWL4AI_URL is set."""
        return bool(_crawl4ai_url())

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Crawl4AI has no search endpoint — this should never be called."""
        return {"success": False, "error": "Crawl4AI does not support search"}

    async def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from one or more URLs via Crawl4AI.

        Async; sends all URLs in a single POST /crawl request. Each URL result
        is checked against website-access policy. Falls back gracefully on
        per-URL failures.

        Accepted kwargs (others ignored):
          - ``format``: ``"markdown"`` or ``"html"``; default is markdown.
        """
        from tools.interrupt import is_interrupted as _is_interrupted

        if _is_interrupted():
            return [{"url": u, "error": "Interrupted", "title": ""} for u in urls]

        base_url = _crawl4ai_url()
        if not base_url:
            return [
                {"url": u, "title": "", "content": "", "error": "CRAWL4AI_URL is not set"}
                for u in urls
            ]

        format_kw = kwargs.get("format", "markdown")

        # Pre-scrape website policy gate
        safe_urls: List[str] = []
        policy_blocks: Dict[str, dict] = {}
        for url in urls:
            blocked = check_website_access(url)
            if blocked:
                policy_blocks[url] = blocked
            else:
                safe_urls.append(url)

        results: List[Dict[str, Any]] = []

        # Emit policy-blocked results
        for url, blocked in policy_blocks.items():
            logger.info(
                "Blocked web_extract for %s by rule %s",
                blocked["host"],
                blocked["rule"],
            )
            results.append(
                {
                    "url": url,
                    "title": "",
                    "content": "",
                    "error": blocked["message"],
                    "blocked_by_policy": {
                        "host": blocked["host"],
                        "rule": blocked["rule"],
                        "source": blocked["source"],
                    },
                }
            )

        if not safe_urls:
            return results

        # Batch scrape via Crawl4AI /crawl endpoint
        try:
            logger.info("Crawl4AI scraping %d URLs via %s", len(safe_urls), base_url)
            resp = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: httpx.post(
                        f"{base_url}/crawl",
                        json={"urls": safe_urls},
                        timeout=60,
                        headers={"Content-Type": "application/json"},
                    ),
                ),
                timeout=90,
            )
            resp.raise_for_status()
            data = resp.json()
        except asyncio.TimeoutError:
            logger.warning("Crawl4AI request timed out")
            return [
                {
                    "url": u,
                    "title": "",
                    "content": "",
                    "error": "Crawl4AI request timed out after 90s",
                }
                for u in safe_urls
            ] + results
        except Exception as exc:  # noqa: BLE001
            logger.warning("Crawl4AI scrape error: %s", exc)
            return [
                {
                    "url": u,
                    "title": "",
                    "content": "",
                    "error": f"Crawl4AI scrape failed: {exc}",
                }
                for u in safe_urls
            ] + results

        # Parse results
        crawl_results = data.get("results", []) if isinstance(data, dict) else []

        # Build a url→result map from Crawl4AI response
        result_map: Dict[str, dict] = {}
        for item in crawl_results:
            if isinstance(item, dict) and item.get("url"):
                result_map[item["url"]] = item

        for url in safe_urls:
            item = result_map.get(url)
            if not item or not item.get("success", False):
                error_msg = item.get("error", "Crawl4AI returned unsuccessful result") if item else "URL not found in Crawl4AI response"
                results.append(
                    {
                        "url": url,
                        "title": "",
                        "content": "",
                        "error": error_msg,
                    }
                )
                continue

            metadata = item.get("metadata", {}) or {}
            title = metadata.get("title", "")
            final_url = url

            # Re-check policy after redirect
            final_blocked = check_website_access(final_url)
            if final_blocked:
                logger.info(
                    "Blocked redirected web_extract for %s by rule %s",
                    final_blocked["host"],
                    final_blocked["rule"],
                )
                results.append(
                    {
                        "url": final_url,
                        "title": title,
                        "content": "",
                        "error": final_blocked["message"],
                        "blocked_by_policy": {
                            "host": final_blocked["host"],
                            "rule": final_blocked["rule"],
                            "source": final_blocked["source"],
                        },
                    }
                )
                continue

            # Choose content by format. Crawl4AI's /crawl response may nest
            # markdown/html under a dict (`item["markdown"]["raw_markdown"]`) or
            # under `item["content"]` as a dict. Normalize to a flat string.
            def _flatten(field_name: str) -> str:
                val = item.get(field_name)
                if isinstance(val, dict):
                    keys = ["raw_markdown", "markdown_with_citations", "fit_markdown", "markdown"] \
                        if field_name in ("markdown", "content") else \
                        ["fit_html", "cleaned_html", "html"]
                    return next((val.get(k) for k in keys if val.get(k)), "") or ""
                return val or ""

            md_val = _flatten("markdown") or _flatten("content")
            html_val = _flatten("html") or _flatten("cleaned_html") or md_val

            if format_kw == "html":
                content = html_val
            else:
                content = md_val

            results.append(
                {
                    "url": final_url,
                    "title": title,
                    "content": content,
                    "raw_content": content,
                    "metadata": metadata,
                }
            )

        return results

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Crawl4AI",
            "badge": "free · self-hosted",
            "tag": (
                "Open-source LLM-friendly scraper. Set CRAWL4AI_URL at your "
                "Docker instance. Extract-only — pair with a search provider."
            ),
            "env_vars": [
                {
                    "key": "CRAWL4AI_URL",
                    "prompt": "Crawl4AI instance URL (e.g. http://192.168.1.232:11235)",
                    "url": "https://github.com/unclecode/crawl4ai",
                },
            ],
        }
