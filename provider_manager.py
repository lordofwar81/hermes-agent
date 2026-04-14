#!/usr/bin/env python3
"""
Provider Manager Module

This module provides a standalone ProviderManager class that handles all
provider/client management operations including:
- OpenAI client lifecycle (creation, closing, credential refresh)
- Credential pool rotation
- Fallback provider switching
- Thread-safe client management

Extracted from AIAgent to improve modularity and separation of concerns.
"""

import logging
import os
import platform as _plat
import threading
from typing import Any, Dict, Optional, Callable
from openai import OpenAI

from agent.error_classifier import FailoverReason

logger = logging.getLogger(__name__)

# Constants for Qwen Portal headers
_QWEN_CODE_VERSION = "1.30.0"


def _qwen_portal_headers() -> dict:
    """Return default HTTP headers required by Qwen Portal API."""
    _ua = f"QwenCode/{_QWEN_CODE_VERSION} ({_plat.system().lower()}; {_plat.machine()})"
    return {
        "User-Agent": _ua,
        "X-DashScope-CacheControl": "enable",
        "X-DashScope-UserAgent": _ua,
        "X-DashScope-AuthType": "qwen-oauth",
    }


class ProviderManager:
    """
    Manages AI provider client lifecycle and credential rotation.

    This class encapsulates all provider/client management logic including:
    - Creating and closing OpenAI/Anthropic clients
    - Managing thread-safe client access
    - Handling credential rotation and pools
    - Managing fallback provider switching
    - Detecting and cleaning up dead connections

    The ProviderManager is designed to be used by AIAgent and other
    components that need direct access to provider clients without
    the full conversation management overhead.
    """

    # Which error types indicate a transient transport failure worth
    # one more attempt with a rebuilt client / connection pool.
    _TRANSIENT_TRANSPORT_ERRORS = frozenset(
        {
            "ReadTimeout",
            "ConnectTimeout",
            "PoolTimeout",
            "ConnectError",
            "RemoteProtocolError",
            "APIConnectionError",
            "APITimeoutError",
        }
    )

    def __init__(
        self,
        base_url: str,
        api_key: str,
        provider: str,
        api_mode: str,
        model: str = "",
        use_prompt_caching: bool = False,
        cache_ttl: str = "5m",
        credential_pool: Any = None,
        anthropic_api_key: Optional[str] = None,
        anthropic_base_url: Optional[str] = None,
        fallback_chain: Optional[list] = None,
        client_kwargs: Optional[dict] = None,
    ):
        """
        Initialize the ProviderManager.

        Args:
            base_url: Base URL for the model API
            api_key: API key for authentication
            provider: Provider identifier (e.g., "openrouter", "anthropic", "openai-codex")
            api_mode: API mode: "chat_completions", "codex_responses", or "anthropic_messages"
            model: Model name (optional, for logging)
            use_prompt_caching: Whether to use prompt caching
            cache_ttl: Cache TTL for prompt caching
            credential_pool: Optional credential pool for rotation
            anthropic_api_key: Anthropic-specific API key (for anthropic_messages mode)
            anthropic_base_url: Anthropic-specific base URL (for anthropic_messages mode)
            fallback_chain: List of fallback provider configs
            client_kwargs: Additional kwargs for client creation
        """
        self._base_url = base_url or ""
        self._base_url_lower = self._base_url.lower() if self._base_url else ""
        self.api_key = api_key or ""
        self.provider = provider or ""
        self.api_mode = api_mode or "chat_completions"
        self.model = model or ""

        self._use_prompt_caching = use_prompt_caching
        self._cache_ttl = cache_ttl
        self._credential_pool = credential_pool

        # Anthropic-specific state
        self._anthropic_api_key = anthropic_api_key or ""
        self._anthropic_base_url = anthropic_base_url or ""
        self._anthropic_client = None
        self._is_anthropic_oauth = False

        # Client state
        self._client_lock: Optional[threading.RLock] = None
        self.client: Optional[Any] = None
        self._client_kwargs = client_kwargs or {}

        # Fallback state
        self._fallback_chain = fallback_chain or []
        self._fallback_index = 0
        self._fallback_activated = False

        # Primary runtime snapshot for restoration
        self._primary_runtime: Dict[str, Any] = {}

    @property
    def base_url(self) -> str:
        """Get the current base URL."""
        return self._base_url

    @base_url.setter
    def base_url(self, value: str) -> None:
        """Set the base URL and update the lowercase cache."""
        self._base_url = value
        self._base_url_lower = value.lower() if value else ""

    @property
    def is_openrouter(self) -> bool:
        """Return True when the base URL targets OpenRouter."""
        return "openrouter" in self._base_url_lower

    @property
    def is_qwen_portal(self) -> bool:
        """Return True when the base URL targets Qwen Portal."""
        return "portal.qwen.ai" in self._base_url_lower

    def _is_direct_openai_url(self, base_url: Optional[str] = None) -> bool:
        """Return True when a base URL targets OpenAI's native API."""
        url = (base_url or self._base_url_lower).lower()
        return "api.openai.com" in url and "openrouter" not in url

    def _is_openrouter_url(self) -> bool:
        """Return True when the base URL targets OpenRouter."""
        return self.is_openrouter

    def _is_qwen_portal_url(self) -> bool:
        """Return True when the base URL targets Qwen Portal."""
        return self.is_qwen_portal

    def _thread_identity(self) -> str:
        """Get a string identifying the current thread."""
        thread = threading.current_thread()
        return f"{thread.name}:{thread.ident}"

    def _client_log_context(self) -> str:
        """Get a log context string identifying the current client configuration."""
        provider = getattr(self, "provider", "unknown")
        base_url = getattr(self, "base_url", "unknown")
        model = getattr(self, "model", "unknown")
        return (
            f"thread={self._thread_identity()} provider={provider} "
            f"base_url={base_url} model={model}"
        )

    def _openai_client_lock(self) -> threading.RLock:
        """Get or create the client lock for thread-safe operations."""
        lock = getattr(self, "_client_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._client_lock = lock
        return lock

    @staticmethod
    def _is_openai_client_closed(client: Any) -> bool:
        """Check if an OpenAI client is closed.

        Handles both property and method forms of is_closed:
        - httpx.Client.is_closed is a bool property
        - openai.OpenAI.is_closed is a method returning bool

        Prior bug: getattr(client, "is_closed", False) returned the bound method,
        which is always truthy, causing unnecessary client recreation on every call.
        """
        from unittest.mock import Mock

        if isinstance(client, Mock):
            return False

        is_closed_attr = getattr(client, "is_closed", None)
        if is_closed_attr is not None:
            # Handle method (openai SDK) vs property (httpx)
            if callable(is_closed_attr):
                if is_closed_attr():
                    return True
            elif bool(is_closed_attr):
                return True

        http_client = getattr(client, "_client", None)
        if http_client is not None:
            return bool(getattr(http_client, "is_closed", False))
        return False

    def _create_openai_client(
        self, client_kwargs: dict, *, reason: str, shared: bool
    ) -> Any:
        """Create an OpenAI client with the given kwargs.

        Args:
            client_kwargs: Keyword arguments to pass to OpenAI client constructor
            reason: Log context for why the client is being created
            shared: Whether this is a shared (primary) client or request-specific

        Returns:
            The created OpenAI client (or CopilotACPClient for copilot-acp provider)
        """
        if self.provider == "copilot-acp" or str(
            client_kwargs.get("base_url", "")
        ).startswith("acp://copilot"):
            from agent.copilot_acp_client import CopilotACPClient

            client: Any = CopilotACPClient(**client_kwargs)
            logger.info(
                "Copilot ACP client created (%s, shared=%s) %s",
                reason,
                shared,
                self._client_log_context(),
            )
            return client
        client = OpenAI(**client_kwargs)
        logger.info(
            "OpenAI client created (%s, shared=%s) %s",
            reason,
            shared,
            self._client_log_context(),
        )
        return client

    @staticmethod
    def _force_close_tcp_sockets(client: Any) -> int:
        """Force-close underlying TCP sockets to prevent CLOSE-WAIT accumulation.

        When a provider drops a connection mid-stream, httpx's ``client.close()``
        performs a graceful shutdown which leaves sockets in CLOSE-WAIT until the
        OS times them out (often minutes).  This method walks the httpx transport
        pool and issues ``socket.shutdown(SHUT_RDWR)`` + ``socket.close()`` to
        force an immediate TCP RST, freeing the file descriptors.

        Returns the number of sockets force-closed.
        """
        import socket as _socket

        closed = 0
        try:
            http_client = getattr(client, "_client", None)
            if http_client is None:
                return 0
            transport = getattr(http_client, "_transport", None)
            if transport is None:
                return 0
            pool = getattr(transport, "_pool", None)
            if pool is None:
                return 0
            # httpx uses httpcore connection pools; connections live in
            # _connections (list) or _pool (list) depending on version.
            connections = (
                getattr(pool, "_connections", None)
                or getattr(pool, "_pool", None)
                or []
            )
            for conn in list(connections):
                stream = getattr(conn, "_network_stream", None) or getattr(
                    conn, "_stream", None
                )
                if stream is None:
                    continue
                sock = getattr(stream, "_sock", None)
                if sock is None:
                    sock = getattr(stream, "stream", None)
                    if sock is not None:
                        sock = getattr(sock, "_sock", None)
                if sock is None:
                    continue
                try:
                    sock.shutdown(_socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass
                closed += 1
        except Exception as exc:
            logger.debug("Force-close TCP sockets sweep error: %s", exc)
        return closed

    def _close_openai_client(self, client: Any, *, reason: str, shared: bool) -> None:
        """Close an OpenAI client and force-close TCP sockets.

        Args:
            client: The client to close
            reason: Log context for why the client is being closed
            shared: Whether this was a shared (primary) client or request-specific
        """
        if client is None:
            return
        # Force-close TCP sockets first to prevent CLOSE-WAIT accumulation,
        # then do the graceful SDK-level close.
        force_closed = self._force_close_tcp_sockets(client)
        try:
            client.close()
            logger.info(
                "OpenAI client closed (%s, shared=%s, tcp_force_closed=%d) %s",
                reason,
                shared,
                force_closed,
                self._client_log_context(),
            )
        except Exception as exc:
            logger.debug(
                "OpenAI client close failed (%s, shared=%s) %s error=%s",
                reason,
                shared,
                self._client_log_context(),
                exc,
            )

    def _replace_primary_openai_client(self, *, reason: str) -> bool:
        """Replace the primary OpenAI client with a new one.

        Args:
            reason: Log context for why the client is being replaced

        Returns:
            True if the client was successfully replaced, False otherwise
        """
        with self._openai_client_lock():
            old_client = getattr(self, "client", None)
            try:
                new_client = self._create_openai_client(
                    self._client_kwargs, reason=reason, shared=True
                )
            except Exception as exc:
                logger.warning(
                    "Failed to rebuild shared OpenAI client (%s) %s error=%s",
                    reason,
                    self._client_log_context(),
                    exc,
                )
                return False
            self.client = new_client
        self._close_openai_client(old_client, reason=f"replace:{reason}", shared=True)
        return True

    def _ensure_primary_openai_client(self, *, reason: str) -> Any:
        """Ensure the primary OpenAI client exists and is not closed.

        Args:
            reason: Log context for why the client is being ensured

        Returns:
            The primary OpenAI client

        Raises:
            RuntimeError: If the client could not be recreated
        """
        with self._openai_client_lock():
            client = getattr(self, "client", None)
            if client is not None and not self._is_openai_client_closed(client):
                return client

        logger.warning(
            "Detected closed shared OpenAI client; recreating before use (%s) %s",
            reason,
            self._client_log_context(),
        )
        if not self._replace_primary_openai_client(reason=f"recreate_closed:{reason}"):
            raise RuntimeError("Failed to recreate closed OpenAI client")
        with self._openai_client_lock():
            return self.client

    def _cleanup_dead_connections(self) -> bool:
        """Detect and clean up dead TCP connections on the primary client.

        Inspects the httpx connection pool for sockets in unhealthy states
        (CLOSE-WAIT, errors).  If any are found, force-closes all sockets
        and rebuilds the primary client from scratch.

        Returns True if dead connections were found and cleaned up.
        """
        client = getattr(self, "client", None)
        if client is None:
            return False
        try:
            http_client = getattr(client, "_client", None)
            if http_client is None:
                return False
            transport = getattr(http_client, "_transport", None)
            if transport is None:
                return False
            pool = getattr(transport, "_pool", None)
            if pool is None:
                return False
            connections = (
                getattr(pool, "_connections", None)
                or getattr(pool, "_pool", None)
                or []
            )
            dead_count = 0
            for conn in list(connections):
                # Check for connections that are idle but have closed sockets
                stream = getattr(conn, "_network_stream", None) or getattr(
                    conn, "_stream", None
                )
                if stream is None:
                    continue
                sock = getattr(stream, "_sock", None)
                if sock is None:
                    sock = getattr(stream, "stream", None)
                    if sock is not None:
                        sock = getattr(sock, "_sock", None)
                if sock is None:
                    continue
                # Probe socket health with a non-blocking recv peek
                import socket as _socket

                try:
                    sock.setblocking(False)
                    data = sock.recv(1, _socket.MSG_PEEK | _socket.MSG_DONTWAIT)
                    if data == b"":
                        dead_count += 1
                except BlockingIOError:
                    pass  # No data available — socket is healthy
                except OSError:
                    dead_count += 1
                finally:
                    try:
                        sock.setblocking(True)
                    except OSError:
                        pass
            if dead_count > 0:
                logger.warning(
                    "Found %d dead connection(s) in client pool — rebuilding client",
                    dead_count,
                )
                self._replace_primary_openai_client(reason="dead_connection_cleanup")
                return True
        except Exception as exc:
            logger.debug("Dead connection check error: %s", exc)
        return False

    def _create_request_openai_client(self, *, reason: str) -> Any:
        """Create a request-specific OpenAI client.

        Args:
            reason: Log context for why the client is being created

        Returns:
            A new OpenAI client for this request
        """
        from unittest.mock import Mock

        primary_client = self._ensure_primary_openai_client(reason=reason)
        if isinstance(primary_client, Mock):
            return primary_client
        with self._openai_client_lock():
            request_kwargs = dict(self._client_kwargs)
        return self._create_openai_client(request_kwargs, reason=reason, shared=False)

    def _close_request_openai_client(self, client: Any, *, reason: str) -> None:
        """Close a request-specific OpenAI client.

        Args:
            client: The client to close
            reason: Log context for why the client is being closed
        """
        self._close_openai_client(client, reason=reason, shared=False)

    def _try_refresh_codex_client_credentials(self, *, force: bool = True) -> bool:
        """Refresh Codex client credentials.

        Args:
            force: Whether to force a refresh even if cached credentials exist

        Returns:
            True if credentials were refreshed successfully, False otherwise
        """
        if self.api_mode != "codex_responses" or self.provider != "openai-codex":
            return False

        try:
            from hermes_cli.auth import resolve_codex_runtime_credentials

            creds = resolve_codex_runtime_credentials(force_refresh=force)
        except Exception as exc:
            logger.debug("Codex credential refresh failed: %s", exc)
            return False

        api_key = creds.get("api_key")
        base_url = creds.get("base_url")
        if not isinstance(api_key, str) or not api_key.strip():
            return False
        if not isinstance(base_url, str) or not base_url.strip():
            return False

        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/")
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url

        if not self._replace_primary_openai_client(reason="codex_credential_refresh"):
            return False

        return True

    def _try_refresh_nous_client_credentials(self, *, force: bool = True) -> bool:
        """Refresh Nous client credentials.

        Args:
            force: Whether to force a refresh even if cached credentials exist

        Returns:
            True if credentials were refreshed successfully, False otherwise
        """
        if self.api_mode != "chat_completions" or self.provider != "nous":
            return False

        try:
            from hermes_cli.auth import resolve_nous_runtime_credentials

            creds = resolve_nous_runtime_credentials(
                min_key_ttl_seconds=max(
                    60, int(os.getenv("HERMES_NOUS_MIN_KEY_TTL_SECONDS", "1800"))
                ),
                timeout_seconds=float(os.getenv("HERMES_NOUS_TIMEOUT_SECONDS", "15")),
                force_mint=force,
            )
        except Exception as exc:
            logger.debug("Nous credential refresh failed: %s", exc)
            return False

        api_key = creds.get("api_key")
        base_url = creds.get("base_url")
        if not isinstance(api_key, str) or not api_key.strip():
            return False
        if not isinstance(base_url, str) or not base_url.strip():
            return False

        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/")
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        # Nous requests should not inherit OpenRouter-only attribution headers.
        self._client_kwargs.pop("default_headers", None)

        if not self._replace_primary_openai_client(reason="nous_credential_refresh"):
            return False

        return True

    def _try_refresh_anthropic_client_credentials(self) -> bool:
        """Refresh Anthropic client credentials.

        Returns:
            True if credentials were refreshed successfully, False otherwise
        """
        if self.api_mode != "anthropic_messages" or not hasattr(
            self, "_anthropic_api_key"
        ):
            return False
        # Only refresh credentials for the native Anthropic provider.
        # Other anthropic_messages providers (MiniMax, Alibaba, etc.) use their own keys.
        if self.provider != "anthropic":
            return False

        try:
            from agent.anthropic_adapter import (
                resolve_anthropic_token,
                build_anthropic_client,
            )

            new_token = resolve_anthropic_token()
        except Exception as exc:
            logger.debug("Anthropic credential refresh failed: %s", exc)
            return False

        if not isinstance(new_token, str) or not new_token.strip():
            return False
        new_token = new_token.strip()
        if new_token == self._anthropic_api_key:
            return False

        try:
            if self._anthropic_client is not None:
                self._anthropic_client.close()
        except Exception:
            pass

        try:
            self._anthropic_client = build_anthropic_client(
                new_token, getattr(self, "_anthropic_base_url", None) or ""
            )
        except Exception as exc:
            logger.warning(
                "Failed to rebuild Anthropic client after credential refresh: %s", exc
            )
            return False

        self._anthropic_api_key = new_token
        # Update OAuth flag — token type may have changed (API key ↔ OAuth)
        from agent.anthropic_adapter import _is_oauth_token

        self._is_anthropic_oauth = _is_oauth_token(new_token)
        return True

    def _apply_client_headers_for_base_url(self, base_url: str) -> None:
        """Apply provider-specific HTTP headers based on the base URL.

        Args:
            base_url: The base URL to apply headers for
        """
        from agent.auxiliary_client import _OR_HEADERS

        normalized = (base_url or "").lower()
        if "openrouter" in normalized:
            self._client_kwargs["default_headers"] = dict(_OR_HEADERS)
        elif "api.githubcopilot.com" in normalized:
            from hermes_cli.models import copilot_default_headers

            self._client_kwargs["default_headers"] = copilot_default_headers()
        elif "api.kimi.com" in normalized:
            self._client_kwargs["default_headers"] = {"User-Agent": "KimiCLI/1.30.0"}
        elif "portal.qwen.ai" in normalized:
            self._client_kwargs["default_headers"] = _qwen_portal_headers()
        else:
            self._client_kwargs.pop("default_headers", None)

    def _swap_credential(self, entry) -> None:
        """Swap to a new credential from the credential pool.

        Args:
            entry: The credential pool entry to swap to
        """
        runtime_key = getattr(entry, "runtime_api_key", None) or getattr(
            entry, "access_token", ""
        )
        runtime_base = (
            getattr(entry, "runtime_base_url", None)
            or getattr(entry, "base_url", None)
            or self.base_url
        )

        if self.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_client, _is_oauth_token

            try:
                if self._anthropic_client is not None:
                    self._anthropic_client.close()
            except Exception:
                pass

            self._anthropic_api_key = runtime_key or ""
            self._anthropic_base_url = runtime_base or ""
            self._anthropic_client = build_anthropic_client(runtime_key or "", runtime_base or "")
            self._is_anthropic_oauth = (
                _is_oauth_token(runtime_key or "") if self.provider == "anthropic" else False
            )
            self.api_key = runtime_key or ""
            self.base_url = runtime_base or ""
            return

        self.api_key = runtime_key or ""
        self.base_url = (
            runtime_base.rstrip("/") if isinstance(runtime_base, str) else runtime_base
        )
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        self._apply_client_headers_for_base_url(self.base_url)
        self._replace_primary_openai_client(reason="credential_rotation")

    def _recover_with_credential_pool(
        self,
        *,
        status_code: Optional[int],
        has_retried_429: bool,
        classified_reason: Optional[FailoverReason] = None,
        error_context: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, bool]:
        """Attempt credential recovery via pool rotation.

        Args:
            status_code: HTTP status code from the failed request
            has_retried_429: Whether we've already retried after a 429
            classified_reason: Structured error classification from error_classifier
            error_context: Additional error context for logging

        Returns:
            Tuple of (recovered, has_retried_429) where:
            - recovered: True if recovery was successful
            - has_retried_429: Updated retry state for 429s

        On rate limits: first occurrence retries same credential (sets flag True).
                        second consecutive failure rotates to next credential.
        On billing exhaustion: immediately rotates.
        On auth failures: attempts token refresh before rotating.

        `classified_reason` lets the recovery path honor the structured error
        classifier instead of relying only on raw HTTP codes. This matters for
        providers that surface billing/rate-limit/auth conditions under a
        different status code, such as Anthropic returning HTTP 400 for
        "out of extra usage".
        """
        pool = self._credential_pool
        if pool is None:
            return False, has_retried_429

        effective_reason = classified_reason
        if effective_reason is None:
            if status_code == 402:
                effective_reason = FailoverReason.billing
            elif status_code == 429:
                effective_reason = FailoverReason.rate_limit
            elif status_code == 401:
                effective_reason = FailoverReason.auth

        if effective_reason == FailoverReason.billing:
            rotate_status = status_code if status_code is not None else 402
            next_entry = pool.mark_exhausted_and_rotate(
                status_code=rotate_status, error_context=error_context
            )
            if next_entry is not None:
                logger.info(
                    "Credential %s (billing) — rotated to pool entry %s",
                    rotate_status,
                    getattr(next_entry, "id", "?"),
                )
                self._swap_credential(next_entry)
                return True, False
            return False, has_retried_429

        if effective_reason == FailoverReason.rate_limit:
            if not has_retried_429:
                return False, True
            rotate_status = status_code if status_code is not None else 429
            next_entry = pool.mark_exhausted_and_rotate(
                status_code=rotate_status, error_context=error_context
            )
            if next_entry is not None:
                logger.info(
                    "Credential %s (rate limit) — rotated to pool entry %s",
                    rotate_status,
                    getattr(next_entry, "id", "?"),
                )
                self._swap_credential(next_entry)
                return True, False
            return False, True

        if effective_reason == FailoverReason.auth:
            refreshed = pool.try_refresh_current()
            if refreshed is not None:
                logger.info(
                    f"Credential auth failure — refreshed pool entry {getattr(refreshed, 'id', '?')}"
                )
                self._swap_credential(refreshed)
                return True, has_retried_429
            # Refresh failed — rotate to next credential instead of giving up.
            # The failed entry is already marked exhausted by try_refresh_current().
            rotate_status = status_code if status_code is not None else 401
            next_entry = pool.mark_exhausted_and_rotate(
                status_code=rotate_status, error_context=error_context
            )
            if next_entry is not None:
                logger.info(
                    "Credential %s (auth refresh failed) — rotated to pool entry %s",
                    rotate_status,
                    getattr(next_entry, "id", "?"),
                )
                self._swap_credential(next_entry)
                return True, False

        return False, has_retried_429

    def snapshot_primary_runtime(self, compressor_state: Optional[dict] = None) -> None:
        """Snapshot the current primary runtime state for later restoration.

        Args:
            compressor_state: Optional dict with context compressor state
        """
        self._primary_runtime = {
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "api_mode": self.api_mode,
            "api_key": getattr(self, "api_key", ""),
            "client_kwargs": dict(self._client_kwargs),
            "use_prompt_caching": self._use_prompt_caching,
        }
        if compressor_state:
            self._primary_runtime.update(compressor_state)
        if self.api_mode == "anthropic_messages":
            self._primary_runtime.update(
                {
                    "anthropic_api_key": self._anthropic_api_key,
                    "anthropic_base_url": self._anthropic_base_url,
                    "is_anthropic_oauth": self._is_anthropic_oauth,
                }
            )

    def restore_primary_runtime(self) -> bool:
        """Restore the primary runtime from the saved snapshot.

        Returns:
            True if restoration was successful, False otherwise
        """
        if not self._primary_runtime:
            return False

        rt = self._primary_runtime
        try:
            # ── Core runtime state ──
            self.model = rt["model"]
            self.provider = rt["provider"]
            self.base_url = rt["base_url"]  # setter updates _base_url_lower
            self.api_mode = rt["api_mode"]
            self.api_key = rt["api_key"]
            self._client_kwargs = dict(rt["client_kwargs"])
            self._use_prompt_caching = rt["use_prompt_caching"]

            # ── Rebuild client for the primary provider ──
            if self.api_mode == "anthropic_messages":
                from agent.anthropic_adapter import build_anthropic_client

                self._anthropic_api_key = rt["anthropic_api_key"]
                self._anthropic_base_url = rt["anthropic_base_url"]
                self._anthropic_client = build_anthropic_client(
                    rt["anthropic_api_key"] or "",
                    rt["anthropic_base_url"] or "",
                )
                self._is_anthropic_oauth = rt["is_anthropic_oauth"]
                self.client = None
            else:
                self.client = self._create_openai_client(
                    dict(rt["client_kwargs"]),
                    reason="restore_primary",
                    shared=True,
                )

            # ── Reset fallback state ──
            self._fallback_activated = False
            self._fallback_index = 0

            logging.info(
                "Primary runtime restored for new turn: %s (%s)",
                self.model,
                self.provider,
            )
            return True
        except Exception as e:
            logging.warning("Failed to restore primary runtime: %s", e)
            return False

    def _try_activate_fallback(self, status_callback: Optional[Callable] = None) -> bool:
        """Switch to the next fallback model/provider in the chain.

        Called when the current model is failing after retries.  Swaps the
        OpenAI client, model slug, and provider in-place so the retry loop
        can continue with the new backend.  Advances through the chain on
        each call; returns False when exhausted.

        Uses the centralized provider router (resolve_provider_client) for
        auth resolution and client construction — no duplicated provider→key
        mappings.

        Args:
            status_callback: Optional callback for status messages

        Returns:
            True if a fallback was activated, False if chain is exhausted
        """
        if self._fallback_index >= len(self._fallback_chain):
            return False

        fb = self._fallback_chain[self._fallback_index]
        self._fallback_index += 1
        fb_provider = (fb.get("provider") or "").strip().lower()
        fb_model = (fb.get("model") or "").strip()
        if not fb_provider or not fb_model:
            return self._try_activate_fallback(status_callback)  # skip invalid, try next

        # Use centralized router for client construction.
        # raw_codex=True because the main agent needs direct responses.stream()
        # access for Codex providers.
        try:
            from agent.auxiliary_client import resolve_provider_client

            # Pass base_url and api_key from fallback config so custom
            # endpoints (e.g. Ollama Cloud) resolve correctly instead of
            # falling through to OpenRouter defaults.
            fb_base_url_hint = (fb.get("base_url") or "").strip() or None
            fb_api_key_hint = (fb.get("api_key") or "").strip() or None
            # For Ollama Cloud endpoints, pull OLLAMA_API_KEY from env
            # when no explicit key is in the fallback config.
            if (
                fb_base_url_hint
                and "ollama.com" in fb_base_url_hint.lower()
                and not fb_api_key_hint
            ):
                fb_api_key_hint = os.getenv("OLLAMA_API_KEY") or None
            fb_client, _resolved_fb_model = resolve_provider_client(
                fb_provider,
                model=fb_model,
                raw_codex=True,
                explicit_base_url=fb_base_url_hint or "",
                explicit_api_key=fb_api_key_hint or "",
            )
            if fb_client is None:
                logging.warning(
                    "Fallback to %s failed: provider not configured", fb_provider
                )
                return self._try_activate_fallback(status_callback)  # try next in chain
            try:
                from hermes_cli.model_normalize import normalize_model_for_provider

                fb_model = normalize_model_for_provider(fb_model, fb_provider)
            except Exception:
                pass

            # Determine api_mode from provider / base URL
            fb_api_mode = "chat_completions"
            fb_base_url = str(fb_client.base_url)
            if fb_provider == "openai-codex":
                fb_api_mode = "codex_responses"
            elif fb_provider == "anthropic" or fb_base_url.rstrip("/").lower().endswith(
                "/anthropic"
            ):
                fb_api_mode = "anthropic_messages"
            elif self._is_direct_openai_url(fb_base_url):
                fb_api_mode = "codex_responses"

            old_model = self.model
            self.model = fb_model
            self.provider = fb_provider
            self.base_url = fb_base_url
            self.api_mode = fb_api_mode
            self._fallback_activated = True

            if fb_api_mode == "anthropic_messages":
                # Build native Anthropic client instead of using OpenAI client
                from agent.anthropic_adapter import (
                    build_anthropic_client,
                    resolve_anthropic_token,
                    _is_oauth_token,
                )

                effective_key = (
                    (fb_client.api_key or resolve_anthropic_token() or "")
                    if fb_provider == "anthropic"
                    else (fb_client.api_key or "")
                )
                self.api_key = effective_key
                self._anthropic_api_key = effective_key
                self._anthropic_base_url = fb_base_url
                self._anthropic_client = build_anthropic_client(
                    effective_key, self._anthropic_base_url
                )
                self._is_anthropic_oauth = _is_oauth_token(effective_key)
                self.client = None
                self._client_kwargs = {}
            else:
                # Swap OpenAI client and config in-place
                self.api_key = fb_client.api_key
                self.client = fb_client
                # Preserve provider-specific headers that
                # resolve_provider_client() may have baked into
                # fb_client via the default_headers kwarg.  The OpenAI
                # SDK stores these in _custom_headers.  Without this,
                # subsequent request-client rebuilds (via
                # _create_request_openai_client) drop the headers,
                # causing 403s from providers like Kimi Coding that
                # require a User-Agent sentinel.
                fb_headers = getattr(fb_client, "_custom_headers", None)
                if not fb_headers:
                    fb_headers = getattr(fb_client, "default_headers", None)
                self._client_kwargs = {
                    "api_key": fb_client.api_key,
                    "base_url": fb_base_url,
                    **({"default_headers": dict(fb_headers)} if fb_headers else {}),
                }

            # Re-evaluate prompt caching for the new provider/model
            is_native_anthropic = fb_api_mode == "anthropic_messages"
            self._use_prompt_caching = (
                "openrouter" in fb_base_url.lower() and "claude" in fb_model.lower()
            ) or is_native_anthropic

            if status_callback:
                status_callback(
                    f"Primary model failed — switching to fallback: "
                    f"{fb_model} via {fb_provider}"
                )
            logging.info(
                "Fallback activated: %s → %s (%s)",
                old_model,
                fb_model,
                fb_provider,
            )
            return True
        except Exception as e:
            logging.error("Failed to activate fallback %s: %s", fb_model, e)
            return self._try_activate_fallback(status_callback)  # try next in chain

    def _try_recover_primary_transport(
        self,
        api_error: Exception,
        *,
        retry_count: int,
        max_retries: int,
    ) -> bool:
        """Attempt one extra primary-provider recovery cycle for transient transport failures.

        After ``max_retries`` exhaust, rebuild the primary client (clearing
        stale connection pools) and give it one more attempt before falling
        back.  This is most useful for direct endpoints (custom, Z.AI,
        Anthropic, OpenAI, local models) where a TCP-level hiccup does not
        mean the provider is down.

        Skipped for proxy/aggregator providers (OpenRouter, Nous) which
        already manage connection pools and retries server-side — if our
        retries through them are exhausted, one more rebuilt client won't help.

        Args:
            api_error: The exception that occurred
            retry_count: Current retry count
            max_retries: Maximum retries allowed

        Returns:
            True if recovery was attempted and should retry, False otherwise
        """
        if self._fallback_activated:
            return False

        # Only for transient transport errors
        error_type = type(api_error).__name__
        if error_type not in self._TRANSIENT_TRANSPORT_ERRORS:
            return False

        # Skip for aggregator providers — they manage their own retry infra
        if self._is_openrouter_url():
            return False
        provider_lower = (self.provider or "").strip().lower()
        if provider_lower in ("nous", "nous-research"):
            return False

        try:
            # Close existing client to release stale connections
            if getattr(self, "client", None) is not None:
                try:
                    self._close_openai_client(
                        self.client,
                        reason="primary_recovery",
                        shared=True,
                    )
                except Exception:
                    pass

            # Rebuild from primary snapshot
            rt = self._primary_runtime
            if not rt:
                return False
            self._client_kwargs = dict(rt["client_kwargs"])
            self.model = rt["model"]
            self.provider = rt["provider"]
            self.base_url = rt["base_url"]

            # Rebuild the client
            self.client = self._create_openai_client(
                self._client_kwargs, reason="primary_recovery", shared=True
            )

            logging.info(
                "Primary transport recovery successful after %d retries — "
                "rebuilt client for one more attempt",
                retry_count,
            )
            return True
        except Exception as exc:
            logging.warning("Primary transport recovery failed: %s", exc)
            return False

    def initialize_client(self) -> None:
        """Initialize the primary client based on current configuration.

        This method should be called after construction to create the initial
        client based on the provider, api_mode, and credentials.
        """
        if self.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import (
                build_anthropic_client,
                resolve_anthropic_token,
                _is_oauth_token,
            )

            # Only fall back to ANTHROPIC_TOKEN when the provider is actually Anthropic.
            # Other anthropic_messages providers (MiniMax, Alibaba, etc.) must use their own API key.
            _is_native_anthropic = self.provider == "anthropic"
            effective_key = (
                (self.api_key or resolve_anthropic_token() or "")
                if _is_native_anthropic
                else (self.api_key or "")
            )
            self.api_key = effective_key
            self._anthropic_api_key = effective_key
            self._anthropic_base_url = self._base_url
            self._is_anthropic_oauth = _is_oauth_token(effective_key)
            self._anthropic_client = build_anthropic_client(effective_key, self._base_url or "")
            # No OpenAI client needed for Anthropic mode
            self.client = None
            self._client_kwargs = {}
        else:
            self._apply_client_headers_for_base_url(self._base_url)
            self._client_kwargs["api_key"] = self.api_key
            self._client_kwargs["base_url"] = self._base_url
            self.client = self._create_openai_client(
                self._client_kwargs, reason="provider_init", shared=True
            )
