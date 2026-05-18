"""Turn-level model router with latency-aware fallback and circuit breaker.

Routes each turn to either a local (fast/cheap) or primary (strong) model
based on message complexity, per-session latency tracking, and endpoint
health via a circuit breaker.

States:
  closed   — endpoint is healthy, traffic flows normally
  open     — endpoint failed too many times; all traffic diverted
  half-open — cooldown elapsed; next call is a probe to test recovery
"""

from __future__ import annotations

import enum
import logging
import time
from typing import Dict, Optional
from urllib.request import urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task classification
# ---------------------------------------------------------------------------

_COMPLEX_KEYWORDS = frozenset({
    "debug", "debugging", "implement", "implementation", "refactor",
    "patch", "traceback", "stacktrace", "exception", "error", "analyze",
    "analysis", "investigate", "architecture", "design", "compare",
    "benchmark", "optimize", "optimise", "review", "terminal", "shell",
    "tool", "tools", "pytest", "test", "tests", "plan", "planning",
    "delegate", "subagent", "cron", "docker", "kubernetes", "deploy",
    "build", "compile", "configure", "server", "database", "api",
    "code", "function", "class", "module", "script",
})


class TaskCategory(enum.Enum):
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    COMPLEX = "complex"


def classify_turn(message: str) -> TaskCategory:
    text = (message or "").strip()
    if not text:
        return TaskCategory.TRIVIAL

    lowered = text.lower()
    words = {token.strip(".,:;!?()[]{}\"'`") for token in lowered.split()}
    if words & _COMPLEX_KEYWORDS:
        return TaskCategory.COMPLEX

    if "`" in text or "```" in text:
        return TaskCategory.COMPLEX

    return TaskCategory.SIMPLE


# ---------------------------------------------------------------------------
# Latency tracking
# ---------------------------------------------------------------------------

_COLD_START_THRESHOLD_S = 30.0
_LOW_CACHE_THRESHOLD = 0.10


class LatencyRecord:
    __slots__ = ("_is_slow", "cold_start_count")

    def __init__(self) -> None:
        self._is_slow: bool = False
        self.cold_start_count: int = 0

    def record(self, latency_s: float, cache_pct: float) -> None:
        if latency_s > _COLD_START_THRESHOLD_S and cache_pct < _LOW_CACHE_THRESHOLD:
            self.cold_start_count += 1
            self._is_slow = True
        else:
            self._is_slow = False

    @property
    def is_slow(self) -> bool:
        return self._is_slow


# ---------------------------------------------------------------------------
# Circuit breaker (per-endpoint)
# ---------------------------------------------------------------------------

_CIRCUIT_FAILURE_THRESHOLD = 2
_CIRCUIT_COOLDOWN_SEC = 60.0


class CircuitState:
    __slots__ = ("failures", "opened_at")

    def __init__(self) -> None:
        self.failures: int = 0
        self.opened_at: float = 0.0

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= _CIRCUIT_FAILURE_THRESHOLD:
            self.opened_at = time.monotonic()

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = 0.0

    @property
    def is_open(self) -> bool:
        if self.failures < _CIRCUIT_FAILURE_THRESHOLD:
            return False
        age = time.monotonic() - self.opened_at
        if age < _CIRCUIT_COOLDOWN_SEC:
            return True
        return False

    @property
    def is_half_open(self) -> bool:
        if self.failures < _CIRCUIT_FAILURE_THRESHOLD:
            return False
        age = time.monotonic() - self.opened_at
        return age >= _CIRCUIT_COOLDOWN_SEC


# ---------------------------------------------------------------------------
# Endpoint probe
# ---------------------------------------------------------------------------

def probe_local_endpoint(base_url: str, timeout: float = 3.0) -> bool:
    url = base_url.rstrip("/") + "/models"
    try:
        req = urlopen(url, timeout=timeout)
        return req.status == 200
    except (URLError, OSError, TimeoutError):
        return False


# ---------------------------------------------------------------------------
# TurnRouter
# ---------------------------------------------------------------------------

class TurnRouter:
    def __init__(
        self,
        *,
        primary_provider: str,
        primary_model: str,
        primary_base_url: str,
        primary_api_key: str,
        local_base_url: str,
        local_provider: str,
        local_model: str,
        local_api_key: str,
    ) -> None:
        self._primary = {
            "provider": primary_provider,
            "model": primary_model,
            "base_url": primary_base_url,
            "api_key": primary_api_key,
        }
        self._local = {
            "provider": local_provider,
            "model": local_model,
            "base_url": local_base_url,
            "api_key": local_api_key,
        }

        self._latency_tracker: Dict[str, LatencyRecord] = {}
        self._slow_sessions: set = set()
        self._circuits: Dict[str, CircuitState] = {}

    def _circuit(self, base_url: str) -> CircuitState:
        if base_url not in self._circuits:
            self._circuits[base_url] = CircuitState()
        return self._circuits[base_url]

    def get_route(self, message: str, session_key: Optional[str] = None) -> dict:
        category = classify_turn(message)

        # Complex messages always go to primary
        if category == TaskCategory.COMPLEX:
            return dict(self._primary)

        local_url = self._local["base_url"]

        # Circuit breaker check — if local endpoint is open, go primary
        circuit = self._circuit(local_url)
        if circuit.is_open:
            return dict(self._primary)

        # Latency fallback check — if this session had a cold start, go primary
        if session_key and session_key in self._slow_sessions:
            return dict(self._primary)

        # Probe local endpoint (can be mocked in tests)
        if not probe_local_endpoint(local_url):
            return dict(self._primary)

        # Route to local for simple/trivial
        return dict(self._local)

    def record_latency(
        self,
        session_key: str,
        latency_s: float,
        cache_pct: float,
        base_url: str,
    ) -> None:
        if not session_key:
            return
        # Only track latency for local endpoint calls
        if base_url == self._primary["base_url"]:
            return

        rec = self._latency_tracker.get(session_key)
        if rec is None:
            rec = LatencyRecord()
            self._latency_tracker[session_key] = rec
        rec.record(latency_s, cache_pct)

        if rec.is_slow:
            self._slow_sessions.add(session_key)
        else:
            self._slow_sessions.discard(session_key)

    def record_outcome(self, base_url: str, success: bool) -> None:
        circuit = self._circuit(base_url)
        if success:
            circuit.record_success()
        else:
            circuit.record_failure()
            logger.info(
                "Circuit breaker: %s failure #%d (threshold=%d)",
                base_url, circuit.failures, _CIRCUIT_FAILURE_THRESHOLD,
            )

    def reset_session(self, session_key: str) -> None:
        self._latency_tracker.pop(session_key, None)
        self._slow_sessions.discard(session_key)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_router_instance: Optional[TurnRouter] = None


def get_router_instance() -> Optional[TurnRouter]:
    return _router_instance
