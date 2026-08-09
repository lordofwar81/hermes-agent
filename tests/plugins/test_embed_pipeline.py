"""Tests for the embed pipeline: retry-on-400 halving and error branching.

Locks in the 2026-08-08 fix to EmbedClient.embed():
  - Retry-on-400 halving: HTTP 400 (context overflow) → retry with half the text
  - HTTP 401/403 → auth failure message, NO server-down marking
  - HTTP 500+ → server error message, NO server-down marking (HTTP errors
    don't mark down — only connection/timeout errors do)
  - Connection errors → mark server down + record timestamp

Uses unittest.mock to mock urllib.request.urlopen — no real server needed.

Run: venv/bin/python -m pytest tests/plugins/test_embed_pipeline.py -v
"""

import sys
import os
import json
import urllib.error
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_response(data: dict, status: int = 200):
    """Create a mock HTTP response usable as a context manager."""
    resp = MagicMock()
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    resp.status = status
    resp.read = MagicMock(return_value=json.dumps(data).encode())
    return resp


def _make_http_error(code: int, msg: str = "Error") -> urllib.error.HTTPError:
    """Create a mock HTTPError with the given status code."""
    return urllib.error.HTTPError("http://fake/url", code, msg, {}, None)


class TestRetryOn400Halving:
    """Verify the retry-on-400 halving logic."""

    def test_400_then_success_retries_with_half_text(self):
        """On 400, embed retries with half the text. If second attempt succeeds, returns vector."""
        import numpy as np
        from plugins.memory.holographic.store import EmbedClient

        c = EmbedClient(
            url="http://fake:9999/v1/embeddings",
            model="test-model",
            api_key="key123",
        )
        c._alive = True

        original_text = "A" * 1000  # 1000 chars
        half_text_len = 500

        # First call: 400 error. Second call: success.
        error_resp = _make_http_error(400, "Bad Request")
        success_resp = _make_response({"data": [{"embedding": [0.1] * 4096}]})

        with patch("urllib.request.urlopen", side_effect=[error_resp, success_resp]) as mock_open:
            result = c.embed(original_text)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) == 4096

        # Verify two calls were made
        assert mock_open.call_count == 2

        # Second call should have sent half the text
        second_req = mock_open.call_args_list[1][0][0]  # second call, first positional arg
        second_body = json.loads(second_req.data.decode())
        assert len(second_body["input"]) == half_text_len, (
            f"Expected {half_text_len} chars after halving, got {len(second_body['input'])}"
        )

    def test_400_three_halvings_max(self):
        """After 3 halvings (4 attempts total), 400 gives up and returns None."""
        from plugins.memory.holographic.store import EmbedClient

        c = EmbedClient(
            url="http://fake:9999/v1/embeddings",
            model="test-model",
            api_key="key123",
        )
        c._alive = True

        # All 4 attempts return 400
        errors = [_make_http_error(400) for _ in range(4)]

        with patch("urllib.request.urlopen", side_effect=errors) as mock_open:
            result = c.embed("X" * 8000)

        assert result is None
        # 1 original + 3 halvings = 4 attempts
        assert mock_open.call_count == 4

    def test_halving_reduces_text_geometrically(self):
        """Each halving cuts text in half: 1000 → 500 → 250 → 125."""
        from plugins.memory.holographic.store import EmbedClient

        c = EmbedClient(
            url="http://fake:9999/v1/embeddings",
            model="m",
            api_key="k",
        )
        c._alive = True

        lengths_seen = []
        original = "Y" * 1000

        def track_and_fail(req, *args, **kwargs):
            body = json.loads(req.data.decode())
            lengths_seen.append(len(body["input"]))
            raise _make_http_error(400)

        with patch("urllib.request.urlopen", side_effect=track_and_fail):
            c.embed(original)

        # Expected: 1000, 500, 250, 125
        assert lengths_seen == [1000, 500, 250, 125], (
            f"Halving sequence wrong: {lengths_seen}"
        )

    def test_non_400_error_no_retry(self):
        """Non-400 HTTP errors should NOT trigger halving retries."""
        from plugins.memory.holographic.store import EmbedClient

        c = EmbedClient(
            url="http://fake:9999/v1/embeddings",
            model="m",
            api_key="k",
        )
        c._alive = True

        with patch("urllib.request.urlopen", side_effect=_make_http_error(404)) as mock_open:
            result = c.embed("test text here")

        assert result is None
        assert mock_open.call_count == 1, "Non-400 error should not retry"


class TestErrorBranching:
    """Verify HTTP error codes produce correct diagnostic branching."""

    def test_401_does_not_mark_down(self):
        """HTTP 401 (auth failure) must NOT mark server as down."""
        from plugins.memory.holographic.store import EmbedClient

        c = EmbedClient(
            url="http://fake:9999/v1/embeddings",
            api_key="bad-key",
        )
        c._alive = True
        c._alive_false_ts = 0.0

        with patch("urllib.request.urlopen", side_effect=_make_http_error(401)):
            result = c.embed("test")

        assert result is None
        assert c._alive is True, "401 must not mark server down (config bug, not server down)"
        assert c._alive_false_ts == 0.0, "401 must not set down timestamp"

    def test_403_does_not_mark_down(self):
        """HTTP 403 (forbidden) must NOT mark server as down."""
        from plugins.memory.holographic.store import EmbedClient

        c = EmbedClient(
            url="http://fake:9999/v1/embeddings",
            api_key="key",
        )
        c._alive = True
        c._alive_false_ts = 0.0

        with patch("urllib.request.urlopen", side_effect=_make_http_error(403)):
            result = c.embed("test")

        assert result is None
        assert c._alive is True

    def test_500_does_not_mark_down(self):
        """HTTP 500 (server error) must NOT mark server as down.

        HTTP errors indicate the server is reachable (it responded with
        an error code). Only connection-level failures mark it down.
        """
        from plugins.memory.holographic.store import EmbedClient

        c = EmbedClient(
            url="http://fake:9999/v1/embeddings",
            api_key="key",
        )
        c._alive = True
        c._alive_false_ts = 0.0

        with patch("urllib.request.urlopen", side_effect=_make_http_error(500)):
            result = c.embed("test")

        assert result is None
        assert c._alive is True, "HTTP 500 must not mark server down (server is reachable)"

    def test_503_does_not_mark_down(self):
        """HTTP 503 (service unavailable) must NOT mark server as down."""
        from plugins.memory.holographic.store import EmbedClient

        c = EmbedClient(
            url="http://fake:9999/v1/embeddings",
            api_key="key",
        )
        c._alive = True
        c._alive_false_ts = 0.0

        with patch("urllib.request.urlopen", side_effect=_make_http_error(503)):
            result = c.embed("test")

        assert result is None
        assert c._alive is True

    def test_connection_error_marks_down(self):
        """Connection errors MUST mark server down and record timestamp."""
        from plugins.memory.holographic.store import EmbedClient

        c = EmbedClient(
            url="http://fake:9999/v1/embeddings",
            api_key="key",
        )
        c._alive = True
        c._alive_false_ts = 0.0

        with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError("refused")):
            result = c.embed("test")

        assert result is None
        assert c._alive is False, "Connection error must mark server down"
        assert c._alive_false_ts > 0.0

    def test_timeout_marks_down(self):
        """Timeout MUST mark server down."""
        import socket
        from plugins.memory.holographic.store import EmbedClient

        c = EmbedClient(
            url="http://fake:9999/v1/embeddings",
            api_key="key",
            timeout=1,
        )
        c._alive = True
        c._alive_false_ts = 0.0

        with patch("urllib.request.urlopen", side_effect=socket.timeout("timed out")):
            result = c.embed("test")

        assert result is None
        assert c._alive is False

    def test_400_retries_then_500_returns_none(self):
        """Mixed error scenario: 400 (retry) then 500 (give up)."""
        from plugins.memory.holographic.store import EmbedClient

        c = EmbedClient(
            url="http://fake:9999/v1/embeddings",
            api_key="key",
        )
        c._alive = True

        with patch(
            "urllib.request.urlopen",
            side_effect=[_make_http_error(400), _make_http_error(500)],
        ) as mock_open:
            result = c.embed("test text here")

        assert result is None
        assert mock_open.call_count == 2  # 1 original + 1 retry
        assert c._alive is True  # 500 doesn't mark down


class TestSuccessfulEmbed:
    """Verify the success path still works after all the error handling."""

    def test_returns_vector_on_success(self):
        import numpy as np
        from plugins.memory.holographic.store import EmbedClient

        c = EmbedClient(
            url="http://fake:9999/v1/embeddings",
            model="test-model",
            api_key="key123",
        )
        c._alive = True

        fake_vec = [0.1] * 4096
        fake_resp = _make_response({"data": [{"embedding": fake_vec}]})

        with patch("urllib.request.urlopen", return_value=fake_resp):
            result = c.embed("hello world")

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == 4096

    def test_returns_none_when_server_not_alive(self):
        """If alive check fails, embed returns None immediately."""
        from plugins.memory.holographic.store import EmbedClient

        c = EmbedClient(
            url="http://fake:9999/v1/embeddings",
            api_key="key",
        )
        c._alive = False
        c._alive_false_ts = float("inf")  # never expires

        with patch("urllib.request.urlopen") as mock_open:
            result = c.embed("test")

        assert result is None
        assert mock_open.call_count == 0  # should not even try
