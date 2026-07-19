"""Mock-HTTP tests for EmbedClient — kills the 56 mutation survivors in embed/_probe.

These test the HTTP paths (payload construction, auth header, response parsing,
error handling) by mocking urllib.request.urlopen. The existing regression tests
only cover the alive-TTL logic, not the actual HTTP call paths.

Run: venv/bin/python -m pytest tests/test_embed_http.py -v
"""
import sys
import os
import json
import time
import urllib.error
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_response(data: dict, status: int = 200):
    """Create a mock HTTP response object usable as a context manager."""
    resp = MagicMock()
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    resp.status = status
    resp.read = MagicMock(return_value=json.dumps(data).encode())
    return resp


def test_embed_returns_vector_on_success():
    """embed() must parse the response and return a float32 numpy vector."""
    import numpy as np
    from plugins.memory.holographic.store import EmbedClient
    c = EmbedClient(url="http://fake:9999/v1/embeddings", model="test-model", api_key="key123")
    c._alive = True  # skip the alive check
    fake_vec = [0.1] * 4096
    fake_resp = _make_response({"data": [{"embedding": fake_vec}]})
    with patch("urllib.request.urlopen", return_value=fake_resp):
        result = c.embed("hello world")
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert len(result) == 4096
    assert result[0] == 0.1


def test_embed_sends_correct_payload():
    """embed() must send the text + model in the JSON payload (kills payload mutants)."""
    from plugins.memory.holographic.store import EmbedClient
    c = EmbedClient(url="http://fake:9999/v1/embeddings", model="test-model", api_key="key123")
    c._alive = True
    fake_resp = _make_response({"data": [{"embedding": [0.1] * 4096}]})
    with patch("urllib.request.urlopen", return_value=fake_resp) as mock_urlopen:
        c.embed("test text payload")
    # Verify the request was constructed correctly
    call_args = mock_urlopen.call_args
    req = call_args[0][0]  # first positional arg = the Request object
    body = json.loads(req.data.decode())
    assert body["input"] == "test text payload", "payload must include the input text"
    assert body["model"] == "test-model", "payload must include the model name"


def test_embed_sends_auth_header():
    """embed() must send the Authorization header with the API key (kills header mutants)."""
    from plugins.memory.holographic.store import EmbedClient
    c = EmbedClient(url="http://fake:9999/v1/embeddings", model="m", api_key="secret-key-456")
    c._alive = True
    fake_resp = _make_response({"data": [{"embedding": [0.1] * 4096}]})
    with patch("urllib.request.urlopen", return_value=fake_resp) as mock_urlopen:
        c.embed("auth test")
    req = mock_urlopen.call_args[0][0]
    assert req.has_header("Authorization"), "must send Authorization header"
    auth_val = req.get_header("Authorization")
    assert "secret-key-456" in auth_val, "auth header must contain the API key"


def test_embed_returns_none_on_http_401():
    """embed() must return None on HTTP 401 WITHOUT marking server down (audit H-2 split)."""
    from plugins.memory.holographic.store import EmbedClient
    c = EmbedClient(url="http://fake:9999/v1/embeddings", api_key="bad-key")
    c._alive = True
    c._alive_false_ts = 0.0
    with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError("url", 401, "Unauthorized", {}, None)):
        result = c.embed("test")
    assert result is None, "401 should return None"
    # KEY ASSERTION: 401 is a config bug, NOT a down server — alive must NOT flip to False
    assert c._alive is True, "401 must not mark server as down (audit H-2 HTTPError split)"


def test_embed_marks_down_on_connection_error():
    """embed() must return None AND mark server down on connection errors (not HTTP errors)."""
    from plugins.memory.holographic.store import EmbedClient
    c = EmbedClient(url="http://fake:9999/v1/embeddings", api_key="key")
    c._alive = True
    c._alive_false_ts = 0.0
    with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError("connection refused")):
        result = c.embed("test")
    assert result is None, "connection error should return None"
    assert c._alive is False, "connection error must mark server down"
    assert c._alive_false_ts > 0, "must record the failure timestamp for TTL"


def test_embed_marks_down_on_timeout():
    """embed() must return None AND mark server down on timeout."""
    from plugins.memory.holographic.store import EmbedClient
    import socket
    c = EmbedClient(url="http://fake:9999/v1/embeddings", api_key="key", timeout=1)
    c._alive = True
    c._alive_false_ts = 0.0
    with patch("urllib.request.urlopen", side_effect=socket.timeout("timed out")):
        result = c.embed("test")
    assert result is None
    assert c._alive is False, "timeout must mark server down"


def test_probe_returns_true_on_200():
    """_probe() must return True when the server responds successfully."""
    from plugins.memory.holographic.store import EmbedClient
    c = EmbedClient(url="http://fake:9999/v1/embeddings", api_key="key")
    fake_resp = _make_response({"data": []}, status=200)
    with patch("urllib.request.urlopen", return_value=fake_resp):
        assert c._probe() is True


def test_probe_returns_false_on_connection_error():
    """_probe() must return False on connection error (server unreachable)."""
    from plugins.memory.holographic.store import EmbedClient
    c = EmbedClient(url="http://fake:9999/v1/embeddings", api_key="key")
    with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError("refused")):
        assert c._probe() is False


def test_probe_sends_auth_header():
    """_probe() must send the auth header (kills the probe-header-removal mutant)."""
    from plugins.memory.holographic.store import EmbedClient
    c = EmbedClient(url="http://fake:9999/v1/embeddings", api_key="probe-key")
    fake_resp = _make_response({"data": []})
    with patch("urllib.request.urlopen", return_value=fake_resp) as mock_urlopen:
        c._probe()
    req = mock_urlopen.call_args[0][0]
    assert req.has_header("Authorization"), "probe must send auth header"
    assert "probe-key" in req.get_header("Authorization")


def test_probe_hits_models_endpoint():
    """_probe() must hit the /models endpoint (not /embeddings) — kills URL mutants."""
    from plugins.memory.holographic.store import EmbedClient
    c = EmbedClient(url="http://fake:9999/v1/embeddings", api_key="key")
    fake_resp = _make_response({"data": []})
    with patch("urllib.request.urlopen", return_value=fake_resp) as mock_urlopen:
        c._probe()
    req = mock_urlopen.call_args[0][0]
    # The probe replaces /embeddings with /models
    assert "/models" in req.full_url, f"probe must hit /models, got {req.full_url}"
    assert "/embeddings" not in req.full_url, "probe must NOT hit /embeddings"


def test_embed_batch_returns_list():
    """embed_batch() must return a list of results (one per text)."""
    import numpy as np
    from plugins.memory.holographic.store import EmbedClient
    c = EmbedClient(url="http://fake:9999/v1/embeddings", api_key="key")
    c._alive = True
    fake_resp = _make_response({"data": [{"embedding": [0.1] * 4096}]})
    with patch("urllib.request.urlopen", return_value=fake_resp):
        results = c.embed_batch(["text1", "text2", "text3"])
    assert isinstance(results, list)
    assert len(results) == 3
    assert all(r is not None for r in results)
