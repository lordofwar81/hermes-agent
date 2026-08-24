"""A+ Leap 3: embed pipeline round-trip regression (2026-08-23).

Guards the bug class that lived for weeks undetected: silently-dead or
misconfigured embeddings (wrong endpoint, auth failure, zero vectors) while
the memory system kept writing. Runs against the live local embed server
(:11434, qwen3-embed-8b); skips when unreachable so offline runs stay green.
"""

import os
import urllib.request

import pytest

from plugins.memory.holographic.store import EmbedClient

EMBED_URL = os.environ.get("EMBED_SERVER_URL", "http://localhost:11434/v1/embeddings")


def _server_up() -> bool:
    try:
        with urllib.request.urlopen(
            EMBED_URL.replace("/embeddings", "/models"), timeout=2
        ):
            return True
    except Exception:
        return False


# The server requires auth for /embeddings (the /models probe is keyless —
# liveness alone is not proof embeds will work). Production injects
# EMBED_SERVER_KEY via systemd env; tests read it (or LLAMA_API_KEY) from
# the environment — source /etc/default/llama-cluster before running.
EMBED_KEY = os.environ.get("EMBED_SERVER_KEY") or os.environ.get("LLAMA_API_KEY", "")

pytestmark = pytest.mark.skipif(
    not _server_up() or not EMBED_KEY,
    reason="local embed server :11434 not reachable or no embed API key in env",
)


@pytest.fixture
def client():
    return EmbedClient(url=EMBED_URL, api_key=EMBED_KEY, timeout=10)


def _cos(u, v) -> float:
    import numpy as np

    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


class TestEmbedPipelineRoundTrip:
    def test_embed_returns_nonzero_vector(self, client):
        vec = client.embed("BPC-157 peptide dose tracking")
        assert vec is not None
        assert vec.shape[0] > 0
        assert float(abs(vec).sum()) > 0.0

    def test_similar_texts_closer_than_distinct(self, client):
        a = client.embed("morning resting heart rate trend")
        b = client.embed("resting heart rate this morning")
        c = client.embed("kubernetes cluster upgrade procedure")
        assert a is not None and b is not None and c is not None
        assert _cos(a, b) > _cos(a, c)

    def test_embed_batch_matches_single(self, client):
        """Batch and single embeddings must be equivalent — NOT bit-identical:
        llama-server embeds vary ~1e-3 between calls (float/batching
        nondeterminism), so the invariant is cosine equivalence."""
        texts = ["alpha fact about the fleet", "beta fact about the router"]
        batch = client.embed_batch(texts)
        single = [client.embed(t) for t in texts]
        assert all(v is not None for v in batch)
        assert all(v is not None for v in single)
        for vb, vs in zip(batch, single):
            assert _cos(vb, vs) > 0.999


class TestEmbedClientResilience:
    """Offline behavior — these run even when the server is down."""

    def test_dead_endpoint_returns_none_not_crash(self):
        dead = EmbedClient(url="http://127.0.0.1:1/v1/embeddings", timeout=1)
        dead._alive = False
        dead._alive_false_ts = dead._time.time()  # inside TTL — no probe
        assert dead.embed("anything") is None

    def test_alive_false_expires_and_reprobes(self):
        dead = EmbedClient(url="http://127.0.0.1:1/v1/embeddings", timeout=1)
        dead._alive = False
        dead._alive_false_ts = dead._time.time() - 120  # TTL expired
        assert dead.alive is False  # re-probed (and still dead — probe ran)
        assert dead._alive_false_ts > 0
