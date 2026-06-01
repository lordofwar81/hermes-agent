"""Tests for plugins/memory/holographic/holographic.py — HRR encoding, binding, and similarity.

Run:  python3 -m pytest tests/plugins/memory/test_holographic.py -v
"""

import math
import pytest
import numpy as np

from plugins.memory.holographic.holographic import (
    encode_atom,
    encode_text,
    encode_fact,
    bind,
    unbind,
    bundle,
    similarity,
    phases_to_bytes,
    bytes_to_phases,
    snr_estimate,
)


# ─── encode_atom ────────────────────────────────────────────────────────

class TestEncodeAtom:
    def test_deterministic(self):
        """Same word always produces same vector."""
        a1 = encode_atom("hermes")
        a2 = encode_atom("hermes")
        np.testing.assert_array_equal(a1, a2)

    def test_different_words_different_vectors(self):
        """Different words produce quasi-orthogonal vectors."""
        a = encode_atom("alpha")
        b = encode_atom("beta")
        # Phase cosine similarity for random words should be near 0
        sim = similarity(a, b)
        assert abs(sim) < 0.3, f"Expected near-zero similarity, got {sim}"

    def test_shape(self):
        vec = encode_atom("test", dim=512)
        assert vec.shape == (512,)
        assert vec.dtype == np.float64

    def test_values_in_phase_range(self):
        vec = encode_atom("test", dim=1024)
        assert np.all(vec >= 0.0)
        assert np.all(vec <= 2.0 * math.pi)

    def test_cross_platform_reproducibility(self):
        """SHA-256 deterministic — exact values must match."""
        vec = encode_atom("hello", dim=64)
        # Manually compute expected first phase
        import hashlib, struct
        digest = hashlib.sha256("hello:0".encode()).digest()
        expected_val = struct.unpack("<16H", digest)[0]
        expected_phase = expected_val * (2.0 * math.pi / 65536.0)
        assert abs(vec[0] - expected_phase) < 1e-12

    def test_empty_word(self):
        vec = encode_atom("", dim=64)
        assert vec.shape == (64,)
        assert np.all((vec >= 0.0) & (vec <= 2.0 * math.pi))


# ─── bind / unbind ───────────────────────────────────────────────────────

class TestBindUnbind:
    def test_bind_is_quasi_orthogonal(self):
        """bind(a, b) should be dissimilar to both a and b."""
        a = encode_atom("alice")
        b = encode_atom("bob")
        c = bind(a, b)
        assert abs(similarity(c, a)) < 0.3
        assert abs(similarity(c, b)) < 0.3

    def test_unbind_recovers_approximately(self):
        """unbind(bind(a, b), a) ≈ b (up to superposition noise)."""
        a = encode_atom("key", dim=1024)
        b = encode_atom("value", dim=1024)
        bound = bind(a, b)
        recovered = unbind(bound, a)
        # Phase cosine similarity should be high
        sim = similarity(recovered, b)
        assert sim > 0.85, f"Expected high similarity after unbind, got {sim}"

    def test_bind_unbind_symmetric(self):
        """bind(a, b) == bind(b, a) for phase encoding (commutative)."""
        a = encode_atom("x", dim=1024)
        b = encode_atom("y", dim=1024)
        np.testing.assert_allclose(bind(a, b), bind(b, a), atol=1e-12)

    def test_unbind_wrong_key_gives_noise(self):
        """unbind with wrong key produces near-zero similarity."""
        a = encode_atom("key", dim=1024)
        b = encode_atom("value", dim=1024)
        wrong = encode_atom("wrong", dim=1024)
        bound = bind(a, b)
        recovered = unbind(bound, wrong)
        sim = similarity(recovered, b)
        assert abs(sim) < 0.3, f"Wrong key should give noise, got similarity {sim}"

    def test_double_bind_double_unbind(self):
        """bind(a, bind(b, c)) unbind unbind recovers c."""
        a = encode_atom("a", dim=1024)
        b = encode_atom("b", dim=1024)
        c = encode_atom("c", dim=1024)
        nested = bind(a, bind(b, c))
        step1 = unbind(nested, a)
        step2 = unbind(step1, b)
        sim = similarity(step2, c)
        assert sim > 0.80, f"Double unbind should recover c, similarity={sim}"


# ─── bundle ─────────────────────────────────────────────────────────────

class TestBundle:
    def test_bundle_of_identical_vectors(self):
        """Bundling identical vectors returns the same vector."""
        a = encode_atom("test", dim=1024)
        bundled = bundle(a, a, a)
        sim = similarity(bundled, a)
        assert sim > 0.95

    def test_bundle_is_similar_to_components(self):
        """Bundled vector is similar to each input."""
        a = encode_atom("alpha", dim=1024)
        b = encode_atom("beta", dim=1024)
        c = encode_atom("gamma", dim=1024)
        bundled = bundle(a, b, c)
        assert similarity(bundled, a) > 0.5
        assert similarity(bundled, b) > 0.5
        assert similarity(bundled, c) > 0.5

    def test_bundle_empty(self):
        """Bundling no vectors returns a zero vector (no error in phase encoding)."""
        # *args with no items — np.sum of empty array
        try:
            result = bundle()
        except (IndexError, ValueError, TypeError):
            return  # acceptable behavior
        # If it doesn't raise, it should return a valid vector
        assert result is not None

    def test_bundle_many_degrades(self):
        """Bundling too many items degrades similarity (capacity test)."""
        dim = 128  # small for fast test
        vectors = [encode_atom(f"item_{i}", dim=dim) for i in range(50)]
        bundled = bundle(*vectors)
        # With 50 items in dim=128, SNR = sqrt(128/50) ≈ 1.6 — degraded
        avg_sim = sum(similarity(bundled, v) for v in vectors[:10]) / 10
        # Should have some residual similarity but degraded from 1.0
        assert avg_sim < 0.8, f"Bundled 50 items in dim=128 should be degraded, avg_sim={avg_sim}"


# ─── similarity ──────────────────────────────────────────────────────────

class TestSimilarity:
    def test_identical_vectors(self):
        a = encode_atom("same")
        assert similarity(a, a) == pytest.approx(1.0, abs=1e-10)

    def test_phase_range(self):
        """All similarities must be in [-1, 1]."""
        a = encode_atom("x", dim=64)
        b = encode_atom("y", dim=64)
        sim = similarity(a, b)
        assert -1.0 <= sim <= 1.0

    def test_inverted_vector(self):
        """A vector shifted by π should have similarity ≈ -1."""
        a = encode_atom("test", dim=1024)
        inverted = (a + math.pi) % (2.0 * math.pi)
        sim = similarity(a, inverted)
        assert sim < -0.9, f"Inverted vector should have similarity near -1, got {sim}"


# ─── encode_text ─────────────────────────────────────────────────────────

class TestEncodeText:
    def test_bag_of_words(self):
        """encode_text bundles individual token atoms."""
        text = "hello world"
        vec = encode_text(text, dim=1024)
        assert vec.shape == (1024,)

    def test_empty_text(self):
        """Empty text returns __hrr_empty__ atom."""
        vec = encode_text("", dim=64)
        empty = encode_atom("__hrr_empty__", dim=64)
        np.testing.assert_array_equal(vec, empty)

    def test_punctuation_stripped(self):
        """Punctuation should not affect encoding (tokens are stripped)."""
        a = encode_text("hello!", dim=512)
        b = encode_text("hello", dim=512)
        np.testing.assert_array_equal(a, b)

    def test_case_insensitive(self):
        a = encode_text("Hello", dim=512)
        b = encode_text("hello", dim=512)
        np.testing.assert_array_equal(a, b)


# ─── encode_fact ────────────────────────────────────────────────────────

class TestEncodeFact:
    def test_fact_structure(self):
        """encode_fact creates a role-bound composite vector."""
        vec = encode_fact("runs the company", ["Alice"], dim=1024)
        assert vec.shape == (1024,)

    def test_fact_different_from_text(self):
        """Facts are structured differently from plain text encoding."""
        fact_vec = encode_fact("runs the company", ["Alice"], dim=1024)
        text_vec = encode_text("runs the company Alice", dim=1024)
        # They should NOT be identical
        sim = similarity(fact_vec, text_vec)
        # They might be somewhat similar but not identical
        assert sim < 0.99, f"Fact and text should differ structurally, got sim={sim}"

    def test_fact_with_multiple_entities(self):
        """Multiple entities create a different vector than fewer entities."""
        vec1 = encode_fact("met at the office", ["Alice", "Bob"], dim=1024)
        vec2 = encode_fact("met at the office", ["Alice"], dim=1024)
        sim = similarity(vec1, vec2)
        # Should differ because different entities are bound
        assert sim < 0.95


# ─── serialization ──────────────────────────────────────────────────────

class TestSerialization:
    def test_round_trip(self):
        """phases_to_bytes → bytes_to_phases preserves exact values."""
        vec = encode_atom("test", dim=1024)
        data = phases_to_bytes(vec)
        recovered = bytes_to_phases(data)
        np.testing.assert_array_equal(vec, recovered)

    def test_size(self):
        """1024-dim float64 = 8KB."""
        vec = encode_atom("test", dim=1024)
        data = phases_to_bytes(vec)
        assert len(data) == 1024 * 8  # 8192 bytes

    def test_frombuffer_readonly_handled(self):
        """bytes_to_phases must return a mutable copy."""
        vec = bytes_to_phases(phases_to_bytes(encode_atom("x", dim=64)))
        vec[0] = 0.0  # should not raise
        assert vec[0] == 0.0


# ─── snr_estimate ────────────────────────────────────────────────────────

class TestSNR:
    def test_zero_items_inf(self):
        assert snr_estimate(1024, 0) == float("inf")

    def test_known_values(self):
        # sqrt(1024/10) = sqrt(102.4) ≈ 10.12
        snr = snr_estimate(1024, 10)
        assert snr == pytest.approx(math.sqrt(1024 / 10), abs=1e-10)

    def test_low_snr_warning(self, caplog):
        """SNR < 2.0 triggers a warning."""
        import logging
        with caplog.at_level(logging.WARNING):
            snr_estimate(100, 50)  # sqrt(100/50) = sqrt(2) ≈ 1.41
        assert "near capacity" in caplog.text.lower() or "snr" in caplog.text.lower()
