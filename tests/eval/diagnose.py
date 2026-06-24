#!/usr/bin/env python3
"""Diagnostic: show per-category cosine similarities for the eval cases.

Helps tune the semantic centroids by revealing WHY each message classifies the
way it does — which categories are close, where the confusion clusters.
"""
from __future__ import annotations
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml
from agent.routing import TaskClassifier, Category


def main():
    evalset = Path(__file__).parent / "evalset.yaml"
    cases = yaml.safe_load(evalset.read_text())["cases"]

    centroids = TaskClassifier._compute_centroids()
    if centroids is None:
        print("ERROR: embedding server unavailable"); sys.exit(2)

    from tools.vector_memory import get_embedding

    print(f"\n{'input':<55} {'expect':<10} {'top-3 categories (cosine sim)'}")
    print("-" * 110)
    for c in cases:
        msg = c["input"]
        expected = c["expected_category"]
        vec = get_embedding(msg)
        if vec is None:
            print(f"{msg[:54]:<55} {expected:<10} EMBED_FAILED")
            continue
        sims = sorted(
            ((cat.value, TaskClassifier._cosine_similarity(vec, cent)) for cat, cent in centroids.items()),
            key=lambda x: -x[1],
        )
        top3 = "  ".join(f"{n}={s:.3f}" for n, s in sims[:3])
        predicted = TaskClassifier.classify_semantic(msg)
        mark = "✓" if predicted and predicted.value == expected else "✗"
        print(f"{mark} {msg[:53]:<55} {expected:<10} {top3}")

if __name__ == "__main__":
    main()
