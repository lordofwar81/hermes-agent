#!/usr/bin/env python3
"""
Contradiction Detector - Automatically detect and resolve memory contradictions.

Workflow:
1. Search for similar existing memories (top-5, similarity > 0.6)
2. Use LLM to check for contradictions
3. Present resolution options:
   - keep_new: Mark old as "contradicted", add new
   - keep_old: Discard new
   - keep_both: Mark both as "contradicted", link them
   - discard_new: Same as keep_old

Usage:
    from tools.contradiction_detector import (
        handle_memory_with_contradiction_check,
        resolve_contradiction,
    )

    # Add with automatic contradiction checking
    result = handle_memory_with_contradiction_check(new_text, vector_memory_store)

    # Resolve contradiction
    resolve = resolve_contradiction(new_text, old_id, "keep_new", vector_memory_store)
"""

import json
import logging
import re
import sys
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning(
        "requests module not available, LLM-based contradiction detection disabled"
    )

try:
    import lancedb

    HAS_LANCEDB = True
except ImportError:
    HAS_LANCEDB = False
    logger.warning("lancedb module not available, vector similarity search disabled")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Embedding endpoint (same as adaptive_context_manager.py)
EMBED_ENDPOINT = "http://localhost:11434/v1"
EMBED_MODEL = "mxbai-embed-large-v1"

# LLM endpoint for contradiction detection (use smallest local model)
LLM_ENDPOINT = "http://localhost:8100/v1"
LLM_MODEL = "Qwen3-Coder-30B-APEX-I-Compact"
LLM_API_KEY = "notempty"

# Similarity threshold for considering memories "similar"
SIMILARITY_THRESHOLD = 0.6

# Maximum similar memories to check
TOP_K = 5

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding vector for text."""
    if not HAS_REQUESTS:
        return None

    # Truncate to avoid exceeding embedding model's token limit
    if len(text) > 800:
        text = text[:800]
        logger.debug("Truncated text for embedding (max 800 chars)")

    try:
        response = requests.post(
            f"{EMBED_ENDPOINT}/embeddings",
            headers={"Content-Type": "application/json"},
            json={"model": EMBED_MODEL, "input": text},
            timeout=30,
        )
        if response.status_code == 200:
            result = response.json()
            return result["data"][0]["embedding"]
        else:
            logger.warning(
                f"Embedding request failed: {response.status_code} - {response.text}"
            )
            return None
    except Exception as e:
        logger.warning(f"Embedding exception: {e}")
        return None


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def find_similar_memories(
    text: str,
    vector_store,
    threshold: float = SIMILARITY_THRESHOLD,
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    """
    Find memories similar to the given text using vector similarity.

    Args:
        text: Text to compare
        vector_store: LanceDB table object
        threshold: Minimum similarity score (0.0-1.0)
        top_k: Maximum number of results

    Returns:
        List of memory dicts with similarity scores
    """
    if not HAS_LANCEDB or vector_store is None:
        return []

    embedding = get_embedding(text)
    if not embedding:
        return []

    try:
        # Perform vector search
        results = vector_store.search(embedding).limit(top_k).to_pandas()

        similar = []
        for _, row in results.iterrows():
            # Compute similarity (LanceDB search returns sorted by distance)
            # For now, assume results are already ranked by similarity
            # We'll compute actual cosine similarity
            vec = row.get("vector")
            if vec is not None:
                sim = cosine_similarity(
                    embedding, vec.tolist() if hasattr(vec, "tolist") else vec
                )
                if sim >= threshold:
                    similar.append(
                        {
                            "id": row.get("id", ""),
                            "text": row.get("text", ""),
                            "similarity": sim,
                            "epistemic_status": row.get("epistemic_status", "stated"),
                            "confidence": row.get("confidence", 0.5),
                        }
                    )

        return similar
    except Exception as e:
        logger.warning(f"Vector search failed: {e}")
        return []


def detect_contradiction_llm(
    new_text: str,
    old_text: str,
    llm_endpoint: str = LLM_ENDPOINT,
    llm_model: str = LLM_MODEL,
) -> Dict[str, Any]:
    """
    Use LLM to detect contradiction between two statements.

    Returns dict with keys:
        - is_contradiction: bool
        - confidence: float (0.0-1.0)
        - reasoning: str
        - resolution_suggestion: one of "keep_new", "keep_old", "keep_both", "discard_new"
    """
    if not HAS_REQUESTS:
        return {
            "is_contradiction": False,
            "confidence": 0.0,
            "reasoning": "LLM detection not available",
            "resolution_suggestion": "keep_new",
        }

    prompt = f"""You are a contradiction detection system. Compare these two statements:

Statement A: "{old_text}"

Statement B: "{new_text}"

Answer the following questions:
1. Do these statements contradict each other? (yes/no)
2. How confident are you? (0.0-1.0)
3. Brief reasoning (1-2 sentences)
4. If they contradict, which statement is more likely correct? (A/B/both/neither)

Format your response exactly as:
CONTRADICTION: yes/no
CONFIDENCE: 0.0-1.0
REASONING: <reasoning>
SUGGESTION: A/B/both/neither
"""

    try:
        response = requests.post(
            f"{llm_endpoint}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LLM_API_KEY}",
            },
            json={
                "model": llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
                "temperature": 0.1,
                "stream": False,
            },
            timeout=60,
        )

        if response.status_code != 200:
            logger.warning(
                f"LLM request failed: {response.status_code} - {response.text}"
            )
            return {
                "is_contradiction": False,
                "confidence": 0.0,
                "reasoning": f"LLM request failed: {response.status_code}",
                "resolution_suggestion": "keep_new",
            }

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # Parse response
        contradiction = False
        confidence = 0.5
        reasoning = ""
        suggestion = "keep_new"

        lines = content.strip().split("\n")
        for line in lines:
            line_lower = line.lower()
            if line_lower.startswith("contradiction:"):
                if "yes" in line_lower:
                    contradiction = True
            elif line_lower.startswith("confidence:"):
                try:
                    confidence = float(re.search(r"[\d.]+", line).group())
                except:
                    pass
            elif line_lower.startswith("reasoning:"):
                reasoning = line[10:].strip()
            elif line_lower.startswith("suggestion:"):
                if "a" in line_lower:
                    suggestion = "keep_old"
                elif "b" in line_lower:
                    suggestion = "keep_new"
                elif "both" in line_lower:
                    suggestion = "keep_both"
                elif "neither" in line_lower:
                    suggestion = "discard_new"

        return {
            "is_contradiction": contradiction,
            "confidence": confidence,
            "reasoning": reasoning,
            "resolution_suggestion": suggestion,
        }

    except Exception as e:
        logger.warning(f"LLM contradiction detection failed: {e}")
        return {
            "is_contradiction": False,
            "confidence": 0.0,
            "reasoning": f"Exception: {e}",
            "resolution_suggestion": "keep_new",
        }


# -----------------------------------------------------------------------------
# Main API
# -----------------------------------------------------------------------------


def handle_memory_with_contradiction_check(
    new_text: str,
    vector_store,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """
    Add a new memory with automatic contradiction checking.

    Returns dict with keys:
        - added: bool (whether memory was added)
        - contradiction_found: bool
        - similar_memories: list of similar memory dicts
        - contradiction_analysis: dict from detect_contradiction_llm (if any)
        - resolution: str (action taken)
        - new_memory_id: str (if added)
    """
    result = {
        "added": False,
        "contradiction_found": False,
        "similar_memories": [],
        "contradiction_analysis": None,
        "resolution": "none",
        "new_memory_id": None,
    }

    # Find similar memories
    similar = find_similar_memories(new_text, vector_store, similarity_threshold)
    result["similar_memories"] = similar

    if not similar:
        # No similar memories, add without contradiction check
        # TODO: Actually add to vector store
        result["added"] = True
        result["resolution"] = "keep_new"
        return result

    # Check for contradictions with each similar memory
    contradictions = []
    for old_memory in similar:
        if use_llm:
            analysis = detect_contradiction_llm(new_text, old_memory["text"])
        else:
            # Simple keyword-based contradiction detection (placeholder)
            analysis = {
                "is_contradiction": False,
                "confidence": 0.0,
                "reasoning": "LLM detection disabled",
                "resolution_suggestion": "keep_new",
            }

        if analysis["is_contradiction"]:
            contradictions.append(
                {
                    "old_memory": old_memory,
                    "analysis": analysis,
                }
            )

    if not contradictions:
        # No contradictions found, add new memory
        # TODO: Actually add to vector store
        result["added"] = True
        result["resolution"] = "keep_new"
        return result

    # Found contradictions - use the first one for resolution
    result["contradiction_found"] = True
    result["contradiction_analysis"] = contradictions[0]["analysis"]

    suggestion = contradictions[0]["analysis"]["resolution_suggestion"]
    result["resolution"] = suggestion

    # Apply resolution
    if suggestion == "keep_new":
        # Mark old memory as contradicted
        old_id = contradictions[0]["old_memory"]["id"]
        # TODO: Update old memory's epistemic_status to "contradicted"
        # TODO: Add new memory
        result["added"] = True
    elif suggestion == "keep_old":
        # Discard new memory
        result["added"] = False
    elif suggestion == "keep_both":
        # Mark both as contradicted
        old_id = contradictions[0]["old_memory"]["id"]
        # TODO: Update old memory's epistemic_status to "contradicted"
        # TODO: Add new memory with epistemic_status "contradicted"
        result["added"] = True
    else:  # discard_new
        result["added"] = False

    return result


def resolve_contradiction(
    new_text: str,
    old_memory_id: str,
    resolution: str,
    vector_store,
) -> Dict[str, Any]:
    """
    Manually resolve a contradiction with specified resolution.

    Args:
        new_text: New memory text
        old_memory_id: ID of existing memory
        resolution: "keep_new", "keep_old", "keep_both", "discard_new"
        vector_store: LanceDB table object

    Returns:
        Dict with result of resolution
    """
    # TODO: Implement actual resolution logic
    # For now, return placeholder
    return {
        "success": False,
        "error": "Not yet implemented",
        "resolution": resolution,
        "old_memory_id": old_memory_id,
        "new_text": new_text,
    }


# -----------------------------------------------------------------------------
# Integration with VectorMemoryStore (monkey-patch)
# -----------------------------------------------------------------------------


def add_epistemic_methods_to_class(cls):
    """
    Monkey-patch a VectorMemoryStore class to add epistemic methods.

    Adds methods:
        - verify(memory_id, content)
        - contradict(memory_id, old_text)
        - retract(memory_id, old_text)
    """

    def verify(self, memory_id, content):
        """Verify a memory (upgrade epistemic status to 'verified')."""
        # TODO: Implement
        return {"success": False, "error": "Not implemented"}

    def contradict(self, memory_id, old_text):
        """Mark a memory as contradicted."""
        # TODO: Implement
        return {"success": False, "error": "Not implemented"}

    def retract(self, memory_id, old_text):
        """Retract a contradicted memory."""
        # TODO: Implement
        return {"success": False, "error": "Not implemented"}

    cls.verify = verify
    cls.contradict = contradict
    cls.retract = retract

    return cls


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Contradiction detection test")
    parser.add_argument("text", help="New memory text")
    parser.add_argument("--old", help="Old memory text to compare")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM detection")
    args = parser.parse_args()

    if args.old:
        # Direct comparison mode
        analysis = detect_contradiction_llm(args.text, args.old)
        print("Contradiction analysis:")
        print(f"  Contradiction: {analysis['is_contradiction']}")
        print(f"  Confidence: {analysis['confidence']}")
        print(f"  Reasoning: {analysis['reasoning']}")
        print(f"  Suggestion: {analysis['resolution_suggestion']}")
    else:
        # TODO: Connect to vector store and run full check
        print("Full contradiction check requires vector store connection")
        print("Use --old to compare two texts directly")
