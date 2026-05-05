#!/usr/bin/env python3
"""
Entity Extractor - Named entity recognition for memory indexing.

Provides pattern-based entity extraction with optional LLM fallback.
Used to populate the 'entities' field in vector memory records.

Entity Types:
- person: Names, aliases (e.g., "John Doe", "Alice")
- project: Project names, repos, tools (e.g., "Hermes-Agent", "Project X")
- tech: Languages, frameworks, libraries (e.g., "Python", "React", "PostgreSQL")
- concept: Ideas, patterns, methodologies (e.g., "machine learning", "REST API")
- location: Places, paths, URLs (e.g., "New York", "/home/user", "https://...")
- org: Companies, teams, organizations (e.g., "Google", "Engineering Team")

Usage:
    from tools.entity_extractor import extract_entities

    # Pattern-based (fast)
    entities = extract_entities("I'm working on Python with John Doe")

    # LLM-based (accurate, requires LLM client)
    entities = extract_entities("I'm working on Python with John Doe", use_llm=True)
"""

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Regex patterns
# -----------------------------------------------------------------------------

# Capitalized multi-word phrases (e.g., "John Doe", "New York")
_RE_CAPITALIZED = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

# Double-quoted terms (e.g., "Python", "Project X")
_RE_DOUBLE_QUOTE = re.compile(r'"([^"]+)"')

# Single-quoted terms (e.g., 'pytest', 'API')
_RE_SINGLE_QUOTE = re.compile(r"'([^']+)'")

# AKA patterns (e.g., "Guido aka BDFL")
_RE_AKA = re.compile(
    r"(\w+(?:\s+\w+)*)\s+(?:aka|also known as)\s+(\w+(?:\s+\w+)*)", re.IGNORECASE
)

# GitHub-style references (e.g., "user/repo", "org/project")
_RE_GITHUB_REF = re.compile(r"\b([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)\b")

# URLs
_RE_URL = re.compile(r"\b(https?://[^\s]+)\b")

# File paths (Unix and Windows)
_RE_FILE_PATH = re.compile(r"\b(?:/|[A-Za-z]:\\)[^\s]+\b")

# Tech keywords (common languages, frameworks, tools)
_TECH_KEYWORDS = {
    # Languages
    "python",
    "javascript",
    "typescript",
    "java",
    "c++",
    "c#",
    "go",
    "rust",
    "ruby",
    "php",
    "swift",
    "kotlin",
    "scala",
    "haskell",
    "elixir",
    "clojure",
    "perl",
    # Frameworks & Libraries
    "react",
    "vue",
    "angular",
    "node.js",
    "express",
    "django",
    "flask",
    "fastapi",
    "spring",
    "rails",
    "laravel",
    "symfony",
    "tensorflow",
    "pytorch",
    "keras",
    "pandas",
    "numpy",
    "scikit-learn",
    "docker",
    "kubernetes",
    "terraform",
    "ansible",
    "jenkins",
    "git",
    "postgresql",
    "mysql",
    "mongodb",
    "redis",
    "elasticsearch",
    "kafka",
    "rabbitmq",
    "graphql",
    "rest",
    "grpc",
    # Tools & Platforms
    "linux",
    "windows",
    "macos",
    "aws",
    "azure",
    "gcp",
    "github",
    "gitlab",
    "jira",
    "slack",
    "vscode",
    "vim",
    "emacs",
    "intellij",
    "pycharm",
}

# Organization keywords (common company names, team names)
_ORG_KEYWORDS = {
    "google",
    "microsoft",
    "apple",
    "amazon",
    "facebook",
    "twitter",
    "netflix",
    "uber",
    "airbnb",
    "spotify",
    "github",
    "gitlab",
    "docker",
    "hashicorp",
    "redhat",
    "canonical",
    "apache",
    "mozilla",
    "gnu",
    "linux foundation",
}

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def _is_tech_term(term: str) -> bool:
    """Check if a term is a known technology keyword."""
    term_lower = term.lower()
    # Direct match
    if term_lower in _TECH_KEYWORDS:
        return True
    # Suffix matches (e.g., "Pythonic" → "python")
    for keyword in _TECH_KEYWORDS:
        if term_lower.startswith(keyword) or term_lower.endswith(keyword):
            return True
    return False


def _is_org_term(term: str) -> bool:
    """Check if a term is a known organization keyword."""
    term_lower = term.lower()
    if term_lower in _ORG_KEYWORDS:
        return True
    # Common suffixes
    if (
        term_lower.endswith(" inc")
        or term_lower.endswith(" corp")
        or term_lower.endswith(" ltd")
    ):
        return True
    return False


def _classify_entity(name: str) -> str:
    """Classify entity type based on name patterns."""
    name.lower()

    # Check for tech terms
    if _is_tech_term(name):
        return "tech"

    # Check for org terms
    if _is_org_term(name):
        return "org"

    # GitHub-style references are projects
    if _RE_GITHUB_REF.match(name):
        return "project"

    # URLs are locations
    if _RE_URL.match(name):
        return "location"

    # File paths are locations
    if _RE_FILE_PATH.match(name):
        return "location"

    # Person names typically 2-3 words, all capitalized
    words = name.split()
    if 2 <= len(words) <= 3 and all(w[0].isupper() for w in words):
        return "person"

    # Default to concept for abstract terms
    return "concept"


# -----------------------------------------------------------------------------
# Main extraction function
# -----------------------------------------------------------------------------


def extract_entities(text: str, use_llm: bool = False) -> List[Dict[str, Any]]:
    """
    Extract named entities from text.

    Args:
        text: Input text to analyze
        use_llm: Whether to use LLM for more accurate extraction (not implemented yet)

    Returns:
        List of entity dicts with keys: 'name', 'type', 'source'
    """
    if not text or not isinstance(text, str):
        return []

    seen = set()
    entities = []

    def _add(name: str, source: str):
        name = name.strip()
        if not name:
            return
        # Normalize: lowercase for deduplication
        key = name.lower()
        if key in seen:
            return
        seen.add(key)

        # Classify entity type
        entity_type = _classify_entity(name)

        entities.append(
            {
                "name": name,
                "type": entity_type,
                "source": source,
            }
        )

    # Extract capitalized multi-word phrases
    for match in _RE_CAPITALIZED.finditer(text):
        _add(match.group(1), "capitalized")

    # Extract double-quoted terms
    for match in _RE_DOUBLE_QUOTE.finditer(text):
        _add(match.group(1), "double_quote")

    # Extract single-quoted terms
    for match in _RE_SINGLE_QUOTE.finditer(text):
        _add(match.group(1), "single_quote")

    # Extract AKA patterns (both sides)
    for match in _RE_AKA.finditer(text):
        _add(match.group(1), "aka")
        _add(match.group(2), "aka")

    # Extract GitHub references
    for match in _RE_GITHUB_REF.finditer(text):
        _add(match.group(1), "github_ref")

    # Extract URLs
    for match in _RE_URL.finditer(text):
        _add(match.group(1), "url")

    # Extract file paths
    for match in _RE_FILE_PATH.finditer(text):
        _add(match.group(1), "file_path")

    # LLM fallback
    if use_llm and not entities:
        try:
            from .llm_extractor import extract_entities_llm

            llm_entities = extract_entities_llm(text)
            # Convert to our format
            for llm_entity in llm_entities:
                _add(llm_entity["name"], "llm")
                # Update type if provided by LLM
                if "type" in llm_entity:
                    # Find the entity we just added and update its type
                    for i, entity in enumerate(entities):
                        if entity["name"] == llm_entity["name"]:
                            entities[i]["type"] = llm_entity["type"]
                            break
        except ImportError as e:
            logger.warning(f"LLM extractor not available: {e}")
        except Exception as e:
            logger.warning(f"LLM-based entity extraction failed: {e}")

    return entities


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python entity_extractor.py <text> [--llm]")
        sys.exit(1)

    text = sys.argv[1]
    use_llm = "--llm" in sys.argv

    entities = extract_entities(text, use_llm)

    print(f"Input: {text}")
    print(f"Entities found: {len(entities)}")
    for entity in entities:
        print(f"  - {entity['name']} ({entity['type']}) [source: {entity['source']}]")

    # Also output as JSON for piping
    print("\nJSON output:")
    print(json.dumps(entities, indent=2))
