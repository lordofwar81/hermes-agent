#!/usr/bin/env python3
"""
LLM Extractor - Shared client for LLM-based extraction tasks (temporal, relationships, entities).

Uses LFM2-24B (port 8101) for all extraction tasks.
Implements pattern-first design: regex/rule-based extraction first, LLM fallback only when needed.
"""

import json
import logging
import re
import time
from typing import Dict, List, Any, Optional, Union
import requests

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# LFM2-24B endpoint (port 8101, same APEX stack)
LLM_ENDPOINT = "http://localhost:8101/v1"
LLM_MODEL = "LFM2-24B-A2B-APEX-I-Compact"
LLM_API_KEY = "notempty"  # same as llama.cpp servers

# Timeout settings
LLM_TIMEOUT = 60  # seconds
MAX_RETRIES = 2

# -----------------------------------------------------------------------------
# Base LLM Client
# -----------------------------------------------------------------------------


class LLMExtractor:
    """Shared LLM client for extraction tasks."""

    def __init__(
        self,
        endpoint: str = LLM_ENDPOINT,
        model: str = LLM_MODEL,
        api_key: str = LLM_API_KEY,
    ):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key

    def call_llm(
        self,
        prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.1,
        json_mode: bool = True,
    ) -> Optional[Union[str, Dict, List]]:
        """
        Call LLM with prompt and return parsed response.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            json_mode: If True, expect JSON response and parse it

        Returns:
            Parsed JSON if json_mode=True, otherwise raw text
            Returns None on failure
        """
        for attempt in range(MAX_RETRIES):
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }

                messages = [{"role": "user", "content": prompt}]

                # Add JSON response format instruction if needed
                if json_mode:
                    system_msg = "You are a data extraction assistant. Always respond with valid JSON."
                    messages.insert(0, {"role": "system", "content": system_msg})

                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False,
                }

                response = requests.post(
                    f"{self.endpoint}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=LLM_TIMEOUT,
                )

                if response.status_code != 200:
                    logger.warning(
                        f"LLM request failed (attempt {attempt + 1}): {response.status_code} - {response.text}"
                    )
                    time.sleep(1)
                    continue

                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()

                if json_mode:
                    try:
                        # Try to parse as JSON
                        parsed = json.loads(content)
                        return parsed
                    except json.JSONDecodeError:
                        # Try to extract JSON from text
                        json_match = re.search(r"\{.*\}", content, re.DOTALL)
                        if json_match:
                            try:
                                parsed = json.loads(json_match.group())
                                return parsed
                            except json.JSONDecodeError:
                                logger.warning(
                                    f"Failed to parse JSON from response: {content}"
                                )
                                return None
                        else:
                            logger.warning(f"No JSON found in response: {content}")
                            return None
                else:
                    return content

            except requests.exceptions.Timeout:
                logger.warning(f"LLM request timeout (attempt {attempt + 1})")
                time.sleep(2)
            except Exception as e:
                logger.warning(f"LLM request exception (attempt {attempt + 1}): {e}")
                time.sleep(1)

        logger.error(f"All {MAX_RETRIES} LLM attempts failed")
        return None


# -----------------------------------------------------------------------------
# Temporal Extraction
# -----------------------------------------------------------------------------


def extract_temporal_patterns(text: str) -> Dict[str, Any]:
    """
    Pattern-based temporal extraction (regex/rule-based).

    Returns dict with keys:
        events: list of {"text": str, "start": str|null, "end": str|null, "confidence": float}
        has_explicit_date: bool
    """
    events = []

    # Common date patterns
    date_patterns = [
        # YYYY-MM-DD
        (
            r"\b(\d{4})-(\d{2})-(\d{2})\b",
            lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}",
        ),
        # MM/DD/YYYY or DD/MM/YYYY (ambiguous)
        (
            r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b",
            lambda m: f"{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}",
        ),
        # Month day, year
        (
            r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})\b",
            lambda m: (
                f"{m.group(3)}-{month_to_num(m.group(1)):02d}-{int(m.group(2)):02d}"
            ),
        ),
    ]

    # Relative time patterns
    relative_patterns = [
        (r"\b(yesterday)\b", "yesterday"),
        (r"\b(today)\b", "today"),
        (r"\b(tomorrow)\b", "tomorrow"),
        (r"\b(last\s+week)\b", "last_week"),
        (r"\b(this\s+week)\b", "this_week"),
        (r"\b(next\s+week)\b", "next_week"),
        (r"\b(last\s+month)\b", "last_month"),
        (r"\b(this\s+month)\b", "this_month"),
        (r"\b(next\s+month)\b", "next_month"),
        (r"\b(past\s+(\d+)\s+days?)\b", lambda m: f"past_{m.group(2)}_days"),
        (r"\b(since\s+(\d{4}-\d{2}))\b", lambda m: f"since_{m.group(2)}"),
    ]

    # Find explicit dates
    explicit_dates = []
    for pattern, converter in date_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                date_str = converter(match) if callable(converter) else converter
                explicit_dates.append(
                    {
                        "text": match.group(0),
                        "date": date_str,
                        "start": date_str,
                        "end": None,
                    }
                )
            except (ValueError, TypeError):
                pass

    # Find relative time expressions
    relative_times = []
    for pattern, converter in relative_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            rel_str = converter(match) if callable(converter) else converter
            relative_times.append(
                {
                    "text": match.group(0),
                    "expression": rel_str,
                }
            )

    # Create events from found patterns
    if explicit_dates:
        for date_info in explicit_dates:
            events.append(
                {
                    "text": date_info["text"],
                    "start": date_info["start"],
                    "end": None,
                    "confidence": 0.9,
                    "source": "pattern",
                }
            )

    if relative_times:
        for rel_info in relative_times:
            events.append(
                {
                    "text": rel_info["text"],
                    "start": None,  # Will be resolved relative to current date
                    "end": None,
                    "confidence": 0.7,
                    "source": "pattern",
                }
            )

    return {
        "events": events,
        "has_explicit_date": len(explicit_dates) > 0,
        "explicit_dates": [d["date"] for d in explicit_dates if "date" in d],
        "relative_expressions": [r["expression"] for r in relative_times],
    }


def month_to_num(month_str: str) -> int:
    """Convert month name to number (1-12)."""
    months = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    return months.get(month_str.lower()[:3], 1)


def extract_temporal_llm(
    text: str, llm_extractor: LLMExtractor = None
) -> Dict[str, Any]:
    """
    LLM-based temporal extraction (fallback when patterns insufficient).

    Returns dict with same structure as extract_temporal_patterns.
    """
    if llm_extractor is None:
        llm_extractor = LLMExtractor()

    prompt = f"""Extract explicit or implied timestamps from this text:

"{text}"

Return JSON with this exact structure:
{{
  "events": [
    {{
      "text": "excerpt mentioning time",
      "start": "YYYY-MM-DD or null if unknown",
      "end": "YYYY-MM-DD or null if single point",
      "confidence": 0.0-1.0
    }}
  ],
  "has_explicit_date": true/false
}}

If no timestamps are found, return empty events list.
"""

    result = llm_extractor.call_llm(prompt, max_tokens=500)
    if result is None:
        return {"events": [], "has_explicit_date": False}

    # Ensure structure
    if not isinstance(result, dict):
        return {"events": [], "has_explicit_date": False}

    if "events" not in result:
        result["events"] = []

    return result


def extract_temporal(text: str, use_llm_fallback: bool = True) -> Dict[str, Any]:
    """
    Main temporal extraction function (pattern-first, LLM fallback).

    Args:
        text: Input text
        use_llm_fallback: Whether to use LLM if patterns find nothing

    Returns:
        Dict with temporal information
    """
    # First try patterns
    pattern_result = extract_temporal_patterns(text)

    # If patterns found something, return it
    if pattern_result["events"]:
        return pattern_result

    # If no patterns and LLM fallback enabled, try LLM
    if use_llm_fallback:
        llm_result = extract_temporal_llm(text)
        return llm_result

    return {"events": [], "has_explicit_date": False}


# -----------------------------------------------------------------------------
# Relationship Extraction
# -----------------------------------------------------------------------------


def extract_relationship_patterns(fact_a: str, fact_b: str) -> Optional[Dict[str, Any]]:
    """
    Pattern-based relationship extraction.

    Returns dict with keys:
        relationship: str (is_a, part_of, related_to, contradicts, precedes, same_as, author_of, located_in)
        confidence: float
        direction: str (forward, backward, bidirectional)
    or None if no pattern matches.
    """
    # Common relationship patterns
    patterns = [
        # is_a patterns
        (r"\b(\w+)\s+is\s+(?:a|an)\s+(\w+)\b", "is_a", "forward"),
        (r"\b(\w+)\s+are\s+(?:a|an)\s+(\w+)\b", "is_a", "forward"),
        # part_of patterns
        (r"\b(\w+)\s+is\s+part\s+of\s+(\w+)\b", "part_of", "forward"),
        (r"\b(\w+)\s+includes?\s+(\w+)\b", "part_of", "backward"),
        (r"\b(\w+)\s+contains?\s+(\w+)\b", "part_of", "backward"),
        # precedes patterns
        (r"\b(\w+)\s+before\s+(\w+)\b", "precedes", "forward"),
        (r"\b(\w+)\s+after\s+(\w+)\b", "precedes", "backward"),
        (r"\b(\w+)\s+then\s+(\w+)\b", "precedes", "forward"),
        # same_as patterns
        (
            r"\b(\w+)\s+(?:also known as|aka|called)\s+(\w+)\b",
            "same_as",
            "bidirectional",
        ),
        (r"\b(\w+)\s+\((\w+)\)\b", "same_as", "bidirectional"),
        # contradicts patterns
        (r"\b(\w+)\s+but\s+not\s+(\w+)\b", "contradicts", "forward"),
        (r"\b(\w+)\s+vs\.?\s+(\w+)\b", "contradicts", "bidirectional"),
    ]

    # Check both facts for patterns
    for pattern, rel_type, direction in patterns:
        if re.search(pattern, fact_a, re.IGNORECASE) or re.search(
            pattern, fact_b, re.IGNORECASE
        ):
            return {
                "relationship": rel_type,
                "confidence": 0.7,
                "direction": direction,
                "source": "pattern",
            }

    return None


def extract_relationship_llm(
    fact_a: str, fact_b: str, llm_extractor: LLMExtractor = None
) -> Optional[Dict[str, Any]]:
    """
    LLM-based relationship extraction.
    """
    if llm_extractor is None:
        llm_extractor = LLMExtractor()

    prompt = f"""Given two facts, identify the relationship type from: is_a, part_of, related_to, contradicts, precedes, same_as, author_of, located_in.

Fact 1: "{fact_a}"
Fact 2: "{fact_b}"

Return JSON with this exact structure:
{{
  "relationship": "relationship_type",
  "confidence": 0.0-1.0,
  "direction": "forward|backward|bidirectional",
  "reasoning": "brief explanation"
}}

If no clear relationship exists, set relationship to "related_to" with low confidence.
"""

    result = llm_extractor.call_llm(prompt, max_tokens=300)
    if result is None:
        return None

    # Ensure required fields
    if not isinstance(result, dict):
        return None

    if "relationship" not in result:
        result["relationship"] = "related_to"

    if "confidence" not in result:
        result["confidence"] = 0.5

    if "direction" not in result:
        result["direction"] = "bidirectional"

    return result


def extract_relationship(
    fact_a: str, fact_b: str, use_llm_fallback: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Main relationship extraction function (pattern-first, LLM fallback).
    """
    # First try patterns
    pattern_result = extract_relationship_patterns(fact_a, fact_b)
    if pattern_result:
        return pattern_result

    # If no patterns and LLM fallback enabled, try LLM
    if use_llm_fallback:
        llm_result = extract_relationship_llm(fact_a, fact_b)
        return llm_result

    return None


# -----------------------------------------------------------------------------
# Entity Extraction (LLM fallback)
# -----------------------------------------------------------------------------


def extract_entities_llm(
    text: str, llm_extractor: LLMExtractor = None
) -> List[Dict[str, Any]]:
    """
    LLM-based entity extraction (fallback for entity_extractor.py).
    """
    if llm_extractor is None:
        llm_extractor = LLMExtractor()

    prompt = f"""Extract named entities from this text:

"{text}"

Return JSON with this exact structure:
{{
  "entities": [
    {{
      "name": "entity name",
      "type": "person|project|tech|concept|location|org",
      "confidence": 0.0-1.0
    }}
  ]
}}
"""

    result = llm_extractor.call_llm(prompt, max_tokens=500)
    if result is None:
        return []

    if not isinstance(result, dict) or "entities" not in result:
        return []

    return result["entities"]


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python llm_extractor.py <command> [args]")
        print("Commands:")
        print("  temporal <text>          Extract temporal information")
        print("  relationship <fact1> <fact2>  Extract relationship")
        print("  entities <text>          Extract entities (LLM only)")
        sys.exit(1)

    command = sys.argv[1]

    if command == "temporal":
        text = " ".join(sys.argv[2:])
        result = extract_temporal(text, use_llm_fallback=True)
        print(json.dumps(result, indent=2))

    elif command == "relationship":
        if len(sys.argv) < 4:
            print("Usage: python llm_extractor.py relationship <fact1> <fact2>")
            sys.exit(1)
        fact_a = sys.argv[2]
        fact_b = sys.argv[3]
        result = extract_relationship(fact_a, fact_b, use_llm_fallback=True)
        print(json.dumps(result, indent=2))

    elif command == "entities":
        text = " ".join(sys.argv[2:])
        result = extract_entities_llm(text)
        print(json.dumps(result, indent=2))

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
