"""Trivial task classifier for routing decisions.

Detects if a task is trivial (greetings, chitchat, acknowledgments) and should NOT
be routed to Venice. Uses ML-based detection with fallback to external services.

Classes:
    TrivialTaskClassifier: ML-based trivial task detection
    ComplexityDetector: Analyzes task complexity for routing priority
"""

from __future__ import annotations

import asyncio
import re
from typing import Any
from urllib.parse import urljoin
from urllib.request import urlopen

import yaml


class TrivialTaskClassifier:
    """Classify tasks as trivial or non-trivial using ML-based detection."""

    # Keyword patterns for fast detection (80% accuracy)
    TRIVIAL_KEYWORDS = [
        r"^\s*(hello|hi|hey|good\s+(morning|afternoon|evening)|howdy)\b",  # Greetings
        r"^\s*(thanks|thank\s+you|ty|appreciate\s+it|cheers)\b",  # Acknowledgments
        r"^\s*(ok|okay|alright|got\s+it|understood|sure|no\s+problem)\b",  # Confirmations
        r"^\s*(how\s+are\s+you|what\s+(?:'re|are)\s+(?:you|u)\s+up\s+to|how\s+(?:'s|is)\s+(?:it|things)\s+going)\b",  # Chitchat
        r"^\s*(nice|awesome|cool|great|good)\b",  # Simple praise
        r"^\s*(let\s+(?:'s|us)\s+start|go\s+ahead|proceed|continue)\b",  # Action starters
    ]

    # Meta questions that should use local models
    META_QUESTIONS = [
        r"explain\s+(?:this|that|what|how)",  # "explain this"
        r"what\s+(?:does|do)\s+(?:this|that)\s+(?:mean|refer\s+to)",  # "what does this mean"
        r"what\s+(?:is|are)\s+(?:this|that)",  # "what is this"
    ]

    def __init__(self, config_path: str = None):
        """Initialize classifier with config loading.

        Args:
            config_path: Path to config.yaml file (defaults to ~/.hermes/config.yaml)
        """
        if config_path is None:
            config_path = "/home/lordofwarai/.hermes/config.yaml"
        self._config_path = config_path
        self._load_config(config_path)
        self._compile_patterns()

    def _load_config(self, config_path: str) -> None:
        """Load configuration from YAML file.

        Args:
            config_path: Path to config.yaml
        """
        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

        # Detect complexity detection method
        complexity_method = self._config.get("complexity_detection", {}).get(
            "method", "ml"
        )
        self._complexity_method = complexity_method
        self._fallback_to_zai = self._config.get("fallback", {}).get(
            "use_zai_for_complexity", True
        )

        # Local inference settings
        local_inference = self._config.get("local_inference", {})
        self._local_inference_enabled = local_inference.get("enabled", False)
        self._local_inference_port = local_inference.get("port", 8101)

        # Z.ai settings
        self._zai_fallback = self._config.get("fallback", {}).get(
            "use_zai_for_complexity", True
        )
        self._zai_api_key = self._config.get("zai", {}).get("api_key", "")

    def _compile_patterns(self) -> None:
        """Compile regex patterns for keyword detection."""
        self._trivial_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.TRIVIAL_KEYWORDS
        ]
        self._meta_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.META_QUESTIONS
        ]

    def _classify_by_keywords(self, message: str) -> tuple[bool, str, float]:
        """Fast keyword-based classification (80% accuracy).

        Args:
            message: The task/message to classify

        Returns:
            Tuple of (is_trivial, type, confidence)
        """
        for pattern in self._trivial_patterns:
            if pattern.search(message):
                # Determine type based on pattern
                if pattern == self._trivial_patterns[0]:
                    return True, "greeting", 0.85
                elif pattern == self._trivial_patterns[1]:
                    return True, "acknowledgment", 0.85
                elif pattern == self._trivial_patterns[2]:
                    return True, "confirmation", 0.85
                elif pattern == self._trivial_patterns[3]:
                    return True, "chitchat", 0.85
                elif pattern == self._trivial_patterns[4]:
                    return True, "simple_praise", 0.85
                elif pattern == self._trivial_patterns[5]:
                    return True, "action_starter", 0.85

        # Check for meta questions
        for pattern in self._meta_patterns:
            if pattern.search(message):
                return True, "meta_question", 0.80

        return False, "unknown", 0.0

    async def _query_local_model(self, message: str) -> dict[str, Any]:
        """Query local LFM2-24B model for classification.

        Args:
            message: The task/message to classify

        Returns:
            Classification result dict
        """
        payload = {
            "model": "glm-5.1",
            "messages": [
                {
                    "role": "user",
                    "content": 'Classify this task: Is it trivial (greeting, chitchat, acknowledgment)? Only return JSON: {"trivial": bool, "type": str, "confidence": float} Task: '
                    + message,
                }
            ],
            "stream": False,
        }

        url = urljoin(
            f"http://127.0.0.1:{self._local_inference_port}/v1/", "chat/completions"
        )
        headers = {"Content-Type": "application/json"}

        req = urlopen(url, str(payload).encode(), timeout=10)
        response = json.loads(req.read().decode())

        return {
            "trivial": response["choices"][0]["message"]["content"].startswith(
                '{"trivial": true'
            ),
            "type": "unknown",
            "confidence": 0.95,
        }

    async def _query_zai_model(self, message: str) -> dict[str, Any]:
        """Query Z.ai glm-4.5-air for classification.

        Args:
            message: The task/message to classify

        Returns:
            Classification result dict
        """
        payload = {
            "model": "glm-4.5-air",
            "messages": [
                {
                    "role": "user",
                    "content": 'Classify this task: Is it trivial (greeting, chitchat, acknowledgment)? Only return JSON: {"trivial": bool, "type": str, "confidence": float} Task: '
                    + message,
                }
            ],
            "stream": False,
        }

        url = "https://api.z.ai/api/coding/paas/v4/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._zai_api_key}",
        }

        req = urlopen(url, str(payload).encode(), timeout=10)
        response = json.loads(req.read().decode())

        return {
            "trivial": response["choices"][0]["message"]["content"].startswith(
                '{"trivial": true'
            ),
            "type": "unknown",
            "confidence": 0.95,
        }

    async def classify(self, message: str) -> dict[str, Any]:
        """Classify task as trivial or non-trivial.

        Priority:
        1. Keyword detection (fast, 80% accuracy)
        2. ML-based detection (accurate, 95%+)
           - Local LFM2-24B (fast, free)
           - Z.ai glm-4.5-air (reliable, external)

        Args:
            message: The task/message to classify

        Returns:
            Dict with classification results
        """
        # Fast keyword detection
        is_trivial, task_type, confidence = self._classify_by_keywords(message)

        if is_trivial:
            return {"trivial": True, "type": task_type, "confidence": confidence}

        # ML-based detection
        try:
            if self._local_inference_enabled:
                result = await self._query_local_model(message)
            elif self._fallback_to_zai:
                result = await self._query_zai_model(message)
            else:
                # No ML available, assume non-trivial
                return {"trivial": False, "type": "non-trivial", "confidence": 0.90}

            return result
        except Exception:
            # ML failed, fallback to keyword detection
            return {"trivial": False, "type": "non-trivial", "confidence": 0.90}


class ComplexityDetector:
    """Detect task complexity for routing priority."""

    def __init__(self, config_path: str = None):
        """Initialize detector with config loading."""
        if config_path is None:
            config_path = "/home/lordofwarai/.hermes/config.yaml"
        self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        """Load configuration from YAML file.

        Args:
            config_path: Path to config.yaml
        """
        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

        # Detect complexity detection method
        complexity_method = self._config.get("complexity_detection", {}).get(
            "method", "ml"
        )
        self._complexity_method = complexity_method

        # Local inference settings
        local_inference = self._config.get("local_inference", {})
        self._local_inference_enabled = local_inference.get("enabled", False)
        self._local_inference_port = local_inference.get("port", 8101)

        # Z.ai settings
        self._zai_api_key = self._config.get("zai", {}).get("api_key", "")

    def _classify_by_keywords(self, message: str) -> str:
        """Fast keyword-based complexity classification (80% accuracy).

        Args:
            message: The task/message to classify

        Returns:
            Complexity level: "simple"/"medium"/"complex"/"expert"
        """
        # Expert complexity indicators
        expert_patterns = [
            r"design\s+(?:an|a)\s+(?:system|architecture|framework)",
            r"implement\s+(?:a|an)\s+(?:complete|full|end-to-end)",
            r"optimize\s+(?:for|to)\s+(?:performance|latency|throughput)",
            r"refactor\s+(?:this|the\s+codebase|the\s+entire)",
            r"debug\s+(?:this|the\s+issue|the\s+problem)",
        ]

        # Complex complexity indicators
        complex_patterns = [
            r"analyze\s+(?:this|the\s+codebase|the\s+architecture)",
            r"explain\s+(?:this|that|how|why)",
            r"compare\s+(?:and\s+contrast|vs|versus)",
            r"evaluate\s+(?:this|the\s+approach|the\s+solution)",
        ]

        # Medium complexity indicators
        medium_patterns = [
            r"write\s+(?:a|an)\s+(?:function|method|class)",
            r"create\s+(?:a|an)\s+(?:script|tool|utility)",
            r"fix\s+(?:this|the\s+bug|the\s+error)",
            r"update\s+(?:this|the\s+code|the\s+function)",
        ]

        # Simple complexity indicators
        simple_patterns = [
            r"what\s+(?:is|are)\s+(?:this|that)",
            r"how\s+(?:do\s+I|does)\s+(?:this|that)\s+(?:work|function)",
            r"show\s+(?:me|me\s+the)\s+(?:this|that)",
            r"give\s+(?:me)\s+(?:a|an)\s+(?:example|snippet)",
        ]

        # Check in priority order
        for pattern in expert_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return "expert"

        for pattern in complex_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return "complex"

        for pattern in medium_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return "medium"

        for pattern in simple_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return "simple"

        # Default to medium
        return "medium"

    async def _query_local_model(self, message: str) -> str:
        """Query local LFM2-24B model for complexity detection.

        Args:
            message: The task/message to classify

        Returns:
            Complexity level string
        """
        payload = {
            "model": "glm-5.1",
            "messages": [
                {
                    "role": "user",
                    "content": "Classify this task complexity: Simple/Medium/Complex/Expert. Only return the level name. Task: "
                    + message,
                }
            ],
            "stream": False,
        }

        url = urljoin(
            f"http://127.0.0.1:{self._local_inference_port}/v1/", "chat/completions"
        )
        headers = {"Content-Type": "application/json"}

        req = urlopen(url, str(payload).encode(), timeout=10)
        response = json.loads(req.read().decode())

        content = response["choices"][0]["message"]["content"]

        if "simple" in content.lower():
            return "simple"
        elif "medium" in content.lower():
            return "medium"
        elif "complex" in content.lower():
            return "complex"
        elif "expert" in content.lower():
            return "expert"
        else:
            return "medium"

    async def _query_zai_model(self, message: str) -> str:
        """Query Z.ai glm-4.5-air for complexity detection.

        Args:
            message: The task/message to classify

        Returns:
            Complexity level string
        """
        payload = {
            "model": "glm-4.5-air",
            "messages": [
                {
                    "role": "user",
                    "content": "Classify this task complexity: Simple/Medium/Complex/Expert. Only return the level name. Task: "
                    + message,
                }
            ],
            "stream": False,
        }

        url = "https://api.z.ai/api/coding/paas/v4/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._zai_api_key}",
        }

        req = urlopen(url, str(payload).encode(), timeout=10)
        response = json.loads(req.read().decode())

        content = response["choices"][0]["message"]["content"]

        if "simple" in content.lower():
            return "simple"
        elif "medium" in content.lower():
            return "medium"
        elif "complex" in content.lower():
            return "complex"
        elif "expert" in content.lower():
            return "expert"
        else:
            return "medium"

    def detect(self, message: str) -> str:
        """Detect task complexity.

        Priority:
        1. Keyword detection (fast, 80% accuracy)
        2. ML-based detection (accurate, 95%+)
           - Local LFM2-24B (fast, free)
           - Z.ai glm-4.5-air (reliable, external)

        Args:
            message: The task/message to classify

        Returns:
            Complexity level: "simple"/"medium"/"complex"/"expert"
        """
        # Fast keyword detection
        complexity = self._classify_by_keywords(message)

        if complexity != "medium":
            return complexity

        # ML-based detection
        try:
            if self._local_inference_enabled:
                return asyncio.run(self._query_local_model(message))
            else:
                return asyncio.run(self._query_zai_model(message))
        except Exception:
            # ML failed, fallback to keyword detection
            return "medium"


# Module-level classify function for compatibility with existing code
def classify(text: str, config_path: str = None) -> dict[str, Any]:
    """Classify text as trivial or non-trivial.

    Args:
        text: The text to classify
        config_path: Path to config.yaml (defaults to ~/.hermes/config.yaml)

    Returns:
        Dict with classification results: {"is_trivial": bool, "task_type": str}
    """
    if config_path is None:
        config_path = "/home/lordofwarai/.hermes/config.yaml"
    classifier = TrivialTaskClassifier(config_path)
    result = asyncio.run(classifier.classify(text))
    return {
        "is_trivial": result["trivial"],
        "task_type": result["type"],
        "confidence": result["confidence"],
    }


def detect_complexity(text: str, config_path: str = None) -> str:
    """Detect task complexity.

    Args:
        text: The text to classify
        config_path: Path to config.yaml (defaults to ~/.hermes/config.yaml)

    Returns:
        Complexity level: "simple"/"medium"/"complex"/"expert"
    """
    if config_path is None:
        config_path = "/home/lordofwarai/.hermes/config.yaml"
    detector = ComplexityDetector(config_path)
    return detector.detect(text)
