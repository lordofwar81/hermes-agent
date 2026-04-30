"""Fallback chain for model selection with graceful degradation.

Implements a comprehensive fallback chain that prioritizes free/fast models
before using budgeted Venice models. Handles failures gracefully with retry
logic and automatic model switching.

Classes:
    ModelFallbackChain: Comprehensive fallback chain with retry logic
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import urljoin
from urllib.request import urlopen

import yaml


class ModelFallbackChain:
    """Implement fallback chain for model selection."""

    # Local model priority order (free, fast)
    LOCAL_MODELS = [
        {"port": 8100, "model": "Carnice-Qwen3.6-MoE-35B-A3B-APEX-I-Compact", "cost": 0.0},
        {"port": 8104, "model": "Qwen3.5-122B-A10B-APEX-I-Compact", "cost": 0.0},
    ]

    # Z.ai model priority order (free, unlimited)
    ZAI_MODELS = [
        {"model": "glm-5.1", "cost": 0.0},
        {"model": "glm-5-turbo", "cost": 0.0},
        {"model": "glm-5", "cost": 0.0},
        {"model": "glm-4.7", "cost": 0.0},
        {"model": "glm-4.6", "cost": 0.0},
        {"model": "glm-4.5", "cost": 0.0},
        {"model": "glm-4.5-air", "cost": 0.0},
    ]

    # Venice model priority order (budgeted, last resort)
    VENICE_MODELS = [
        {"model": "deepseek-v3.2", "cost": 0.008},
        {"model": "zai-org-glm-4.7-flash", "cost": 0.007},
        {"model": "zai-org-glm-5", "cost": 0.04},
        {"model": "zai-org-glm-4.7", "cost": 0.03},
        {"model": "qwen-3-6-plus", "cost": 0.05},
        {"model": "qwen3-coder-480b-a35b-instruct", "cost": 0.04},
        {"model": "qwen3-5-35b-a3b", "cost": 0.02},
        {"model": "claude-sonnet-4-6", "cost": 0.22},
        {"model": "grok-4-20-beta", "cost": 0.10},
        {"model": "venice-uncensored", "cost": 0.01},
    ]

    def __init__(self, config_path: str = None):
        """Initialize fallback chain with config loading.

        Args:
            config_path: Path to config.yaml file (defaults to ~/.hermes/config.yaml)
        """
        if config_path is None:
            config_path = "/home/lordofwarai/.hermes/config.yaml"
        self._config_path = config_path
        self._load_config()

        # Initialize fallback chain
        self._local_models = self._get_local_models()
        self._zai_models = self._get_zai_models()
        self._venice_models = self._get_venice_models()

        # Retry settings
        self._max_retries = 3
        self._retry_delay = 1.0  # seconds

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        with open(self._config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

        # Get API keys and settings
        self._zai_api_key = self._config.get("zai", {}).get("api_key", "")
        self._venice_api_key = self._config.get("venice", {}).get("api_key", "")

        # Local inference settings
        local_inference = self._config.get("local_inference", {})
        self._local_inference_enabled = local_inference.get("enabled", False)
        self._local_inference_port = local_inference.get("port", 8101)

        # Fallback settings
        fallback_config = self._config.get("fallback", {})
        self._use_zai_for_complexity = fallback_config.get(
            "use_zai_for_complexity", True
        )
        self._use_venice_as_last_resort = fallback_config.get(
            "use_venice_as_last_resort", True
        )

    def _get_local_models(self) -> list[dict[str, Any]]:
        """Get local models configuration.

        Returns:
            List of local model configurations
        """
        models = []
        if self._local_inference_enabled:
            models = self.LOCAL_MODELS
        return models

    def _get_zai_models(self) -> list[dict[str, Any]]:
        """Get Z.ai models configuration.

        Returns:
            List of Z.ai model configurations
        """
        if not self._zai_api_key:
            return []
        return self.ZAI_MODELS

    def _get_venice_models(self) -> list[dict[str, Any]]:
        """Get Venice models configuration.

        Returns:
            List of Venice model configurations
        """
        if not self._venice_api_key:
            return []
        return self.VENICE_MODELS

    def get_fallback_order(self) -> list[dict[str, Any]]:
        """Get complete fallback order.

        Priority:
        1. Local models (free, fastest)
        2. Z.ai models (free, fast)
        3. Venice models (budgeted, last resort)

        Returns:
            List of model configurations in fallback order
        """
        return self._local_models + self._zai_models + self._venice_models

    async def _query_local_model(
        self, message: str, model_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Query local model with retry logic.

        Args:
            message: The message to send
            model_config: Model configuration dict

        Returns:
            Response dict with content
        """
        payload = {
            "model": model_config["model"],
            "messages": [{"role": "user", "content": message}],
            "stream": False,
        }

        url = urljoin(
            f"http://127.0.0.1:{model_config['port']}/v1/", "chat/completions"
        )
        headers = {"Content-Type": "application/json"}

        for attempt in range(self._max_retries):
            try:
                req = urlopen(url, str(payload).encode(), timeout=30)
                response = json.loads(req.read().decode())
                return {
                    "success": True,
                    "content": response["choices"][0]["message"]["content"],
                    "model": model_config["model"],
                    "cost": model_config["cost"],
                }
            except Exception as e:
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay)
                else:
                    return {
                        "success": False,
                        "error": str(e),
                        "model": model_config["model"],
                    }

    async def _query_zai_model(
        self, message: str, model_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Query Z.ai model with retry logic.

        Args:
            message: The message to send
            model_config: Model configuration dict

        Returns:
            Response dict with content
        """
        payload = {
            "model": model_config["model"],
            "messages": [{"role": "user", "content": message}],
            "stream": False,
        }

        url = "https://api.z.ai/api/coding/paas/v4/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._zai_api_key}",
        }

        for attempt in range(self._max_retries):
            try:
                req = urlopen(url, str(payload).encode(), timeout=30)
                response = json.loads(req.read().decode())
                return {
                    "success": True,
                    "content": response["choices"][0]["message"]["content"],
                    "model": model_config["model"],
                    "cost": model_config["cost"],
                }
            except Exception as e:
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay)
                else:
                    return {
                        "success": False,
                        "error": str(e),
                        "model": model_config["model"],
                    }

    async def _query_venice_model(
        self, message: str, model_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Query Venice model with retry logic.

        Args:
            message: The message to send
            model_config: Model configuration dict

        Returns:
            Response dict with content
        """
        payload = {
            "model": model_config["model"],
            "messages": [{"role": "user", "content": message}],
            "stream": False,
        }

        url = "https://api.venice.ai/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._venice_api_key}",
        }

        for attempt in range(self._max_retries):
            try:
                req = urlopen(url, str(payload).encode(), timeout=60)
                response = json.loads(req.read().decode())
                return {
                    "success": True,
                    "content": response["choices"][0]["message"]["content"],
                    "model": model_config["model"],
                    "cost": model_config["cost"],
                }
            except Exception as e:
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay)
                else:
                    return {
                        "success": False,
                        "error": str(e),
                        "model": model_config["model"],
                    }

    async def execute_with_fallback(
        self, message: str, use_venice: bool = False
    ) -> dict[str, Any]:
        """Execute message with comprehensive fallback chain.

        Priority:
        1. Local models (free, fastest)
        2. Z.ai models (free, fast)
        3. Venice models (budgeted, last resort) - only if use_venice=True

        Args:
            message: The message to send
            use_venice: Whether to allow Venice models

        Returns:
            Dict with execution results
        """
        results = []

        # Try local models first
        if self._local_models:
            for model_config in self._local_models:
                result = await self._query_local_model(message, model_config)
                results.append(result)
                if result["success"]:
                    return {
                        "success": True,
                        "message": message,
                        "response": result["content"],
                        "model": result["model"],
                        "cost": result["cost"],
                        "fallback_chain": results,
                    }

        # Try Z.ai models
        if self._zai_models:
            for model_config in self._zai_models:
                result = await self._query_zai_model(message, model_config)
                results.append(result)
                if result["success"]:
                    return {
                        "success": True,
                        "message": message,
                        "response": result["content"],
                        "model": result["model"],
                        "cost": result["cost"],
                        "fallback_chain": results,
                    }

        # Try Venice models only if allowed
        if use_venice and self._venice_models:
            for model_config in self._venice_models:
                result = await self._query_venice_model(message, model_config)
                results.append(result)
                if result["success"]:
                    return {
                        "success": True,
                        "message": message,
                        "response": result["content"],
                        "model": result["model"],
                        "cost": result["cost"],
                        "fallback_chain": results,
                    }

        # All fallbacks failed
        return {
            "success": False,
            "message": message,
            "response": None,
            "model": None,
            "cost": None,
            "fallback_chain": results,
            "error": "all fallback chains failed",
        }

    async def execute_with_complexity_fallback(
        self, message: str, complexity: str, trivial: bool = False
    ) -> dict[str, Any]:
        """Execute message with complexity-aware fallback chain.

        Complexity-aware routing:
        1. Trivial tasks: local models only (never Venice)
        2. Simple/medium: local → Z.ai (never Venice)
        3. Complex: local → Z.ai → Venice (if failed)
        4. Expert: local → Z.ai → Venice (always allowed)

        Args:
            message: The message to send
            complexity: Task complexity level
            trivial: Whether task is trivial

        Returns:
            Dict with execution results
        """
        # Determine if Venice is allowed
        use_venice = False
        if trivial:
            use_venice = False
        elif complexity in ("simple", "medium"):
            use_venice = False
        elif complexity == "complex":
            use_venice = True  # Only if fallback chain fails
        elif complexity == "expert":
            use_venice = True  # Always allowed

        return await self.execute_with_fallback(message, use_venice=use_venice)

    def get_fallback_route(self, message: str) -> Optional[dict[str, Any]]:
        """Get fallback route without executing (for routing decisions).

        Returns a routing decision based on the fallback chain priority:
        1. Local models (free, fast)
        2. Z.ai models (free, unlimited)
        3. Venice models (budgeted, last resort)

        Args:
            message: The message to route

        Returns:
            Dict with routing decision or None if no fallback available
        """
        # Try local models first
        for model_config in self._local_models:
            route = {
                "model": model_config["model"],
                "reason": f"Local model (free, fast) - {model_config['model']}",
                "provider": "local",
                "cost": model_config["cost"],
                "runtime": {
                    "api_key": model_config.get("api_key"),
                    "base_url": model_config.get("base_url", ""),
                    "provider": "local",
                    "api_mode": model_config.get("api_mode", ""),
                    "command": model_config.get("command", ""),
                    "args": tuple(model_config.get("args") or ()),
                    "credential_pool": None,
                },
            }
            return route

        # Try Z.ai models
        for model_config in self._zai_models:
            route = {
                "model": model_config["model"],
                "reason": f"Z.ai model (free, unlimited) - {model_config['model']}",
                "provider": "zai",
                "cost": model_config["cost"],
                "runtime": {
                    "api_key": model_config.get("api_key"),
                    "base_url": model_config.get("base_url", ""),
                    "provider": "zai",
                    "api_mode": model_config.get("api_mode", ""),
                    "command": model_config.get("command", ""),
                    "args": tuple(model_config.get("args") or ()),
                    "credential_pool": None,
                },
            }
            return route

        # Try Venice models (budgeted, last resort)
        for model_config in self._venice_models:
            route = {
                "model": model_config["model"],
                "reason": f"Venice model (budgeted) - {model_config['model']}",
                "provider": "venice",
                "cost": model_config["cost"],
                "runtime": {
                    "api_key": model_config.get("api_key"),
                    "base_url": model_config.get("base_url", ""),
                    "provider": "venice",
                    "api_mode": model_config.get("api_mode", ""),
                    "command": model_config.get("command", ""),
                    "args": tuple(model_config.get("args") or ()),
                    "credential_pool": None,
                },
            }
            return route

        # No fallback available - return None to let caller handle it
        return None
