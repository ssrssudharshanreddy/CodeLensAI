"""
Local Ollama API client for CodeLens AI.
Handles connection verification, model availability, and structured chat completions.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any, Optional
from pydantic import BaseModel


class OllamaClientError(Exception):
    """Base exception for Ollama client errors."""
    pass


class OllamaConnectionError(OllamaClientError):
    """Raised when Ollama server cannot be reached."""
    pass


class ModelNotFoundError(OllamaClientError):
    """Raised when the specified model is not installed in Ollama."""
    pass


class ModelInfo(BaseModel):
    name: str
    parameter_size: Optional[str] = None
    family: Optional[str] = None
    context_length: Optional[int] = None
    quantization_level: Optional[str] = None


class OllamaClient:
    """Client for local Ollama HTTP API."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", default_model: str = "gemma4:e2b", timeout: int = 90):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout

    def check_connection(self) -> bool:
        """Verify that Ollama server is running and accessible."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except Exception as e:
            raise OllamaConnectionError(
                f"Could not connect to Ollama at {self.base_url}. "
                f"Ensure Ollama is running ('ollama serve'). Underlying error: {e}"
            )

    def list_models(self) -> list[ModelInfo]:
        """Fetch list of all installed models in Ollama."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
                models = []
                for item in data.get("models", []):
                    details = item.get("details", {})
                    models.append(
                        ModelInfo(
                            name=item.get("name", ""),
                            parameter_size=details.get("parameter_size"),
                            family=details.get("family"),
                            context_length=details.get("context_length"),
                            quantization_level=details.get("quantization_level"),
                        )
                    )
                return models
        except urllib.error.URLError as e:
            raise OllamaConnectionError(f"Connection to Ollama failed: {e}")
        except Exception as e:
            raise OllamaClientError(f"Failed to list Ollama models: {e}")

    def verify_model(self, model_name: Optional[str] = None) -> ModelInfo:
        """
        Verify that the requested model is available locally.
        Raises ModelNotFoundError if missing.
        """
        target = model_name or self.default_model
        installed = self.list_models()
        target_clean = target.lower().strip()

        for m in installed:
            name_clean = m.name.lower().strip()
            # Match either exact name 'gemma4:e2b' or prefix
            if name_clean == target_clean or name_clean.startswith(f"{target_clean}:"):
                return m

        available = [m.name for m in installed]
        raise ModelNotFoundError(
            f"Requested model '{target}' is not installed in local Ollama.\n"
            f"Installed models: {', '.join(available) if available else 'None'}\n"
            f"To pull it, run: ollama pull {target}"
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        format_json: bool = True,
        temperature: float = 0.1,
        timeout: Optional[int] = None,
    ) -> str:
        """
        Execute a chat completion with Ollama.
        Returns the raw message content string from the assistant.
        """
        selected_model = model or self.default_model
        payload: dict[str, Any] = {
            "model": selected_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if format_json:
            payload["format"] = "json"

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        effective_timeout = timeout or self.timeout
        try:
            with urllib.request.urlopen(req, timeout=effective_timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
                message = result.get("message", {})
                content = message.get("content", "")
                return content
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise OllamaClientError(f"Ollama HTTP {e.code} error: {err_body}")
        except urllib.error.URLError as e:
            raise OllamaConnectionError(f"Failed to communicate with Ollama: {e}")
        except TimeoutError:
            raise OllamaClientError(f"Ollama request timed out after {effective_timeout}s")
        except Exception as e:
            raise OllamaClientError(f"Unexpected error calling Ollama: {e}")
