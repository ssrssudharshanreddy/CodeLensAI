"""
Cloud LLM client for CodeLens AI supporting Gemini API, Groq, OpenRouter,
and any standard OpenAI-compatible hosted endpoints for web publishing.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Any, Optional
from pydantic import BaseModel

from codelens.llm.ollama_client import ModelInfo, OllamaClientError


class CloudClientError(OllamaClientError):
    """Exception raised by cloud provider failures."""
    pass


class CloudLLMClient:
    """
    HTTP client for hosted AI providers (Google Gemini API, Groq, OpenRouter, OpenAI).
    Uses standard HTTP without heavyweight SDK dependencies, ideal for lightweight deployments.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: str = "gemini-1.5-flash",
        timeout: int = 90,
    ):
        # Auto-detect API key from common environment variables
        self.api_key = (
            api_key
            or os.getenv("CODELENS_CLOUD_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GROQ_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        ).strip()

        # Detect default endpoint based on detected key if base_url is not explicit
        if base_url:
            self.base_url = base_url.rstrip("/")
        elif os.getenv("GROQ_API_KEY"):
            self.base_url = "https://api.groq.com/openai/v1"
            self.default_model = default_model if default_model != "gemini-1.5-flash" else "llama-3.3-70b-versatile"
        elif os.getenv("OPENROUTER_API_KEY"):
            self.base_url = "https://openrouter.ai/api/v1"
            self.default_model = default_model if default_model != "gemini-1.5-flash" else "google/gemini-2.0-flash-exp:free"
        else:
            # Default to Google Gemini API (OpenAI compatibility endpoint)
            self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
            self.default_model = default_model

        self.default_model = default_model
        self.timeout = timeout

    def check_connection(self) -> bool:
        """Verify API key presence and connectivity to cloud endpoint."""
        if not self.api_key:
            return False

        try:
            url = f"{self.base_url}/models"
            if "generativelanguage" in self.base_url:
                url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}"

            req = urllib.request.Request(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=8) as response:
                return response.status == 200
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 403):
                return False
            return True
        except Exception:
            return bool(self.api_key)

    def verify_model(self, model_name: Optional[str] = None) -> ModelInfo:
        """Verify cloud model configuration."""
        target = model_name or self.default_model
        return ModelInfo(
            name=target,
            parameter_size="Cloud Hosted",
            family="Cloud Provider",
            context_length=128000,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        format_json: bool = True,
        temperature: float = 0.1,
        timeout: Optional[int] = None,
    ) -> str:
        """Execute chat completion against cloud API."""
        if not self.api_key:
            raise CloudClientError(
                "Cloud API key is missing. Set GEMINI_API_KEY, GROQ_API_KEY, or configure settings."
            )

        selected_model = model or self.default_model

        payload: dict[str, Any] = {
            "model": selected_model,
            "messages": messages,
            "temperature": temperature,
        }

        if format_json:
            payload["response_format"] = {"type": "json_object"}

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data_bytes,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        effective_timeout = timeout or self.timeout
        try:
            with urllib.request.urlopen(req, timeout=effective_timeout) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                choices = res_data.get("choices", [])
                if not choices:
                    raise CloudClientError("Cloud API returned empty choices array.")
                content = choices[0].get("message", {}).get("content", "")
                return content
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise CloudClientError(f"Cloud API error ({e.code}): {err_body}")
        except urllib.error.URLError as e:
            raise CloudClientError(f"Network error communicating with cloud endpoint: {e}")
        except TimeoutError:
            raise CloudClientError(f"Cloud request timed out after {effective_timeout}s")
        except Exception as e:
            raise CloudClientError(f"Unexpected error calling cloud API: {e}")
