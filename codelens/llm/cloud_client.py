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
        default_model: str = "gemini-2.0-flash",
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
        elif self.api_key.startswith("gsk_") or os.getenv("GROQ_API_KEY"):
            self.base_url = "https://api.groq.com/openai/v1"
            self.default_model = default_model if default_model != "gemini-2.0-flash" else "llama-3.3-70b-versatile"
        elif self.api_key.startswith("sk-or-") or os.getenv("OPENROUTER_API_KEY"):
            self.base_url = "https://openrouter.ai/api/v1"
            self.default_model = default_model if default_model != "gemini-2.0-flash" else "google/gemini-2.0-flash-exp:free"
        else:
            # Default to Google Gemini API
            self.base_url = "https://generativelanguage.googleapis.com"
            self.default_model = default_model

        self.default_model = default_model
        self.timeout = timeout
        self._cached_gemini_models: Optional[list[str]] = None

    def is_gemini(self, model: Optional[str] = None) -> bool:
        """Determine whether to route request to native Gemini REST endpoint."""
        target_model = (model or self.default_model or "").lower()
        if self.api_key.startswith("gsk_") or "api.groq.com" in self.base_url:
            return False
        if self.api_key.startswith("sk-or-") or "openrouter.ai" in self.base_url:
            return False
        if "api.openai.com" in self.base_url:
            return False
        if self.api_key.startswith("AIza") or self.api_key.startswith("AQ."):
            return True
        if "generativelanguage.googleapis.com" in self.base_url:
            return True
        if target_model.startswith("gemini"):
            return True
        return False

    def _get_available_gemini_models(self) -> list[str]:
        """Fetch list of active Gemini models supporting generateContent from Google API."""
        if self._cached_gemini_models is not None:
            return self._cached_gemini_models

        models: list[str] = []
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}"
            req = urllib.request.Request(
                url,
                headers={
                    "x-goog-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=6) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    for m in data.get("models", []):
                        methods = m.get("supportedGenerationMethods", [])
                        if "generateContent" in methods:
                            m_name = m.get("name", "")
                            if m_name.startswith("models/"):
                                m_name = m_name[len("models/"):]
                            models.append(m_name)
        except Exception:
            pass

        self._cached_gemini_models = models
        return models

    def check_connection(self) -> bool:
        """Verify API key presence and connectivity to cloud endpoint."""
        if not self.api_key:
            return False

        try:
            if self.is_gemini():
                models = self._get_available_gemini_models()
                return len(models) > 0 or bool(self.api_key)
            else:
                url = f"{self.base_url}/models"
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
            family="Google Gemini" if self.is_gemini(target) else "Cloud Provider",
            context_length=128000,
        )

    def _execute_gemini_post(self, model_name: str, body_bytes: bytes, timeout: int) -> str:
        """Post generateContent payload to Gemini endpoint, trying v1beta then v1."""
        endpoints = [
            f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}",
            f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent?key={self.api_key}",
        ]
        last_error = None
        for url in endpoints:
            req = urllib.request.Request(
                url,
                data=body_bytes,
                headers={
                    "x-goog-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    res_data = json.loads(response.read().decode("utf-8"))
                    candidates = res_data.get("candidates", [])
                    if not candidates:
                        feedback = res_data.get("promptFeedback", {})
                        block_reason = feedback.get("blockReason")
                        if block_reason:
                            raise CloudClientError(f"Gemini API blocked request: {block_reason}")
                        raise CloudClientError("Gemini API returned no candidates.")

                    candidate = candidates[0]
                    content_obj = candidate.get("content", {})
                    parts = content_obj.get("parts", [])
                    if not parts:
                        finish_reason = candidate.get("finishReason")
                        raise CloudClientError(f"Gemini API returned empty response (finishReason: {finish_reason}).")

                    return "".join(p.get("text", "") for p in parts)
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace")
                last_error = CloudClientError(f"Gemini API error ({e.code}): {err_body}")
                if e.code == 404:
                    # Try next endpoint / version
                    continue
                raise last_error
            except (urllib.error.URLError, TimeoutError) as e:
                raise CloudClientError(f"Network error communicating with Gemini API: {e}")

        if last_error:
            raise last_error
        raise CloudClientError(f"Failed to generate content with Gemini model '{model_name}'.")

    def _call_gemini_native(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        format_json: bool = True,
        temperature: float = 0.1,
        timeout: Optional[int] = None,
    ) -> str:
        """Execute chat completion using Google's native Gemini generateContent REST API."""
        clean_model = (model or self.default_model).strip()
        if ":" in clean_model or clean_model in ("gemma4:e2b", "gemma2", "llama3", "mistral", "phi3"):
            clean_model = self.default_model
        if clean_model.startswith("models/"):
            clean_model = clean_model[len("models/"):]

        available = self._get_available_gemini_models()
        if available:
            models_to_try = []
            if clean_model in available:
                models_to_try.append(clean_model)
            for preferred in ("gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash-latest", "gemini-1.5-flash-002", "gemini-1.5-flash", "gemini-1.5-pro"):
                if preferred in available and preferred not in models_to_try:
                    models_to_try.append(preferred)
            for m in available:
                if m not in models_to_try:
                    models_to_try.append(m)
        else:
            # Fallback sequence
            models_to_try = [clean_model]
            for fallback in ("gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash-latest", "gemini-1.5-flash", "gemini-1.5-pro"):
                if fallback not in models_to_try:
                    models_to_try.append(fallback)

        system_parts: list[dict[str, str]] = []
        contents: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append({"text": content})
                continue

            gemini_role = "model" if role in ("assistant", "model") else "user"

            if contents and contents[-1]["role"] == gemini_role:
                contents[-1]["parts"].append({"text": content})
            else:
                contents.append({"role": gemini_role, "parts": [{"text": content}]})

        if not contents:
            contents.append({"role": "user", "parts": [{"text": "Proceed."}]})
        elif contents[0]["role"] == "model":
            contents.insert(0, {"role": "user", "parts": [{"text": "Task context:"}]})

        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
            },
        }

        if system_parts:
            body["systemInstruction"] = {
                "parts": system_parts
            }

        if format_json:
            body["generationConfig"]["responseMimeType"] = "application/json"

        data_bytes = json.dumps(body).encode("utf-8")
        effective_timeout = timeout or self.timeout

        last_exc: Optional[Exception] = None
        for m_name in models_to_try:
            try:
                return self._execute_gemini_post(m_name, data_bytes, effective_timeout)
            except CloudClientError as err:
                last_exc = err
                if "404" in str(err):
                    # Model not available in this generation tier, try next fallback
                    continue
                raise err
            except Exception as e:
                if isinstance(e, CloudClientError):
                    raise
                raise CloudClientError(f"Unexpected error calling Gemini API: {e}")

        if last_exc:
            if "404" in str(last_exc) and not available:
                raise CloudClientError(
                    "Google Generative Language API returned 404 for all models. "
                    "This usually means the 'Generative Language API' is not enabled on your Google Cloud Project. "
                    "To fix this, either: 1. Enable 'Generative Language API' in Google Cloud Console, or "
                    "2. Generate a free key directly at Google AI Studio (https://aistudio.google.com/app/apikey), or "
                    "3. Paste a free Groq API key (starts with 'gsk_') in Settings."
                )
            raise last_exc
        raise CloudClientError("Failed to obtain response from Gemini API.")

    def _call_openai_compatible(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        format_json: bool = True,
        temperature: float = 0.1,
        timeout: Optional[int] = None,
    ) -> str:
        """Execute chat completion using standard OpenAI-compatible chat/completions endpoint."""
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
            if isinstance(e, CloudClientError):
                raise
            raise CloudClientError(f"Unexpected error calling cloud API: {e}")

    def chat(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        format_json: bool = True,
        temperature: float = 0.1,
        timeout: Optional[int] = None,
    ) -> str:
        """Execute chat completion against configured cloud API."""
        if not self.api_key:
            raise CloudClientError(
                "Cloud API key is missing. Set GEMINI_API_KEY, GROQ_API_KEY, or configure settings."
            )

        selected_model = model or self.default_model
        if ":" in selected_model or selected_model in ("gemma4:e2b", "gemma2", "llama3", "mistral", "phi3"):
            selected_model = self.default_model

        if self.is_gemini(selected_model):
            return self._call_gemini_native(
                messages=messages,
                model=selected_model,
                format_json=format_json,
                temperature=temperature,
                timeout=timeout,
            )

        return self._call_openai_compatible(
            messages=messages,
            model=selected_model,
            format_json=format_json,
            temperature=temperature,
            timeout=timeout,
        )
