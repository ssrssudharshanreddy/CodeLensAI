"""
Factory for instantiating local Ollama or Cloud LLM clients based on configuration.
"""

from __future__ import annotations

import os
from typing import Union
from codelens.config import CodeLensConfig
from codelens.llm.ollama_client import OllamaClient, OllamaConnectionError
from codelens.llm.cloud_client import CloudLLMClient


def create_llm_client(config: CodeLensConfig) -> Union[OllamaClient, CloudLLMClient]:
    """
    Instantiate the appropriate LLM client.
    Supports local Ollama (gemma4:e2b) or hosted Cloud APIs (Gemini/Groq/OpenAI).
    """
    provider = getattr(config, "provider", "auto").lower()

    if provider == "cloud":
        return CloudLLMClient(
            api_key=config.cloud_api_key,
            base_url=config.cloud_base_url,
            default_model=config.cloud_model,
            timeout=config.llm_timeout,
        )

    if provider == "ollama":
        return OllamaClient(
            base_url=config.ollama_url,
            default_model=config.model_name,
            timeout=config.llm_timeout,
        )

    # Provider == 'auto': Check local Ollama first
    ollama_client = OllamaClient(
        base_url=config.ollama_url,
        default_model=config.model_name,
        timeout=config.llm_timeout,
    )

    try:
        if ollama_client.check_connection():
            return ollama_client
    except Exception:
        pass

    # If local Ollama not accessible, check if Cloud API key is available
    has_cloud_key = bool(
        config.cloud_api_key
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if has_cloud_key:
        return CloudLLMClient(
            api_key=config.cloud_api_key,
            base_url=config.cloud_base_url,
            default_model=config.cloud_model,
            timeout=config.llm_timeout,
        )

    # Default to local Ollama client so user receives standard local guidance
    return ollama_client
