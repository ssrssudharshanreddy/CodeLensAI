"""
Configuration management for CodeLens AI.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

# Load .env if present
load_dotenv()


class CodeLensConfig(BaseModel):
    """Global configuration for CodeLens AI agent execution."""

    ollama_url: str = Field(
        default_factory=lambda: os.getenv("CODELENS_OLLAMA_URL", "http://127.0.0.1:11434")
    )
    model_name: str = Field(
        default_factory=lambda: os.getenv("CODELENS_MODEL", "gemma4:e2b")
    )
    workspace_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("CODELENS_WORKSPACE", ".")).resolve()
    )
    max_iterations: int = Field(
        default_factory=lambda: int(os.getenv("CODELENS_MAX_ITERATIONS", "8"))
    )
    llm_timeout: int = Field(
        default_factory=lambda: int(os.getenv("CODELENS_LLM_TIMEOUT", "90"))
    )
    sandbox_timeout: int = Field(
        default_factory=lambda: int(os.getenv("CODELENS_SANDBOX_TIMEOUT", "8"))
    )
    sandbox_memory_limit_mb: int = Field(
        default_factory=lambda: int(os.getenv("CODELENS_SANDBOX_MEMORY_MB", "256"))
    )
    require_approval: bool = Field(
        default_factory=lambda: os.getenv("CODELENS_AUTO_APPROVE", "0").lower() not in ("1", "true", "yes")
    )
    sandbox_enabled: bool = Field(
        default_factory=lambda: os.getenv("CODELENS_SANDBOX_ENABLED", "1").lower() in ("1", "true", "yes")
    )
    verbose: bool = Field(
        default_factory=lambda: os.getenv("CODELENS_VERBOSE", "0").lower() in ("1", "true", "yes")
    )
    provider: str = Field(
        default_factory=lambda: os.getenv("CODELENS_PROVIDER", "auto")
    )
    cloud_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("CODELENS_CLOUD_API_KEY", os.getenv("GEMINI_API_KEY", os.getenv("GROQ_API_KEY", None)))
    )
    cloud_base_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("CODELENS_CLOUD_BASE_URL", None)
    )
    cloud_model: str = Field(
        default_factory=lambda: os.getenv("CODELENS_CLOUD_MODEL", "gemini-2.0-flash")
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def validate_workspace(self) -> Path:
        """Ensure workspace directory exists and is a directory."""
        resolved = self.workspace_dir.resolve()
        if not resolved.exists():
            resolved.mkdir(parents=True, exist_ok=True)
        if not resolved.is_dir():
            raise ValueError(f"Workspace path is not a directory: {resolved}")
        self.workspace_dir = resolved
        return resolved
