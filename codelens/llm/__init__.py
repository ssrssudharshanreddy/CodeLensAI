"""
LLM integration and structured protocol for CodeLens AI.
"""

from codelens.llm.ollama_client import (
    OllamaClient,
    OllamaClientError,
    OllamaConnectionError,
    ModelNotFoundError,
    ModelInfo,
)
from codelens.llm.protocol import (
    ToolProtocol,
    ToolProtocolError,
    ActionDecision,
)

__all__ = [
    "OllamaClient",
    "OllamaClientError",
    "OllamaConnectionError",
    "ModelNotFoundError",
    "ModelInfo",
    "ToolProtocol",
    "ToolProtocolError",
    "ActionDecision",
]
