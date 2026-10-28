"""
Utilities for CodeLens AI (security, logging, helpers).
"""

from codelens.utils.security import (
    validate_safe_path,
    sanitize_sandbox_env,
    request_user_approval,
    SecurityError,
    PathTraversalError,
    ApprovalDeniedError,
)
from codelens.utils.logger import AgentLogger, console

__all__ = [
    "validate_safe_path",
    "sanitize_sandbox_env",
    "request_user_approval",
    "SecurityError",
    "PathTraversalError",
    "ApprovalDeniedError",
    "AgentLogger",
    "console",
]
