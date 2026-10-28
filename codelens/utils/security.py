"""
Security and safety guardrails for CodeLens AI.
Enforces workspace boundaries, path containment, environment sanitization,
and explicit user approval policies.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional


class SecurityError(Exception):
    """Base exception for security boundary violations."""
    pass


class PathTraversalError(SecurityError):
    """Raised when an operation attempts to escape the configured workspace."""
    pass


class ApprovalDeniedError(SecurityError):
    """Raised when user denies approval for a destructive or modification action."""
    pass


# Sensitive patterns to block from inspection or modification
PROTECTED_PATTERNS = [
    r"^\.env(\..+)?$",
    r"^id_rsa.*$",
    r"^.*\.pem$",
    r"^.*\.key$",
    r"^\.git(/|\\|$)",
]

# Environment variable substrings that must never leak into sandboxes
SENSITIVE_ENV_SUBSTRINGS = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "AUTH",
    "AWS_",
    "AZURE_",
    "GCP_",
    "GOOGLE_",
    "GITHUB_",
    "SSH_",
)


def validate_safe_path(target_path: str | Path, workspace_root: Path) -> Path:
    """
    Validate that target_path resolves strictly within workspace_root.
    Prevents path traversal attacks (e.g. `../`, alternate drives, or symlink escapes).

    Raises:
        PathTraversalError: If the target path escapes the workspace root.
        SecurityError: If the target path points to protected secret files.
    """
    workspace = workspace_root.resolve()
    target = Path(target_path)

    # If target is relative, anchor it to workspace
    if not target.is_absolute():
        resolved = (workspace / target).resolve()
    else:
        resolved = target.resolve()

    # Verify containment using Path.relative_to
    try:
        rel = resolved.relative_to(workspace)
    except ValueError:
        raise PathTraversalError(
            f"Access denied: '{target_path}' resolves outside the allowed workspace: {workspace}"
        )

    # Check against protected patterns (e.g. .env, id_rsa, .git)
    filename = resolved.name
    for pattern in PROTECTED_PATTERNS:
        if re.search(pattern, filename, re.IGNORECASE) or re.search(pattern, str(rel), re.IGNORECASE):
            raise SecurityError(
                f"Security violation: Access to sensitive file or pattern '{filename}' is restricted."
            )

    return resolved


def sanitize_sandbox_env() -> dict[str, str]:
    """
    Produce a strictly sanitized environment dictionary for sandboxed subprocess execution.
    Strips sensitive credentials, auth tokens, and API keys.
    Sets safe Python runtime flags.
    """
    allowed_keys = {
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "PATH",
        "PATHEXT",
        "TEMP",
        "TMP",
        "COMSPEC",
        "WINDIR",
        "LOCALAPPDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
    }

    sanitized: dict[str, str] = {}

    for k, v in os.environ.items():
        k_upper = k.upper()

        # Check if explicitly allowed system var
        if k_upper in allowed_keys:
            sanitized[k] = v
            continue

        # Reject any sensitive credential substrings
        if any(substr in k_upper for substr in SENSITIVE_ENV_SUBSTRINGS):
            continue

    # Set security and isolation flags for Python subprocesses
    sanitized["PYTHONSAFEPATH"] = "1"           # Don't prepend current dir to sys.path automatically
    sanitized["PYTHONDONTWRITEBYTECODE"] = "1"   # Don't clutter with .pyc
    sanitized["PYTHONUNBUFFERED"] = "1"          # Immediate output flushing
    sanitized["PYTHONIOENCODING"] = "utf-8"      # Ensure UTF-8 on Windows
    sanitized["CODELENS_SANDBOX"] = "1"          # Marker for sandboxed runs

    return sanitized


def request_user_approval(action_summary: str, details: Optional[str] = None, auto_approve: bool = False) -> bool:
    """
    Prompt user for confirmation before performing modifications or executing tests.
    If auto_approve is True, returns True immediately.
    """
    if auto_approve:
        return True

    from rich.prompt import Confirm
    from rich.console import Console

    console = Console()
    console.print(f"\n[bold yellow][!] Approval Required:[/bold yellow] {action_summary}")
    if details:
        console.print(f"[dim]{details}[/dim]")

    try:
        approved = Confirm.ask("Do you approve this action?", default=True)
    except (KeyboardInterrupt, EOFError):
        approved = False

    if not approved:
        raise ApprovalDeniedError(f"User denied approval for: {action_summary}")

    return True
