"""
Unit tests for CodeLens AI security guardrails.
"""

import os
from pathlib import Path
import pytest

from codelens.utils.security import (
    validate_safe_path,
    sanitize_sandbox_env,
    PathTraversalError,
    SecurityError,
)


def test_validate_safe_path_valid(tmp_path: Path):
    test_file = tmp_path / "valid.py"
    test_file.write_text("x = 1")
    resolved = validate_safe_path("valid.py", tmp_path)
    assert resolved == test_file.resolve()


def test_validate_safe_path_subfolder(tmp_path: Path):
    sub = tmp_path / "subdir"
    sub.mkdir()
    sub_file = sub / "child.py"
    sub_file.write_text("y = 2")
    resolved = validate_safe_path("subdir/child.py", tmp_path)
    assert resolved == sub_file.resolve()


def test_validate_safe_path_traversal_blocked(tmp_path: Path):
    with pytest.raises(PathTraversalError):
        validate_safe_path("../outside.py", tmp_path)

    with pytest.raises(PathTraversalError):
        validate_safe_path("subdir/../../escape.py", tmp_path)


def test_validate_safe_path_protected_files(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("SECRET=123")
    with pytest.raises(SecurityError):
        validate_safe_path(".env", tmp_path)

    key_file = tmp_path / "id_rsa"
    key_file.write_text("ssh-rsa ...")
    with pytest.raises(SecurityError):
        validate_safe_path("id_rsa", tmp_path)


def test_sanitize_sandbox_env():
    # Inject sensitive mock environment variables
    os.environ["SECRET_API_KEY"] = "super-secret"
    os.environ["AWS_ACCESS_KEY_ID"] = "AKIA123"
    os.environ["GITHUB_TOKEN"] = "ghp_abc"
    os.environ["SAFE_SYSTEM_VAR"] = "ok"

    sanitized = sanitize_sandbox_env()

    # Verify sensitive keys stripped
    assert "SECRET_API_KEY" not in sanitized
    assert "AWS_ACCESS_KEY_ID" not in sanitized
    assert "GITHUB_TOKEN" not in sanitized

    # Verify safe python flags present
    assert sanitized.get("PYTHONSAFEPATH") == "1"
    assert sanitized.get("PYTHONIOENCODING") == "utf-8"
    assert sanitized.get("CODELENS_SANDBOX") == "1"
