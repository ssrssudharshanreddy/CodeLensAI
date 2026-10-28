"""
Live integration test against local Ollama running gemma4:e2b.
Skips if Ollama is unreachable.
"""

from pathlib import Path
import pytest

from codelens.config import CodeLensConfig
from codelens.llm.ollama_client import OllamaClient, OllamaConnectionError
from codelens.orchestrator.agent import CodeLensAgent


def test_live_ollama_connectivity():
    client = OllamaClient(default_model="gemma4:e2b")
    try:
        is_up = client.check_connection()
        assert is_up is True
        info = client.verify_model("gemma4:e2b")
        assert info.name == "gemma4:e2b"
    except OllamaConnectionError:
        pytest.skip("Local Ollama server is not running at 127.0.0.1:11434")


def test_live_agent_inspection(tmp_path: Path):
    client = OllamaClient(default_model="gemma4:e2b")
    try:
        client.check_connection()
        client.verify_model("gemma4:e2b")
    except Exception:
        pytest.skip("Local Ollama or gemma4:e2b model unavailable.")

    # Create simple target file
    (tmp_path / "hello.py").write_text("def greet():\n    return 'hello world'\n")

    cfg = CodeLensConfig(
        workspace_dir=tmp_path,
        max_iterations=3,
        require_approval=False,
        sandbox_enabled=True,
    )
    agent = CodeLensAgent(config=cfg, client=client)
    state = agent.run(task_description="Inspect hello.py and verify its syntax.", target_file="hello.py")

    assert len(state.steps) >= 1
    assert state.steps[0].tool_name in ("inspect_file", "check_syntax", "list_files")
