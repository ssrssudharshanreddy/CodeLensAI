"""
Command-line interface for CodeLens AI.
Provides commands for running autonomous agent workflows, fixing bugs,
interactive student learning sessions, system health checks, and demonstrations.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from codelens.config import CodeLensConfig
from codelens.llm.ollama_client import OllamaClient
from codelens.tools.registry import ToolRegistry
from codelens.orchestrator.agent import CodeLensAgent
from codelens.orchestrator.learning import LearningSession
from codelens.utils.logger import AgentLogger

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="CodeLens AI")
def main():
    """CodeLens AI: Local Autonomous AI Coding & Debugging Agent."""
    pass


@main.command()
@click.argument("task", required=True)
@click.option("--target", "-t", default=None, help="Specific target file to analyze or fix.")
@click.option("--workspace", "-w", default=".", help="Workspace root directory.")
@click.option("--model", "-m", default="gemma4:e2b", help="Ollama model name.")
@click.option("--max-iter", "-i", default=8, type=int, help="Maximum allowed agent iterations.")
@click.option("--auto-approve", is_flag=True, default=False, help="Automatically approve patch modifications.")
@click.option("--disable-sandbox", is_flag=True, default=False, help="Disable isolated code execution.")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable verbose debug logging.")
def run(task: str, target: Optional[str], workspace: str, model: str, max_iter: int, auto_approve: bool, disable_sandbox: bool, verbose: bool):
    """Run the autonomous agent loop on a specified task."""
    cfg = CodeLensConfig(
        model_name=model,
        workspace_dir=Path(workspace).resolve(),
        max_iterations=max_iter,
        require_approval=not auto_approve,
        sandbox_enabled=not disable_sandbox,
        verbose=verbose,
    )
    agent = CodeLensAgent(config=cfg)
    state = agent.run(task_description=task, target_file=target)
    sys.exit(0 if state.status.value in ("SOLVED", "IDLE") else 1)


@main.command()
@click.argument("file_path", required=True)
@click.option("--workspace", "-w", default=".", help="Workspace root directory.")
@click.option("--model", "-m", default="gemma4:e2b", help="Ollama model name.")
@click.option("--auto-approve", is_flag=True, default=False, help="Automatically approve patch modifications.")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable verbose debug logging.")
def fix(file_path: str, workspace: str, model: str, auto_approve: bool, verbose: bool):
    """Analyze, diagnose, test, and safely patch a buggy source file."""
    cfg = CodeLensConfig(
        model_name=model,
        workspace_dir=Path(workspace).resolve(),
        require_approval=not auto_approve,
        verbose=verbose,
    )
    agent = CodeLensAgent(config=cfg)
    task = f"Inspect and fix any bugs in '{file_path}'. Verify the fix with tests."
    state = agent.run(task_description=task, target_file=file_path)
    sys.exit(0 if state.status.value == "SOLVED" else 1)


@main.command()
@click.argument("file_path", required=True)
@click.option("--workspace", "-w", default=".", help="Workspace root directory.")
@click.option("--model", "-m", default="gemma4:e2b", help="Ollama model name.")
@click.option("--problem", "-p", default="Identify and fix the bug in this file.", help="Description of the bug or expected behavior.")
def learn(file_path: str, workspace: str, model: str, problem: str):
    """Interactive student learning mode (Hint -> Guidance -> Solution)."""
    ws = Path(workspace).resolve()
    target = ws / file_path if not Path(file_path).is_absolute() else Path(file_path)

    if not target.exists():
        console.print(f"[bold red]File not found:[/bold red] {file_path}")
        sys.exit(1)

    code_text = target.read_text(encoding="utf-8", errors="replace")
    client = OllamaClient(default_model=model)

    try:
        client.check_connection()
        client.verify_model(model)
    except Exception as e:
        console.print(f"[bold red]Ollama check failed:[/bold red] {e}")
        sys.exit(1)

    session = LearningSession(client=client, model=model)
    session.interactive_walkthrough(code=code_text, problem_description=problem)


@main.command()
@click.option("--url", default="http://127.0.0.1:11434", help="Ollama server URL.")
@click.option("--model", "-m", default="gemma4:e2b", help="Target model to check.")
def check(url: str, model: str):
    """Verify Ollama server connectivity and model availability."""
    console.print(Panel(f"Checking Ollama at [bold cyan]{url}[/bold cyan]...", title="CodeLens Health Check"))
    client = OllamaClient(base_url=url, default_model=model)

    try:
        client.check_connection()
        console.print("[bold green][OK][/bold green] Ollama server is responding at " + url)
    except Exception as e:
        console.print(f"[bold red][FAIL][/bold red] Could not connect to Ollama: {e}")
        sys.exit(1)

    try:
        info = client.verify_model(model)
        table = Table(title=f"Verified Model: {info.name}", border_style="green")
        table.add_column("Property", style="bold cyan")
        table.add_column("Value", style="white")

        table.add_row("Model Name", info.name)
        table.add_row("Family", info.family or "N/A")
        table.add_row("Parameter Size", info.parameter_size or "N/A")
        table.add_row("Quantization", info.quantization_level or "N/A")
        table.add_row("Context Length", str(info.context_length or "N/A"))
        console.print(table)
    except Exception as e:
        console.print(f"[bold red][FAIL][/bold red] Model verification failed: {e}")
        sys.exit(1)


@main.command()
@click.option("--workspace", "-w", default=".", help="Workspace root directory.")
def tools(workspace: str):
    """List all registered tools, their descriptions, and argument schemas."""
    ws = Path(workspace).resolve()
    reg = ToolRegistry.create_default(workspace_root=ws)

    table = Table(title="CodeLens AI Registered Tools", border_style="cyan")
    table.add_column("Tool Name", style="bold green", width=22)
    table.add_column("Description", style="white")
    table.add_column("Arguments", style="dim cyan")

    for name in reg.list_names():
        tool = reg.get(name)
        if not tool:
            continue
        props = tool.args_schema.model_json_schema().get("properties", {})
        args_str = ", ".join(props.keys()) if props else "None"
        table.add_row(name, tool.description, args_str)

    # Add finish pseudo-tool
    table.add_row("finish", "Terminate agent loop and provide final root-cause explanation.", "root_cause, solution_summary, test_results_summary, remaining_uncertainty")

    console.print(table)


@main.command()
@click.option("--workspace", "-w", default=".", help="Workspace root directory.")
@click.option("--model", "-m", default="gemma4:e2b", help="Ollama model name.")
def demo(workspace: str, model: str):
    """Run an automated demonstration debugging a sample buggy program."""
    from codelens.orchestrator.agent import CodeLensAgent

    ws = Path(workspace).resolve()
    console.print(Panel(
        "[bold cyan]CodeLens AI - Autonomous Debugging Demonstration[/bold cyan]\n"
        "This demo runs the agent against a sample program with a logic/off-by-one error,\n"
        "inspects the code, runs tests in the sandbox, generates a patch, verifies with tests, and explains the root cause.",
        border_style="cyan"
    ))

    # Setup sample program in examples/ if not present
    sample_file = ws / "examples" / "off_by_one.py"
    if not sample_file.exists():
        sample_file.parent.mkdir(parents=True, exist_ok=True)
        sample_file.write_text(
            'def find_median(sorted_numbers):\n'
            '    """Return median of a non-empty sorted list of numbers."""\n'
            '    n = len(sorted_numbers)\n'
            '    if n == 0:\n'
            '        raise ValueError("List cannot be empty")\n'
            '    # BUG: Off-by-one error in indexing\n'
            '    mid = n // 2\n'
            '    if n % 2 == 1:\n'
            '        return sorted_numbers[mid + 1]\n'
            '    else:\n'
            '        return (sorted_numbers[mid - 1] + sorted_numbers[mid]) / 2.0\n',
            encoding="utf-8"
        )

    task = (
        f"The function in '{sample_file.relative_to(ws).as_posix()}' produces wrong results for odd-length lists. "
        "Inspect the file, write a test to reproduce the bug, apply a safe patch, run tests to verify, and explain the root cause."
    )

    cfg = CodeLensConfig(
        model_name=model,
        workspace_dir=ws,
        max_iterations=8,
        require_approval=False,  # Auto-approve for demo
        sandbox_enabled=True,
    )
    agent = CodeLensAgent(config=cfg)
    agent.run(task_description=task, target_file=sample_file.relative_to(ws).as_posix())


@main.command()
@click.option("--host", default="127.0.0.1", help="Host interface to bind server.")
@click.option("--port", default=8000, type=int, help="Port to listen on.")
@click.option("--no-browser", is_flag=True, default=False, help="Do not open browser automatically.")
def web(host: str, port: int, no_browser: bool):
    """Launch the clean, attractive CodeLens AI web interface."""
    import uvicorn
    import webbrowser
    url = f"http://{host}:{port}"
    console.print(Panel(
        f"[bold cyan]Launching CodeLens AI Web App[/bold cyan]\n"
        f"Server URL: [bold green]{url}[/bold green]\n"
        f"Open in your browser to interact with the full-stack agent interface.",
        border_style="cyan"
    ))
    if not no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    uvicorn.run("codelens.web.app:app", host=host, port=port)


if __name__ == "__main__":
    main()
