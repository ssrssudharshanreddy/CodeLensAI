"""
Structured console and file logging using Rich for CodeLens AI.
Clearly distinguishes tool observations (ground truth) from model assumptions.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text


console = Console()
error_console = Console(stderr=True)


class AgentLogger:
    """Provides structured, formatted terminal output and logging."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.console = console

    def banner(self, title: str, subtitle: Optional[str] = None):
        """Render CodeLens banner."""
        text = Text()
        text.append(f"  {title}\n", style="bold cyan")
        if subtitle:
            text.append(f"  {subtitle}", style="dim")
        panel = Panel(text, border_style="cyan", padding=(1, 2))
        self.console.print(panel)

    def log_thought(self, iteration: int, max_iter: int, thought: str, plan: Optional[str] = None):
        """Display model reasoning and plan."""
        content = Text()
        content.append(f"[Iteration {iteration}/{max_iter}]\n", style="bold blue")
        if plan:
            content.append("Plan: ", style="bold magenta")
            content.append(f"{plan}\n\n", style="italic")
        content.append("Reasoning: ", style="bold")
        content.append(thought)
        
        panel = Panel(
            content,
            title=f"[bold blue]Step {iteration}: Agent Analysis[/bold blue]",
            border_style="blue",
            padding=(0, 1),
        )
        self.console.print(panel)

    def log_tool_call(self, tool_name: str, args: dict[str, Any]):
        """Display tool dispatch."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="bold yellow")
        table.add_column("Value", style="cyan")

        for k, v in args.items():
            val_str = str(v)
            if len(val_str) > 100:
                val_str = val_str[:97] + "..."
            table.add_row(f"{k}:", val_str)

        panel = Panel(
            table,
            title=f"[bold yellow]-> Executing Tool: [green]{tool_name}[/green][/bold yellow]",
            border_style="yellow",
            padding=(0, 1),
        )
        self.console.print(panel)

    def log_tool_observation(self, tool_name: str, success: bool, output: str, is_assumption: bool = False):
        """
        Display tool observation.
        CRITICAL: Clearly highlights whether this is an actual verified tool observation or model assumption.
        """
        status_label = "[green]SUCCESS[/green]" if success else "[red]FAILURE[/red]"
        tag = "[bold cyan][VERIFIED TOOL OBSERVATION][/bold cyan]" if not is_assumption else "[bold red][UNVERIFIED MODEL ASSUMPTION][/bold red]"
        
        # Truncate overly long output in summary, full output in verbose
        display_output = output
        if not self.verbose and len(output) > 800:
            display_output = output[:797] + "\n... [truncated, use --verbose to see full output]"

        panel = Panel(
            display_output,
            title=f"{tag} - {tool_name} ({status_label})",
            border_style="green" if success else "red",
            padding=(0, 1),
        )
        self.console.print(panel)

    def log_diff(self, filename: str, diff_text: str):
        """Render a unified code diff."""
        syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=True)
        panel = Panel(
            syntax,
            title=f"[bold magenta]Proposed Patch: {filename}[/bold magenta]",
            border_style="magenta",
            padding=(0, 1),
        )
        self.console.print(panel)

    def log_solution(self, root_cause: str, explanation: str, test_summary: str, uncertainty: str):
        """Display the final structured explanation."""
        table = Table(title="[bold green]Task Resolution Summary[/bold green]", border_style="green")
        table.add_column("Category", style="bold cyan", width=22)
        table.add_column("Details", style="white")

        table.add_row("Root Cause", root_cause)
        table.add_row("Correction Explanation", explanation)
        table.add_row("Verified Test Results", test_summary)
        table.add_row("Remaining Uncertainty", uncertainty)

        self.console.print(table)

    def log_error(self, message: str):
        """Display error banner."""
        error_console.print(f"[bold red][ERROR][/bold red] {message}")

    def log_warning(self, message: str):
        """Display warning banner."""
        self.console.print(f"[bold yellow][WARNING][/bold yellow] {message}")

    def log_info(self, message: str):
        """Display informational note."""
        self.console.print(f"[bold cyan][INFO][/bold cyan] {message}")
