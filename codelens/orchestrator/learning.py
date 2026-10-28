"""
Student learning mode for CodeLens AI.
Provides progressive three-stage assistance:
  1. Hint (conceptual clue without revealing fix)
  2. Guidance (suggested next step / area to inspect)
  3. Solution (root cause, explanation, and corrected code)
The user explicitly controls when the agent reveals each stage.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from codelens.llm.ollama_client import OllamaClient


class LearningStage(str, Enum):
    HINT = "hint"
    GUIDANCE = "guidance"
    SOLUTION = "solution"


class LearningSession:
    """Manages progressive multi-stage learning assistance for students."""

    def __init__(self, client: OllamaClient, model: Optional[str] = None):
        self.client = client
        self.model = model or client.default_model
        self.console = Console()

    def generate_hint(self, code: str, problem_description: str) -> str:
        """Stage 1: Generate a conceptual clue without spoiling the solution."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a computer science tutor helping a student learn debugging. "
                    "Provide ONLY a conceptual clue or hint. "
                    "CRITICAL RULES: "
                    "1. DO NOT give away the fix or write corrected code. "
                    "2. DO NOT cite the exact line number to edit. "
                    "3. Highlight the general programming concept, boundary condition, or invariant to rethink. "
                    "Keep it to 2-4 sentences."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Problem:\n{problem_description}\n\n"
                    f"Student Code:\n{code}\n\n"
                    "Give me a conceptual hint."
                ),
            },
        ]
        return self.client.chat(messages=messages, model=self.model, format_json=False, temperature=0.2)

    def generate_guidance(self, code: str, problem_description: str, hint: str) -> str:
        """Stage 2: Suggest a concrete next step or specific area to inspect."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a computer science tutor. The student reviewed the conceptual hint and wants guidance. "
                    "Provide a targeted next step: "
                    "1. Point to the specific function, loop, or variable expression that requires attention. "
                    "2. Suggest what edge case, input value, or mental trace to test. "
                    "3. DO NOT write the complete replacement code yet."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Problem:\n{problem_description}\n\n"
                    f"Student Code:\n{code}\n\n"
                    f"Prior Hint:\n{hint}\n\n"
                    "Provide targeted guidance."
                ),
            },
        ]
        return self.client.chat(messages=messages, model=self.model, format_json=False, temperature=0.2)

    def generate_solution(self, code: str, problem_description: str) -> str:
        """Stage 3: Deliver the complete root cause, correction explanation, and code fix."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert pair programmer. Provide the complete solution for the student: "
                    "1. Explain the root cause clearly. "
                    "2. Provide the corrected code snippet. "
                    "3. Explain why the correction works and how to prevent similar bugs in the future."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Problem:\n{problem_description}\n\n"
                    f"Student Code:\n{code}\n\n"
                    "Provide the complete solution with explanation."
                ),
            },
        ]
        return self.client.chat(messages=messages, model=self.model, format_json=False, temperature=0.1)

    def interactive_walkthrough(self, code: str, problem_description: str):
        """Interactive CLI session walking the student through Hint -> Guidance -> Solution."""
        self.console.print("\n[bold cyan]=== CodeLens Student Learning Mode ===[/bold cyan]")
        self.console.print("[dim]Learn by debugging. You control when each assistance level is revealed.[/dim]\n")

        # Stage 1: Hint
        self.console.print("[bold yellow][Stage 1: Conceptual Hint][/bold yellow]")
        with self.console.status("[bold green]Generating conceptual hint with local model..."):
            hint = self.generate_hint(code, problem_description)

        self.console.print(Panel(hint.strip(), title="[bold yellow]Hint[/bold yellow]", border_style="yellow"))

        choice = Prompt.ask(
            "\nWould you like guidance for the next step?",
            choices=["next", "solve", "quit"],
            default="next",
        )
        if choice == "quit":
            self.console.print("[dim]Learning session paused. Happy coding![/dim]")
            return

        # Stage 2: Guidance
        if choice in ("next", "guidance"):
            self.console.print("\n[bold magenta][Stage 2: Targeted Guidance][/bold magenta]")
            with self.console.status("[bold green]Generating guidance..."):
                guidance = self.generate_guidance(code, problem_description, hint)

            self.console.print(Panel(guidance.strip(), title="[bold magenta]Guidance[/bold magenta]", border_style="magenta"))

            choice = Prompt.ask(
                "\nWould you like to reveal the full solution?",
                choices=["solve", "quit"],
                default="solve",
            )
            if choice == "quit":
                self.console.print("[dim]Good luck investigating the bug![/dim]")
                return

        # Stage 3: Solution
        self.console.print("\n[bold green][Stage 3: Full Solution & Explanation][/bold green]")
        with self.console.status("[bold green]Generating solution..."):
            solution = self.generate_solution(code, problem_description)

        self.console.print(Panel(solution.strip(), title="[bold green]Solution[/bold green]", border_style="green"))
