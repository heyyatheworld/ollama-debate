"""Command-line interface for the Ollama Debate project (CLI layer)."""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from arena import (
    Arena,
    BattleResult,
    Participant,
    check_ollama_running,
    ensure_models_available,
    load_config,
    save_debate_to_md,
)


console = Console()
PANEL_WIDTH = console.width

DEFAULT_TOPIC = (
    "What is better for society: total state control or complete anarchy and absence of vertical power structure"
)


def _error_exit(message: str, *, title: str = "Error") -> None:
    """Print a red Rich panel and exit with code 1."""
    console.print(
        Panel(
            message,
            title=f"[bold red]{title}[/]",
            border_style="red",
            width=min(72, PANEL_WIDTH),
        )
    )
    sys.exit(1)


def _print_settings_table(args: argparse.Namespace) -> None:
    table = Table(title="Debate settings", show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="dim")
    table.add_column("Value")
    table.add_row("Topic", args.topic)
    table.add_row("Rounds", str(args.rounds))
    table.add_row("Machiavelli (model)", args.model_m)
    table.add_row("Socrates (model)", args.model_s)
    table.add_row("Judge (model)", args.judge)
    console.print(table)
    console.print()


def _print_speech(entry: Dict[str, Any]) -> None:
    """Print one participant's speech in a Rich panel."""
    name = entry["name"]
    icon = entry["icon"]
    think = (entry.get("think") or "").strip()
    speech = entry["speech"]
    border_style = "magenta" if name == "Machiavelli" else "cyan"

    body = Text()
    if think:
        body.append("🔍 Thoughts: ", style="dim")
        body.append(think, style="dim italic")
        body.append("\n\n")
    body.append(speech)

    console.print(
        Panel(
            body,
            title=f"{icon} {name.upper()}",
            border_style=border_style,
            width=PANEL_WIDTH,
        )
    )
    p = entry.get("prompt_tokens")
    c = entry.get("completion_tokens")
    if p is not None and c is not None:
        console.print(f"[dim]Tokens: prompt: {p}, completion: {c}, total: {p + c}[/]")
    console.print()


def parse_args(config: Dict[str, Any]) -> argparse.Namespace:
    """Parse CLI args; defaults come from config so CLI overrides config."""
    models = config.get("models") or {}
    settings = config.get("settings") or {}
    parser = argparse.ArgumentParser(
        description="Run a historical court debate between Socrates and Machiavelli using Ollama models."
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=DEFAULT_TOPIC,
        help="Debate topic (required or use default).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=settings.get("default_rounds", 2),
        help="Number of debate rounds (default from config).",
    )
    parser.add_argument(
        "--model_m",
        type=str,
        default=models.get("machiavelli", "llama3:latest"),
        help="Ollama model for Machiavelli (default from config).",
    )
    parser.add_argument(
        "--model_s",
        type=str,
        default=models.get("socrates", "qwen2.5-coder:7b"),
        help="Ollama model for Socrates (default from config).",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default=models.get("judge", "llama3.2:latest"),
        help="Ollama model for the judge (default from config).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the CLI."""
    try:
        config = load_config()
    except FileNotFoundError as e:
        _error_exit(str(e), title="Config error")
    except ValueError as e:
        _error_exit(str(e), title="Config error")
    except Exception as e:  # pragma: no cover - defensive
        _error_exit(f"Failed to load config: {e}", title="Config error")

    args = parse_args(config)

    try:
        check_ollama_running()
    except Exception as e:
        _error_exit(f"Ollama server is not running: {e}\n\nPlease start Ollama app or run 'ollama serve'.")

    try:
        ensure_models_available(args.model_m, args.model_s, args.judge)
    except Exception as e:
        _error_exit(f"Model error: {e}")

    _print_settings_table(args)

    settings = config.get("settings") or {}
    llm_options = {
        "num_predict": settings.get("num_predict", 350),
        "temperature": settings.get("temperature", 0.8),
        "num_ctx": settings.get("num_ctx", 2048),
    }
    prompts = config.get("prompts") or {}

    machiavelli = Participant(
        name="Machiavelli",
        model=args.model_m,
        system_prompt=prompts.get(
            "machiavelli",
            "You are Machiavelli. Speak English. You are a cynical pragmatist. Defend state interest and order at any cost.",
        ),
        icon="🦊",
    )
    socrates = Participant(
        name="Socrates",
        model=args.model_s,
        system_prompt=prompts.get(
            "socrates",
            "You are Socrates. Speak English. Use Socratic method: ask short, probing questions. Be humble but ironic.",
        ),
        icon="🏛",
    )
    judge = Participant(
        name="Judge",
        model=args.judge,
        system_prompt=prompts.get(
            "judge",
            "You are the Supreme Judge. Analyze the debate. Who won: Socrates or Machiavelli? Answer briefly and strictly in English.",
        ),
        icon="⚖️",
    )

    arena = Arena(machiavelli=machiavelli, socrates=socrates, judge=judge, llm_options=llm_options)

    console.print()
    console.print(
        Panel(
            f"[bold cyan]«{args.topic}»[/bold cyan]",
            title="🏛  HISTORICAL COURT",
            border_style="cyan",
            width=PANEL_WIDTH,
        )
    )
    console.print()

    try:
        result: BattleResult = arena.run_battle(args.topic, rounds=int(args.rounds))
    except KeyboardInterrupt:  # pragma: no cover - interactive
        console.print("[yellow]Debate interrupted by user.[/]")
        sys.exit(130)
    except Exception as e:  # pragma: no cover - defensive
        _error_exit(f"Unexpected error while running debate: {e}")

    for entry in result.transcript_entries:
        _print_speech(entry)

    console.print(
        Panel(
            Text(result.verdict, style="bold"),
            title="⚖️  VERDICT",
            border_style="gold1",
            width=PANEL_WIDTH,
        )
    )
    console.print(
        f"[dim]Tokens: prompt: {result.token_prompt}, completion: {result.token_completion}, "
        f"total: {result.token_total}[/]"
    )
    console.print()

    debates_dir = settings.get("debates_dir", "debates")
    try:
        filepath = save_debate_to_md(
            topic=result.topic,
            model_m=result.machiavelli_model,
            model_s=result.socrates_model,
            model_judge=result.judge_model,
            transcript_entries=result.transcript_entries,
            verdict=result.verdict,
            token_stats={
                "prompt": result.token_prompt,
                "completion": result.token_completion,
                "total": result.token_total,
            },
            debates_dir=debates_dir,
        )
        console.print(f"[dim]Debate saved to {filepath}[/]")
    except OSError as e:
        _error_exit(f"Could not save debate log: {e}", title="I/O error")


if __name__ == "__main__":
    main()

