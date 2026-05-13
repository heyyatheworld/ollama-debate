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
    build_participants,
    check_ollama_running,
    ensure_models_available,
    llm_options_from_config,
    load_config,
    save_debate_to_md,
    SpeechTurn,
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


def _print_speech(entry: SpeechTurn, *, round_num: int, total_rounds: int) -> None:
    """Print one participant's speech in a Rich panel."""
    name = entry.name
    icon = entry.icon
    think = (entry.think or "").strip()
    speech = entry.speech
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
            subtitle=f"Round {round_num} of {total_rounds}",
            subtitle_align="right",
            border_style=border_style,
            width=PANEL_WIDTH,
        )
    )
    p = entry.prompt_tokens
    c = entry.completion_tokens
    console.print(f"[dim]This reply — prompt: {p}, completion: {c}, subtotal: {p + c}[/]")
    console.print()


def _print_token_usage_table(rows: list[Dict[str, Any]]) -> None:
    """Print a Rich table of per-call token usage (round, speaker, counts)."""
    table = Table(title="Token usage by call", show_header=True, header_style="bold cyan")
    table.add_column("Round", style="dim", justify="right")
    table.add_column("Speaker", style="bold")
    table.add_column("Prompt", justify="right")
    table.add_column("Completion", justify="right")
    table.add_column("Subtotal", justify="right")
    for r in rows:
        table.add_row(
            str(r["round"]),
            r["speaker"],
            str(r["prompt"]),
            str(r["completion"]),
            str(r["subtotal"]),
        )
    console.print(table)
    console.print()


def _print_session_token_footer(result: BattleResult) -> None:
    """Print aggregate token counts for the whole run."""
    console.print(
        "[dim]Session total (all LLM calls) — prompt: "
        f"{result.token_prompt}, completion: {result.token_completion}, "
        f"total: {result.token_total}[/]"
    )
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
    parser.add_argument(
        "--token-table",
        action="store_true",
        help="After the debate, print a Rich table of per-call token usage.",
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
    llm_options = llm_options_from_config(config)

    machiavelli, socrates, judge = build_participants(
        config,
        model_m=args.model_m,
        model_s=args.model_s,
        model_judge=args.judge,
    )

    arena = Arena(machiavelli=machiavelli, socrates=socrates, judge=judge, llm_options=llm_options)

    n_rounds = int(args.rounds)
    speech_idx = 0
    token_rows: list[Dict[str, Any]] = []

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

    def on_speech(entry: SpeechTurn) -> None:
        nonlocal speech_idx
        speech_idx += 1
        round_num = (speech_idx - 1) // 2 + 1
        if (speech_idx - 1) % 2 == 0:
            console.rule(f"[bold cyan]Round {round_num} / {n_rounds}[/]", align="center")
        _print_speech(entry, round_num=round_num, total_rounds=n_rounds)
        if args.token_table:
            token_rows.append(
                {
                    "round": round_num,
                    "speaker": entry.name,
                    "prompt": entry.prompt_tokens,
                    "completion": entry.completion_tokens,
                    "subtotal": entry.prompt_tokens + entry.completion_tokens,
                }
            )

    def on_verdict(text: str, p: int, c: int) -> None:
        console.rule("[bold yellow]Judge[/]", align="center")
        console.print(
            Panel(
                Text(text, style="bold"),
                title="⚖️  VERDICT",
                border_style="gold1",
                width=PANEL_WIDTH,
            )
        )
        console.print(f"[dim]Judge call only — prompt: {p}, completion: {c}, subtotal: {p + c}[/]")
        console.print()
        if args.token_table:
            token_rows.append(
                {
                    "round": "—",
                    "speaker": "Judge",
                    "prompt": p,
                    "completion": c,
                    "subtotal": p + c,
                }
            )

    try:
        result: BattleResult = arena.run_battle(
            args.topic,
            rounds=n_rounds,
            on_speech=on_speech,
            on_verdict=on_verdict,
        )
    except KeyboardInterrupt:  # pragma: no cover - interactive
        console.print("[yellow]Debate interrupted by user.[/]")
        sys.exit(130)
    except Exception as e:  # pragma: no cover - defensive
        _error_exit(f"Unexpected error while running debate: {e}")

    if args.token_table and token_rows:
        _print_token_usage_table(token_rows)
    _print_session_token_footer(result)

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

