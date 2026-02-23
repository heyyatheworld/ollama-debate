"""Ollama Debate: historical court debate between Socrates and Machiavelli via Ollama models."""
import argparse
import os
import re
import sys
from datetime import date
from pathlib import Path

import ollama
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()
PANEL_WIDTH = console.width
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def _error_exit(title, message):
    """Print a red Rich panel and exit with code 1."""
    console.print(Panel(message, title=f"[bold red]{title}[/]", border_style="red", width=min(72, PANEL_WIDTH)))
    sys.exit(1)


def _show_error(title, message):
    """Print a red Rich panel (no exit)."""
    console.print(Panel(message, title=f"[bold red]{title}[/]", border_style="red", width=min(72, PANEL_WIDTH)))


def check_ollama_running():
    """Verify Ollama server is reachable; exit with Rich error if not."""
    try:
        ollama.list()
    except Exception as e:
        _error_exit(
            "Error: Ollama server is not running.",
            f"[red]{e!s}[/]\n\n"
            "Please start Ollama app or run [bold]ollama serve[/] in a terminal.",
        )


def _model_in_list(model_name, listed_names):
    """Check if model_name is available (exact or as base name)."""
    if model_name in listed_names:
        return True
    base = model_name.split(":")[0] if ":" in model_name else model_name
    return any(n == model_name or n.startswith(base + ":") for n in listed_names)


def ensure_models_available(model_m, model_s, model_judge):
    """Ensure all three models exist; pull any missing one with rich.status."""
    try:
        data = ollama.list()
    except Exception as e:
        _error_exit("Ollama error", str(e))
    models = data.get("models") if isinstance(data, dict) else data
    if not isinstance(models, list):
        models = []
    listed_names = set()
    for m in models:
        name = (m.get("name") or m.get("model") or "")
        if name:
            listed_names.add(name)
    for _label, model_name in [("Machiavelli", model_m), ("Socrates", model_s), ("Judge", model_judge)]:
        if not _model_in_list(model_name, listed_names):
            with console.status(f"Model [bold]{model_name}[/] not found. Pulling from Ollama registry...", spinner="dots"):
                try:
                    ollama.pull(model_name)
                except Exception as e:
                    _error_exit(f"Failed to pull model {model_name}", str(e))


def load_config():
    """Load config.yaml; exit with a Rich error if the file is missing. Returns config dict."""
    if not CONFIG_PATH.is_file():
        console.print(
            Panel(
                f"[red]Config file not found:[/] [bold]{CONFIG_PATH}[/]\n\n"
                "Create [bold]config.yaml[/] in the project root with [cyan]models[/], "
                "[cyan]prompts[/], and [cyan]settings[/] sections. See the project README or "
                "copy from an example.",
                title="[bold red]Error[/]",
                border_style="red",
            )
        )
        sys.exit(1)
    with open(CONFIG_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data:
        console.print("[red]config.yaml is empty.[/]")
        sys.exit(1)
    return data


def clean_text(text):
    """Removes excessive line breaks and whitespace."""
    text = re.sub(r'\n+', '\n', text).strip()
    return text

def extract_think(text):
    """Separates <think> tags from the main response."""
    think = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    content = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    think_text = think.group(1).strip() if think else ""
    if len(think_text) > 200:
        think_text = think_text[:200] + "..."
        
    return clean_text(think_text), clean_text(content)

def _token_counts(response):
    """Extract prompt and completion token counts from Ollama response."""
    prompt = response.get("prompt_eval_count") or 0
    completion = response.get("eval_count") or 0
    return prompt, completion


def print_speech(name, think, speech, icon, border_style, prompt_tokens=None, completion_tokens=None):
    """Prints a participant's speech block in a rich Panel."""
    body = Text()
    if think:
        body.append("🔍 Thoughts: ", style="dim")
        body.append(think, style="dim italic")
        body.append("\n\n")
    body.append(speech)
    console.print(Panel(body, title=f"{icon} {name.upper()}", border_style=border_style, width=PANEL_WIDTH))
    if prompt_tokens is not None and completion_tokens is not None:
        total = prompt_tokens + completion_tokens
        console.print(f"[dim]Tokens: prompt: {prompt_tokens}, completion: {completion_tokens}, total: {total}[/]")
    console.print()

def _topic_to_slug(topic):
    """Convert topic to a short filename-safe slug."""
    slug = topic.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[-\s]+", "_", slug)
    return slug[:240] if slug else "debate"

def _build_markdown(topic, model_m, model_s, model_judge, transcript_entries, verdict, token_stats=None):
    """Build full Markdown content for the debate file."""
    lines = [
        f"# Debate: {topic}",
        "",
        "## Participants",
        "",
        f"- **Socrates:** `{model_s}`",
        f"- **Machiavelli:** `{model_m}`",
        f"- **Judge:** `{model_judge}`",
        "",
        "## Transcript",
        "",
    ]
    for entry in transcript_entries:
        name = entry["name"]
        icon = entry["icon"]
        think = entry.get("think", "").strip()
        speech = entry["speech"]
        if think:
            lines.append("<details><summary>Thoughts</summary>")
            lines.append("")
            lines.append(think)
            lines.append("")
            lines.append("</details>")
            lines.append("")
        lines.append(f"> **{icon} {name}:**")
        for line in speech.split("\n"):
            lines.append(f"> {line}")
        lines.append("")
    lines.extend(["## Verdict", "", (verdict or "").strip(), ""])
    if token_stats:
        lines.extend([
            "",
            "## Token usage",
            "",
            f"- **Prompt tokens:** {token_stats['prompt']}",
            f"- **Completion tokens:** {token_stats['completion']}",
            f"- **Total:** {token_stats['total']}",
            "",
        ])
    return "\n".join(lines)

def save_debate_to_md(topic, model_m, model_s, model_judge, transcript_entries, verdict, token_stats=None, debates_dir="debates"):
    """Save debate to debates_dir/YYYY-MM-DD_slug.md; create dir if missing. Returns path (str).
    Raises OSError on write failure (caller should show Rich error)."""
    out_dir = Path(debates_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    slug = _topic_to_slug(topic)
    filename = f"{today}_{slug}.md"
    filepath = out_dir / filename
    md = _build_markdown(topic, model_m, model_s, model_judge, transcript_entries, verdict or "", token_stats)
    filepath.write_text(md, encoding="utf-8")
    return str(filepath)

def _make_result(topic, model_m, model_s, model_judge, transcript_entries, verdict_text, total_prompt, total_completion):
    """Build the result dict for start_court (full or partial)."""
    token_stats = {
        "prompt": total_prompt,
        "completion": total_completion,
        "total": total_prompt + total_completion,
    }
    return {
        "topic": topic,
        "model_m": model_m,
        "model_s": model_s,
        "model_judge": model_judge,
        "transcript_entries": transcript_entries,
        "verdict": verdict_text,
        "token_stats": token_stats,
    }


def start_court(model_m, model_s, model_judge, topic, rounds=3, config_prompts=None, llm_options=None):
    """Run the debate. Returns (result_dict, interrupted_by_user)."""
    if config_prompts is None:
        config_prompts = {}
    prompts = {
        "Socrates": config_prompts.get("socrates", ""),
        "Machiavelli": config_prompts.get("machiavelli", ""),
    }
    judge_system_prompt = config_prompts.get(
        "judge",
        "You are the Supreme Judge. Analyze the debate. Who won: Socrates or Machiavelli? Answer briefly and strictly in English.",
    )
    if llm_options is None:
        llm_options = {"num_predict": 350, "temperature": 0.8, "num_ctx": 2048}

    history_m = []
    history_s = []
    transcript_plain = []
    transcript_entries = []
    total_prompt, total_completion = 0, 0

    console.print()
    console.print(Panel(
        f"[bold cyan]«{topic}»[/bold cyan]",
        title="🏛  HISTORICAL COURT",
        border_style="cyan",
        width=PANEL_WIDTH,
    ))
    console.print()

    current_input = f"Start a debate on the topic: {topic}. State your position briefly."

    try:
        for i in range(rounds):
            # Machiavelli's turn
            history_m.append({"role": "user", "content": current_input})
            with console.status("[bold magenta]Machiavelli is thinking...[/]", spinner="dots"):
                res_m = ollama.chat(
                    model=model_m,
                    messages=[{"role": "system", "content": prompts["Machiavelli"]}] + history_m,
                    options=llm_options,
                )
            prompt_m, completion_m = _token_counts(res_m)
            total_prompt += prompt_m
            total_completion += completion_m
            think_m, speech_m = extract_think(res_m["message"]["content"])
            history_m.append({"role": "assistant", "content": speech_m})
            transcript_plain.append(f"Machiavelli: {speech_m}")
            transcript_entries.append({"name": "Machiavelli", "icon": "🦊", "think": think_m, "speech": speech_m})
            print_speech("Machiavelli", think_m, speech_m, "🦊", "magenta", prompt_m, completion_m)

            # Socrates's turn
            history_s.append({"role": "user", "content": speech_m})
            with console.status("[bold cyan]Socrates is thinking...[/]", spinner="dots"):
                res_s = ollama.chat(
                    model=model_s,
                    messages=[{"role": "system", "content": prompts["Socrates"]}] + history_s,
                    options=llm_options,
                )
            prompt_s, completion_s = _token_counts(res_s)
            total_prompt += prompt_s
            total_completion += completion_s
            think_s, speech_s = extract_think(res_s["message"]["content"])
            history_s.append({"role": "assistant", "content": speech_s})
            transcript_plain.append(f"Socrates: {speech_s}")
            transcript_entries.append({"name": "Socrates", "icon": "🏛", "think": think_s, "speech": speech_s})
            print_speech("Socrates", think_s, speech_s, "🏛", "cyan", prompt_s, completion_s)

            current_input = speech_s

        # Judge's verdict
        full_text = "\n".join(transcript_plain)
        with console.status("[bold yellow]Judge is deliberating...[/]", spinner="dots"):
            res_j = ollama.chat(
                model=model_judge,
                messages=[
                    {"role": "system", "content": judge_system_prompt},
                    {"role": "user", "content": full_text},
                ],
            )
        prompt_j, completion_j = _token_counts(res_j)
        total_prompt += prompt_j
        total_completion += completion_j
        verdict_text = res_j["message"]["content"].strip()
        console.print(Panel(Text(verdict_text, style="bold"), title="⚖️  VERDICT", border_style="gold1", width=PANEL_WIDTH))
        console.print(f"[dim]Tokens: prompt: {prompt_j}, completion: {completion_j}, total: {prompt_j + completion_j}[/]")
        console.print()
        return _make_result(topic, model_m, model_s, model_judge, transcript_entries, verdict_text, total_prompt, total_completion), False

    except KeyboardInterrupt:
        verdict_partial = "(Debate interrupted by user.)"
        return _make_result(topic, model_m, model_s, model_judge, transcript_entries, verdict_partial, total_prompt, total_completion), True

DEFAULT_TOPIC = "What is better for society: total state control or complete anarchy and absence of vertical power structure"


def parse_args(config):
    """Parse CLI args; defaults come from config so CLI overrides config."""
    models = config.get("models") or {}
    settings = config.get("settings") or {}
    parser = argparse.ArgumentParser(description="Run a historical court debate between Socrates and Machiavelli using Ollama models.")
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


if __name__ == "__main__":
    try:
        config = load_config()
        args = parse_args(config)

        check_ollama_running()
        ensure_models_available(args.model_m, args.model_s, args.judge)

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

        settings = config.get("settings") or {}
        llm_options = {
            "num_predict": settings.get("num_predict", 350),
            "temperature": settings.get("temperature", 0.8),
            "num_ctx": settings.get("num_ctx", 2048),
        }
        result, interrupted = start_court(
            model_m=args.model_m,
            model_s=args.model_s,
            model_judge=args.judge,
            topic=args.topic,
            rounds=args.rounds,
            config_prompts=config.get("prompts"),
            llm_options=llm_options,
        )
        if interrupted:
            console.print("[yellow]Debate interrupted by user. Saving partial log...[/]")

        debates_dir = settings.get("debates_dir", "debates")
        try:
            filepath = save_debate_to_md(
                topic=result["topic"],
                model_m=result["model_m"],
                model_s=result["model_s"],
                model_judge=result["model_judge"],
                transcript_entries=result["transcript_entries"],
                verdict=result["verdict"],
                token_stats=result.get("token_stats"),
                debates_dir=debates_dir,
            )
            console.print(f"[dim]Debate saved to {filepath}[/]")
        except OSError as e:
            _show_error("Error writing debate log", f"Could not save file: {e}")
            sys.exit(1)
    except KeyboardInterrupt:
        console.print("[yellow]Debate interrupted by user. Saving partial log...[/]")
        sys.exit(130)
    except Exception as e:
        _show_error("Error", f"{type(e).__name__}: {e}")
        sys.exit(1)
