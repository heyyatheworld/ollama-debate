"""Core business logic for the Ollama Debate project.

This module contains:
- Config and environment helpers (config.yaml, Ollama availability, models).
- Pure debate logic in the Arena class (no CLI or web UI code).
- Utilities for saving debate transcripts to Markdown.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ollama
import yaml
import re


CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def clean_text(text: str) -> str:
    """Remove excessive line breaks and surrounding whitespace."""
    text = re.sub(r"\n+", "\n", text).strip()
    return text


def extract_think(text: str) -> Tuple[str, str]:
    """Separate <think>...</think> block from the visible content."""
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    content = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    think_text = think_match.group(1).strip() if think_match else ""
    if len(think_text) > 200:
        think_text = think_text[:200] + "..."

    return clean_text(think_text), clean_text(content)


def token_counts(response: Dict[str, Any]) -> Tuple[int, int]:
    """Extract prompt and completion token counts from an Ollama response dict."""
    prompt = response.get("prompt_eval_count") or 0
    completion = response.get("eval_count") or 0
    return int(prompt), int(completion)


def topic_to_slug(topic: str) -> str:
    """Convert a topic to a filename-safe slug (max 240 chars)."""
    slug = topic.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[-\s]+", "_", slug)
    return slug[:240] if slug else "debate"


def build_markdown(
    topic: str,
    model_m: str,
    model_s: str,
    model_judge: str,
    transcript_entries: List[Dict[str, Any]],
    verdict: str,
    token_stats: Optional[Dict[str, int]] = None,
) -> str:
    """Build full Markdown content for the debate file."""
    lines: List[str] = [
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
        think = (entry.get("think") or "").strip()
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
        lines.extend(
            [
                "",
                "## Token usage",
                "",
                f"- **Prompt tokens:** {token_stats['prompt']}",
                f"- **Completion tokens:** {token_stats['completion']}",
                f"- **Total:** {token_stats['total']}",
                "",
            ]
        )
    return "\n".join(lines)


def save_debate_to_md(
    topic: str,
    model_m: str,
    model_s: str,
    model_judge: str,
    transcript_entries: List[Dict[str, Any]],
    verdict: str,
    token_stats: Optional[Dict[str, int]] = None,
    debates_dir: str = "debates",
) -> str:
    """Save debate to debates_dir/YYYY-MM-DD_slug.md, creating directory if needed."""
    out_dir = Path(debates_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    slug = topic_to_slug(topic)
    filename = f"{today}_{slug}.md"
    filepath = out_dir / filename
    md = build_markdown(topic, model_m, model_s, model_judge, transcript_entries, verdict or "", token_stats)
    filepath.write_text(md, encoding="utf-8")
    return str(filepath)


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load config.yaml and return its dict; raise on error."""
    cfg_path = path or CONFIG_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data:
        raise ValueError("config.yaml is empty.")
    return data


def check_ollama_running() -> None:
    """Raise RuntimeError if Ollama server is not reachable."""
    try:
        ollama.list()
    except Exception as e:  # pragma: no cover - depends on external service
        raise RuntimeError(f"Ollama server is not running: {e}") from e


def _model_in_list(model_name: str, listed_names: List[str]) -> bool:
    if model_name in listed_names:
        return True
    base = model_name.split(":")[0] if ":" in model_name else model_name
    return any(n == model_name or n.startswith(base + ":") for n in listed_names)


def ensure_models_available(model_m: str, model_s: str, model_judge: str) -> None:
    """Ensure all three models are present locally; pull any missing ones."""
    try:
        data = ollama.list()
    except Exception as e:  # pragma: no cover - depends on external service
        raise RuntimeError(f"Ollama error: {e}") from e
    models = data.get("models") if isinstance(data, dict) else data
    if not isinstance(models, list):
        models = []
    listed_names: List[str] = []
    for m in models:
        name = (m.get("name") or m.get("model") or "")
        if name:
            listed_names.append(name)
    for model_name in (model_m, model_s, model_judge):
        if not _model_in_list(model_name, listed_names):
            try:  # pragma: no cover - depends on network / local registry
                ollama.pull(model_name)
            except Exception as e:
                raise RuntimeError(f"Failed to pull model {model_name}: {e}") from e


@dataclass
class Participant:
    """Represents a debate participant or judge."""

    name: str
    model: str
    system_prompt: str
    icon: str = ""


@dataclass
class BattleResult:
    """Structured result of a single debate."""

    topic: str
    machiavelli_model: str
    socrates_model: str
    judge_model: str
    transcript_entries: List[Dict[str, Any]]
    verdict: str
    token_prompt: int
    token_completion: int
    interrupted: bool = False

    @property
    def token_total(self) -> int:
        return self.token_prompt + self.token_completion


class Arena:
    """Coordinates a debate between two participants and a judge."""

    def __init__(
        self,
        machiavelli: Participant,
        socrates: Participant,
        judge: Participant,
        llm_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.machiavelli = machiavelli
        self.socrates = socrates
        self.judge = judge
        self.llm_options = llm_options or {"num_predict": 350, "temperature": 0.8, "num_ctx": 2048}

    def run_battle(
        self,
        topic: str,
        rounds: int = 3,
        on_speech: Optional[Any] = None,
        on_verdict: Optional[Any] = None,
    ) -> BattleResult:
        """Run the full debate loop and return a BattleResult.

        If on_speech is provided, it is called after each participant reply with
        the transcript entry dict. If on_verdict is provided, it is called once
        with (verdict_text, prompt_tokens, completion_tokens) for the judge.
        """
        history_m: List[Dict[str, str]] = []
        history_s: List[Dict[str, str]] = []
        transcript_plain: List[str] = []
        transcript_entries: List[Dict[str, Any]] = []
        total_prompt = 0
        total_completion = 0

        current_input = f"Start a debate on the topic: {topic}. State your position briefly."

        try:
            for _i in range(rounds):
                # Machiavelli turn
                history_m.append({"role": "user", "content": current_input})
                res_m = ollama.chat(
                    model=self.machiavelli.model,
                    messages=[{"role": "system", "content": self.machiavelli.system_prompt}] + history_m,
                    options=self.llm_options,
                )
                prompt_m, completion_m = token_counts(res_m)
                total_prompt += prompt_m
                total_completion += completion_m
                think_m, speech_m = extract_think(res_m["message"]["content"])
                history_m.append({"role": "assistant", "content": speech_m})
                transcript_plain.append(f"{self.machiavelli.name}: {speech_m}")
                entry_m = {
                    "name": self.machiavelli.name,
                    "icon": self.machiavelli.icon,
                    "think": think_m,
                    "speech": speech_m,
                    "prompt_tokens": prompt_m,
                    "completion_tokens": completion_m,
                }
                transcript_entries.append(entry_m)
                if on_speech is not None:
                    on_speech(entry_m)

                # Socrates turn
                history_s.append({"role": "user", "content": speech_m})
                res_s = ollama.chat(
                    model=self.socrates.model,
                    messages=[{"role": "system", "content": self.socrates.system_prompt}] + history_s,
                    options=self.llm_options,
                )
                prompt_s, completion_s = token_counts(res_s)
                total_prompt += prompt_s
                total_completion += completion_s
                think_s, speech_s = extract_think(res_s["message"]["content"])
                history_s.append({"role": "assistant", "content": speech_s})
                transcript_plain.append(f"{self.socrates.name}: {speech_s}")
                entry_s = {
                    "name": self.socrates.name,
                    "icon": self.socrates.icon,
                    "think": think_s,
                    "speech": speech_s,
                    "prompt_tokens": prompt_s,
                    "completion_tokens": completion_s,
                }
                transcript_entries.append(entry_s)
                if on_speech is not None:
                    on_speech(entry_s)

                current_input = speech_s

            # Judge verdict
            full_text = "\n".join(transcript_plain)
            res_j = ollama.chat(
                model=self.judge.model,
                messages=[
                    {"role": "system", "content": self.judge.system_prompt},
                    {"role": "user", "content": full_text},
                ],
            )
            prompt_j, completion_j = token_counts(res_j)
            total_prompt += prompt_j
            total_completion += completion_j
            verdict_text = res_j["message"]["content"].strip()
            if on_verdict is not None:
                on_verdict(verdict_text, prompt_j, completion_j)
            return BattleResult(
                topic=topic,
                machiavelli_model=self.machiavelli.model,
                socrates_model=self.socrates.model,
                judge_model=self.judge.model,
                transcript_entries=transcript_entries,
                verdict=verdict_text,
                token_prompt=total_prompt,
                token_completion=total_completion,
                interrupted=False,
            )

        except KeyboardInterrupt:  # pragma: no cover - interactive behaviour
            verdict_text = "(Debate interrupted by user.)"
            return BattleResult(
                topic=topic,
                machiavelli_model=self.machiavelli.model,
                socrates_model=self.socrates.model,
                judge_model=self.judge.model,
                transcript_entries=transcript_entries,
                verdict=verdict_text,
                token_prompt=total_prompt,
                token_completion=total_completion,
                interrupted=True,
            )

