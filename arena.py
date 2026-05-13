"""Core business logic for the Ollama Debate project.

This module is the single source of debate semantics and file export:

- **Config** — `load_config`, `check_ollama_running`, `ensure_models_available`.
- **Setup from YAML** — `build_participants`, `llm_options_from_config` (shared by CLI and Streamlit).
- **Domain types** — `Participant`, `SpeechTurn` (one debater reply), `BattleResult` (transcript + verdict + token sums).
- **Orchestration** — `Arena.run_battle`: for each round, Machiavelli then Socrates via `ollama.chat`; then the Judge on the full plain transcript. Optional `on_speech` / `on_verdict` hooks allow UIs to stream output without importing Ollama in the front ends. With ``stream=True``, Ollama returns chunked responses; optional ``on_stream_begin`` / ``on_stream_chunk`` expose deltas before the full reply is finalized.
- **Artifacts** — `build_markdown`, `save_debate_to_md` (filename `YYYY-MM-DD_slug.md` under a configurable directory).

CLI and web UI only construct `Arena`, wire callbacks, and call save/download helpers; they do not duplicate the turn loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import ollama
import yaml
import re


CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

_DEFAULT_PROMPTS: Dict[str, str] = {
    "machiavelli": (
        "You are Machiavelli. Speak English. You are a cynical pragmatist. "
        "Defend state interest and order at any cost."
    ),
    "socrates": (
        "You are Socrates. Speak English. Use Socratic method: ask short, probing questions. "
        "Be humble but ironic."
    ),
    "judge": (
        "You are the Supreme Judge. Analyze the debate. Who won: Socrates or Machiavelli? "
        "Answer briefly and strictly in English."
    ),
}


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


def token_counts(response: Any) -> Tuple[int, int]:
    """Extract prompt and completion token counts from an Ollama response dict."""
    prompt = response.get("prompt_eval_count") or 0
    completion = response.get("eval_count") or 0
    return int(prompt), int(completion)


def _chunk_assistant_delta(chunk: Any) -> str:
    """Return incremental assistant ``content`` from one streamed chat chunk."""
    msg = chunk.get("message") if hasattr(chunk, "get") else getattr(chunk, "message", None)
    if msg is None:
        return ""
    if hasattr(msg, "get"):
        return msg.get("content") or ""
    return getattr(msg, "content", None) or ""


def _chat_message_and_token_totals(
    model: str,
    messages: List[Dict[str, str]],
    options: Optional[Dict[str, Any]],
    *,
    stream: bool,
    role_name: str,
    on_stream_begin: Optional[Callable[[str], None]],
    on_stream_chunk: Optional[Callable[[str, str], None]],
) -> Tuple[str, int, int]:
    """Run ``ollama.chat`` once; return assistant text and token counts."""
    if not stream:
        res: Any = ollama.chat(model=model, messages=messages, options=options)
        text = res["message"]["content"]
        p, c = token_counts(res)
        return text, p, c

    if on_stream_begin is not None:
        on_stream_begin(role_name)

    stream_iter = ollama.chat(model=model, messages=messages, options=options, stream=True)
    parts: List[str] = []
    last: Any = None
    for chunk in stream_iter:
        last = chunk
        piece = _chunk_assistant_delta(chunk)
        if piece:
            parts.append(piece)
            if on_stream_chunk is not None:
                on_stream_chunk(role_name, piece)

    if last is None:
        return "", 0, 0
    p, c = token_counts(last)
    return "".join(parts), p, c


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
    transcript_entries: List[SpeechTurn],
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
        name = entry.name
        icon = entry.icon
        think = (entry.think or "").strip()
        speech = entry.speech
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
    transcript_entries: List[SpeechTurn],
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


def build_participants(
    config: Dict[str, Any],
    *,
    model_m: str,
    model_s: str,
    model_judge: str,
) -> Tuple[Participant, Participant, Participant]:
    """Build Machiavelli, Socrates, and Judge from config prompts and the given models."""
    prompts = config.get("prompts") or {}
    machiavelli = Participant(
        name="Machiavelli",
        model=model_m,
        system_prompt=prompts.get("machiavelli", _DEFAULT_PROMPTS["machiavelli"]),
        icon="🦊",
    )
    socrates = Participant(
        name="Socrates",
        model=model_s,
        system_prompt=prompts.get("socrates", _DEFAULT_PROMPTS["socrates"]),
        icon="🏛",
    )
    judge = Participant(
        name="Judge",
        model=model_judge,
        system_prompt=prompts.get("judge", _DEFAULT_PROMPTS["judge"]),
        icon="⚖️",
    )
    return machiavelli, socrates, judge


def llm_options_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Derive Ollama chat options from config settings (same defaults as Arena)."""
    settings = config.get("settings") or {}
    return {
        "num_predict": settings.get("num_predict", 350),
        "temperature": settings.get("temperature", 0.8),
        "num_ctx": settings.get("num_ctx", 2048),
    }


@dataclass
class SpeechTurn:
    """One participant reply in the debate transcript (not the judge verdict)."""

    name: str
    icon: str
    think: str
    speech: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class BattleResult:
    """Structured result of a single debate."""

    topic: str
    machiavelli_model: str
    socrates_model: str
    judge_model: str
    transcript_entries: List[SpeechTurn]
    verdict: str
    token_prompt: int
    token_completion: int
    interrupted: bool = False

    @property
    def token_total(self) -> int:
        return self.token_prompt + self.token_completion


class Arena:
    """Run a fixed-format debate: Machiavelli and Socrates alternate; then the Judge.

    Each debater maintains its own `ollama.chat` message history (system + thread).
    The Judge receives a single user message: the joined plain transcript (no
    structured JSON). Token counts from every `ollama.chat` response are summed
    into `BattleResult.token_prompt` and `token_completion`.

    Front ends should use `build_participants` and `llm_options_from_config`
    with a loaded config dict, then pass the resulting `Participant` instances
    and options into this class. Pass ``stream=True`` and optional stream
    callbacks on ``run_battle`` when the UI should show token-by-token output.
    """

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
        *,
        stream: bool = False,
        on_stream_begin: Optional[Callable[[str], None]] = None,
        on_stream_chunk: Optional[Callable[[str, str], None]] = None,
    ) -> BattleResult:
        """Run `rounds` exchanges (each exchange: Machiavelli, then Socrates), then the Judge.

        The first user message to Machiavelli asks him to open the debate on `topic`.
        Each following round uses Socrates’s last speech as the next prompt for Machiavelli.

        **Callbacks**

        - ``on_speech(entry: SpeechTurn)`` — invoked after each debater completion with
          name, icon, optional extracted ``think`` snippet, visible ``speech``, and
          token counts for that call only.
        - ``on_verdict(text, prompt_tokens, completion_tokens)`` — invoked once after
          the judge model returns; these token counts are for the judge call only,
          while the returned ``BattleResult`` includes session-wide totals.

        **Interruption**

        On ``KeyboardInterrupt``, returns a ``BattleResult`` with ``interrupted=True``,
        a placeholder verdict string, and whatever transcript was built so far (no
        ``on_verdict`` call in that path).

        **Streaming**

        If ``stream`` is True, ``ollama.chat`` runs with ``stream=True``. Optional
        ``on_stream_begin(role_name)`` runs before each streamed reply; optional
        ``on_stream_chunk(role_name, delta)`` runs for each non-empty content
        fragment (same ``role_name`` as the participant or ``Judge``). Full text
        is still passed to ``on_speech`` / ``on_verdict`` after the stream ends.

        Returns:
            ``BattleResult`` with models used, ``transcript_entries`` (debater turns only),
            ``verdict`` text, aggregated token fields, and ``interrupted`` flag.
        """
        history_m: List[Dict[str, str]] = []
        history_s: List[Dict[str, str]] = []
        transcript_plain: List[str] = []
        transcript_entries: List[SpeechTurn] = []
        total_prompt = 0
        total_completion = 0

        current_input = f"Start a debate on the topic: {topic}. State your position briefly."

        try:
            for _i in range(rounds):
                # Machiavelli turn
                history_m.append({"role": "user", "content": current_input})
                raw_m, prompt_m, completion_m = _chat_message_and_token_totals(
                    self.machiavelli.model,
                    [{"role": "system", "content": self.machiavelli.system_prompt}] + history_m,
                    self.llm_options,
                    stream=stream,
                    role_name=self.machiavelli.name,
                    on_stream_begin=on_stream_begin,
                    on_stream_chunk=on_stream_chunk,
                )
                total_prompt += prompt_m
                total_completion += completion_m
                think_m, speech_m = extract_think(raw_m)
                history_m.append({"role": "assistant", "content": speech_m})
                transcript_plain.append(f"{self.machiavelli.name}: {speech_m}")
                entry_m = SpeechTurn(
                    name=self.machiavelli.name,
                    icon=self.machiavelli.icon,
                    think=think_m,
                    speech=speech_m,
                    prompt_tokens=prompt_m,
                    completion_tokens=completion_m,
                )
                transcript_entries.append(entry_m)
                if on_speech is not None:
                    on_speech(entry_m)

                # Socrates turn
                history_s.append({"role": "user", "content": speech_m})
                raw_s, prompt_s, completion_s = _chat_message_and_token_totals(
                    self.socrates.model,
                    [{"role": "system", "content": self.socrates.system_prompt}] + history_s,
                    self.llm_options,
                    stream=stream,
                    role_name=self.socrates.name,
                    on_stream_begin=on_stream_begin,
                    on_stream_chunk=on_stream_chunk,
                )
                total_prompt += prompt_s
                total_completion += completion_s
                think_s, speech_s = extract_think(raw_s)
                history_s.append({"role": "assistant", "content": speech_s})
                transcript_plain.append(f"{self.socrates.name}: {speech_s}")
                entry_s = SpeechTurn(
                    name=self.socrates.name,
                    icon=self.socrates.icon,
                    think=think_s,
                    speech=speech_s,
                    prompt_tokens=prompt_s,
                    completion_tokens=completion_s,
                )
                transcript_entries.append(entry_s)
                if on_speech is not None:
                    on_speech(entry_s)

                current_input = speech_s

            # Judge verdict
            full_text = "\n".join(transcript_plain)
            raw_j, prompt_j, completion_j = _chat_message_and_token_totals(
                self.judge.model,
                [
                    {"role": "system", "content": self.judge.system_prompt},
                    {"role": "user", "content": full_text},
                ],
                self.llm_options,
                stream=stream,
                role_name=self.judge.name,
                on_stream_begin=on_stream_begin,
                on_stream_chunk=on_stream_chunk,
            )
            total_prompt += prompt_j
            total_completion += completion_j
            verdict_text = raw_j.strip()
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

