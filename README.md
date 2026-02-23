# Ollama Debate

**Version:** v1.0.0  
**Last updated:** February 2025

A Python script that runs a historical court debate between two [Ollama](https://ollama.com/) models (Socrates and Machiavelli) on a topic you choose, with a third model acting as the judge to deliver a verdict.

## Requirements

- [Ollama](https://ollama.com/) installed and running locally
- Python 3.x
- Python dependencies: **ollama**, **rich**, **PyYAML**, **pytest** (for tests)

## Setup

```bash
pip install -r requirements.txt
```

Make sure Ollama is running (e.g. start the Ollama app or run `ollama serve`). Default models in `config.yaml` are `llama3:latest`, `qwen2.5-coder:7b`, and `llama3.2:latest`. Pull them if needed:

```bash
ollama pull llama3:latest
ollama pull qwen2.5-coder:7b
ollama pull llama3.2:latest
```

## Usage

Run with defaults from `config.yaml`:

```bash
python main.py
```

Override topic, rounds, or models via CLI (CLI takes precedence over config):

```bash
python main.py --topic "Your debate topic" --rounds 3 --model_m llama3 --model_s qwen2.5-coder:7b --judge llama3.2:latest
```

- **--topic** — Debate question or statement (default from config).
- **--rounds** — Number of back-and-forth exchanges (default from config).
- **--model_m** — Ollama model for Machiavelli.
- **--model_s** — Ollama model for Socrates.
- **--judge** — Ollama model for the Judge.

You can also edit `config.yaml` to change default models, system prompts, and settings (e.g. `debates_dir`, `num_ctx`). The `debates/` folder is created automatically on first run when a debate is saved.

## Testing

Run the test suite with [pytest](https://pytest.org/):

```bash
pytest
```

Verbose:

```bash
pytest -v
```

Single file:

```bash
pytest tests/test_debate.py -v
pytest tests/test_main.py -v
```

Tests cover config loading, log filename formatting, argument parsing, and token handling; they use mocks so no real Ollama models are started.

## How it works

1. **Character setup** — Each model receives a character-specific system prompt from `config.yaml` (Socrates: Socratic method; Machiavelli: cynical pragmatist; Judge: verdict).

2. **Debate flow** — Machiavelli opens; each round alternates Machiavelli → Socrates. After all rounds, the Judge model reads the transcript and delivers a verdict.

3. **Output** — Rich panels in the terminal (🦊 Machiavelli, 🏛 Socrates, ⚖️ Judge), token counts per reply, and a Markdown log saved under `debates/` (path configurable in `config.yaml`).

4. **Performance** — Default settings in config suit limited RAM (e.g. 8GB); you can adjust `num_ctx`, `num_predict`, and `temperature` in `config.yaml`.
