# Ollama Debate

A Python script that runs a historical court debate between two [Ollama](https://ollama.com/) models (Socrates and Machiavelli) on a topic you choose, with a third model acting as the judge to deliver a verdict.

## Requirements

- [Ollama](https://ollama.com/) installed and running locally
- Three Ollama models pulled (e.g. `llama3:latest`, `qwen2.5-coder:7b`, `llama3.2:latest`)
- Python 3.x
- The `ollama` Python package

## Setup

```bash
pip install ollama
```

Make sure Ollama is running (e.g. start the Ollama app or run `ollama serve`) and that you have pulled the models you want to use:

```bash
ollama pull llama3:latest
ollama pull qwen2.5-coder:7b
ollama pull llama3.2:latest
```

## Testing

The project uses [pytest](https://pytest.org/) for tests. Install dependencies (including `pytest`), then run:

```bash
pytest
```

Run with verbose output:

```bash
pytest -v
```

Run only tests in a specific file:

```bash
pytest tests/test_debate.py -v
pytest tests/test_main.py -v
```

Tests cover config loading, log filename formatting, argument parsing, and token handling; they use mocks so no real Ollama models are started.

## Usage

Run the default court debate (topic and models are set in `main.py`):

```bash
python main.py
```

### Customizing in code

Edit the `if __name__ == "__main__":` block in `main.py`:

- **topic** — The debate question or statement.
- **model_m** — Ollama model name for Machiavelli (pragmatic defender of state control).
- **model_s** — Ollama model name for Socrates (uses Socratic method with probing questions).
- **model_judge** — Ollama model name for the Judge (delivers final verdict).
- **rounds** — Number of back-and-forth exchanges (default is 3).

Example:

```python
start_court(
    model_m='llama3:latest',
    model_s='qwen2.5-coder:7b',
    model_judge='llama3.2:latest',
    topic='What is better for society: total state control or complete anarchy',
    rounds=2
)
```

You can also call `start_court()` from your own script with different topics and models.

## How it works

1. **Character Setup**: Each model receives a character-specific system prompt:
   - **Socrates**: Uses the Socratic method, asking short, probing questions. Humble but ironic.
   - **Machiavelli**: A cynical pragmatist who defends state interest and order at any cost.

2. **Debate Flow**: 
   - Machiavelli starts with an opening statement on the topic.
   - Each round: Machiavelli replies, then Socrates responds to Machiavelli's statement.
   - The debate continues for the specified number of rounds.

3. **Judge's Verdict**: After all rounds, the Judge model analyzes the full transcript and delivers a verdict determining who won the debate.

4. **Output Formatting**: The script displays color-coded output with icons (🦊 for Machiavelli, 🏛 for Socrates, ⚖️ for Judge). If models include `<think>` tags in their responses, these are extracted and displayed separately as "Thoughts".

5. **Performance**: The script uses optimized settings for 8GB RAM systems (limited context window and response length).

Output is printed to the terminal with formatted speech blocks showing each participant's thoughts and responses.
