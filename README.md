# Ollama Debate

A small Python script that runs a debate between two [Ollama](https://ollama.com/) models on a topic you choose. The models take turns replying to each other for a set number of rounds.

## Requirements

- [Ollama](https://ollama.com/) installed and running locally
- Two Ollama models pulled (e.g. `llama3.2`, `qwen2.5-coder:7b`)
- Python 3.x
- The `ollama` Python package

## Setup

```bash
pip install ollama
```

Make sure Ollama is running (e.g. start the Ollama app or run `ollama serve`) and that you have pulled the models you want to use:

```bash
ollama pull llama3.2
ollama pull qwen2.5-coder:7b
```

## Usage

Run the default debate (topic and models are set in `main.py`):

```bash
python main.py
```

### Customizing in code

Edit the `if __name__ == "__main__":` block in `main.py`:

- **topic** — The debate question or statement.
- **model_a** — Ollama model name for the first speaker.
- **model_b** — Ollama model name for the second speaker.
- **rounds** — Number of back-and-forth exchanges (default in the function is 5).

Example:

```python
start_dialogue(
    model_a="llama3.2",
    model_b="qwen2.5-coder:7b",
    topic="Does humanity need artificial intelligence?",
    rounds=3,
)
```

You can also call `start_dialogue()` from your own script with different topics and models.

## How it works

1. Both models get a **system prompt** that states the debate topic and asks for short, clear answers.
2. The first message to model A is an opening line that includes the topic.
3. Each round: model A replies to the current message, then model B replies to A’s reply. The last reply becomes the next “current message” for the following round.
4. The topic is kept in the system prompt for every turn, so both models stay on topic.

Output is printed to the terminal with the model names and their replies.
