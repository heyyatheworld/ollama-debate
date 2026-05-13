"""Pytest tests for config loading, log filename formatting, and token handling."""
import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import arena


# --- Config loading ---

def test_load_config_missing_file_exits_with_error():
    """When config.yaml does not exist, load_config raises FileNotFoundError."""
    with patch.object(arena, "CONFIG_PATH", Path("/nonexistent/config.yaml")):
        with pytest.raises(FileNotFoundError):
            arena.load_config()


def test_load_config_reads_valid_yaml():
    """When config file exists with valid YAML, load_config returns dict with models, prompts, settings."""
    yaml_content = """
models:
  machiavelli: "m:latest"
  socrates: "s:7b"
  judge: "j:latest"
prompts:
  socrates: "You are Socrates."
  machiavelli: "You are Machiavelli."
  judge: "You are the Judge."
settings:
  default_rounds: 3
  debates_dir: "debates"
  num_ctx: 2048
"""
    fake_path = MagicMock()
    fake_path.is_file.return_value = True
    with patch.object(arena, "CONFIG_PATH", fake_path):
        with patch("arena.CONFIG_PATH.open", mock_open(read_data=yaml_content)):
            data = arena.load_config()
    assert "models" in data
    assert data["models"]["machiavelli"] == "m:latest"
    assert data["models"]["socrates"] == "s:7b"
    assert "prompts" in data
    assert "settings" in data
    assert data["settings"]["default_rounds"] == 3
    assert data["settings"]["debates_dir"] == "debates"


def test_load_config_empty_yaml_exits():
    """When config file exists but is empty/invalid YAML, load_config raises ValueError."""
    fake_path = MagicMock()
    fake_path.is_file.return_value = True
    with patch.object(arena, "CONFIG_PATH", fake_path):
        with patch("arena.CONFIG_PATH.open", mock_open(read_data="")):
            with pytest.raises(ValueError):
                arena.load_config()


# --- Log filename formatting ---

def test_log_filename_includes_date_and_slug():
    """Log filename has form YYYY-MM-DD_slug.md and uses only safe characters."""
    topic = "What is justice? Why?"
    slug = arena.topic_to_slug(topic)
    assert slug == "what_is_justice_why"
    today = date.today().isoformat()
    filename = f"{today}_{slug}.md"
    assert filename.startswith(today)
    assert filename.endswith(".md")
    assert " " not in filename
    assert "?" not in filename
    assert ":" not in filename


def test_log_filename_no_forbidden_chars():
    """Slug strips punctuation and uses only word chars and underscores."""
    assert arena.topic_to_slug("Hello, World!") == "hello_world"
    # & and . are removed; letters (including accented) are kept
    assert arena.topic_to_slug("Café & Co.") == "café_co"
    assert arena.topic_to_slug("a-b c") == "a_b_c"


def test_log_filename_fallback_for_empty_topic():
    """Empty or invalid topic yields slug 'debate' so filename is still valid."""
    slug = arena.topic_to_slug("")
    assert slug == "debate"
    slug = arena.topic_to_slug("???")
    assert slug == "debate"


# --- Token handling (unit test for _token_counts) ---

def test_token_counts_from_response_dict():
    """_token_counts extracts prompt_eval_count and eval_count from Ollama response dict."""
    # Simulated Ollama API response
    mock_response = {
        "model": "llama3:latest",
        "message": {"role": "assistant", "content": "Hello"},
        "prompt_eval_count": 42,
        "eval_count": 15,
    }
    prompt, completion = arena.token_counts(mock_response)
    assert prompt == 42
    assert completion == 15


def test_token_counts_missing_keys_default_to_zero():
    """_token_counts returns 0 for missing prompt_eval_count or eval_count."""
    mock_response = {"message": {"content": "Hi"}}
    prompt, completion = arena.token_counts(mock_response)
    assert prompt == 0
    assert completion == 0


def test_build_participants_and_llm_options_from_config():
    """Participants and LLM options are derived from config with correct defaults and overrides."""
    cfg = {
        "prompts": {"machiavelli": "Custom M prompt"},
        "settings": {"num_predict": 100, "temperature": 0.5, "num_ctx": 4096},
    }
    m, s, j = arena.build_participants(cfg, model_m="m1", model_s="s1", model_judge="j1")
    assert m.model == "m1" and m.system_prompt == "Custom M prompt"
    assert s.model == "s1" and "Socrates" in s.system_prompt
    assert j.model == "j1" and "Supreme Judge" in j.system_prompt
    opts = arena.llm_options_from_config(cfg)
    assert opts == {"num_predict": 100, "temperature": 0.5, "num_ctx": 4096}
    empty = arena.llm_options_from_config({})
    assert empty == {"num_predict": 350, "temperature": 0.8, "num_ctx": 2048}


def test_token_counts_none_values_treated_as_zero():
    """_token_counts treats None as 0 (response.get can return None)."""
    mock_response = {"prompt_eval_count": None, "eval_count": None}
    prompt, completion = arena.token_counts(mock_response)
    assert prompt == 0
    assert completion == 0


@patch("arena.ollama.chat")
def test_chat_message_streaming_accumulates_text_and_tokens(mock_chat):
    """Streaming chat concatenates deltas and reads token counts from the final chunk."""
    mock_chat.return_value = iter(
        [
            {"message": {"content": "Hel"}},
            {"message": {"content": "lo"}},
            {"message": {"content": ""}, "prompt_eval_count": 9, "eval_count": 2},
        ]
    )
    chunks: list[tuple[str, str]] = []
    text, p, c = arena._chat_message_and_token_totals(
        "m",
        [{"role": "user", "content": "x"}],
        {"num_predict": 10},
        stream=True,
        role_name="TestRole",
        on_stream_begin=lambda r: chunks.append(("begin", r)),
        on_stream_chunk=lambda r, d: chunks.append((r, d)),
    )
    assert text == "Hello"
    assert p == 9
    assert c == 2
    assert ("begin", "TestRole") in chunks
    assert chunks.count(("TestRole", "Hel")) == 1
    assert chunks.count(("TestRole", "lo")) == 1
    mock_chat.assert_called_once()
    call_kw = mock_chat.call_args.kwargs
    assert call_kw.get("stream") is True


# --- Mocked Ollama API response: full flow ---

def test_processing_mocked_ollama_chat_response():
    """Code correctly processes a fake ollama.chat() response (tokens + message content)."""
    # Simulated full response from ollama.chat()
    fake_response = {
        "model": "llama3:latest",
        "message": {
            "role": "assistant",
            "content": "<think>Considering the topic</think>\nOrder is preferable to chaos for society.",
        },
        "prompt_eval_count": 100,
        "eval_count": 25,
    }
    prompt, completion = arena.token_counts(fake_response)
    assert prompt == 100
    assert completion == 25
    think, speech = arena.extract_think(fake_response["message"]["content"])
    assert "Considering" in think
    assert "Order is preferable" in speech
