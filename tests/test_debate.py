"""Pytest tests for config loading, log filename formatting, and token handling."""
import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main as main_module


# --- Config loading ---

def test_load_config_missing_file_exits_with_error():
    """When config.yaml does not exist, load_config exits with code 1."""
    with patch.object(main_module, "CONFIG_PATH", "/nonexistent/config.yaml"):
        with patch("main.os.path.isfile", return_value=False):
            with pytest.raises(SystemExit) as exc_info:
                main_module.load_config()
    assert exc_info.value.code == 1


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
    with patch("main.os.path.isfile", return_value=True):
        with patch("main.open", mock_open(read_data=yaml_content)):
            data = main_module.load_config()
    assert "models" in data
    assert data["models"]["machiavelli"] == "m:latest"
    assert data["models"]["socrates"] == "s:7b"
    assert "prompts" in data
    assert "settings" in data
    assert data["settings"]["default_rounds"] == 3
    assert data["settings"]["debates_dir"] == "debates"


def test_load_config_empty_yaml_exits():
    """When config file exists but is empty/invalid YAML, load_config exits."""
    with patch("main.os.path.isfile", return_value=True):
        with patch("main.open", mock_open(read_data="")):
            with pytest.raises(SystemExit):
                main_module.load_config()


# --- Log filename formatting ---

def test_log_filename_includes_date_and_slug():
    """Log filename has form YYYY-MM-DD_slug.md and uses only safe characters."""
    topic = "What is justice? Why?"
    slug = main_module._topic_to_slug(topic)
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
    assert main_module._topic_to_slug("Hello, World!") == "hello_world"
    # & and . are removed; letters (including accented) are kept
    assert main_module._topic_to_slug("Café & Co.") == "café_co"
    assert main_module._topic_to_slug("a-b c") == "a_b_c"


def test_log_filename_fallback_for_empty_topic():
    """Empty or invalid topic yields slug 'debate' so filename is still valid."""
    slug = main_module._topic_to_slug("")
    assert slug == "debate"
    slug = main_module._topic_to_slug("???")
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
    prompt, completion = main_module._token_counts(mock_response)
    assert prompt == 42
    assert completion == 15


def test_token_counts_missing_keys_default_to_zero():
    """_token_counts returns 0 for missing prompt_eval_count or eval_count."""
    mock_response = {"message": {"content": "Hi"}}
    prompt, completion = main_module._token_counts(mock_response)
    assert prompt == 0
    assert completion == 0


def test_token_counts_none_values_treated_as_zero():
    """_token_counts treats None as 0 (response.get can return None)."""
    mock_response = {"prompt_eval_count": None, "eval_count": None}
    prompt, completion = main_module._token_counts(mock_response)
    assert prompt == 0
    assert completion == 0


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
    prompt, completion = main_module._token_counts(fake_response)
    assert prompt == 100
    assert completion == 25
    think, speech = main_module.extract_think(fake_response["message"]["content"])
    assert "Considering" in think
    assert "Order is preferable" in speech
