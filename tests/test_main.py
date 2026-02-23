"""Simple tests for log filename creation and argument parsing."""
import sys
from datetime import date
from unittest.mock import patch

import pytest

# Import after potential sys.path fix so main is loadable
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main as main_module


def test_topic_to_slug_basic():
    """Topic is lowercased, spaces and punctuation become underscores."""
    assert main_module._topic_to_slug("What is justice?") == "what_is_justice"
    assert main_module._topic_to_slug("  State vs Anarchy  ") == "state_vs_anarchy"


def test_topic_to_slug_empty_and_fallback():
    """Empty or only-punctuation topic yields 'debate'."""
    assert main_module._topic_to_slug("") == "debate"
    assert main_module._topic_to_slug("???") == "debate"


def test_topic_to_slug_length_cap():
    """Slug is capped at 240 characters."""
    long_topic = "a" * 300
    slug = main_module._topic_to_slug(long_topic)
    assert len(slug) == 240
    assert slug == "a" * 240


def test_log_filename_format():
    """Log filename is YYYY-MM-DD_slug.md and path is debates_dir/filename."""
    topic = "Test Topic"
    slug = main_module._topic_to_slug(topic)
    today = date.today().isoformat()
    expected_name = f"{today}_{slug}.md"
    assert expected_name.endswith(".md")
    assert expected_name.startswith(today)
    assert "_" in expected_name
    assert "test_topic" in expected_name


def test_parse_args_uses_config_defaults():
    """When no CLI args are given, parse_args returns config values."""
    config = {
        "models": {
            "machiavelli": "custom-m:latest",
            "socrates": "custom-s:7b",
            "judge": "custom-j:latest",
        },
        "settings": {"default_rounds": 5},
    }
    with patch.object(sys, "argv", ["main.py"]):
        args = main_module.parse_args(config)
    assert args.model_m == "custom-m:latest"
    assert args.model_s == "custom-s:7b"
    assert args.judge == "custom-j:latest"
    assert args.rounds == 5


def test_parse_args_cli_overrides_config():
    """CLI arguments override config defaults."""
    config = {
        "models": {"machiavelli": "from-config", "socrates": "socrates-config", "judge": "judge-config"},
        "settings": {"default_rounds": 2},
    }
    with patch.object(sys, "argv", ["main.py", "--model_m", "cli-model", "--rounds", "3"]):
        args = main_module.parse_args(config)
    assert args.model_m == "cli-model"
    assert args.model_s == "socrates-config"
    assert args.rounds == 3
