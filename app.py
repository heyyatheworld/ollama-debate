"""Streamlit web UI for Ollama Debate."""
from pathlib import Path
import sys

import streamlit as st

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent))
import main as main_module

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
DEBATES_DIR = "debates"


def load_config_safe():
    """Load config; return None and set st.error if missing."""
    if not CONFIG_PATH.is_file():
        st.error(f"Config file not found: {CONFIG_PATH}. Create config.yaml in the project root.")
        return None
    import yaml
    with open(CONFIG_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data:
        st.error("config.yaml is empty.")
        return None
    return data


def ensure_ollama():
    """Return True if Ollama is reachable; show st.error and return False otherwise."""
    try:
        main_module.ollama.list()
        return True
    except Exception as e:
        st.error(f"Ollama server is not running: {e}. Start Ollama or run `ollama serve`.")
        return False


def ensure_models(model_m, model_s, model_judge):
    """Ensure models exist; pull if missing. Return True on success."""
    try:
        main_module.ensure_models_available(model_m, model_s, model_judge, exit_on_error=False)
        return True
    except Exception as e:
        st.error(f"Model error: {e}")
        return False


def render_speech(entry, prompt_tokens, completion_tokens):
    """Render one speech block in Streamlit."""
    name = entry["name"]
    icon = entry["icon"]
    think = entry.get("think", "").strip()
    speech = entry["speech"]
    color = "#c71585" if name == "Machiavelli" else "#0696a0"
    st.markdown(f"### {icon} {name.upper()}")
    if think:
        with st.expander("🔍 Thoughts", expanded=False):
            st.caption(think)
    st.markdown(speech)
    if prompt_tokens is not None and completion_tokens is not None:
        total = prompt_tokens + completion_tokens
        st.caption(f"Tokens: prompt {prompt_tokens}, completion {completion_tokens}, total {total}")
    st.divider()


def run_debate_ui(topic, rounds, model_m, model_s, model_judge, config_prompts, llm_options):
    """Run debate and stream output into Streamlit."""
    def status_ctx(msg):
        return st.spinner(msg)

    def on_speech(entry, p, c):
        render_speech(entry, p, c)

    def on_verdict(text, p, c):
        st.markdown("### ⚖️ VERDICT")
        st.markdown(f"**{text}**")
        st.caption(f"Tokens: prompt {p}, completion {c}, total {p + c}")

    result, interrupted = main_module.run_debate_core(
        model_m, model_s, model_judge, topic, rounds,
        config_prompts=config_prompts, llm_options=llm_options,
        status_context=status_ctx, on_speech=on_speech, on_verdict=on_verdict,
    )
    return result, interrupted


def main():
    st.set_page_config(page_title="Ollama Debate", page_icon="🏛", layout="wide")
    st.title("🏛 Ollama Debate")
    st.caption("Historical court: Socrates vs Machiavelli, with a Judge. Powered by Ollama.")

    config = load_config_safe()
    if not config:
        st.stop()

    models_cfg = config.get("models") or {}
    settings_cfg = config.get("settings") or {}

    with st.sidebar:
        st.header("Settings")
        topic = st.text_area(
            "Debate topic",
            value="What is better for society: total state control or complete anarchy?",
            height=80,
        )
        rounds = st.number_input("Rounds", min_value=1, max_value=10, value=settings_cfg.get("default_rounds", 2), step=1)
        model_m = st.text_input("Machiavelli model", value=models_cfg.get("machiavelli", "llama3:latest"))
        model_s = st.text_input("Socrates model", value=models_cfg.get("socrates", "qwen2.5-coder:7b"))
        model_judge = st.text_input("Judge model", value=models_cfg.get("judge", "llama3.2:latest"))
        st.divider()
        run_clicked = st.button("Start debate", type="primary", use_container_width=True)
        st.caption("Ollama must be running. Missing models will be pulled on first run.")

    if not run_clicked:
        st.info("Set the topic and models in the sidebar, then click **Start debate**.")
        st.stop()

    if not topic.strip():
        st.warning("Please enter a debate topic.")
        st.stop()

    if not ensure_ollama():
        st.stop()

    with st.spinner("Checking / pulling models..."):
        if not ensure_models(model_m, model_s, model_judge):
            st.stop()

    st.markdown("---")
    st.markdown(f"**Topic:** {topic}")
    st.markdown(f"*Rounds: {rounds} · Machiavelli: {model_m} · Socrates: {model_s} · Judge: {model_judge}*")
    st.divider()

    llm_options = {
        "num_predict": settings_cfg.get("num_predict", 350),
        "temperature": settings_cfg.get("temperature", 0.8),
        "num_ctx": settings_cfg.get("num_ctx", 2048),
    }
    result, interrupted = run_debate_ui(
        topic.strip(), rounds, model_m, model_s, model_judge,
        config.get("prompts"), llm_options,
    )

    if interrupted:
        st.warning("Debate interrupted by user. Partial transcript above.")

    debates_dir = settings_cfg.get("debates_dir", DEBATES_DIR)
    try:
        filepath = main_module.save_debate_to_md(
            topic=result["topic"],
            model_m=result["model_m"],
            model_s=result["model_s"],
            model_judge=result["model_judge"],
            transcript_entries=result["transcript_entries"],
            verdict=result["verdict"],
            token_stats=result.get("token_stats"),
            debates_dir=debates_dir,
        )
        st.success(f"Debate saved to **{filepath}**")
    except OSError as e:
        st.error(f"Could not save file: {e}")


if __name__ == "__main__":
    main()
