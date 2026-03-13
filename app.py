"""Streamlit web UI for Ollama Debate (web interface layer)."""
from pathlib import Path

import streamlit as st

from arena import (
    Arena,
    BattleResult,
    Participant,
    check_ollama_running,
    ensure_models_available,
    load_config,
    save_debate_to_md,
)


DEBATES_DIR = "debates"


def load_config_safe():
    """Load config for web UI; return None and report errors via Streamlit."""
    try:
        return load_config()
    except FileNotFoundError as e:
        st.error(str(e))
        return None
    except ValueError as e:
        st.error(str(e))
        return None
    except Exception as e:  # pragma: no cover - defensive
        st.error(f"Error loading config: {e}")
        return None


def ensure_ollama() -> bool:
    """Return True if Ollama is reachable; show st.error and return False otherwise."""
    try:
        check_ollama_running()
        return True
    except Exception as e:
        st.error(f"Ollama server is not running: {e}. Start Ollama or run `ollama serve`.")
        return False


def ensure_models(model_m: str, model_s: str, model_judge: str) -> bool:
    """Ensure models exist; pull if missing. Return True on success."""
    try:
        ensure_models_available(model_m, model_s, model_judge)
        return True
    except Exception as e:
        st.error(f"Model error: {e}")
        return False


def render_speech(entry: dict) -> None:
    """Render one speech block in Streamlit."""
    name = entry["name"]
    icon = entry["icon"]
    think = (entry.get("think") or "").strip()
    speech = entry["speech"]
    st.markdown(f"### {icon} {name.upper()}")
    if think:
        with st.expander("🔍 Thoughts", expanded=False):
            st.caption(think)
    st.markdown(speech)
    p = entry.get("prompt_tokens")
    c = entry.get("completion_tokens")
    if p is not None and c is not None:
        st.caption(f"Tokens: prompt {p}, completion {c}, total {p + c}")
    st.divider()


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

    prompts_cfg = config.get("prompts") or {}
    machiavelli = Participant(
        name="Machiavelli",
        model=model_m,
        system_prompt=prompts_cfg.get(
            "machiavelli",
            "You are Machiavelli. Speak English. You are a cynical pragmatist. Defend state interest and order at any cost.",
        ),
        icon="🦊",
    )
    socrates = Participant(
        name="Socrates",
        model=model_s,
        system_prompt=prompts_cfg.get(
            "socrates",
            "You are Socrates. Speak English. Use Socratic method: ask short, probing questions. Be humble but ironic.",
        ),
        icon="🏛",
    )
    judge = Participant(
        name="Judge",
        model=model_judge,
        system_prompt=prompts_cfg.get(
            "judge",
            "You are the Supreme Judge. Analyze the debate. Who won: Socrates or Machiavelli? Answer briefly and strictly in English.",
        ),
        icon="⚖️",
    )

    arena = Arena(machiavelli=machiavelli, socrates=socrates, judge=judge, llm_options=llm_options)
    collected_result: dict = {}

    def on_speech(entry: dict) -> None:
        render_speech(entry)

    def on_verdict(text: str, p: int, c: int) -> None:
        st.markdown("### ⚖️ VERDICT")
        st.markdown(f"**{text}**")
        st.caption(f"Tokens: prompt {p}, completion {c}, total {p + c}")

    result: BattleResult = arena.run_battle(
        topic.strip(),
        rounds=int(rounds),
        on_speech=on_speech,
        on_verdict=on_verdict,
    )

    if result.interrupted:
        st.warning("Debate interrupted by user. Partial transcript above.")

    debates_dir = settings_cfg.get("debates_dir", DEBATES_DIR)
    try:
        filepath = save_debate_to_md(
            topic=result.topic,
            model_m=result.machiavelli_model,
            model_s=result.socrates_model,
            model_judge=result.judge_model,
            transcript_entries=result.transcript_entries,
            verdict=result.verdict,
            token_stats={
                "prompt": result.token_prompt,
                "completion": result.token_completion,
                "total": result.token_total,
            },
            debates_dir=debates_dir,
        )
        st.success(f"Debate saved to **{filepath}**")
    except OSError as e:
        st.error(f"Could not save file: {e}")


if __name__ == "__main__":
    main()
