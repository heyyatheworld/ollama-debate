"""Streamlit web UI for Ollama Debate (web interface layer)."""
from datetime import date
from pathlib import Path

import streamlit as st

from arena import (
    Arena,
    BattleResult,
    build_markdown,
    build_participants,
    check_ollama_running,
    ensure_models_available,
    llm_options_from_config,
    load_config,
    save_debate_to_md,
    SpeechTurn,
    topic_to_slug,
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


def render_speech(entry: SpeechTurn, *, round_num: int, total_rounds: int) -> None:
    """Render one speech block in Streamlit."""
    name = entry.name
    icon = entry.icon
    think = (entry.think or "").strip()
    speech = entry.speech
    st.markdown(f"### {icon} {name.upper()}")
    st.caption(f"Round **{round_num}** of **{total_rounds}**")
    if think:
        with st.expander("🔍 Thoughts", expanded=False):
            st.caption(think)
    st.markdown(speech)
    p = entry.prompt_tokens
    c = entry.completion_tokens
    st.caption(f"This reply — prompt **{p}**, completion **{c}**, subtotal **{p + c}**")
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
        stream_output = st.checkbox("Stream model output", value=False)
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
    if stream_output:
        st.caption("Streaming shows partial text as it arrives; the full reply is still rendered below when each turn completes.")
    st.divider()

    llm_options = llm_options_from_config(config)

    machiavelli, socrates, judge = build_participants(
        config,
        model_m=model_m,
        model_s=model_s,
        model_judge=model_judge,
    )

    arena = Arena(machiavelli=machiavelli, socrates=socrates, judge=judge, llm_options=llm_options)

    n_rounds = int(rounds)
    total_llm_steps = n_rounds * 2 + 1
    speech_idx = 0

    stream_area = st.empty()
    stream_accum = {"text": ""}

    def on_stream_begin(role: str) -> None:
        stream_accum["text"] = ""
        stream_area.markdown(f"**{role}** is generating…")

    def on_stream_chunk(role: str, delta: str) -> None:
        stream_accum["text"] += delta
        stream_area.markdown(f"### {role}\n\n{stream_accum['text']}▍")

    prog = st.progress(0, text="Preparing debate…")
    with st.status("Debate in progress…", expanded=True) as run_status:
        def on_speech(entry: SpeechTurn) -> None:
            nonlocal speech_idx
            speech_idx += 1
            round_num = (speech_idx - 1) // 2 + 1
            run_status.update(
                label=f"Round {round_num}/{n_rounds} — {entry.name} replied",
                state="running",
            )
            prog.progress(
                min(speech_idx / total_llm_steps, 1.0),
                text=f"Round {round_num}/{n_rounds} · {entry.name}",
            )
            if stream_output:
                stream_area.empty()
            render_speech(entry, round_num=round_num, total_rounds=n_rounds)

        def on_verdict(text: str, p: int, c: int) -> None:
            nonlocal speech_idx
            speech_idx += 1
            prog.progress(1.0, text="Judge verdict")
            run_status.update(label="Judge finished", state="running")
            if stream_output:
                stream_area.empty()
            st.markdown("### ⚖️ VERDICT")
            st.markdown(f"**{text}**")
            st.caption(
                f"Judge call only — prompt **{p}**, completion **{c}**, subtotal **{p + c}** "
                "(not the full session total)"
            )

        result: BattleResult = arena.run_battle(
            topic.strip(),
            rounds=n_rounds,
            on_speech=on_speech,
            on_verdict=on_verdict,
            stream=stream_output,
            on_stream_begin=on_stream_begin if stream_output else None,
            on_stream_chunk=on_stream_chunk if stream_output else None,
        )

        run_status.update(label="Debate finished", state="complete")

    st.subheader("Session token usage")
    st.caption(
        f"Sum of every LLM call in this run (all rounds + judge): "
        f"prompt **{result.token_prompt}**, completion **{result.token_completion}**, "
        f"total **{result.token_total}**"
    )

    if result.interrupted:
        st.warning("Debate interrupted by user. Partial transcript above.")

    debates_dir = settings_cfg.get("debates_dir", DEBATES_DIR)
    token_stats = {
        "prompt": result.token_prompt,
        "completion": result.token_completion,
        "total": result.token_total,
    }
    md_content = build_markdown(
        topic=result.topic,
        model_m=result.machiavelli_model,
        model_s=result.socrates_model,
        model_judge=result.judge_model,
        transcript_entries=result.transcript_entries,
        verdict=result.verdict,
        token_stats=token_stats,
    )

    filepath_str: str | None = None
    try:
        filepath_str = save_debate_to_md(
            topic=result.topic,
            model_m=result.machiavelli_model,
            model_s=result.socrates_model,
            model_judge=result.judge_model,
            transcript_entries=result.transcript_entries,
            verdict=result.verdict,
            token_stats=token_stats,
            debates_dir=debates_dir,
        )
        st.success(f"Debate saved to **{filepath_str}**")
    except OSError as e:
        st.error(f"Could not save file: {e}")

    download_name = (
        Path(filepath_str).name
        if filepath_str
        else f"{date.today().isoformat()}_{topic_to_slug(result.topic)}.md"
    )
    st.download_button(
        "Download transcript (.md)",
        data=md_content.encode("utf-8"),
        file_name=download_name,
        mime="text/markdown",
        type="primary",
        use_container_width=True,
    )
    with st.expander("Markdown preview", expanded=False):
        st.code(md_content, language="markdown")


if __name__ == "__main__":
    main()
