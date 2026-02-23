import argparse
import os
import re
from datetime import date

import ollama

# Color codes for terminal output
CYAN = "\033[96m"
YELLOW = "\033[93m"
GRAY = "\033[90m"
WHITE = "\033[97m"
BOLD = "\033[1m"
RESET = "\033[0m"

def clean_text(text):
    """Removes excessive line breaks and whitespace."""
    text = re.sub(r'\n+', '\n', text).strip()
    return text

def extract_think(text):
    """Separates <think> tags from the main response."""
    think = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    content = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    think_text = think.group(1).strip() if think else ""
    if len(think_text) > 200:
        think_text = think_text[:200] + "..."
        
    return clean_text(think_text), clean_text(content)

def print_speech(name, think, speech, color, icon):
    """Prints a participant's speech block with formatting."""
    print(f"{BOLD}{color}{icon} {name.upper()}:{RESET}")
    if think:
        print(f"{GRAY}   🔍 Thoughts: {think}{RESET}")
    indented_speech = "\n".join([f"   {line}" for line in speech.split('\n')])
    print(f"{WHITE}{indented_speech}{RESET}\n")
    print(f"{GRAY}{'—' * 60}{RESET}\n")

def _topic_to_slug(topic):
    """Convert topic to a short filename-safe slug."""
    slug = topic.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[-\s]+", "_", slug)
    return slug[:240] if slug else "debate"

def _build_markdown(topic, model_m, model_s, model_judge, transcript_entries, verdict):
    """Build full Markdown content for the debate file."""
    lines = [
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
        name = entry["name"]
        icon = entry["icon"]
        think = entry.get("think", "").strip()
        speech = entry["speech"]
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
    lines.extend(["## Verdict", "", verdict.strip(), ""])
    return "\n".join(lines)

def save_debate_to_md(topic, model_m, model_s, model_judge, transcript_entries, verdict):
    """Save debate to debates/YYYY-MM-DD_slug.md; create directory if needed. Returns path like debates/filename.md."""
    os.makedirs("debates", exist_ok=True)
    today = date.today().isoformat()
    slug = _topic_to_slug(topic)
    filename = f"{today}_{slug}.md"
    filepath = os.path.join("debates", filename)
    md = _build_markdown(topic, model_m, model_s, model_judge, transcript_entries, verdict)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md)
    return filepath

def start_court(model_m, model_s, model_judge, topic, rounds=3):
    prompts = {
        "Socrates": "You are Socrates. Speak English. Use Socratic method: ask short, probing questions. Be humble but ironic.",
        "Machiavelli": "You are Machiavelli. Speak English. You are a cynical pragmatist. Defend state interest and order at any cost."
    }

    history_m = []
    history_s = []
    transcript_plain = []
    transcript_entries = []

    # Optimized settings for 8GB RAM
    llm_options = {
        "num_predict": 350,
        "temperature": 0.8,
        "num_ctx": 2048
    }

    print(f"\n{BOLD}🏛  HISTORICAL COURT ON THE TOPIC:{RESET}")
    print(f"{CYAN}«{topic}»{RESET}\n")
    print(f"{GRAY}{'=' * 60}{RESET}\n")

    current_input = f"Start a debate on the topic: {topic}. State your position briefly."

    for i in range(rounds):
        # Machiavelli's turn
        history_m.append({"role": "user", "content": current_input})
        res_m = ollama.chat(model=model_m, 
                            messages=[{"role": "system", "content": prompts["Machiavelli"]}] + history_m,
                            options=llm_options)
        
        think_m, speech_m = extract_think(res_m["message"]["content"])
        history_m.append({"role": "assistant", "content": speech_m})
        transcript_plain.append(f"Machiavelli: {speech_m}")
        transcript_entries.append({"name": "Machiavelli", "icon": "🦊", "think": think_m, "speech": speech_m})
        
        print_speech("Machiavelli", think_m, speech_m, YELLOW, "🦊")

        # Socrates's turn
        history_s.append({"role": "user", "content": speech_m})
        res_s = ollama.chat(model=model_s, 
                            messages=[{"role": "system", "content": prompts["Socrates"]}] + history_s,
                            options=llm_options)
        
        think_s, speech_s = extract_think(res_s["message"]["content"])
        history_s.append({"role": "assistant", "content": speech_s})
        transcript_plain.append(f"Socrates: {speech_s}")
        transcript_entries.append({"name": "Socrates", "icon": "🏛", "think": think_s, "speech": speech_s})
        
        print_speech("Socrates", think_s, speech_s, CYAN, "🏛")

        current_input = speech_s

    # Judge's verdict
    print(f"{BOLD}⚖️  JUDGE DELIVERING VERDICT...{RESET}\n")
    
    judge_prompt = "You are the Supreme Judge. Analyze the debate. Who won: Socrates or Machiavelli? Answer briefly and strictly in English."
    full_text = "\n".join(transcript_plain)
    
    res_j = ollama.chat(model=model_judge, 
                        messages=[{"role": "system", "content": judge_prompt}, 
                                  {"role": "user", "content": full_text}])
    
    verdict_text = res_j["message"]["content"].strip()
    print(f"{BOLD}VERDICT:{RESET}")
    print(f"{WHITE}{verdict_text}{RESET}\n")

    return {
        "topic": topic,
        "model_m": model_m,
        "model_s": model_s,
        "model_judge": model_judge,
        "transcript_entries": transcript_entries,
        "verdict": verdict_text,
    }

DEFAULT_TOPIC = "What is better for society: total state control or complete anarchy and absence of vertical power structure"

def parse_args():
    parser = argparse.ArgumentParser(description="Run a historical court debate between Socrates and Machiavelli using Ollama models.")
    parser.add_argument(
        "--topic",
        type=str,
        default=DEFAULT_TOPIC,
        help="Debate topic (required or use default).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Number of debate rounds between the two speakers (default: 2).",
    )
    parser.add_argument(
        "--model_m",
        type=str,
        default="llama3:latest",
        help="Ollama model for Machiavelli (default: llama3:latest).",
    )
    parser.add_argument(
        "--model_s",
        type=str,
        default="qwen2.5-coder:7b",
        help="Ollama model for Socrates (default: qwen2.5-coder:7b).",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default="llama3.2:latest",
        help="Ollama model for the judge verdict (default: llama3.2:latest).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"{BOLD}Settings:{RESET}")
    print(f"  Topic:  {args.topic}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Machiavelli model: {args.model_m}")
    print(f"  Socrates model:    {args.model_s}")
    print(f"  Judge model:       {args.judge}")
    print(f"{GRAY}{'—' * 60}{RESET}\n")

    result = start_court(
        model_m=args.model_m,
        model_s=args.model_s,
        model_judge=args.judge,
        topic=args.topic,
        rounds=args.rounds,
    )
    filepath = save_debate_to_md(
        topic=result["topic"],
        model_m=result["model_m"],
        model_s=result["model_s"],
        model_judge=result["model_judge"],
        transcript_entries=result["transcript_entries"],
        verdict=result["verdict"],
    )
    print(f"Debate saved to {filepath}")
