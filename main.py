import ollama
import re

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

def start_court(model_m, model_s, model_judge, topic, rounds=3):
    prompts = {
        "Socrates": "You are Socrates. Speak English. Use Socratic method: ask short, probing questions. Be humble but ironic.",
        "Machiavelli": "You are Machiavelli. Speak English. You are a cynical pragmatist. Defend state interest and order at any cost."
    }

    history_m = []
    history_s = []
    transcript = []

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
        transcript.append(f"Machiavelli: {speech_m}")
        
        print_speech("Machiavelli", think_m, speech_m, YELLOW, "🦊")

        # Socrates's turn
        history_s.append({"role": "user", "content": speech_m})
        res_s = ollama.chat(model=model_s, 
                            messages=[{"role": "system", "content": prompts["Socrates"]}] + history_s,
                            options=llm_options)
        
        think_s, speech_s = extract_think(res_s["message"]["content"])
        history_s.append({"role": "assistant", "content": speech_s})
        transcript.append(f"Socrates: {speech_s}")
        
        print_speech("Socrates", think_s, speech_s, CYAN, "🏛")

        current_input = speech_s

    # Judge's verdict
    print(f"{BOLD}⚖️  JUDGE DELIVERING VERDICT...{RESET}\n")
    
    judge_prompt = "You are the Supreme Judge. Analyze the debate. Who won: Socrates or Machiavelli? Answer briefly and strictly in English."
    full_text = "\n".join(transcript)
    
    res_j = ollama.chat(model=model_judge, 
                        messages=[{"role": "system", "content": judge_prompt}, 
                                  {"role": "user", "content": full_text}])
    
    print(f"{BOLD}VERDICT:{RESET}")
    print(f"{WHITE}{res_j['message']['content'].strip()}{RESET}\n")

if __name__ == "__main__":
    start_court(
        model_m='llama3:latest',
        model_s='qwen2.5-coder:7b',
        model_judge='llama3.2:latest',
        topic='What is better for society: total state control or complete anarchy and absence of vertical power structure',
        rounds=2
    )
