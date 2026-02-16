import ollama

def start_dialogue(model_a, model_b, topic, rounds=5):
    system_prompt = (
        f"You are a debate participant. The topic is: {topic}. "
        "Answer briefly (1-2 paragraphs), clearly and always end your thought with punctuation."
    )
    # First message to start the dialogue (so the opening is never empty)
    current_message = f"The debate topic is: '{topic}'. What do you think about it? Reply briefly."

    print(f"--- Starting discussion on topic: {topic} ---\n")

    for i in range(rounds):
        # Model A responds
        response_a = ollama.chat(
            model=model_a,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": current_message},
            ],
            options={"num_predict": 80, "temperature": 0.2},
        )
        msg_a = response_a["message"]["content"]
        print(f"🤖 {model_a.upper()}: {msg_a}\n")
        print("-" * 30)

        # Model B gets A's reply; topic is in system prompt
        response_b = ollama.chat(
            model=model_b,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg_a},
            ],
            options={"num_predict": 150, "temperature": 0.9},
        )
        msg_b = response_b["message"]["content"]
        print(f"🍦 {model_b.upper()}: {msg_b}\n")
        print("-" * 30)

        # Update message for next round
        current_message = msg_b

if __name__ == "__main__":

    topic = 'Does humanity need artificial intelligence?'
    start_dialogue(
        model_a='llama3.2', 
        model_b='qwen2.5-coder:7b', 
        topic=topic, 
        rounds=3
    )