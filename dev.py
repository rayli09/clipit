from src.prompts import build_prompt


if __name__ == "__main__":
    q = "What is the capital of France?"
    prompt = build_prompt(q)
    print(prompt)
