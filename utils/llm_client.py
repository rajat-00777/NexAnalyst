"""
This file handles the connection to the AI model (Groq, free tier —
see https://console.groq.com, no credit card needed).

Free tier limits are roughly 30 requests per minute and 14,400 per day
as of 2026 — check console.groq.com for the current numbers.

How to use it elsewhere in the project:
    from utils.llm_client import get_llm
    llm = get_llm()
    response = llm.invoke("Explain this in one sentence: ...")
"""



import os
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_MODEL = "openai/gpt-oss-120b"


def get_llm(temperature: float = 0.2):
    """
    Returns a configured ChatGroq instance.

    Kept as a thin factory function (not a module-level singleton) so agents
    can request different temperatures if needed later (e.g. low temp for
    validation, slightly higher for recommendations).
    """
    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY not set. Copy .env.example to .env and add your "
            "free API key from https://console.groq.com"
        )

    model = os.getenv("GROQ_MODEL", _DEFAULT_MODEL)

    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=api_key,
    )


if __name__ == "__main__":
    # Quick manual test: python utils/llm_client.py
    llm = get_llm()
    result = llm.invoke("Reply with exactly the word: OK")
    print(result.content)