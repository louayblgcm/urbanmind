import logging

import requests

from config.config import GROQ_API_KEY, GROQ_API_URL, GROQ_MODEL, LLM_TIMEOUT_SECONDS


LOGGER = logging.getLogger(__name__)
UNAVAILABLE_MESSAGE = "Urban intelligence generation temporarily unavailable."


def groq_is_configured() -> bool:
    return bool(
        GROQ_API_URL
        and GROQ_API_KEY
        and GROQ_API_KEY != "YOUR_GROQ_API_KEY_HERE"
    )


def generate_groq_response(prompt: str, max_tokens: int = 180) -> str:
    """Generate text through Groq's OpenAI-compatible chat-completions API."""
    if not groq_is_configured():
        return UNAVAILABLE_MESSAGE

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the customer-facing voice of an urban intelligence product. "
                    "Write in plain language, sound calm and helpful, and avoid technical jargon. "
                    "Prefer rounded numbers, practical interpretation, and short readable sentences. "
                    "Do not invent facts absent from the data and do not present probabilities as certainties."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        content = (
            response.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return content.strip() or UNAVAILABLE_MESSAGE
    except (requests.RequestException, ValueError, KeyError, TypeError):
        LOGGER.exception("Groq request failed")
        return UNAVAILABLE_MESSAGE
