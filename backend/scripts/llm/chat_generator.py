from scripts.llm.groq_client import generate_groq_response


def generate_chat_response(area_intelligence, user_question):
    prompt = f"""
Answer the user's question using only the area intelligence below. Be concise,
practical, consumer-friendly, and explicit when the data cannot answer it.
Never present model probability as certainty.

AREA INTELLIGENCE
{area_intelligence}

USER QUESTION
{user_question}
"""
    return generate_groq_response(prompt)
