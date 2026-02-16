import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def get_support_message(user_text, label, results):

    # Safety: avoid crash if key missing
    if not os.environ.get("GROQ_API_KEY"):
        return "Support assistant unavailable."

    system_prompt = """
You are a calm mental-health support assistant.

Rules:
- Provide supportive, non-medical guidance.
- If suicidal signals exist, encourage seeking real help.
- Do NOT diagnose.
- Keep response short (4-6 sentences).
- Tone: calm, validating, grounding.
"""

    user_prompt = f"""
User text:
{user_text}

Model prediction:
Primary label = {label}
Probabilities = {results}

Generate a supportive message appropriate to risk level.
"""

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",   # better emotional reasoning
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.4,
            max_tokens=200
        )

        return response.choices[0].message.content

    except Exception:
        return "Support response temporarily unavailable."
