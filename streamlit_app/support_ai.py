import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_support_message(user_text, label, results):

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

    response = client.chat.completions.create(
        model="gpt-5.2-mini",
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt}
        ]
    )

    return response.choices[0].message.content
