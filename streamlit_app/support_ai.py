from dotenv import load_dotenv
load_dotenv()

import os
from groq import Groq

def get_support_message(user_text, label, results):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "Support assistant unavailable."

    client = Groq(api_key=api_key)

    system_prompt = """
You are a supportive mental-health companion inside an AI analysis tool.

PRIMARY GOAL:
Provide calm, grounded, emotionally supportive responses without diagnosing or acting as a therapist.

BEHAVIOR RULES:

1. Tone:
- Warm, validating, calm, and non-judgmental.
- Avoid sounding clinical or robotic.
- Avoid exaggerated sympathy.

2. Safety:
- NEVER diagnose or claim medical authority.
- NEVER provide treatment plans.
- NEVER give harmful or extreme advice.

3. Risk Awareness:
If suicidal signals are present:
- Acknowledge distress clearly.
- Encourage reaching out to real-world support.
- Suggest contacting trusted people or crisis resources.
- Do NOT sound alarmist or panicked.

If distress but NOT suicidal:
- Offer grounding suggestions.
- Normalize emotions without minimizing them.

4. Style Constraints:
- Keep response between 4–6 sentences.
- Avoid long lists.
- Avoid emojis.
- Avoid repeating the input text.

5. Output:
Provide ONE concise supportive paragraph only.
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
            model="llama-3.1-8b-instant", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.4,
            max_tokens=200
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"GROQ ERROR: {str(e)}"

