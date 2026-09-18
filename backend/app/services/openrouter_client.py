import json
import os
import urllib.request


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openrouter/free"


def ask_openrouter(prompt: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    payload = {
    "model": OPENROUTER_MODEL,
    "messages": [
        {
            "role": "user",
            "content": prompt,
        }
    ],
    "temperature": 0,
    "max_tokens": 1200,
}
    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        OPENROUTER_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))

        content = result["choices"][0]["message"].get("content")

    if not content:
        raise ValueError("OpenRouter returned an empty response.")

    return content.strip()
