import os
import httpx
from dotenv import set_key, dotenv_values

ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")


def load_provider_config():
    """Read current provider settings from .env."""
    values = dotenv_values(ENV_PATH)
    return {
        "provider": values.get("LLM_PROVIDER", "ollama").lower(),
        "api_key": values.get("GEMINI_API_KEY", ""),
    }


def save_provider_config(provider, api_key=""):
    """Write provider settings back to .env."""
    set_key(ENV_PATH, "LLM_PROVIDER", provider)
    if provider == "gemini" and api_key:
        set_key(ENV_PATH, "GEMINI_API_KEY", api_key)


def validate_gemini_key(api_key, model="gemini-2.5-flash"):
    """
    Make a minimal test call to the Gemini API.
    Returns (True, "") on success or (False, error_message) on failure.
    """
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "Reply with one word: OK"}]}],
        "generationConfig": {"temperature": 0.1},
    }
    try:
        response = httpx.post(url, json=payload, timeout=15)
        response.raise_for_status()
        return True, ""
    except httpx.HTTPStatusError as e:
        return False, f"HTTP {e.response.status_code}: {e.response.text[:200]}"
    except httpx.RequestError as e:
        return False, f"Network error: {e}"