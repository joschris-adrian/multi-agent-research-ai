import os
from dotenv import load_dotenv
import httpx

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY", "")
model = "gemini-2.5-flash"

if not api_key:
    print("GEMINI_API_KEY is not set in .env")
    exit(1)

print(f"Testing Gemini connection...")
print(f"Model: {model}")
print(f"Key:   {api_key[:8]}{'*' * (len(api_key) - 8)}")

url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

payload = {
    "contents": [{"role": "user", "parts": [{"text": "Reply with one word: OK"}]}],
    "generationConfig": {"temperature": 0.1},
}


try:
    response = httpx.post(url, json=payload, timeout=15)
    response.raise_for_status()
    text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    print(f"Response: {text.strip()}")
    print("Connection successful.")
except httpx.HTTPStatusError as e:
    print(f"HTTP error {e.response.status_code}: {e.response.text}")
except httpx.RequestError as e:
    print(f"Network error: {e}")