import os
import time
import requests
import httpx
from ..mcp.client.mcp_client import MCPClient
from dotenv import load_dotenv
load_dotenv()
def _get_env(key, default=""):
    return os.environ.get(key, default)

OLLAMA_HOST = _get_env("OLLAMA_HOST", "http://localhost:11434")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

class BaseAgent:
    def __init__(self, role, goal, model="llama3.2", gemini_model="gemini-2.5-flash", temperature=0.7, max_tokens=500):
        self.role = role
        self.goal = goal
        self.model = model
        self.gemini_model = gemini_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.mcp = MCPClient()

    def _build_system_prompt(self):
        return f"""You are a specialised AI agent in a multi-agent research pipeline.
Role: {self.role}
Goal: {self.goal}
Guidelines:
- Focus strictly on your role - do not attempt tasks outside your goal
- Be precise and factual - avoid speculation unless explicitly asked
- Structure your output clearly so the next agent in the pipeline can use it
- If information is unavailable or unclear, say so rather than guessing"""

    def run(self, prompt):
        if os.environ.get("LLM_PROVIDER", "ollama").lower() == "gemini":
            return self._call_gemini(prompt)
        return self._call_ollama(prompt)

    def _call_ollama(self, prompt):
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        response = requests.post(
            f"{ollama_host}/api/generate",
            json={
                "model": self.model,
                "system": self._build_system_prompt(),
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "stop": ["\n\n\n"],
                }
            },
        )
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"[base_agent] Ollama error: {data['error']}")
        return data["response"]

    def _call_gemini(self, prompt):
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. "
                "Export your Google AI Studio key before using LLM_PROVIDER=gemini."
            )
        url = (
            f"{GEMINI_BASE_URL}/models/{self.gemini_model}"
            f":generateContent?key={api_key}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": self._build_system_prompt()}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature},
        }
        try:
            response = httpx.post(url, json=payload, timeout=120)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Gemini request failed: {exc}"
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(
                "Could not reach Gemini API. Check your network connection."
            ) from exc
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]