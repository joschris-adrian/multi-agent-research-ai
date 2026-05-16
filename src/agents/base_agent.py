import os
import requests
from ..mcp.client.mcp_client import MCPClient

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


class BaseAgent:
    def __init__(self, role, goal, model="llama3.2", temperature=0.7, max_tokens=500):
        self.role = role
        self.goal = goal
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.mcp = MCPClient()

    def run(self, prompt):
        system_prompt = f"""You are a specialised AI agent in a multi-agent research pipeline.

Role: {self.role}
Goal: {self.goal}

Guidelines:
- Focus strictly on your role - do not attempt tasks outside your goal
- Be precise and factual - avoid speculation unless explicitly asked
- Structure your output clearly so the next agent in the pipeline can use it
- If information is unavailable or unclear, say so rather than guessing"""

        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": self.model,
                "system": system_prompt,
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
