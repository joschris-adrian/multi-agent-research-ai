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
        system_prompt = f"""You are an AI agent.
Role: {self.role}
Goal: {self.goal}

Respond clearly and concisely."""

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
