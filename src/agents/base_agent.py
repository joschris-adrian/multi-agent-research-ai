import os
import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


class BaseAgent:
    def __init__(self, role, goal, model="llama3.2"):
        self.role = role
        self.goal = goal
        self.model = model

    def run(self, prompt):
        system_prompt = f"""You are an AI agent.
Role: {self.role}
Goal: {self.goal}

Respond clearly and concisely."""

        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": self.model,
                "prompt": system_prompt + "\n\n" + prompt,
                "stream": False,
            },
        )
        return response.json()["response"]
