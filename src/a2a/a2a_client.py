"""
A2A client — allows any agent to call any other agent by name
via HTTP rather than direct Python import.
"""

import httpx

AGENT_BASE_URL = "http://localhost:8004"

AGENT_ENDPOINTS = {
    "planner": "/agent/planner",
    "researcher": "/agent/researcher",
    "analyst": "/agent/analyst",
    "writer": "/agent/writer",
    "critic": "/agent/critic",
    "graph_builder": "/agent/graph_builder",
}


class A2AClient:
    def __init__(self, base_url=AGENT_BASE_URL):
        self.base_url = base_url

    def call_agent(self, agent_name, payload):
        endpoint = AGENT_ENDPOINTS.get(agent_name)
        if not endpoint:
            raise ValueError(f"Unknown agent: {agent_name}")
        try:
            response = httpx.post(
                f"{self.base_url}{endpoint}",
                json=payload,
                timeout=300.0
            )
            data = response.json()
            if "error" in data:
                print(f"[a2a] {agent_name} returned error: {data['error']}")
            return data.get("result")
        except Exception as e:
            print(f"[a2a] failed to call {agent_name}: {e}")
            return None