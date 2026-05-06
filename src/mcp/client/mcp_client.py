import httpx

SERVER_PORTS = {
    "vector_store": "http://localhost:8001",
    "web_search": "http://localhost:8002",
}

class MCPClient:
    def call_tool(self, server, tool, arguments):
        base_url = SERVER_PORTS.get(server, "http://localhost:8001")
        timeout = 60.0 if server == "vector_store" else 10.0
        try:
            response = httpx.post(
                f"{base_url}/{server}/{tool}",
                json=arguments,
                timeout=timeout
            )
            data = response.json()
            return data.get("result", []) or []
        except Exception:
            return []