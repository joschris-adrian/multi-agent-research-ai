import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

SERVER_URLS = {
    "vector_store": "http://localhost:8011/mcp",
    "web_search": "http://localhost:8012/mcp",
    "arxiv": "http://localhost:8013/mcp",
}

TOOL_NAMES = {
    ("vector_store", "search"): "mcp_search",
    ("vector_store", "add"): "mcp_add",
    ("web_search", "search"): "mcp_web_search",
    ("arxiv", "search"): "mcp_arxiv_search",
}

TIMEOUTS = {
    "vector_store": 60.0,
    "web_search": 10.0,
    "arxiv": 10.0,
}


class MCPClient:
    def call_tool(self, server: str, tool: str, arguments: dict):
        url = SERVER_URLS.get(server)
        if url is None:
            print(f"[mcp_client] unknown server: {server}")
            return []
        mcp_tool_name = TOOL_NAMES.get((server, tool))
        if mcp_tool_name is None:
            print(f"[mcp_client] unknown tool: {server}/{tool}")
            return []
        try:
            return asyncio.run(self._call(url, mcp_tool_name, arguments))
        except Exception as e:
            print(f"[mcp_client] call failed for {server}/{tool}: {e}")
            return []

    async def _call(self, url: str, tool_name: str, arguments: dict):
        async with streamablehttp_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                for block in result.content:
                    if hasattr(block, "text"):
                        try:
                            import ast
                            return ast.literal_eval(block.text)
                        except Exception:
                            return block.text
        return []