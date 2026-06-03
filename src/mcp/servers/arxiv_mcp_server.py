import uvicorn
from mcp.server.fastmcp import FastMCP
from src.mcp.servers.arxiv_server import _fetch_arxiv

mcp = FastMCP("arxiv")

@mcp.tool()
def mcp_arxiv_search(topic: str, max_results: int = 5) -> str:
    """Search arXiv for recent academic papers sorted by submission date."""
    try:
        documents = _fetch_arxiv(topic, max_results)
        return str(documents)
    except Exception as e:
        print(f"[arxiv_mcp_server] search failed: {e}")
        return str([])

if __name__ == "__main__":
    uvicorn.run(mcp.streamable_http_app(), host="0.0.0.0", port=8013)