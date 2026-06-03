import time
import uvicorn
from mcp.server.fastmcp import FastMCP
from ddgs import DDGS

mcp = FastMCP("web-search")

@mcp.tool()
def mcp_web_search(query: str, max_results: int = 3, retries: int = 3, delay: int = 2) -> str:
    """Search the web using DuckDuckGo and return chunked document results."""
    for attempt in range(retries):
        try:
            documents = []
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_results)
                for r in results:
                    body = r["body"]
                    chunk_size = 200
                    overlap = 50
                    chunks = [
                        body[i:i + chunk_size]
                        for i in range(0, len(body), chunk_size - overlap)
                        if body[i:i + chunk_size].strip()
                    ]
                    for chunk in chunks:
                        documents.append({
                            "title": r["title"],
                            "content": chunk,
                            "source": r["href"]
                        })
            if documents:
                return str(documents)
        except Exception as e:
            print(f"[web_search_mcp_server] attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    return str([])

if __name__ == "__main__":
    uvicorn.run(mcp.streamable_http_app(), host="0.0.0.0", port=8012)