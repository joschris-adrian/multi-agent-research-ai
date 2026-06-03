import uvicorn
from mcp.server.fastmcp import FastMCP
from src.memory.vector_store import VectorStore

store = VectorStore()
mcp = FastMCP("vector-store")

@mcp.tool()
def mcp_search(query: str, top_k: int = 5) -> str:
    """Semantically search stored document chunks using ChromaDB with optional cross-encoder reranking."""
    try:
        results = store.search(query, top_k=top_k)
        return str(results)
    except Exception as e:
        raise ValueError(str(e))

@mcp.tool()
def mcp_add(documents: list) -> str:
    """Add document chunks to the ChromaDB vector store."""
    try:
        store.add_documents(documents)
        return "ok"
    except Exception as e:
        raise ValueError(str(e))

if __name__ == "__main__":
    uvicorn.run(mcp.streamable_http_app(), host="0.0.0.0", port=8011)