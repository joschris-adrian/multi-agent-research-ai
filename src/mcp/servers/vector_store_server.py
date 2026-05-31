from fastapi import FastAPI
from src.memory.vector_store import VectorStore
from pydantic import BaseModel
from typing import List, Dict, Any
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, CallToolResult

app = FastAPI()
store = VectorStore()
mcp = FastMCP("vector-store")

@app.on_event("startup")
async def startup():
    print("[vector_store_server] pre-warming reranker...")
    store.search("warmup", top_k=1)
    print("[vector_store_server] reranker ready")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    rerank: bool = True

@app.post("/vector_store/search")
def search(request: SearchRequest):
    results = store.search(request.query, top_k=request.top_k)
    return {"result": results, "reranked": store.reranker is not None}

class AddRequest(BaseModel):
    documents: List[Dict[str, Any]]

@app.post("/vector_store/add")
def add(request: AddRequest):
    store.add_documents(request.documents)
    return {"result": "ok"}

@app.post("/vector_store/search/compare")
def search_compare(request: SearchRequest):
    original_reranker = store.reranker
    store.reranker = None
    cosine_results = store.search(request.query, top_k=request.top_k)
    store.reranker = original_reranker
    reranked_results = store.search(request.query, top_k=request.top_k)
    return {
        "query": request.query,
        "cosine_only": [
            {"content": r["content"][:100], "score": round(r["score"], 3)}
            for r in cosine_results
        ],
        "reranked": [
            {"content": r["content"][:100], "rerank_score": round(r.get("rerank_score", 0), 3), "score": round(r["score"], 3)}
            for r in reranked_results
        ]
    }

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
    
app.mount("/mcp", mcp.sse_app())