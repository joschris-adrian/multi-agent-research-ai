from fastapi import FastAPI
from src.memory.vector_store import VectorStore
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI()
store = VectorStore()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/vector_store/search")
def search(request: SearchRequest):
    results = store.search(request.query, top_k=request.top_k)
    return {"result": results}


class AddRequest(BaseModel):
    documents: List[Dict[str, Any]]

@app.post("/vector_store/add")
def add(request: AddRequest):
    store.add_documents(request.documents)
    return {"result": "ok"}