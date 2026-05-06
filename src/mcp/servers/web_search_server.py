import time
from fastapi import FastAPI
from ddgs import DDGS
from pydantic import BaseModel

app = FastAPI()


class SearchRequest(BaseModel):
    query: str
    max_results: int = 3
    retries: int = 3
    delay: int = 2

@app.post("/web_search/search")
def search(request: SearchRequest):
    query = request.query
    max_results = request.max_results
    retries = request.retries
    delay = request.delay
    for attempt in range(retries):
        try:
            documents = []
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_results)
                for r in results:
                    documents.append({
                        "title": r["title"],
                        "content": r["body"][:500],
                        "source": r["href"]
                    })
            if documents:
                return {"result": documents}
        except Exception as e:
            print(f"[web_search_server] attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    return {"result": []}