import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP
import xml.etree.ElementTree as ET
import urllib.parse

app = FastAPI()
mcp = FastMCP("arxiv")

ARXIV_API_URL = "https://export.arxiv.org/api/query"


class ArxivRequest(BaseModel):
    topic: str
    max_results: int = 5


def parse_arxiv_response(xml_text: str) -> list:
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    documents = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns)
        summary = entry.find("atom:summary", ns)
        link = entry.find("atom:id", ns)
        published = entry.find("atom:published", ns)
        if title is not None and summary is not None:
            documents.append({
                "title": title.text.strip(),
                "content": summary.text.strip()[:500],
                "source": link.text.strip() if link is not None else "",
                "published": published.text.strip()[:10] if published is not None else ""
            })
    return documents


def _fetch_arxiv(topic: str, max_results: int) -> list:
    topic = topic.strip()
    topic_quoted = f'"{topic}"'
    search_query = f"ti:{topic_quoted} OR abs:{topic_quoted}"
    encoded_query = urllib.parse.quote(search_query)
    url = (
        f"{ARXIV_API_URL}"
        f"?search_query={encoded_query}"
        f"&start=0"
        f"&max_results={max_results}"
        f"&sortBy=submittedDate"
        f"&sortOrder=descending"
    )
    print(f"[arxiv_server] requesting: {url}")
    response = httpx.get(
        url,
        timeout=30.0,
        follow_redirects=True,
        headers={"User-Agent": "multi-agent-research-assistant/1.0 (research project; python/httpx)"}
    )
    print(f"[arxiv_server] status: {response.status_code}")
    return parse_arxiv_response(response.text)


@app.post("/arxiv/search")
def search(request: ArxivRequest):
    try:
        documents = _fetch_arxiv(request.topic, request.max_results)
        return {"result": documents}
    except Exception as e:
        print(f"[arxiv_server] search failed: {e}")
        return {"result": []}


@mcp.tool()
def mcp_arxiv_search(topic: str, max_results: int = 5) -> str:
    """Search arXiv for recent academic papers sorted by submission date."""
    try:
        documents = _fetch_arxiv(topic, max_results)
        return str(documents)
    except Exception as e:
        print(f"[arxiv_server] mcp search failed: {e}")
        return str([])


app.mount("/mcp", mcp.sse_app())