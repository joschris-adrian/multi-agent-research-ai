"""
Supercharge mode - bulk ingest documents into ChromaDB vector memory
without running the full research pipeline.

Fetches documents across multiple query variations of the topic from
both DuckDuckGo web search and arXiv academic papers, deduplicates by
source URL, chunks via the MCP web search server and stores in batches
via the MCP vector store server. Run this before the main pipeline to
give the analyst richer context from the first query.

Usage:
    python scripts/supercharge.py --topic "renewable energy"
    python scripts/supercharge.py --topic "large language models" --max_results 10

Requirements:
    - MCP vector store server running on port 8001
    - MCP web search server running on port 8002
    - MCP arXiv server running on port 8003
"""

import argparse
import httpx
import sys

VECTOR_STORE_URL = "http://localhost:8001"
WEB_SEARCH_URL = "http://localhost:8002"
ARXIV_URL = "http://localhost:8003"


def fetch_documents(topic, max_results=10):
    queries = [
        topic,
        f"{topic} latest trends",
        f"{topic} research and development",
        f"{topic} market statistics",
    ]

    all_documents = []
    seen_sources = set()

    for query in queries:
        try:
            response = httpx.post(
                f"{WEB_SEARCH_URL}/web_search/search",
                json={
                    "query": query,
                    "max_results": max_results,
                    "retries": 3,
                    "delay": 2
                },
                timeout=30.0
            )
            docs = response.json().get("result", [])
            for doc in docs:
                if doc["source"] not in seen_sources:
                    seen_sources.add(doc["source"])
                    all_documents.append(doc)
            print(f"  [web] [{query}] fetched {len(docs)} chunks")
        except Exception as e:
            print(f"  [web] [{query}] failed: {e}")

    try:
        arxiv_response = httpx.post(
            f"{ARXIV_URL}/arxiv/search",
            json={"topic": topic, "max_results": max_results},
            timeout=15.0
        )
        arxiv_docs = arxiv_response.json().get("result", [])
        for doc in arxiv_docs:
            if doc["source"] not in seen_sources:
                seen_sources.add(doc["source"])
                all_documents.append(doc)
        print(f"  [arxiv] fetched {len(arxiv_docs)} papers")
    except Exception as e:
        print(f"  [arxiv] unavailable: {e}")

    return all_documents


def store_documents(documents, batch_size=5):
    total_stored = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        try:
            response = httpx.post(
                f"{VECTOR_STORE_URL}/vector_store/add",
                json={"documents": batch},
                timeout=120.0
            )
            assert response.status_code == 200
            total_stored += len(batch)
            print(f"  stored batch {i // batch_size + 1} ({total_stored}/{len(documents)} chunks)")
        except Exception as e:
            print(f"  failed to store batch {i // batch_size + 1}: {e}")
            print(f"  retrying batch individually...")
            for doc in batch:
                try:
                    httpx.post(
                        f"{VECTOR_STORE_URL}/vector_store/add",
                        json={"documents": [doc]},
                        timeout=120.0
                    )
                    total_stored += 1
                except Exception as e2:
                    print(f"  failed to store individual chunk: {e2}")
    print(f"  stored {total_stored}/{len(documents)} chunks in ChromaDB")

def check_servers():
    for name, url, payload in [
        ("vector store", VECTOR_STORE_URL, ("POST", "/vector_store/search", {"query": "health check"})),
        ("web search", WEB_SEARCH_URL, ("POST", "/web_search/search", {"query": "health check", "max_results": 1, "retries": 1, "delay": 0})),
        ("arxiv", ARXIV_URL, ("POST", "/arxiv/search", {"topic": "health check", "max_results": 1})),
    ]:
        try:
            r = httpx.post(f"{url}{payload[1]}", json=payload[2], timeout=60.0)
            assert r.status_code == 200
            print(f"  {name} server reachable at {url}")
        except Exception as e:
            print(f"  {name} server not reachable at {url}: {e}")
            port_map = {"vector store": "8001", "web search": "8002", "arxiv": "8003"}
            server_map = {"vector store": "vector_store", "web search": "web_search", "arxiv": "arxiv"}
            port = port_map.get(name, "800x")
            server = server_map.get(name, name)
            print(f"  start it with: uvicorn src.mcp.servers.{server}_server:app --port {port} --reload")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Bulk ingest documents into ChromaDB vector memory")
    parser.add_argument("--topic", required=True, help="Topic to research and ingest")
    parser.add_argument("--max_results", type=int, default=5, help="Max results per query (default: 5)")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  Supercharge mode — topic: {args.topic}")
    print(f"{'='*55}")

    print("\n  Checking servers...")
    check_servers()

    print(f"\n  Fetching documents for: {args.topic}")
    documents = fetch_documents(args.topic, args.max_results)

    if not documents:
        print("  no documents fetched — check MCP web search server")
        sys.exit(1)

    print(f"\n  Total unique chunks fetched: {len(documents)}")

    print("\n  Storing in ChromaDB...")
    store_documents(documents)

    print(f"\n  Done. ChromaDB now has {len(documents)} chunks on '{args.topic}'")
    print("  Run the pipeline on this topic to benefit from pre-populated memory.\n")


if __name__ == "__main__":
    main()