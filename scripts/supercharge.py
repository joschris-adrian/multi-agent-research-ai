"""
Supercharge mode — bulk ingest documents into ChromaDB vector memory
without running the full research pipeline.

Usage:
    python scripts/supercharge.py --topic "renewable energy" --max_results 10
    python scripts/supercharge.py --topic "large language models" --source both
    python scripts/supercharge.py --topic "battery storage" --source web

Sources:
    web   - DuckDuckGo web search (default)
    both  - DuckDuckGo + additional query variations

Requirements:
    - MCP vector store server running on port 8001
    - MCP web search server running on port 8002
"""

import argparse
import httpx
import sys

VECTOR_STORE_URL = "http://localhost:8001"
WEB_SEARCH_URL = "http://localhost:8002"


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
            print(f"  [{query}] fetched {len(docs)} chunks")
        except Exception as e:
            print(f"  [{query}] failed: {e}")

    return all_documents


def store_documents(documents):
    try:
        response = httpx.post(
            f"{VECTOR_STORE_URL}/vector_store/add",
            json={"documents": documents},
            timeout=60.0
        )
        assert response.status_code == 200
        print(f"  stored {len(documents)} chunks in ChromaDB")
    except Exception as e:
        print(f"  failed to store documents: {e}")
        sys.exit(1)


def check_servers():
    for name, url, payload in [
        ("vector store", VECTOR_STORE_URL, ("POST", "/vector_store/search", {"query": "health check"})),
        ("web search", WEB_SEARCH_URL, ("POST", "/web_search/search", {"query": "health check", "max_results": 1, "retries": 1, "delay": 0})),
    ]:
        try:
            r = httpx.post(f"{url}{payload[1]}", json=payload[2], timeout=60.0)
            assert r.status_code == 200
            print(f"  {name} server reachable at {url}")
        except Exception as e:
            print(f"  {name} server not reachable at {url}: {e}")
            print(f"  start it with: uvicorn src.mcp.servers.{'vector_store' if 'vector' in name else 'web_search'}_server:app --port {'8001' if 'vector' in name else '8002'} --reload")
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