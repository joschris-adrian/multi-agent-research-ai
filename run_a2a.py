"""
A2A pipeline CLI — runs the dynamic agent-to-agent research pipeline.

Usage:
    python run_a2a.py
    python run_a2a.py --question "What are the latest trends in AI Agents?"

Requirements:
    - ollama serve
    - uvicorn src.mcp.servers.vector_store_server:app --port 8001 --reload
    - uvicorn src.mcp.servers.web_search_server:app --port 8002 --reload
    - uvicorn src.mcp.servers.arxiv_server:app --port 8003 --reload
    - uvicorn src.a2a.agent_server:app --port 8004 --reload
"""

import os
import argparse
import requests
import sys

def check_servers():
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    servers = []
    if provider != "gemini":
        servers.append(("Ollama", "http://localhost:11434", "GET"))
    servers += [
        ("Vector store MCP", "http://localhost:8001/vector_store/search", "POST"),
        ("Web search MCP", "http://localhost:8002/web_search/search", "POST"),
        ("arXiv MCP", "http://localhost:8003/arxiv/search", "POST"),
        ("A2A agent server", "http://localhost:8004/agent/planner", "POST"),
    ]

    payloads = {
        "http://localhost:8001/vector_store/search": {"query": "health"},
        "http://localhost:8002/web_search/search": {"query": "health", "max_results": 1, "retries": 1, "delay": 0},
        "http://localhost:8003/arxiv/search": {"topic": "health", "max_results": 1},
        "http://localhost:8004/agent/planner": {"question": "health check"},
    }

    print("\nChecking servers...")
    all_ok = True
    for name, url, method in servers:
        try:
            if method == "GET":
                r = requests.get(url, timeout=5)
            else:
                r = requests.post(url, json=payloads.get(url, {}), timeout=120)
            assert r.status_code == 200
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name} — {e}")
            all_ok = False

    if not all_ok:
        print("\nSome servers are not running. Start them and try again.")
        sys.exit(1)

    print("  All servers ready.\n")


def check_provider():
    from dotenv import load_dotenv
    load_dotenv()
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    print(f"LLM provider: {provider}")
    if provider == "gemini":
        if not os.environ.get("GEMINI_API_KEY", ""):
            print("Error: GEMINI_API_KEY is not set. Add it to your .env file.")
            sys.exit(1)
        print("Note: 4-second inter-agent delay active (Gemini free tier limit).")
    else:
        print("Note: Ollama must be running locally (ollama serve).")


def main():
    parser = argparse.ArgumentParser(
        description="Run the A2A dynamic research pipeline"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Research question to investigate"
    )
    args = parser.parse_args()

    question = args.question
    if not question:
        print("Multi-Agent Research Assistant (A2A mode)")
        print("=" * 50)
        question = input("Enter your research question: ").strip()
        if not question:
            print("No question provided. Exiting.")
            sys.exit(1)

    check_provider()
    check_servers()

    from src.a2a.a2a_pipeline import A2AResearchSystem
    system = A2AResearchSystem()
    result = system.run(question)

    print("\n" + "=" * 55)
    print("  FINAL REPORT")
    print("=" * 55)
    print(result["report"])

    print("\n" + "=" * 55)
    print("  CRITIC FEEDBACK")
    print("=" * 55)
    print(result["critic_feedback"])

    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print(f"  Documents retrieved : {len(result['documents'])}")
    companies = result['entities'].get('companies', [])
    trends = result['entities'].get('trends', [])
    technologies = result['entities'].get('technologies', [])
    print(f"  Companies identified : {', '.join(companies) if companies else 'none'}")
    print(f"  Trends identified    : {', '.join(trends) if trends else 'none'}")
    print(f"  Technologies found   : {', '.join(technologies) if technologies else 'none'}")


if __name__ == "__main__":
    main()