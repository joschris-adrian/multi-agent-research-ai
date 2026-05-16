import os
import shutil
import httpx
from ..workflow.agent_pipeline import MultiAgentResearchSystem
from .baseline import SingleAgentBaseline
from .evaluator import Evaluator

def main():
    for name, url, payload in [
        ("vector store MCP", "http://localhost:8001/vector_store/search", {"query": "health"}),
        ("web search MCP", "http://localhost:8002/web_search/search", {"query": "health", "max_results": 1, "retries": 1, "delay": 0}),
        ("arxiv MCP", "http://localhost:8003/arxiv/search", {"topic": "health", "max_results": 1}),
    ]:
        try:
            r = httpx.post(url, json=payload, timeout=60.0)
            assert r.status_code == 200
            print(f"  {name} reachable")
        except Exception as e:
            print(f"  {name} not reachable: {e}")
            print(f"  Start all MCP servers before running evaluation.")
            return
    
    question = "What are the latest trends in renewable energy?"

    chroma_path = "./chroma_db"
    backup_path = "./chroma_db_backup"

    if os.path.exists(chroma_path):
        shutil.copytree(chroma_path, backup_path, dirs_exist_ok=True)
        print("[evaluate] backed up chroma_db - evaluation will add to existing store")

    try:
        multi_agent = MultiAgentResearchSystem()
        baseline = SingleAgentBaseline()
        evaluator = Evaluator()
        
        print("\n--- Running Multi-Agent System (with RAG + MCP) ---")
        multi_result = multi_agent.run(question)
        multi_answer = multi_result["report"]
        print(f"  entities found: {list(multi_result['entities'].keys())}")
        print(f"  documents retrieved: {len(multi_result['documents'])}")
    
        print("\n--- Running Single-Agent Baseline ---")
        single_answer = baseline.run(question)

        print("\n--- Evaluating Multi-Agent Output ---")
        multi_eval = evaluator.evaluate(question, multi_answer)
        print(multi_eval)

        print("\n--- Evaluating Single-Agent Output ---")
        single_eval = evaluator.evaluate(question, single_answer)
        print(single_eval)

        print("\n========== RESULTS ==========\n")
        print("MULTI-AGENT OUTPUT:\n")
        print(multi_answer)
        print("\nEVALUATION:\n")
        print(multi_eval)
        print("\n-----------------------------\n")
        print("SINGLE-AGENT OUTPUT:\n")
        print(single_answer)
        print("\nEVALUATION:\n")
        print(single_eval)

    finally:
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
            print("[evaluate] removed chroma_db backup")

if __name__ == "__main__":
    main()